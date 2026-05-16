# AlloyForge — Extension Guide

This document is for researchers extending AlloyForge with Claude Code.
It explains *where to plug in* for the most common extensions, and which
sub-agent to invoke for each.

---

## 1. Adding a new alloy system

**Goal:** apply AlloyForge to a system not represented in your training data.

**Steps:**
1. Drop your CSV into `data/`.
2. Make sure every element column matches a key in `ELEMENT_PROPERTIES`
   (`core/alloyforge/data_pipeline.py`). If you have elements not in the table
   (e.g. Sc, Y, Lu, Hf for rare-earth or refractory work), add them with
   atomic number, mass, radius (Pyykkö covalent), electronegativity
   (Pauling), density, melting point, and VEC.
3. Run `/run-prediction data/your.csv your_targets`.

**Who to ask:** `@forward-modeler` for feature-engineering edits.

---

## 2. Adding a new property target

**Goal:** train on a property the current pipeline doesn't model
(e.g. fatigue life, thermal conductivity, corrosion potential).

**Considerations:**
- Property heteroskedasticity: fatigue and corrosion are noisier than
  hardness/strength. Consider log-transform; the per-target standardization
  in `ForwardModel` handles linear shifts but not nonlinearity.
- Sample efficiency: if you have < 30 samples, treat the target as
  exploratory and report raw σ rather than conformal intervals.

**Who to ask:** `@forward-modeler` for the head, `@validator` for the
calibration treatment.

---

## 3. Adding a new constraint

**Goal:** encode a physical or process rule that the current
`default_checker` doesn't capture.

**Where:** add a function in `core/alloyforge/feasibility.py` that returns a
`Constraint`. Register it in `default_checker()` if it's broadly useful,
or keep it as an opt-in checker for your project.

**Examples:**
- "No more than 1 wt% O" — composition bound.
- "VED in [50, 90] J/mm³" — process bound.
- "(Cr+Mo+V)/C ratio in [4, 8]" — carbide-balance rule for tool steels.
- "ΔH_mix > -15 kJ/mol" — intermetallic-avoidance heuristic.

**Who to ask:** `@thermodynamics-expert`.

---

## 4. Swapping the inverse-design algorithm

**Goal:** replace or augment NSGA-II with a different MO algorithm
(BoTorch qEHVI, MOEA/D, CMA-ES with constraint handling).

**Where:** add a new method on `InverseDesigner` named after the algorithm.
Keep `run_nsga2` in place — it's the dependable baseline. Match the return
contract: a DataFrame with `composition`, `process`, `<target>_mean`,
`<target>_std`, `score`, `feasible`.

**Who to ask:** `@inverse-designer`.

---

## 5. Adding CALPHAD coupling

**Goal:** use thermodynamic phase predictions to constrain or score
candidates.

**Approach:**
- Wrap pycalphad behind an optional import in `core/alloyforge/feasibility.py`.
- Cache phase predictions keyed by composition hash — CALPHAD is expensive.
- Use phase predictions as **filters** (hard constraints on phase fractions)
  or as **features** (input to the forward model).

**Who to ask:** `@thermodynamics-expert`.

---

## 6. Connecting to an experimental database

**Goal:** auto-pull new experimental results into the training set as
they're produced.

**Approach:**
- Add a `data/connectors/` module with a `BaseConnector` interface
  (`fetch_since(timestamp) -> pd.DataFrame`).
- Implement per-database connectors (e.g. AiiDA, Materials Project, LIMS).
- Add a `/api/retrain` endpoint that pulls deltas + re-runs `run-prediction`.

**Who to ask:** the user — this is project-specific scaffolding.

---

## 7. Custom LLM prompts

**Goal:** change how Claude interprets predictions or reviews designs
for your alloy class.

**Where:** edit `prompts/system_prompt_library.md`. The `LLMAssistant`
loads prompts by name; no Python change is needed unless you add a *new*
prompt slot.

**Who to ask:** no sub-agent needed; this is a prompt-engineering task.

---

## Conventions when extending

- Run `pytest tests/` before opening a PR. The smoke test covers fit →
  predict → inverse-design → explain in < 30s on a tiny synthetic set.
- Update `README.md`'s capability table when you add a user-facing feature.
- Touch `CLAUDE.md` *only* when you change a convention or add a sub-agent.
- The `reviewer` sub-agent is a useful pre-PR self-check.
