# Claude prompts for the composition design platform

This catalogue extends the original sample prompts shipped with
`composition_platform.py` and ties each prompt to a concrete endpoint /
Streamlit tab so a Claude-driven workflow can be reproduced end-to-end.
Each section gives:

* the **user-facing goal**,
* a **prompt template** to send to Claude (system + user),
* the **expected reply shape**, and
* the **platform endpoint** that consumes the reply.

All endpoints are mounted under `/api/v1/composition` in `backend/main.py`.
The Streamlit equivalent is on the `7_조성설계_플랫폼` page.

---

## 0. System prompt (always set)

```
You are a materials-design research assistant. You help users interact with
a composition-property prediction and inverse-design platform. Always be
precise: when you suggest a composition, give the element fractions
explicitly (summing to 1.0) and explain the trade-offs. When you have
numerical model output, ground your reasoning in those numbers and call
out where the model's R^2 is low or uncertainty is high so the user knows
where to trust the prediction less.
```

This is the same string as `core.composition_prompts.SYSTEM_PROMPT`. The
LLM wrapper (`core.llm_designer.LLMDesigner`) sets it automatically.

---

## 1. Free-text → target property JSON

**Goal**: turn a user request like "I need ~650 MPa yield strength and
hardness around 200 HB" into a JSON dict the inverse-design endpoint can
consume directly.

**Template** (`core.composition_prompts.parse_target_prompt`):

```
Extract a target property specification from the following user request.
The model knows about exactly these properties:
  ["yield_strength", "hardness", "elongation", "density"]

Reply with ONLY a JSON object of the form
{"property_name": numeric_target, ...} — no prose, no code fences.
If the user did not mention a property, omit it (do not invent a default).

User request:
"""
I need an alloy with yield strength around 650 MPa and hardness near 200 HB.
"""
```

**Expected reply** (Claude):

```json
{"yield_strength": 650, "hardness": 200}
```

**Consumed by**: `POST /api/v1/composition/claude/parse` → use the parsed
dict in the body of `POST /api/v1/composition/design`.

---

## 2. Inverse design + LLM rationale

**Goal**: after running inverse design, ask Claude to compare the top-K
candidates and recommend one.

**Template** (`core.composition_prompts.explain_candidates_prompt`):

```
Below are inverse-design candidates produced by our random-forest based
platform, scored by weighted MSE against the target properties (lower =
better). Compare them, recommend the best one, and explain why. Where the
validation R^2 of a property is below 0.7, explicitly warn the user that
predictions for that property are unreliable.

Be concise (5-8 sentences). End with a one-line recommendation of the form:
`RECOMMENDED: candidate #N`.

Data (JSON):
{
  "target_properties": {"yield_strength": 650, "hardness": 200},
  "candidates": [
    {"composition": {"Fe": 0.42, "Cr": 0.18, "Ni": 0.25, "Co": 0.15},
     "predicted": {"yield_strength": 648, "hardness": 198},
     "uncertainty": {"yield_strength": 12, "hardness": 6},
     "score": 0.0012,
     "rel_errors": {"yield_strength": 0.003, "hardness": 0.010}},
    ...
  ],
  "model_validation_r2": {"yield_strength": 0.88, "hardness": 0.72}
}
```

**Expected reply**: 5–8 sentences ending in `RECOMMENDED: candidate #N`.

**Consumed by**: `POST /api/v1/composition/claude/explain`. The Streamlit
page shows the reply verbatim under tab 6.

---

## 3. Verify a single composition

**Goal**: feed a fixed composition through the trained predictor, compare
to user targets, and produce a short research memo.

**Step 1** — call `POST /api/v1/composition/analyse` with body:

```json
{
  "composition": {"Al": 0.35, "Mg": 0.25, "Zn": 0.20, "Cu": 0.20},
  "target_properties": {"yield_strength": 400, "elongation": 10},
  "tolerance": 0.1
}
```

**Step 2** — Claude prompt (`feasibility_prompt`):

```
Turn the following feasibility analysis into a short research memo (under
200 words) suitable for a materials scientist to read. Cover: (1) what
the model predicts, (2) how it compares to the targets, (3) which property
is the bottleneck, (4) one concrete next experimental step. Do not invent
numbers — only use what is in the payload.

Payload:
{
  "composition": {...},
  "analysis": {
    "predicted": {...},
    "relative_errors": {...},
    "meets_target": {...},
    "overall_feasible": false,
    "recommendation": "..."
  }
}
```

---

## 4. Suggest composition adjustments

**Goal**: when a candidate misses the target, ask Claude for a specific,
sum-preserving tweak.

**Template** (`improvement_prompt`):

```
Given the current composition, predicted properties, and target properties
below, suggest one or two specific composition adjustments
(e.g. "increase Cr by 0.05 and decrease Ni by 0.05") that you expect will
move the predicted properties closer to the targets. Briefly justify each
adjustment with metallurgical reasoning. Respect the constraint that
fractions must sum to 1.0.

Data:
{
  "current_composition": {"Fe": 0.40, "Ni": 0.30, "Cr": 0.20, "Mo": 0.10},
  "target_properties": {"yield_strength": 700, "corrosion_index": 0.9},
  "predicted_properties": {"yield_strength": 640, "corrosion_index": 0.78},
  "available_elements": ["Fe", "Ni", "Cr", "Mo", "Ti"]
}
```

---

## 5. Constrained inverse design

When a user fixes part of the recipe — e.g. "Ni ≥ 20%, Cr at exactly 10%"
— send those constraints to `/api/v1/composition/design` directly:

```json
{
  "target_properties": {"yield_strength": 800, "elongation": 20},
  "min_fraction": {"Ni": 0.20},
  "fixed": {"Cr": 0.10},
  "strategy": "ga",
  "num_candidates": 5000,
  "top_k": 5
}
```

The platform applies the constraints during sampling so Claude only sees
feasible candidates.

---

## 6. End-to-end conversation example

```
User: We want a Co-Cr-Ni-Fe alloy with yield strength ~720 MPa, hardness
~210 HB, and Ni capped at 25% for cost reasons. Suggest 3 candidates and
tell me which to try first.

Claude (via /claude/parse) →
  {"yield_strength": 720, "hardness": 210}

Platform (via /design with max_fraction={"Ni":0.25}, top_k=3) →
  three DesignCandidate objects with composition, predicted, score, etc.

Claude (via /claude/explain) →
  "Candidate #2 best matches both targets; its uncertainty on hardness is
  the lowest of the three. Validation R^2 on hardness is 0.72 — call this
  a screening prediction and confirm with at least 3 hardness tests on
  the printed coupon. RECOMMENDED: candidate #2."
```

---

## 7. Using Claude Code on this codebase

Claude Code (the CLI assistant powering this branch) can extend the
platform further with prompts like:

* "Add an Active-Learning loop that picks the next experiment based on
  predicted-improvement + epistemic uncertainty" — see `core/mobo.py` and
  `core/composition_platform.py` for existing scaffolding.
* "Wire the inverse-design `/design` endpoint into a Slack slash command
  via `backend/main.py`." — keep the FastAPI router structure; add a new
  router under `backend/routers/`.
* "Add a SHAP plot of element importance per property to the Streamlit
  page." — reuse `core/shap_analysis.py`.

When asking Claude Code to extend the platform, always:

1. Name the **exact file and function** to touch.
2. State the **invariant** the change must preserve (composition fractions
   sum to 1, no LLM-only paths break in offline mode, etc.).
3. Ask for **tests** alongside the change so regressions are caught.
