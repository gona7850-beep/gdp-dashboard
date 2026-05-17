# Composition Design Platform

ML-driven **composition → property prediction**, **inverse design**,
**validation**, **feasibility analysis**, and **Claude-assisted
interpretation** — packaged as a Python library, a REST API, a single-page
web UI, a Streamlit workbench, a Docker image, and a Claude Code
sub-agent + slash command set.

## Reference alloy database (38 well-known alloys)

`core/alloyforge/reference_data.py` ships a curated table of ~38
household-name alloys — 304/316L/17-4 PH/Maraging/M2/4140 steels,
Inconel 718/625/Hastelloy X/Waspaloy/CMSX-4 Ni superalloys, Ti-6Al-4V/
Ti-6242/Ti-5553, AA 2024/6061/7075/AlSi10Mg, Cantor + AlCoCrFeNi HEAs,
Stellite 6, MP35N, CuBe, brass, C-103/W/Mo TZM/Ta refractories — each
with yield/UTS/elongation/HV/density/Young's modulus/melting point
compiled from ASM Handbook, MatWeb, and producer datasheets. Use it as:

- **Backstop** when comparing inverse-design candidates to known alloys.
- **Pretrain seed** for a forward model before fine-tuning on your CSV.
- **Sanity reference** to spot-check predictions against published values.

```python
from core.alloyforge import reference_dataset, find_alloy

df = reference_dataset()                 # 38-row training-ready DataFrame
ti = find_alloy("Ti-6Al-4V")             # one record with refs + notes
ti.as_atomic()                            # {"Ti": 0.862, "Al": 0.102, "V": 0.036}
```

## Data ingestion + unit conversion

`core/alloyforge/data_ingestion.py` handles the common accuracy traps in
externally-sourced materials data:

- **Auto-infer** unit per column from header names + value ranges
  (`MPa / ksi / GPa / HV / HRC / HB / K / °C / wt%`).
- **Convert** with ASTM E140 tables for HRC→HV (verified to within 2 HV
  across the standard table) and closed-form formulas for stress /
  temperature.
- **Composition normalisation** from weight % → atomic fraction.
- **Robust outlier flagging** via median-absolute-deviation z-score.
- **Merge multiple sources** with a `source` column added for
  GroupKFold so the same alloy reused across papers can't leak.

```python
from core.alloyforge import merge_datasets, normalize_composition

merged, summary = merge_datasets(
    sources={"my_csv": my_df, "lab_logbook": logbook_df},
    element_columns=["Fe", "Ni", "Cr", "Mo"],
    target_columns=["yield_mpa", "tensile_mpa"],
)
# merged["source"] becomes the group key for Dataset(groups=...)
```

End-to-end demo: `python examples/reference_alloys_demo.py` pretrains
on the 38-alloy reference table, predicts Ti-6Al-4V properties
(recovers ~880/950/340 MPa/HV from the literature values), inverse-
designs a high-strength low-density Ti-rich composition, and reports
the nearest documented alloy in the reference DB as a backstop.

## External data sources

`core/alloyforge/external_data.py` ships four clients with consistent
return schema (`title / authors / year / venue / doi / url / abstract /
source`). All return an empty DataFrame on network failure — they
never crash the platform.

| Provider | Auth | Function |
|---|---|---|
| OpenAlex | none (mailto recommended) | `search_openalex(query)` |
| arXiv | none | `search_arxiv(query)` |
| CrossRef | none (mailto recommended) | `search_crossref(query)` |
| Materials Project | `MP_API_KEY` env var | `materials_project_summary(elements=…)` |

`provider_status()` reports which clients are configured. The
companion FastAPI endpoints sit under `/api/v1/data/external/*`.

## LLM-mediated table extraction

`core/alloyforge/llm_table_extractor.py` accepts raw paper text and
returns structured rows ready for `merge_datasets()`. With
`ANTHROPIC_API_KEY` it uses Claude (default `claude-sonnet-4-6`,
overridable via `CLAUDE_TABLE_EXTRACTOR_MODEL`) and enforces a strict
JSON schema with per-row confidence flags (`high/medium/low`). Without
the key, a regex heuristic captures the simplest patterns and labels
everything `confidence="low"` so users know what to spot-check.

## Accuracy & reliability report

`core/alloyforge/accuracy_report.py` runs every standard model
diagnostic in one call:

* hold-out R² / MAE / RMSE
* K-fold CV mean ± std (group-aware when `Dataset.groups` is set)
* permutation-test p-value (the bar for "model learned something")
* conformal-interval empirical coverage at nominal 90 %
* per-target reliability diagrams
* DoA percentiles
* sanity check predicting every reference alloy

```python
from core.alloyforge import evaluate_model

rep = evaluate_model(model, dataset, targets=["yield_mpa", "tensile_mpa"])
print(rep.summary())   # CV R²=0.91±0.04, perm p=0.02, coverage=87%@90%
rep.overall_grade       # 'A' / 'B' / 'C' / 'D' heuristic
```

End-to-end demo: `python examples/accuracy_report_demo.py`.

## High-throughput compound screening

An alternative to CALPHAD-based alloy design that mirrors the workflow
in Cho 2025 (Kookmin NSM Lab) for Nb-host alloys:

1. **Compound corpus** — bundled curated DB of ~22 Nb-host intermetallics
   (Nb-Si, Nb-Al, Nb-Cr, Nb-Ni, Nb-Co, Nb-Fe, Nb-Ge, boride / carbide
   additives, ternary substitutions). Live OQMD queries are also
   supported via `core/alloyforge/oqmd_client.py`.
2. **Three thermodynamic descriptors** scored independently and combined:
   * **Tie line with host matrix** — does an isothermal tie line exist
     between the compound and the host metal? Required so the compound
     can coexist with the matrix without consuming it.
   * **Standalone stability** — formation enthalpy per atom (negative,
     and lower than competing decomposition products).
   * **Coherency** — **modular** lattice + per-atom-volume mismatch
     with the host (tries k = 1, 2, 3, 4 multiples so super-cell
     matches like Nb₅Si₃ a=6.57 Å ≈ 2 × Nb a=3.30 Å score correctly).
3. **Ranking** — weighted average with adjustable weights; filters for
   required/forbidden elements + minimum tie-line score.
4. **Bridge to ML** — `host_plus_precipitate_composition()` mixes a
   host matrix + a precipitate compound at a given atomic fraction,
   producing a row the forward model can predict on.

```python
from core.alloyforge import rank_compounds, ScoreWeights

# Default: equal weights, all bundled compounds for Nb host
ranked = rank_compounds(host="Nb", top_k=10)

# Custom: emphasise stability + coherency over tie line
ranked = rank_compounds(
    host="Nb",
    weights=ScoreWeights(tie_line=0.5, stability=2.0, coherency=2.0),
    required_elements=["Nb", "Si"],
)
```

REST endpoints: `/api/v1/hts/{hosts, compounds, rank, compound-mix,
oqmd-search, oqmd-rank}`. Streamlit UI at page 10
(`10_HTS_화합물_스크리닝.py`). End-to-end demo:
`python examples/hts_nb_alloy_demo.py`.

For Nb-host the top-ranked compounds are silicide variants:
**(Nb,Hf)₅Si₃, Nb₅Si₃-α, (Nb,Ti)₅Si₃** — the in-situ-composite phases
that the Bewlay et al. body of literature has validated as the primary
strengthening phases for high-temperature Nb-Si alloys.

## Streamlit data-collection page

`app/pages/9_데이터_수집_통합.py` — five-tab UI:

1. browse / filter the 38-alloy reference DB
2. upload CSV/Excel with auto unit detection
3. search OpenAlex / arXiv / CrossRef / Materials Project
4. paste paper text → LLM table extraction → confidence-flagged rows
5. merge everything into one CSV with a `source` group column

## FastAPI data router

Mounted at `/api/v1/data`:

```
GET  /reference-alloys                 # 38 curated alloys
GET  /reference-alloys/{name}          # one alloy
POST /ingest                           # unit-aware merge + dedup
GET  /external/status                  # which providers usable
GET  /external/openalex?q=...
GET  /external/arxiv?q=...
GET  /external/crossref?q=...
GET  /external/materials-project?elements=Fe,Ni
POST /llm-extract                      # LLM table extraction
```

---

Two ML backends share one platform:

| | Lite (`core/composition_platform.py`) | Advanced (`core/alloyforge/`) |
|---|---|---|
| Forward model | RF / GBR / Ridge / MLP | v1 = XGB+GP+Optuna · v2 = stacked XGB+LGBM+MLP+GP, Optuna-tuned per base learner |
| HPO | Fixed defaults | Optuna TPE |
| Uncertainty | RF tree std | v1 = conformal 90% intervals · v2 = epistemic + aleatoric decomposition |
| Multi-task | — | v2 supports sibling-target stacking |
| Physics features | — | v2 + `ExtendedFeaturizer` adds Miedema ΔH_mix, Yang's Ω, VEC-window probabilities, stiffness proxy |
| Benchmarking | — | `compare_v1_vs_v2()` produces leaderboards on your data |
| Inverse design | Dirichlet MC + simple GA | NSGA-II with risk-aware `μ − λσ` |
| Constraints | Element bounds | Hume-Rothery δ · VEC · VED · custom |
| Explainability | — | SHAP + counterfactual search |
| Active learning | — | Uncertainty + qEHVI batch picks |
| Heavy deps | none | xgboost, lightgbm, optuna, pymoo, shap |

### v1 vs v2 — when to pick which

`ForwardModel` (v1) is a single Optuna-tuned XGBoost with a GP residual.
Fast, strong on most tabular alloy problems. Pick it when you have
<200 rows and just want a calibrated mean ± σ.

`ForwardModelV2` is a stacked ensemble of XGBoost + LightGBM + sklearn-MLP
with a Ridge meta-learner and a GP residual on top. Optuna-tunes each
booster, deep-ensembles each (N seeds) for epistemic σ, optionally adds a
multi-task pass where sibling-target predictions feed back as auxiliary
features. Pair with `ExtendedFeaturizer` for ~10 metallurgical features
(Miedema ΔH_mix, Yang's Ω, VEC-window probabilities, stiffness proxy).
Pick it when you need epistemic/aleatoric decomposition (for active
learning), multi-task lift on correlated properties, or a richer feature
representation. Run `python examples/benchmark_v2.py` on your data —
neither is a universal winner; the harness shows which to use.

## Quick start

```bash
pip install -r requirements.txt

# Option 1 — single-page web UI + REST API (everything in one process)
uvicorn backend.main:app --reload
# → open http://localhost:8000

# Option 2 — Streamlit research workbench
streamlit run app/streamlit_app.py
# → page 7 (lite) or page 8 (advanced)

# Option 3 — Docker (both services side by side)
docker compose up

# Option 4 — CLI end-to-end demo on synthetic Fe-Ni-Cr-Mo-Ti data
python examples/alloyforge_demo.py
```

Set `ANTHROPIC_API_KEY` to enable real Claude calls; without it,
deterministic offline heuristics are used and every endpoint still returns
200.

## Legacy Nb-Si scaffolding (broken)

The repository began as an 8-step Nb-Si-Ti / C103 AM research workflow.
That scaffolding shipped with `core/db.py` corrupted on disk (831 null
bytes) and several companion modules missing (`core/features.py`,
`core/models.py`, `core/physics.py`, `core/shap_analysis.py`,
`core/mobo.py`, `core/literature.py`). `backend/main.py` therefore no
longer imports the legacy routers (`data`, `features`, `train`, `shap`,
`mobo`, `literature`). Streamlit pages 1–6 fall in the same bucket. To
revive that path: restore the missing modules and re-enable the imports
in `backend/main.py`.

---

## Original Nb-Si workflow notes (kept for reference)

> ⚠ **Ethics policy**: this platform integrates only legal Open-Access channels (CrossRef, OpenAlex, Semantic Scholar, arXiv, Unpaywall). Sci-Hub and similar piracy mirrors are explicitly excluded — they violate COPE / Elsevier publishing ethics and would compromise paper submission.

## 8-step workflow (Phase 1: Steps 1–5 + 4.5)

| # | Step | Module | Streamlit page |
|---|---|---|---|
| 1 | Hierarchical Database (composition / process / microstructure / properties) | `core/db.py` | `1_데이터베이스` |
| 2 | Physics-Informed Feature Engineering (VEC, Δχ, δ, sd/sp, Larson-Miller, VED) | `core/features.py` + `core/physics.py` | `2_피처엔지니어링` |
| 3 | Multi-algorithm Benchmarking (RF, XGB, LGBM, CatBoost, GPR, PLS1, BR, …) | `core/models.py` + `core/benchmark.py` | `3_벤치마킹` |
| 4 | SHAP-based XAI (global / local / interaction / **physics validation**) | `core/shap_analysis.py` | `4_SHAP_XAI` |
| 5 | Multi-Objective Bayesian Optimization (BoTorch qNEHVI; sklearn fallback) | `core/mobo.py` | `5_MOBO_파레토` |
| 6 | Literature auto-ingestion (legal OA only) | `core/literature.py` | `6_문헌_수집` |

Phase 2 (planned): Active Learning closed-loop · AM process integration · paper storyline.

## Quick start

```bash
pip install -r requirements.txt

# 1) Streamlit front-end (research workbench)
streamlit run app/streamlit_app.py

# 2) FastAPI back-end (programmatic / future React UI)
uvicorn backend.main:app --reload    # http://localhost:8000/docs

# 3) Gradio standalone demo (HF Spaces deployable)
python gradio_apps/mobo_demo.py
```

## CLI scripts

```bash
# Ingest a CSV (composition + property) into the hierarchical DB
python scripts/ingest.py --csv data/raw/HV_184.csv --target HV --method SPS

# Benchmark all available models
python scripts/train_all.py --target HV --condition RT --models RF,XGB,LGBM

# MOBO batch proposal from paired CSV
python scripts/propose_batch.py --paired data/raw/HV_sigma_51.csv \
    --obj1 HV --obj2 sigma_compressive --q 5
```

## Repository layout

```
app/                # Streamlit pages (Korean filenames)
backend/            # FastAPI routers exposing core/* over HTTP
core/               # Framework-independent ML library
  db.py             # SQLite + 5-table M-P-P-P schema
  physics.py        # VEC, Δχ, δ, ΔH_mix (Miedema), Ω, VED, Rosenthal, Larson-Miller
  features.py       # tabular feature-matrix builder
  models.py         # zoo (RF, XGB, LGBM, CatBoost, GPR, PLS1, BR, …)
  benchmark.py      # 5-fold × 10 seeds + group-CV + permutation/Y-rand tests
  shap_analysis.py  # global/local/interaction + physics validation
  mobo.py           # BoTorch qNEHVI w/ sklearn fallback
  literature.py     # CrossRef / OpenAlex / S2 / arXiv / Unpaywall
gradio_apps/        # Standalone HF-Spaces-deployable demos
scripts/            # CLI entry points
tests/              # pytest test suite
materials_design/   # legacy scoring pipeline (kept for protocol reference)
```

## Validation standards

Every benchmark run reports — by default — what SCI top-journal reviewers expect:

- **5-fold × 10 random seeds** → mean ± std on R² / RMSE / MAE
- **LeaveOneGroupOut by alloy class** — generalization to unseen alloy systems
- **Permutation test** — p-value of "model beats random label" (`p < 0.05` required)
- **Y-randomization** — null distribution of R² on shuffled targets
- **Prediction-interval coverage** — for Bayesian / GPR models

## Physics priors used in SHAP validation

The Step 4.5 *Physics Validation* tab cross-checks SHAP signs against rules from Sun (2021) *Intermetallics* 133, Tsakiropoulos (2018) *Materials* 11, Bewlay (2003) *MSE A* — agreements strengthen the Discussion section, conflicts flag candidates for novel-mechanism follow-up.

## Tests

```bash
pytest tests/ -v          # 23 tests, ~6s
```

## Composition design & inverse design module

A self-contained ML composition platform now lives alongside the Nb-Si
workflow. It handles **composition → property prediction**, **inverse
design** (target properties → candidate compositions), **verification**
of a single recipe against targets, and **Claude-assisted** target
parsing / candidate explanation. Designed to be useful without your own
dataset — there is a synthetic-alloy generator for instant demos.

### Core modules

| File | Purpose |
|---|---|
| `core/composition_platform.py` | `PropertyPredictor` (RF / GBR / Ridge / MLP), `CompositionDesigner` (Dirichlet MC + GA inverse design), `DesignConstraints` (per-element min/max/fixed), joblib persistence, RF tree-std uncertainty |
| `core/synthetic_alloy_data.py` | Rule-of-mixtures dataset generator (10 elements × 4 properties) |
| `core/llm_designer.py` | Anthropic Claude wrapper with deterministic offline fallback |
| `core/composition_prompts.py` | Versioned prompt templates (system + task-specific) |

### REST API

Mount path: `/api/v1/composition` (registered in `backend/main.py`).

```
POST /train          # train from CSV path or inline rows
POST /predict        # composition → properties (+ uncertainty if RF)
POST /design         # target properties → top-K candidate compositions
POST /verify         # alias of /predict
POST /analyse        # composition + target → feasibility report
POST /claude/parse   # free-text request → target-property JSON
POST /claude/explain # LLM rationale + recommendation over candidates
GET  /status         # whether a model is loaded + report summary
POST /demo-dataset   # return synthetic alloy rows for instant demos
```

### Streamlit page

`app/pages/7_조성설계_플랫폼.py` — six tabs covering the full workflow
(data → train → predict → design → verify → Claude assistant).

### Quick demo (no real dataset needed)

```bash
pip install -r requirements.txt

# Web UI
streamlit run app/streamlit_app.py
# → open the page "7_조성설계_플랫폼", click "합성 데이터 생성"

# REST API
uvicorn backend.main:app --reload
curl -X POST http://localhost:8000/api/v1/composition/demo-dataset \
     -H 'content-type: application/json' -d '{"n_samples": 200}'
```

### Claude integration

Setting `ANTHROPIC_API_KEY` enables real LLM calls; otherwise every
endpoint still works using the rule-based fallback in
`core/llm_designer.py`. Override the model via `CLAUDE_COMPOSITION_MODEL`
(default is the Claude Sonnet 4.X identifier). Prompt templates live in
`core/composition_prompts.py` and the catalogue with examples is in
[`prompts/claude_composition_prompts.md`](prompts/claude_composition_prompts.md).

### Tests

```bash
pytest tests/test_composition_platform.py -q   # 21 tests, ~9s
```

## AlloyForge — advanced ML stack

A heavier, more sophisticated companion to the composition design module
above. Use when you have a real dataset and want calibrated uncertainty,
SHAP explanations, multi-objective inverse design with constraints, and
active-learning batch picks.

### Capability matrix (advanced vs. lite)

| Capability | `core/composition_platform.py` (lite) | `core/alloyforge/` (advanced) |
|---|---|---|
| Forward model | RF / GBR / Ridge / MLP | **Stacked XGBoost + GP residual head** |
| Hyperparameter tuning | Fixed defaults | **Optuna TPE search** |
| Featurization | Raw element fractions | **Physics-informed (mean/std/min/max/range × 6 properties + δ + entropy)** |
| Uncertainty | RF tree std | **Calibrated GP σ + conformal intervals** |
| Group-aware CV | No | **Yes** (`GroupKFold` on `groups`) |
| Domain of applicability | No | **Yes** (NN-distance percentile) |
| Inverse design | Dirichlet MC + simple GA | **NSGA-II (`pymoo`)** with risk-aware `μ − λσ` |
| Constraints | Element bounds | **Constraint system**: Hume-Rothery δ, VEC window, VED window, custom |
| Explainability | None | **SHAP + counterfactual search** |
| Active learning | None | **Uncertainty + qEHVI Monte Carlo batch picks** |
| Heavy deps | None | `xgboost`, `optuna`, `pymoo`, `shap`, `scipy` |

### Files

| File | Purpose |
|---|---|
| `core/alloyforge/data_pipeline.py` | `CompositionFeaturizer`, `Dataset`, curated `ELEMENT_PROPERTIES` (35 elements) |
| `core/alloyforge/forward_model.py` | Stacked XGB + GP, Optuna-tuned, group-aware CV |
| `core/alloyforge/validation.py` | `ConformalCalibrator`, `DomainOfApplicability`, reliability diagrams |
| `core/alloyforge/feasibility.py` | `Constraint` system + Hume-Rothery / VEC / VED / element bounds |
| `core/alloyforge/inverse_design.py` | NSGA-II driver, `DesignSpec` config |
| `core/alloyforge/explainability.py` | SHAP wrapper + counterfactual search |
| `core/alloyforge/active_learning.py` | Uncertainty + qEHVI batch selection |
| `core/alloyforge/llm_assistant.py` | Claude wrapper with metallurgy-aware prompts |

### REST API (`/api/v1/alloyforge`)

```
POST /fit                  # stacked XGB+GP + Optuna + group-aware CV
POST /predict              # μ, σ, conformal 90% interval, DoA score
POST /feasibility/check    # Hume-Rothery / VEC / VED / bounds
POST /inverse-design       # NSGA-II with risk-aware μ-λσ
POST /explain              # SHAP + LLM-mediated metallurgy interpretation
POST /active-learning      # uncertainty batch picks with diversity penalty
GET  /sessions             # list in-memory sessions
GET  /status               # capability flags + dep check
```

### Streamlit page

`app/pages/8_AlloyForge_고급플랫폼.py` — six tabs (데이터 → 학습 →
예측·신뢰구간 → NSGA-II 역설계 → SHAP → Active Learning).

### Claude Code integration

The `.claude/` directory at the repo root contains:

- **6 specialized sub-agents**: `forward-modeler`, `inverse-designer`,
  `validator`, `thermodynamics-expert`, `doe-planner`, `reviewer`. Each
  has scope rules and invariants encoded so changes go through the right
  reviewer.
- **5 slash commands**: `/run-prediction`, `/inverse-search`,
  `/explain-prediction`, `/validate-design`, `/generate-doe`. Drop-in
  workflows you can invoke from the Claude Code CLI.

Open the repo in Claude Code and these become available automatically.
See `prompts/alloyforge_extension_guide.md` for the recommended
extension workflow and `prompts/alloyforge_system_prompts.md` for the
prompt library used by `LLMAssistant`.

### Quick demo

```bash
pip install -r requirements.txt

# End-to-end demo on synthetic Fe-Ni-Cr-Mo-Ti data
python examples/alloyforge_demo.py

# Streamlit UI
streamlit run app/streamlit_app.py
# → page "8_AlloyForge_고급플랫폼"

# REST API
uvicorn backend.main:app --reload
curl http://localhost:8000/api/v1/alloyforge/status
```

### Tests

```bash
pytest tests/test_alloyforge.py -q     # 11 tests, ~5s
pytest tests/ -q                        # 32 tests total (lite + advanced)
```

## License

See [LICENSE](LICENSE).
