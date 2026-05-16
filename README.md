# Nb-Si AM Alloy Design Platform

End-to-end ML platform for Nb-Si-Ti / Nb-alloy (C103) Powder Bed Fusion (LPBF) and Directed Energy Deposition (DED) additive-manufacturing research. Implements an 8-step physics-informed workflow targeting SCI top journals (Acta Mater., npj Comput. Mater., Addit. Manuf.).

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
