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

## License

See [LICENSE](LICENSE).
