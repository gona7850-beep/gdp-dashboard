---
description: Fit the forward model on a CSV and produce a predictions report with calibrated intervals.
argument-hint: <data_csv> <target_columns_comma_separated> [groups_column]
---

You are running the **run-prediction** workflow for AlloyForge.

## Inputs
- Data file: `$1`
- Targets: `$2`  (comma-separated target column names)
- Groups column: `$3`  (optional; if omitted, no grouping)

## Steps

1. Read `$1` with pandas. Print shape, dtypes, missingness summary.
2. Identify composition columns: any column whose name matches an element
   symbol in `ELEMENT_PROPERTIES` (see `core/alloyforge/data_pipeline.py`). Print which
   ones were detected.
3. Build a `Dataset` with composition, targets, and optional groups.
4. Fit a `ForwardModel` with `n_trials=25` (Optuna). If the dataset is small
   (< 80 rows), reduce to `n_trials=10` and note this.
5. Run `ConformalCalibrator` on a 20% split.
6. Report:
   - Out-of-fold MAE / R² per target.
   - 90% conformal interval width per target.
   - Reliability diagram numbers (printed, not plotted).
   - Top 10 SHAP global features per target.
7. If any target has R² < 0.5, flag it explicitly and recommend either more
   data, feature additions, or that target be treated as exploratory only.

## Output

A structured markdown report at `reports/prediction_<timestamp>.md` plus a
session pickle at `sessions/<session_id>.pkl`. Print the session_id so it can
be reused by `/inverse-search` and `/explain-prediction`.
