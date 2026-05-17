"""End-to-end accuracy & reliability report on the curated reference data.

What this does
--------------

1. Pretrain a v1 forward model on the 38-alloy reference DB.
2. Run :func:`evaluate_model` to compute:
   * Hold-out R² / MAE / RMSE
   * K-fold CV mean ± std (group-aware by alloy family)
   * Permutation-test p-value (the bar for "model learned something")
   * Conformal-interval empirical coverage at 90 % nominal
   * Per-target reliability diagram
   * DoA percentiles
   * Sanity check predicting every reference alloy
3. Pretty-print a human summary + the overall A/B/C/D grade.

Run from the repo root:
    python examples/accuracy_report_demo.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from core.alloyforge import (
    AccuracyReport,
    CompositionFeaturizer,
    Dataset,
    ForwardModel,
    evaluate_model,
    reference_dataset,
    reference_elements,
)


def main() -> None:
    print("=" * 72)
    print("Accuracy & reliability report on the reference alloy DB")
    print("=" * 72)

    df = reference_dataset()
    elements = reference_elements()
    targets = ["yield_mpa", "tensile_mpa", "hardness_hv", "density_gcc"]
    df_train = df.dropna(subset=targets).reset_index(drop=True)
    print(f"\nTraining set: {len(df_train)} rows, {df_train['family'].nunique()} families.\n")

    ds = Dataset(
        compositions=df_train[elements],
        properties=df_train[targets],
        groups=df_train["family"],
    )
    fm = ForwardModel(
        featurizer=CompositionFeaturizer(element_columns=elements),
        targets=targets, n_cv_splits=5,
    )
    fm.fit(ds, n_trials=6)

    print("Running diagnostics (this takes ~1-2 min)…")
    rep = evaluate_model(
        fm, ds,
        targets=targets,
        n_splits=5,
        n_seeds=2,
        n_permutations=10,
        skip_reliability=False,
        include_reference_check=True,
        seed=0,
    )

    print("\n--- Summary ---\n")
    print(rep.summary())
    if rep.notes:
        print("\nNotes:")
        for n in rep.notes:
            print(f"  - {n}")

    print("\n--- Hold-out (single 80/20 split) ---\n")
    for t, m in rep.holdout.items():
        print(f"  {t:14s}  R²={m['r2']:+.3f}  MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}")

    print("\n--- Conformal coverage (target 90%) ---\n")
    for t, c in rep.coverage.items():
        print(f"  {t:14s}  empirical={c['empirical_coverage']:.0%}  "
              f"target={c['nominal_coverage']:.0%}  n={c['n']}")

    if rep.reference_check is not None and not rep.reference_check.empty:
        print("\n--- Worst-error alloys (against reference DB) ---\n")
        rc = rep.reference_check.copy()
        # Rank by yield_mpa relative error if present
        for tgt in targets:
            err_col = f"{tgt}_rel_err"
            if err_col in rc.columns:
                worst = rc.nlargest(3, err_col)[
                    ["alloy_name", "family", f"{tgt}_actual",
                     f"{tgt}_pred", err_col]
                ]
                print(f"  Top-3 by relative error on {tgt}:")
                print(worst.round(3).to_string(index=False))
                print()

    print("Overall grade:", rep.overall_grade)
    print("\nDone. Re-run with your CSV to get a one-shot reliability report.")


if __name__ == "__main__":
    main()
