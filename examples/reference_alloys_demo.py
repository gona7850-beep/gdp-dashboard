"""End-to-end demo using the curated reference alloy database.

What this script does:

1. Load the ~38-alloy curated reference dataset.
2. Train a v1 forward model on it (group-aware CV using alloy family).
3. Forward-predict properties for a user-supplied composition
   (here we use Ti-6Al-4V as if it were unknown).
4. Inverse-design: given target properties, find candidate compositions
   from the trained surrogate.
5. Compare each candidate to the known reference table and report the
   nearest documented alloy as a sanity backstop.

Run from the repo root:
    python examples/reference_alloys_demo.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Make `from core...` work when run as `python examples/foo.py` from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from core.alloyforge.data_pipeline import CompositionFeaturizer, Dataset
from core.alloyforge.feasibility import default_checker
from core.alloyforge.forward_model import ForwardModel
from core.alloyforge.inverse_design import DesignSpec, InverseDesigner
from core.alloyforge.reference_data import (
    PROPERTY_COLUMNS,
    find_alloy,
    reference_dataset,
    reference_elements,
)


def main() -> None:
    print("=" * 72)
    print("Reference alloy database demo")
    print("=" * 72)

    # ----- 1. Load curated reference --------------------------------------
    df = reference_dataset()
    elements = reference_elements()
    print(f"\nLoaded {len(df)} curated alloys, {len(elements)} elements, "
          f"{len(PROPERTY_COLUMNS)} property channels.")
    print(f"Families: {sorted(df['family'].unique())[:8]} … "
          f"({df['family'].nunique()} total)")

    # ----- 2. Train forward model on a subset of properties ----------------
    targets = ["yield_mpa", "tensile_mpa", "hardness_hv", "density_gcc"]
    df_train = df.dropna(subset=targets).reset_index(drop=True)
    print(f"\nTraining on {len(df_train)} rows with full property labels.")

    ds = Dataset(
        compositions=df_train[elements],
        properties=df_train[targets],
        groups=df_train["family"],
    )
    fm = ForwardModel(
        featurizer=CompositionFeaturizer(element_columns=elements),
        targets=targets,
        n_cv_splits=5,
    )
    fm.fit(ds, n_trials=6, verbose=False)
    print("\nGroup-aware (alloy-family) CV metrics:")
    for t, m in fm.metrics_.items():
        print(f"  {t:14s}  MAE={m['cv_mae']:7.1f}  R²={m['cv_r2']:+.3f}")

    # ----- 3. Forward prediction on a known alloy --------------------------
    ti = find_alloy("Ti-6Al-4V")
    comp = ti.as_atomic()
    one_row = pd.DataFrame([{el: comp.get(el, 0.0) for el in elements}])
    preds = fm.predict(one_row).iloc[0]
    print(f"\nForward prediction for Ti-6Al-4V (~86% Ti, 10% Al, 4% V atomic):")
    for t in targets:
        actual = getattr(ti, t)
        mu = preds[f"{t}_mean"]
        sigma = preds[f"{t}_std"]
        err = f"vs ref {actual}" if actual is not None else "(no ref)"
        print(f"  {t:14s}  pred = {mu:7.1f} ± {sigma:5.1f}  {err}")

    # ----- 4. Inverse design: 'high-strength low-density' ------------------
    # Restrict the search to a small relevant element subset; force every
    # other element to 0 via tight (0, 0) bounds so NSGA-II only mutates
    # the design elements. Forward model still uses all 23 columns.
    design_elements = ["Ti", "Al", "V", "Fe", "Cr", "Ni", "Mo"]
    element_bounds = {
        "Ti": (0.50, 0.95), "Al": (0.02, 0.15), "V": (0.0, 0.10),
        "Fe": (0.0, 0.05), "Cr": (0.0, 0.05),
        "Ni": (0.0, 0.05), "Mo": (0.0, 0.05),
    }
    # Zero every non-design element (otherwise NSGA-II explores 23-dim space)
    for el in elements:
        element_bounds.setdefault(el, (0.0, 1e-6))
    spec = DesignSpec(
        objectives=[
            ("tensile_mpa", "max"),
            ("density_gcc", "min"),
            ("yield_mpa", "max"),
        ],
        element_bounds=element_bounds,
        risk_lambda=0.5,
        feasibility=default_checker(design_elements),
    )
    print("\nRunning NSGA-II inverse design (Ti-rich, high-strength + low-density)…")
    # Use a custom-element-set forward-model wrapper: predict on full 23-col
    # vector but the designer only mutates the 7 design elements.
    designer = InverseDesigner(model=fm, spec=spec, element_columns=elements)
    front = designer.run_nsga2(pop_size=40, n_gen=25, seed=0)
    if front.empty or "agg_score" not in front.columns:
        print("  (no feasible candidates — try relaxing constraints)")
    else:
        top = front.sort_values("agg_score").head(3)
        print(f"\nTop 3 candidates:")
        show_cols = ["Ti", "Al", "V", "Fe", "Mo", "tensile_mpa_mean",
                     "density_gcc_mean", "yield_mpa_mean", "agg_score"]
        show = [c for c in show_cols if c in top.columns]
        print(top[show].round(3).to_string(index=False))

    # ----- 5. Match each candidate to the nearest reference alloy ----------
    if not (front.empty or "agg_score" not in front.columns):
        print("\nBack-data sanity: nearest documented alloy in our reference DB")
        print("(Euclidean distance in element-fraction space).")
        ref_X = df[elements].to_numpy(dtype=float)
        top = front.sort_values("agg_score").head(3)
        for i, (_, row) in enumerate(top.iterrows()):
            cand = np.array([row[el] for el in elements])
            d = np.linalg.norm(ref_X - cand[None, :], axis=1)
            j = int(np.argmin(d))
            print(f"  Candidate #{i + 1}: closest = {df.iloc[j]['alloy_name']} "
                  f"(family {df.iloc[j]['family']}, distance {d[j]:.3f})")

    print("\nDone. Pretrain on the curated reference, fine-tune on your CSV "
          "with `merge_datasets(...)` in core.alloyforge.data_ingestion.")


if __name__ == "__main__":
    main()
