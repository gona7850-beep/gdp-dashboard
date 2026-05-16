"""End-to-end AlloyForge demo on a synthetic Fe-Ni-Cr-Mo-Ti dataset.

Run from the repo root:
    python examples/alloyforge_demo.py

Trains a stacked XGBoost + GP forward model, calibrates conformal intervals,
runs NSGA-II inverse design, and produces SHAP-based explanations for the
top candidate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.alloyforge.data_pipeline import CompositionFeaturizer, Dataset
from core.alloyforge.explainability import Explainer
from core.alloyforge.feasibility import default_checker
from core.alloyforge.forward_model import ForwardModel
from core.alloyforge.inverse_design import DesignSpec, InverseDesigner
from core.alloyforge.validation import ConformalCalibrator, DomainOfApplicability


def make_synthetic_alloys(n: int = 150, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.dirichlet([3.0, 1.0, 1.0, 0.5, 0.3], size=n)
    df = pd.DataFrame(base, columns=["Fe", "Ni", "Cr", "Mo", "Ti"])

    rng2 = np.random.default_rng(seed + 1)
    df["hardness_hv"] = (
        220 + 850 * df["Mo"] + 600 * df["Cr"]
        + 1200 * df["Ti"] - 90 * df["Ni"] + rng2.normal(0, 15, n)
    )
    df["tensile_mpa"] = (
        450 + 1100 * df["Mo"] + 700 * df["Cr"]
        + 900 * df["Ti"] + 200 * df["Ni"] + rng2.normal(0, 30, n)
    )
    df["elongation_pct"] = (
        45 - 80 * df["Mo"] - 60 * df["Ti"] + 40 * df["Ni"] + rng2.normal(0, 3, n)
    ).clip(2, 60)
    df["heat_id"] = [f"H{i // 5:03d}" for i in range(n)]
    return df


def main() -> None:
    print("=" * 70)
    print("AlloyForge demo — Fe-Ni-Cr-Mo-Ti synthetic dataset")
    print("=" * 70)

    df = make_synthetic_alloys()
    print(f"\nDataset: {len(df)} rows, {df['heat_id'].nunique()} heats")

    elements = ["Fe", "Ni", "Cr", "Mo", "Ti"]
    targets = ["hardness_hv", "tensile_mpa", "elongation_pct"]
    dataset = Dataset(
        compositions=df[elements],
        properties=df[targets],
        groups=df["heat_id"],
    )

    print("\nFitting ForwardModel (small Optuna budget for demo)…")
    fm = ForwardModel(
        featurizer=CompositionFeaturizer(element_columns=elements),
        targets=targets,
        n_cv_splits=5,
    )
    fm.fit(dataset, n_trials=8)
    print("  CV metrics:")
    for t, m in fm.metrics_.items():
        print(f"    {t:18s}  MAE={m['cv_mae']:7.2f}  R²={m['cv_r2']:5.3f}")

    print("\nCalibrating conformal intervals (α=0.1, target 90% coverage)…")
    calib = ConformalCalibrator(alpha=0.10).calibrate(fm, dataset)
    for t in targets:
        print(f"    {t:18s}  q_hat={calib.q_hat_[t]:.3f}")

    doa = DomainOfApplicability().fit(fm, dataset)
    print(f"\n  DoA reference quantile: {doa.train_nn_quantile_:.4f}")

    spec = DesignSpec(
        objectives=[("hardness_hv", "max"), ("tensile_mpa", "max")],
        element_bounds={
            "Fe": (0.50, 0.85),
            "Ni": (0.05, 0.25),
            "Cr": (0.05, 0.25),
            "Mo": (0.00, 0.10),
            "Ti": (0.00, 0.05),
        },
        risk_lambda=1.0,
        feasibility=default_checker(elements),
    )
    print("\nRunning NSGA-II inverse design…")
    designer = InverseDesigner(model=fm, spec=spec, element_columns=elements)
    front = designer.run_nsga2(pop_size=32, n_gen=20, seed=0)
    print(f"  Pareto front size: {len(front)}")

    show_cols = elements + [f"{t}_mean" for t in ["hardness_hv", "tensile_mpa"]] + ["agg_score"]
    print("\nTop-5 candidates:")
    print(front.sort_values("agg_score").head(5)[show_cols].to_string(index=False))

    top = front.sort_values("agg_score").head(1)[elements].copy()
    expl = Explainer(model=fm)
    sv = expl.explain(top, target="hardness_hv", background_df=df[elements])
    print("\nSHAP top-5 features for top candidate (hardness_hv):")
    print(sv.assign(abs_shap=lambda d: d["shap"].abs())
          .sort_values("abs_shap", ascending=False)
          .head(5)[["feature", "value", "shap"]].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
