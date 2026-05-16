"""Head-to-head v1 vs v2 benchmark on synthetic Fe-Ni-Cr-Mo-Ti data.

What this example demonstrates:

1. ``ForwardModelV2`` produces **decomposed uncertainty** (epistemic +
   aleatoric) that ``ForwardModel`` v1 cannot. This matters more for
   design than raw R²: in inverse design we want to flag candidates
   whose σ is dominated by *epistemic* (model ignorance, fixable with
   experiments) vs *aleatoric* (intrinsic noise, fundamental).

2. ``ForwardModelV2`` with ``share_targets=True`` adds **multi-task
   learning** — sibling-property OOF predictions become aux features.
   When properties correlate, harder targets get lifted.

3. ``ExtendedFeaturizer`` exposes 10 metallurgical features (Miedema
   ΔH_mix, Yang's Ω, VEC-window probabilities, stiffness proxy) on top
   of the base 33. On physics-driven data they help; on clean
   polynomial data they can add noise — the benchmark surfaces this
   honestly per-dataset.

4. Use the ``compare_v1_vs_v2`` harness on **your own data** to decide
   which path to use. The CV is group-aware when ``Dataset.groups`` is
   set.

Run from the repo root:
    python examples/benchmark_v2.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from core.alloyforge.benchmark import compare_v1_vs_v2, leaderboard_pivot
from core.alloyforge.data_pipeline import CompositionFeaturizer, Dataset
from core.alloyforge.forward_model_v2 import ForwardModelV2
from core.alloyforge.physics_features import make_extended


def make_alloys(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    comp = rng.dirichlet([3.0, 1.0, 1.0, 0.5, 0.3], size=n)
    df = pd.DataFrame(comp, columns=["Fe", "Ni", "Cr", "Mo", "Ti"])
    rng2 = np.random.default_rng(seed + 1)
    df["hardness_hv"] = (
        220 + 850 * df["Mo"] + 600 * df["Cr"] + 1200 * df["Ti"]
        - 90 * df["Ni"] + 500 * df["Mo"] * df["Ti"]
        + rng2.normal(0, 15, n)
    )
    df["tensile_mpa"] = (
        450 + 1100 * df["Mo"] + 700 * df["Cr"] + 900 * df["Ti"]
        + 200 * df["Ni"] + 300 * df["Cr"] * df["Mo"]
        + rng2.normal(0, 30, n)
    )
    df["elong_pct"] = (
        45 - 80 * df["Mo"] - 60 * df["Ti"] + 40 * df["Ni"]
        + rng2.normal(0, 3, n)
    ).clip(2, 60)
    df["heat_id"] = [f"H{i // 5:03d}" for i in range(n)]
    return df


def main() -> None:
    print("=" * 72)
    print("V1 (XGB+GP+Optuna) vs V2 (stacked ensemble + multi-task)")
    print("=" * 72)
    df = make_alloys(n=180, seed=11)
    elements = ["Fe", "Ni", "Cr", "Mo", "Ti"]
    targets = ["hardness_hv", "tensile_mpa", "elong_pct"]
    dataset = Dataset(
        compositions=df[elements],
        properties=df[targets],
        groups=df["heat_id"],
    )

    print("\n--- Leaderboard (mean R² over 3-fold group CV) ---\n")
    print("Running 4 models × 3 targets × 3 folds (~60-90 s with Optuna×4)…")
    lb = compare_v1_vs_v2(
        dataset=dataset,
        element_columns=elements,
        targets=targets,
        n_splits=3,
        n_trials_v1=4,
        v2_seeds=2,
    )
    pivot = leaderboard_pivot(lb, metric="r2_mean").round(3)
    print(pivot.to_string())

    print("\n--- Per-target detail ---\n")
    print(lb.to_string(index=False))

    print("\n--- V2 unique capability: decomposed uncertainty ---\n")
    ext = make_extended(elements)
    fm = ForwardModelV2(
        featurizer=ext, targets=targets, n_seeds=2, n_cv_splits=3,
        share_targets=True, n_trials=4, random_state=0,
    )
    fm.fit(dataset)

    # Predict on a few rows and show the decomposition
    head = df[elements].head(3).reset_index(drop=True)
    preds = fm.predict(head, return_decomposed=True)
    print("(epistemic = model ignorance, fixable by data; aleatoric = noise)")
    for i in range(len(head)):
        print(f"\n  Row {i}: {head.iloc[i].to_dict()}")
        for t in targets:
            mu = preds[f"{t}_mean"].iloc[i]
            sig = preds[f"{t}_std"].iloc[i]
            epi = preds[f"{t}_epistemic"].iloc[i]
            ale = preds[f"{t}_aleatoric"].iloc[i]
            print(f"    {t:14s}  μ={mu:8.2f}  total σ={sig:6.2f} "
                  f"(epi={epi:5.2f}  ale={ale:5.2f})")

    print("\n--- Conclusion ---\n")
    print("Pick V1 if a single tuned learner suffices, you want fastest fit.")
    print("Pick V2 if you need:")
    print("  • epistemic/aleatoric decomposition for active learning")
    print("  • multi-task learning when properties correlate")
    print("  • extended physics features (Miedema, Ω, VEC windows)")
    print("Run `compare_v1_vs_v2` on YOUR data — there is no universal winner.")


if __name__ == "__main__":
    main()
