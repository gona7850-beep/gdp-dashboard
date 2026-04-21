"""Simple optimisation routines for alloy composition design."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _predict_scalar(model, X: np.ndarray) -> float:
    return float(np.asarray(model.predict(X)).ravel()[0])


def random_search_optimization(
    model,
    param_ranges: Dict[str, Tuple[float, float]],
    n_samples: int = 1000,
    objective: str = "max",
    enforce_sum: bool = True,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Randomly sample compositions and rank by predicted property."""
    objective = objective.lower()
    if objective not in {"max", "min"}:
        raise ValueError("objective must be 'max' or 'min'")

    rng = np.random.default_rng(random_state)
    feature_names = list(param_ranges.keys())
    samples = []

    for _ in range(n_samples):
        comp = {}
        remaining = 100.0
        for i, feat in enumerate(feature_names):
            lo, hi = param_ranges[feat]
            if enforce_sum and i == len(feature_names) - 1:
                val = max(lo, min(hi, remaining))
            else:
                val = float(rng.uniform(lo, hi))
                if enforce_sum:
                    remaining -= val
            comp[feat] = val

        if enforce_sum and remaining < 0:
            continue

        X = np.array([[comp[f] for f in feature_names]], dtype=float)
        comp["prediction"] = _predict_scalar(model, X)
        samples.append(comp)

    df = pd.DataFrame(samples)
    return df.sort_values(by="prediction", ascending=(objective == "min")).reset_index(drop=True)


def multiobjective_pareto(
    models: tuple[object, object],
    param_ranges: Dict[str, Tuple[float, float]],
    n_samples: int = 2000,
    enforce_sum: bool = True,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Approximate a two-objective Pareto frontier by random sampling."""
    model1, model2 = models
    rng = np.random.default_rng(random_state)
    feature_names = list(param_ranges.keys())
    samples = []

    for _ in range(n_samples):
        comp = {}
        remaining = 100.0
        for i, feat in enumerate(feature_names):
            lo, hi = param_ranges[feat]
            if enforce_sum and i == len(feature_names) - 1:
                val = max(lo, min(hi, remaining))
            else:
                val = float(rng.uniform(lo, hi))
                if enforce_sum:
                    remaining -= val
            comp[feat] = val

        if enforce_sum and remaining < 0:
            continue

        X = np.array([[comp[f] for f in feature_names]], dtype=float)
        comp["pred1"] = _predict_scalar(model1, X)
        comp["pred2"] = _predict_scalar(model2, X)
        samples.append(comp)

    df = pd.DataFrame(samples)
    preds = df[["pred1", "pred2"]].to_numpy()
    pareto_mask = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if pareto_mask[i] and (preds > preds[i]).all(axis=1).any():
            pareto_mask[i] = False

    return (
        df[pareto_mask]
        .copy()
        .sort_values(by=["pred1", "pred2"], ascending=[False, False])
        .reset_index(drop=True)
    )
