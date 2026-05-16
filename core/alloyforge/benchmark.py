"""Head-to-head model benchmarking.

Compare any number of models on the same dataset with consistent CV splits.
Outputs a leaderboard ``pd.DataFrame`` with mean ± std of R^2, MAE, and
RMSE per target, sorted by mean R^2.

Designed to make claims like "v2 beats v1 by X% R²" measurable and
reproducible. Use ``benchmark_models`` for full custom comparison or
``compare_v1_vs_v2`` as the canonical shortcut.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

from .data_pipeline import CompositionFeaturizer, Dataset
from .forward_model import ForwardModel
from .forward_model_v2 import ForwardModelV2
from .physics_features import ExtendedFeaturizer, make_extended

log = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    model_name: str
    target: str
    r2_mean: float
    r2_std: float
    mae_mean: float
    rmse_mean: float
    fit_seconds: float
    extra: Dict[str, object]


# ---------------------------------------------------------------------------
def _make_splits(dataset: Dataset, n_splits: int, seed: int):
    idx = np.arange(len(dataset.compositions))
    if dataset.groups is not None and dataset.groups.nunique() >= n_splits:
        cv = GroupKFold(n_splits=n_splits)
        return list(cv.split(idx, groups=dataset.groups.values))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(idx))


def _eval_one(
    factory: Callable[[], object],
    name: str,
    dataset: Dataset,
    element_columns: Sequence[str],
    targets: Sequence[str],
    n_splits: int,
    seed: int,
) -> List[BenchmarkResult]:
    """Train ``factory()`` model on each CV fold and collect per-fold scores.

    The factory takes no args and returns a fresh model ready to ``.fit``.
    """
    splits = _make_splits(dataset, n_splits, seed)
    # fold_scores[target] -> list of (r2, mae, rmse) tuples
    fold_scores: Dict[str, List[tuple]] = {t: [] for t in targets}
    t0 = time.perf_counter()
    for tr_idx, te_idx in splits:
        ds_tr = Dataset(
            compositions=dataset.compositions.iloc[tr_idx].reset_index(drop=True),
            properties=dataset.properties.iloc[tr_idx].reset_index(drop=True),
            process=(dataset.process.iloc[tr_idx].reset_index(drop=True)
                     if dataset.process is not None else None),
            groups=(dataset.groups.iloc[tr_idx].reset_index(drop=True)
                    if dataset.groups is not None else None),
        )
        mdl = factory()
        mdl.fit(ds_tr)
        comp_te = dataset.compositions.iloc[te_idx].reset_index(drop=True)
        proc_te = (dataset.process.iloc[te_idx].reset_index(drop=True)
                   if dataset.process is not None else None)
        preds = mdl.predict(comp_te, process=proc_te)
        for t in targets:
            y_true = dataset.properties.iloc[te_idx][t].to_numpy(dtype=float)
            y_pred = preds[f"{t}_mean"].to_numpy(dtype=float)
            fold_scores[t].append((
                float(r2_score(y_true, y_pred)),
                float(mean_absolute_error(y_true, y_pred)),
                float(np.sqrt(mean_squared_error(y_true, y_pred))),
            ))
    dt = time.perf_counter() - t0
    results = []
    for t in targets:
        arr = np.array(fold_scores[t])
        results.append(BenchmarkResult(
            model_name=name,
            target=t,
            r2_mean=float(arr[:, 0].mean()),
            r2_std=float(arr[:, 0].std()),
            mae_mean=float(arr[:, 1].mean()),
            rmse_mean=float(arr[:, 2].mean()),
            fit_seconds=dt / max(1, len(splits)),
            extra={"n_folds": len(splits)},
        ))
    return results


# ---------------------------------------------------------------------------
def benchmark_models(
    models: Dict[str, Callable[[], object]],
    dataset: Dataset,
    element_columns: Sequence[str],
    targets: Sequence[str],
    n_splits: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Run every model in ``models`` on the same CV splits.

    ``models`` is a dict mapping name -> factory function (no args).
    Returns a tidy DataFrame sorted by mean R².
    """
    rows: List[BenchmarkResult] = []
    for name, factory in models.items():
        log.info(f"benchmarking {name}")
        rows.extend(_eval_one(
            factory, name, dataset, element_columns, targets, n_splits, seed,
        ))
    df = pd.DataFrame([{
        "model": r.model_name,
        "target": r.target,
        "r2_mean": round(r.r2_mean, 4),
        "r2_std": round(r.r2_std, 4),
        "mae_mean": round(r.mae_mean, 4),
        "rmse_mean": round(r.rmse_mean, 4),
        "fit_seconds": round(r.fit_seconds, 2),
    } for r in rows])
    return df.sort_values(["target", "r2_mean"], ascending=[True, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
def compare_v1_vs_v2(
    dataset: Dataset,
    element_columns: Sequence[str],
    targets: Sequence[str],
    n_splits: int = 5,
    n_trials_v1: int = 8,
    v2_seeds: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Canonical v1-vs-v2 leaderboard on ``dataset``.

    Returns a DataFrame with columns:
        model, target, r2_mean, r2_std, mae_mean, rmse_mean, fit_seconds
    """
    base_feat = lambda: CompositionFeaturizer(element_columns=list(element_columns))
    ext_feat = lambda: make_extended(element_columns)

    def make_v1():
        return ForwardModel(
            featurizer=base_feat(),
            targets=list(targets),
            n_cv_splits=min(n_splits, 5),
        )

    def make_v1_fit():
        m = make_v1()
        # Wrap fit so the factory call also handles trials param
        orig_fit = m.fit
        m.fit = lambda ds, **kw: orig_fit(ds, n_trials=n_trials_v1, **kw)  # type: ignore[method-assign]
        return m

    def make_v2_base():
        return ForwardModelV2(
            featurizer=base_feat(),
            targets=list(targets),
            n_seeds=v2_seeds,
            n_cv_splits=min(n_splits, 5),
            share_targets=False,
            random_state=seed,
        )

    def make_v2_ext():
        return ForwardModelV2(
            featurizer=ext_feat(),
            targets=list(targets),
            n_seeds=v2_seeds,
            n_cv_splits=min(n_splits, 5),
            share_targets=False,
            random_state=seed,
        )

    def make_v2_ext_multitask():
        return ForwardModelV2(
            featurizer=ext_feat(),
            targets=list(targets),
            n_seeds=v2_seeds,
            n_cv_splits=min(n_splits, 5),
            share_targets=True,
            random_state=seed,
        )

    return benchmark_models(
        models={
            "v1_xgb_gp": make_v1_fit,
            "v2_stack_base_feats": make_v2_base,
            "v2_stack_ext_feats": make_v2_ext,
            "v2_stack_ext_multitask": make_v2_ext_multitask,
        },
        dataset=dataset,
        element_columns=element_columns,
        targets=targets,
        n_splits=n_splits,
        seed=seed,
    )


def leaderboard_pivot(df: pd.DataFrame, metric: str = "r2_mean") -> pd.DataFrame:
    """Pivot a benchmark dataframe into a model × target table."""
    pivot = df.pivot(index="model", columns="target", values=metric)
    pivot["__avg__"] = pivot.mean(axis=1)
    return pivot.sort_values("__avg__", ascending=False)


__all__ = [
    "BenchmarkResult",
    "benchmark_models",
    "compare_v1_vs_v2",
    "leaderboard_pivot",
]
