"""Benchmark harness — k-fold × n_seeds + group-CV + permutation/Y-randomization.

The user's research workflow specifies five validation requirements:
  1. 5-fold × 10 random seeds → mean ± std on R², RMSE, MAE
  2. LeaveOneGroupOut by alloy class — generalization to unseen alloy systems
  3. Permutation test → p-value of "model beats random label"
  4. Y-randomization → null distribution of R² on shuffled targets
  5. Prediction-interval coverage (Bayesian / GPR only) — nominal vs empirical

`benchmark()` performs (1)–(2). Use `permutation_pvalue()` and
`y_randomization()` separately because they are expensive (100+ refits).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, permutation_test_score


@dataclass
class FoldResult:
    seed: int
    fold: int
    r2: float
    rmse: float
    mae: float


@dataclass
class ModelResult:
    name: str
    r2_mean: float
    r2_std: float
    rmse_mean: float
    rmse_std: float
    mae_mean: float
    mae_std: float
    n_folds: int
    runtime_s: float
    permutation_p: float | None = None
    y_random_r2_mean: float | None = None
    folds: list[FoldResult] = field(default_factory=list)

    def as_row(self) -> dict:
        d = asdict(self)
        d.pop("folds")
        return d


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _flatten(arr) -> np.ndarray:
    a = np.asarray(arr).reshape(-1)
    return a


def benchmark(
    X: pd.DataFrame,
    y: pd.Series,
    model_factories: dict[str, Callable[[], object]],
    *,
    cv: str = "kfold",
    n_splits: int = 5,
    n_seeds: int = 10,
    groups: pd.Series | None = None,
) -> list[ModelResult]:
    """Run cross-validated benchmark.

    cv ∈ {"kfold", "group"}:
      - "kfold": KFold(n_splits) repeated for `n_seeds` shuffles.
      - "group": LeaveOneGroupOut over `groups` (n_seeds ignored, deterministic).
    """
    Xv = X.values
    yv = y.values
    out: list[ModelResult] = []

    for name, factory in model_factories.items():
        t0 = time.perf_counter()
        folds: list[FoldResult] = []

        try:
            if cv == "group":
                if groups is None or groups.nunique() < 2:
                    raise ValueError("Group CV requires `groups` Series with ≥2 unique groups")
                logo = LeaveOneGroupOut()
                splits = list(logo.split(Xv, yv, groups=groups.values))
                for i, (tr, te) in enumerate(splits):
                    if len(te) == 0:
                        continue
                    est = factory()
                    est.fit(Xv[tr], yv[tr])
                    pred = _flatten(est.predict(Xv[te]))
                    folds.append(FoldResult(0, i, r2_score(yv[te], pred), _rmse(yv[te], pred), mean_absolute_error(yv[te], pred)))
            else:
                for seed in range(n_seeds):
                    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                    for fold, (tr, te) in enumerate(kf.split(Xv)):
                        est = factory()
                        est.fit(Xv[tr], yv[tr])
                        pred = _flatten(est.predict(Xv[te]))
                        folds.append(FoldResult(seed, fold, r2_score(yv[te], pred), _rmse(yv[te], pred), mean_absolute_error(yv[te], pred)))
        except Exception as e:
            out.append(ModelResult(name, float("nan"), float("nan"), float("nan"), float("nan"),
                                   float("nan"), float("nan"), 0, time.perf_counter() - t0))
            print(f"[benchmark] {name} failed: {e}")
            continue

        if not folds:
            continue
        r2 = np.array([f.r2 for f in folds])
        rmse = np.array([f.rmse for f in folds])
        mae = np.array([f.mae for f in folds])
        out.append(ModelResult(
            name=name,
            r2_mean=float(r2.mean()), r2_std=float(r2.std()),
            rmse_mean=float(rmse.mean()), rmse_std=float(rmse.std()),
            mae_mean=float(mae.mean()), mae_std=float(mae.std()),
            n_folds=len(folds),
            runtime_s=time.perf_counter() - t0,
            folds=folds,
        ))
    return out


def permutation_pvalue(
    X: pd.DataFrame, y: pd.Series, factory: Callable[[], object],
    *, n_permutations: int = 100, n_splits: int = 5, random_state: int = 0,
) -> tuple[float, float]:
    """sklearn permutation_test_score wrapper. Returns (score, p-value)."""
    score, _, p = permutation_test_score(
        factory(), X.values, y.values, scoring="r2",
        cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        n_permutations=n_permutations, random_state=random_state, n_jobs=-1,
    )
    return float(score), float(p)


def y_randomization(
    X: pd.DataFrame, y: pd.Series, factory: Callable[[], object],
    *, n_iter: int = 100, n_splits: int = 5, random_state: int = 0,
) -> dict[str, float]:
    """Shuffle y, fit, record R². Reports null distribution mean/std/95%."""
    rng = np.random.default_rng(random_state)
    scores = []
    yv = y.values
    Xv = X.values
    for _ in range(n_iter):
        y_shuf = yv.copy()
        rng.shuffle(y_shuf)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(rng.integers(0, 1_000_000)))
        cv_scores = []
        for tr, te in kf.split(Xv):
            est = factory()
            est.fit(Xv[tr], y_shuf[tr])
            pred = _flatten(est.predict(Xv[te]))
            cv_scores.append(r2_score(y_shuf[te], pred))
        scores.append(np.mean(cv_scores))
    arr = np.array(scores)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def to_dataframe(results: list[ModelResult]) -> pd.DataFrame:
    return pd.DataFrame([r.as_row() for r in results]).sort_values("r2_mean", ascending=False).reset_index(drop=True)
