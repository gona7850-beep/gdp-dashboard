"""Model training and evaluation utilities for the Nb alloy platform."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNetCV
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def within_tolerance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    tol_abs: Optional[float] = None,
    tol_rel: Optional[float] = None,
) -> float:
    """Calculate the fraction of predictions that satisfy tolerance bounds."""
    if tol_abs is None and tol_rel is None:
        raise ValueError("At least one of tol_abs or tol_rel must be provided")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    diffs = np.abs(y_pred - y_true)
    rel_tol = tol_rel * np.abs(y_true) if tol_rel is not None else 0
    abs_tol = tol_abs if tol_abs is not None else 0
    tol = np.maximum(rel_tol, abs_tol)
    return float((diffs <= tol).mean())


def _build_model(algo: str, random_state: int = 0) -> Pipeline:
    algo = algo.lower()
    if algo == "elasticnet":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", ElasticNetCV(cv=5, random_state=random_state)),
        ])
    if algo == "randomforest":
        return Pipeline([
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    random_state=random_state,
                    n_jobs=-1,
                    oob_score=False,
                ),
            )
        ])
    if algo == "svr":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
        ])
    if algo in {"bayesianridge", "bayesian_ridge"}:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", BayesianRidge()),
        ])
    raise ValueError(f"Unknown algorithm: {algo}")


def train_models(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str],
    *,
    group_col: Optional[str] = None,
    algorithms: Optional[Iterable[str]] = None,
    n_splits: int = 5,
    tol_abs: Optional[float] = None,
    tol_rel: Optional[float] = None,
    random_state: int = 0,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Train and evaluate multiple regressors with cross-validation."""
    if algorithms is None:
        algorithms = ["ElasticNet", "RandomForest", "SVR", "BayesianRidge"]

    X = df[list(feature_cols)].to_numpy()
    y = df[target_col].to_numpy()

    if group_col is not None and group_col in df.columns:
        groups = df[group_col].values
        cv = GroupKFold(n_splits=n_splits)
    else:
        groups = None
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        "r2": "r2",
        "rmse": make_scorer(lambda yt, yp: -np.sqrt(mean_squared_error(yt, yp)), greater_is_better=True),
        "mae": make_scorer(lambda yt, yp: -mean_absolute_error(yt, yp), greater_is_better=True),
    }
    if tol_abs is not None or tol_rel is not None:
        scoring["within_tol"] = make_scorer(
            lambda y_true, y_pred: within_tolerance(y_true, y_pred, tol_abs=tol_abs, tol_rel=tol_rel),
            greater_is_better=True,
        )

    models: Dict[str, Any] = {}
    records = []
    for algo in algorithms:
        model = _build_model(algo, random_state)
        cv_results = cross_validate(model, X, y, cv=cv, groups=groups, scoring=scoring, return_train_score=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        models[algo] = model

        record = {"Algorithm": algo}
        for key, vals in cv_results.items():
            if key.startswith("test_"):
                metric = key.replace("test_", "")
                scores = -vals if metric in {"rmse", "mae"} else vals
                record[f"{metric}_mean"] = float(np.mean(scores))
                record[f"{metric}_std"] = float(np.std(scores))
        records.append(record)

    results_df = pd.DataFrame(records)
    if "r2_mean" in results_df.columns:
        results_df = results_df.sort_values(by="r2_mean", ascending=False).reset_index(drop=True)
    return models, results_df
