"""Colab-ready Nb alloy composition-property workflow.

Usage in Google Colab:
1) Paste this whole file into a Colab cell and run.
2) Then call `run_demo()` or `run_pipeline_from_csv('/content/your.csv')`.

This is intentionally self-contained so copy/paste works without package layout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import BayesianRidge, ElasticNetCV
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# ------------------------- Data preprocessing -------------------------

def load_data(file_path: str, format: str = "auto") -> pd.DataFrame:
    fmt = format.lower()
    if fmt == "auto":
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext in {".csv", ".txt"}:
            fmt = "csv"
        elif ext in {".xls", ".xlsx"}:
            fmt = "excel"
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    if fmt == "csv":
        try:
            return pd.read_csv(file_path)
        except Exception:
            return pd.read_csv(file_path, sep=";")
    if fmt == "excel":
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported format: {format}")


def compute_correlations(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[Iterable[str]] = None,
    mic_bins: int = 10,
) -> pd.DataFrame:
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' is not in DataFrame")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = list(feature_cols) if feature_cols is not None else [c for c in numeric_cols if c != target_col]
    if not features:
        raise ValueError("No feature columns provided")

    work = df[features + [target_col]].dropna()
    if len(work) < 3:
        raise ValueError("Need at least 3 valid rows")

    X = work[features].to_numpy()
    y = work[target_col].to_numpy()

    try:
        mi = mutual_info_regression(X, y, random_state=0)
    except Exception:
        y_disc = pd.qcut(y, q=mic_bins, labels=False, duplicates="drop")
        mi = mutual_info_regression(X, y_disc, random_state=0)

    pcc = []
    for i in range(len(features)):
        try:
            p, _ = pearsonr(X[:, i], y)
            pcc.append(abs(float(p)))
        except Exception:
            pcc.append(0.0)

    return pd.DataFrame({"MIC": mi, "PCC": pcc}, index=features).sort_values("MIC", ascending=False)


def select_top_features(corr_df: pd.DataFrame, top_n: int = 5, method: str = "union") -> List[str]:
    method = method.lower()
    if method == "mic":
        return corr_df.sort_values("MIC", ascending=False).head(top_n).index.tolist()
    if method == "pcc":
        return corr_df.sort_values("PCC", ascending=False).head(top_n).index.tolist()
    if method != "union":
        raise ValueError("method must be one of: mic, pcc, union")

    half = max(1, top_n // 2)
    mic_list = corr_df.sort_values("MIC", ascending=False).index.tolist()
    pcc_list = corr_df.sort_values("PCC", ascending=False).index.tolist()
    picks = list(dict.fromkeys(mic_list[:half] + pcc_list[:half]))
    for feat in mic_list + pcc_list:
        if len(picks) >= top_n:
            break
        if feat not in picks:
            picks.append(feat)
    return picks


# ------------------------- Model training -------------------------

def _build_model(name: str, random_state: int = 0) -> Pipeline:
    n = name.lower()
    if n == "elasticnet":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", ElasticNetCV(cv=5, random_state=random_state)),
        ])
    if n == "randomforest":
        return Pipeline([
            ("regressor", RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)),
        ])
    if n == "svr":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
        ])
    if n in {"bayesianridge", "bayesian_ridge"}:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", BayesianRidge()),
        ])
    raise ValueError(f"Unknown algorithm: {name}")


def within_tolerance(y_true: np.ndarray, y_pred: np.ndarray, tol_rel: float = 0.1) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((np.abs(y_pred - y_true) <= tol_rel * np.abs(y_true)).mean())


def train_models(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str],
    algorithms: Iterable[str] = ("ElasticNet", "RandomForest", "SVR", "BayesianRidge"),
    n_splits: int = 5,
    tol_rel: Optional[float] = 0.1,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
    X = df[list(feature_cols)].to_numpy()
    y = df[target_col].to_numpy()

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scoring = {
        "r2": "r2",
        "rmse": make_scorer(lambda yt, yp: -np.sqrt(mean_squared_error(yt, yp)), greater_is_better=True),
        "mae": make_scorer(lambda yt, yp: -mean_absolute_error(yt, yp), greater_is_better=True),
    }
    if tol_rel is not None:
        scoring["within_tol"] = make_scorer(lambda yt, yp: within_tolerance(yt, yp, tol_rel=tol_rel), greater_is_better=True)

    models: Dict[str, Pipeline] = {}
    rows = []
    for algo in algorithms:
        model = _build_model(algo)
        cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
        model.fit(X, y)
        models[algo] = model

        row = {"Algorithm": algo}
        for k, v in cv_res.items():
            if not k.startswith("test_"):
                continue
            metric = k.replace("test_", "")
            scores = -v if metric in {"rmse", "mae"} else v
            row[f"{metric}_mean"] = float(np.mean(scores))
            row[f"{metric}_std"] = float(np.std(scores))
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("r2_mean", ascending=False).reset_index(drop=True)
    return models, results


# ------------------------- Optimization -------------------------

def random_search_optimization(
    model,
    param_ranges: Dict[str, Tuple[float, float]],
    n_samples: int = 2000,
    objective: str = "max",
    random_state: int = 42,
) -> pd.DataFrame:
    objective = objective.lower()
    if objective not in {"max", "min"}:
        raise ValueError("objective must be 'max' or 'min'")

    rng = np.random.default_rng(random_state)
    features = list(param_ranges.keys())
    rows = []
    for _ in range(n_samples):
        sample = {f: float(rng.uniform(*param_ranges[f])) for f in features}
        X = np.array([[sample[f] for f in features]])
        sample["prediction"] = float(np.asarray(model.predict(X)).ravel()[0])
        rows.append(sample)

    out = pd.DataFrame(rows)
    return out.sort_values("prediction", ascending=(objective == "min")).reset_index(drop=True)


# ------------------------- High-level pipeline -------------------------

@dataclass
class PipelineOutput:
    top_features: List[str]
    results_df: pd.DataFrame
    best_algorithm: str
    optimization_df: pd.DataFrame


def run_pipeline(df: pd.DataFrame, target_col: str, top_n_features: int = 5, random_search_samples: int = 2000) -> PipelineOutput:
    corr = compute_correlations(df, target_col=target_col)
    top_features = select_top_features(corr, top_n=top_n_features, method="union")

    models, results = train_models(df, target_col=target_col, feature_cols=top_features)
    best_algo = str(results.iloc[0]["Algorithm"])
    best_model = models[best_algo]

    ranges = {f: (float(df[f].min()), float(df[f].max())) for f in top_features}
    opt = random_search_optimization(best_model, ranges, n_samples=random_search_samples)

    return PipelineOutput(top_features=top_features, results_df=results, best_algorithm=best_algo, optimization_df=opt)


def run_pipeline_from_csv(csv_path: str, target_col: str = "HV") -> PipelineOutput:
    df = load_data(csv_path)
    return run_pipeline(df, target_col=target_col)


def run_demo() -> PipelineOutput:
    demo_df = pd.DataFrame(
        {
            "Nb": [70, 68, 72, 66, 64, 74, 62, 69, 67, 71],
            "Si": [8, 10, 7, 12, 14, 6, 15, 9, 11, 8],
            "Ti": [10, 12, 9, 11, 10, 8, 12, 11, 10, 9],
            "Cr": [4, 3, 5, 4, 5, 4, 4, 3, 4, 5],
            "Al": [3, 2, 2, 3, 3, 3, 4, 3, 4, 2],
            "Hf": [5, 5, 5, 4, 4, 5, 3, 5, 4, 5],
            "HV": [310, 325, 300, 340, 355, 295, 365, 330, 345, 305],
        }
    )
    out = run_pipeline(demo_df, target_col="HV")

    print("Top features:", out.top_features)
    print("\nModel leaderboard:")
    print(out.results_df)
    print(f"\nBest algorithm: {out.best_algorithm}")
    print("\nTop optimization candidates:")
    print(out.optimization_df.head(10))

    return out


if __name__ == "__main__":
    run_demo()
