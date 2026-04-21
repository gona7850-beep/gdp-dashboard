"""Utilities for loading and preprocessing alloy datasets."""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

_ID_CANDIDATES = {"alloy_id", "id", "ref", "reference"}


def load_data(file_path: str, format: str = "auto") -> pd.DataFrame:
    """Load a dataset from CSV or Excel."""
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


def prepare_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide-format alloy table to long format."""
    id_col = next((c for c in df.columns if c.lower() in _ID_CANDIDATES), None)

    comp_cols = []
    for col in df.columns:
        if id_col and col == id_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            col_min, col_max = df[col].min(), df[col].max()
            if pd.notna(col_min) and pd.notna(col_max) and 0 <= col_min and col_max <= 100:
                comp_cols.append(col)

    target_cols = [
        c
        for c in df.columns
        if c not in comp_cols and (not id_col or c != id_col) and pd.api.types.is_numeric_dtype(df[c])
    ]

    long_rows = []
    for idx, row in df.iterrows():
        alloy_id = row[id_col] if id_col else idx
        comp_vals = {col: row[col] for col in comp_cols}
        for tgt in target_cols:
            val = row[tgt]
            if pd.notna(val):
                entry = {"Alloy_ID": alloy_id, "Property": tgt, "Value": val}
                entry.update(comp_vals)
                long_rows.append(entry)

    return pd.DataFrame(long_rows)


def compute_correlations(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[Iterable[str]] = None,
    *,
    mic_bins: int = 10,
) -> pd.DataFrame:
    """Compute mutual information and Pearson correlations for features."""
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' is not present in DataFrame")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = list(feature_cols) if feature_cols is not None else [c for c in numeric_cols if c != target_col]
    if not features:
        raise ValueError("No feature columns available for correlation analysis")

    work = df[features + [target_col]].dropna()
    if len(work) < 3:
        raise ValueError("Not enough non-null rows to compute correlations")

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

    corr_df = pd.DataFrame({"MIC": mi, "PCC": pcc}, index=features)
    return corr_df.sort_values(by="MIC", ascending=False)


def select_top_features(corr_df: pd.DataFrame, top_n: int = 10, method: str = "mic") -> List[str]:
    """Select top features using MIC, PCC, or a union strategy."""
    if top_n <= 0:
        return []
    method = method.lower()
    if method not in {"mic", "pcc", "union"}:
        raise ValueError("method must be 'mic', 'pcc' or 'union'")

    if method == "mic":
        return corr_df.sort_values("MIC", ascending=False).head(top_n).index.tolist()
    if method == "pcc":
        return corr_df.sort_values("PCC", ascending=False).head(top_n).index.tolist()

    half = max(1, top_n // 2)
    mic_list = corr_df.sort_values("MIC", ascending=False).head(top_n).index.tolist()
    pcc_list = corr_df.sort_values("PCC", ascending=False).head(top_n).index.tolist()
    union = list(dict.fromkeys(mic_list[:half] + pcc_list[:half]))
    for feat in mic_list + pcc_list:
        if len(union) >= top_n:
            break
        if feat not in union:
            union.append(feat)
    return union
