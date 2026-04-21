"""SHAP analysis utilities for alloy property models."""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import shap


def _extract_shap_values(shap_values) -> np.ndarray:
    if hasattr(shap_values, "values"):
        return np.asarray(shap_values.values)
    return np.asarray(shap_values)


def analyse_shap(
    model,
    X: np.ndarray,
    feature_names: Iterable[str],
    outdir: str,
    target_name: str,
    max_display: int = 20,
) -> None:
    """Compute SHAP values and save summary/dependence plots."""
    os.makedirs(outdir, exist_ok=True)
    estimator = model.named_steps["regressor"] if hasattr(model, "named_steps") and "regressor" in model.named_steps else model

    try:
        explainer = shap.Explainer(estimator.predict, X)
        shap_values = explainer(X)
    except Exception:
        explainer = shap.KernelExplainer(estimator.predict, X[:100])
        shap_values = explainer.shap_values(X[:200])

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=list(feature_names), max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{target_name}_shap_summary.png"), dpi=300)
    plt.close()

    vals = np.abs(_extract_shap_values(shap_values)).mean(axis=0)
    top_idx = int(np.argmax(vals))
    top_feat = list(feature_names)[top_idx]
    plt.figure()
    shap.dependence_plot(top_feat, shap_values, X, feature_names=list(feature_names), show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{target_name}_shap_dependence_{top_feat}.png"), dpi=300)
    plt.close()
