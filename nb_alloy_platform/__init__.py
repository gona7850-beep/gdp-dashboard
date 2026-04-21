"""Top-level package for the Nb alloy composition–property platform."""

from .data_preprocessing import (
    compute_correlations,
    load_data,
    prepare_long_format,
    select_top_features,
)
from .model_training import train_models, within_tolerance
from .optimization import multiobjective_pareto, random_search_optimization
from .shap_analysis import analyse_shap

__all__ = [
    "load_data",
    "prepare_long_format",
    "compute_correlations",
    "select_top_features",
    "train_models",
    "within_tolerance",
    "analyse_shap",
    "random_search_optimization",
    "multiobjective_pareto",
]
