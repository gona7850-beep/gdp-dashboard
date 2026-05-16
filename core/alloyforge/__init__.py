"""AlloyForge core ML modules."""

from .data_pipeline import (
    CompositionFeaturizer,
    Dataset,
    ELEMENT_PROPERTIES,
    build_preprocessor,
)
from .forward_model import ForwardModel
from .forward_model_v2 import ForwardModelV2
from .physics_features import ExtendedFeaturizer, make_extended
from .benchmark import benchmark_models, compare_v1_vs_v2, leaderboard_pivot
from .validation import ConformalCalibrator, DomainOfApplicability, reliability_diagram
from .feasibility import (
    Constraint,
    FeasibilityChecker,
    FeasibilityResult,
    hume_rothery_size_mismatch,
    vec_window,
    element_bounds,
    composition_sum_equals_one,
    ved_window,
    default_checker,
)
from .inverse_design import DesignSpec, InverseDesigner
from .active_learning import ActiveLearner, pareto_front, hypervolume_2d
from .explainability import Explainer
from .llm_assistant import LLMAssistant

__all__ = [
    "CompositionFeaturizer", "Dataset", "ELEMENT_PROPERTIES", "build_preprocessor",
    "ExtendedFeaturizer", "make_extended",
    "ForwardModel", "ForwardModelV2",
    "benchmark_models", "compare_v1_vs_v2", "leaderboard_pivot",
    "ConformalCalibrator", "DomainOfApplicability", "reliability_diagram",
    "Constraint", "FeasibilityChecker", "FeasibilityResult",
    "hume_rothery_size_mismatch", "vec_window", "element_bounds",
    "composition_sum_equals_one", "ved_window", "default_checker",
    "DesignSpec", "InverseDesigner",
    "ActiveLearner", "pareto_front", "hypervolume_2d",
    "Explainer", "LLMAssistant",
]
