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
from .reference_data import (
    ALLOYS,
    KnownAlloy,
    PROPERTY_COLUMNS,
    atomic_to_weight_pct,
    find_alloy,
    reference_dataset,
    reference_elements,
    reference_families,
    weight_to_atomic_pct,
)
from .data_ingestion import (
    IngestSummary,
    convert_value,
    flag_outliers,
    infer_units,
    merge_datasets,
    normalize_composition,
    normalize_units,
)
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
    "ALLOYS", "KnownAlloy", "PROPERTY_COLUMNS",
    "reference_dataset", "reference_elements", "reference_families",
    "find_alloy", "weight_to_atomic_pct", "atomic_to_weight_pct",
    "IngestSummary", "convert_value", "flag_outliers", "infer_units",
    "merge_datasets", "normalize_composition", "normalize_units",
    "ConformalCalibrator", "DomainOfApplicability", "reliability_diagram",
    "Constraint", "FeasibilityChecker", "FeasibilityResult",
    "hume_rothery_size_mismatch", "vec_window", "element_bounds",
    "composition_sum_equals_one", "ved_window", "default_checker",
    "DesignSpec", "InverseDesigner",
    "ActiveLearner", "pareto_front", "hypervolume_2d",
    "Explainer", "LLMAssistant",
]
