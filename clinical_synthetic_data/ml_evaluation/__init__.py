"""Évaluation par apprentissage automatique (rapport section 8)."""

from .evaluation import (
    build_ml_evaluation_report,
    cross_method_evaluation,
    cross_validate_models,
    evaluate_all_models_on_split,
    evaluate_model,
)
from .models import (
    MODEL_DISPLAY_NAMES,
    MODEL_FACTORIES,
    make_logistic_regression,
    make_mlp,
    make_random_forest,
)
from .preprocessing import (
    NOMINAL_FIELDS,
    ORDINAL_CATEGORIES,
    ORDINAL_FIELDS,
    build_preprocessor,
    prepare_features_and_target,
)

__all__ = [
    "build_preprocessor",
    "prepare_features_and_target",
    "NOMINAL_FIELDS",
    "ORDINAL_FIELDS",
    "ORDINAL_CATEGORIES",
    "make_logistic_regression",
    "make_random_forest",
    "make_mlp",
    "MODEL_FACTORIES",
    "MODEL_DISPLAY_NAMES",
    "evaluate_model",
    "evaluate_all_models_on_split",
    "cross_validate_models",
    "cross_method_evaluation",
    "build_ml_evaluation_report",
]
