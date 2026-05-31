"""Configuration : plages, seuils, paramètres de génération, corrélations."""

from .clinical_ranges import (
    CATEGORICAL_VARIABLES,
    CONTINUOUS_VARIABLES,
    CategoricalVariableSpec,
    ContinuousVariableSpec,
    VariableType,
    hdl_low_threshold,
)
from .correlations import (
    BASE_CORRELATION_MATRIX,
    COPULA_INDEX,
    COPULA_VARIABLES,
    CORRELATION_PAIRS,
    build_correlation_matrix,
    ensure_valid_correlation_matrix,
    is_positive_definite,
    project_to_nearest_correlation,
)
from .diagnostic_criteria import (
    count_cv_risk_factors,
    cv_risk_factors_breakdown,
    is_diabetic,
    is_dyslipidemic,
    is_hypertensive,
    is_obese,
)
from .generation_params import (
    CATEGORICAL_PROBS_BY_CLASS,
    CONTINUOUS_PARAMS_BY_CLASS,
    ContinuousParam,
)

__all__ = [
    # clinical_ranges
    "CONTINUOUS_VARIABLES",
    "CATEGORICAL_VARIABLES",
    "ContinuousVariableSpec",
    "CategoricalVariableSpec",
    "VariableType",
    "hdl_low_threshold",
    # diagnostic_criteria
    "count_cv_risk_factors",
    "cv_risk_factors_breakdown",
    "is_diabetic",
    "is_dyslipidemic",
    "is_hypertensive",
    "is_obese",
    # generation_params
    "ContinuousParam",
    "CONTINUOUS_PARAMS_BY_CLASS",
    "CATEGORICAL_PROBS_BY_CLASS",
    # correlations
    "COPULA_VARIABLES",
    "COPULA_INDEX",
    "CORRELATION_PAIRS",
    "BASE_CORRELATION_MATRIX",
    "build_correlation_matrix",
    "is_positive_definite",
    "project_to_nearest_correlation",
    "ensure_valid_correlation_matrix",
]
