"""Validation : bornes, règles inter-variables, cohérence de classe."""

from .class_coherence import check_class_coherence
from .inter_variable_rules import (
    INTER_VARIABLE_RULES,
    check_friedewald,
    check_glucose_hba1c_coherence,
    check_pulse_pressure,
    check_quetelet,
)
from .physiological_bounds import (
    check_absolute_bounds,
    violated_categorical_modality,
    violated_continuous_bound,
)
from .statistics import RejectionStatistics
from .validator import (
    RULE_BOUNDS,
    RULE_CLASS,
    ValidationResult,
    is_valid,
    validate,
)

__all__ = [
    "validate",
    "is_valid",
    "ValidationResult",
    "RULE_BOUNDS",
    "RULE_CLASS",
    "check_absolute_bounds",
    "violated_continuous_bound",
    "violated_categorical_modality",
    "check_quetelet",
    "check_friedewald",
    "check_pulse_pressure",
    "check_glucose_hba1c_coherence",
    "check_class_coherence",
    "INTER_VARIABLE_RULES",
    "RejectionStatistics",
]
