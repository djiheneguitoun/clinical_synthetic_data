"""Cœur métier : schéma patient et règle d'attribution."""

from .class_assigner import (
    assign_class,
    explain_class_assignment,
    matches_expected_class,
)
from .patient_schema import (
    ActivityLevel,
    AlcoholConsumption,
    CATEGORICAL_FIELDS,
    CONTINUOUS_FIELDS,
    ClassLabel,
    DietQuality,
    Patient,
    Sex,
    SmokingStatus,
)

__all__ = [
    "Patient",
    "ClassLabel",
    "Sex",
    "ActivityLevel",
    "SmokingStatus",
    "AlcoholConsumption",
    "DietQuality",
    "CONTINUOUS_FIELDS",
    "CATEGORICAL_FIELDS",
    "assign_class",
    "matches_expected_class",
    "explain_class_assignment",
]
