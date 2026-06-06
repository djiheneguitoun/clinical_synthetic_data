"""Hiérarchie d'attribution mono-label."""

from typing import Any, Mapping

from ..config.diagnostic_criteria import (
    count_cv_risk_factors,
    is_diabetic,
    is_dyslipidemic,
    is_hypertensive,
    is_obese,
)
from .patient_schema import ClassLabel


PatientValues = Mapping[str, Any]


CV_RISK_MIN_TOTAL_FACTORS: int = 3
CV_RISK_MIN_BIOLOGICAL_FACTORS: int = 2


def assign_class(p: PatientValues) -> ClassLabel:
    """Détermine la classe d'un patient à partir de ses valeurs cliniques."""
    n_total, n_biological = count_cv_risk_factors(p)

    if (n_total >= CV_RISK_MIN_TOTAL_FACTORS
            or n_biological >= CV_RISK_MIN_BIOLOGICAL_FACTORS):
        return ClassLabel.CV_RISK

    if is_diabetic(p):
        return ClassLabel.DIABETES
    if is_dyslipidemic(p):
        return ClassLabel.DYSLIPIDEMIA
    if is_hypertensive(p):
        return ClassLabel.HYPERTENSION
    if is_obese(p):
        return ClassLabel.OBESITY

    return ClassLabel.HEALTHY


def matches_expected_class(p: PatientValues, expected: ClassLabel) -> bool:
    """Renvoie True si la classe inférée par `assign_class` correspond à `expected`."""
    return assign_class(p) == expected


def explain_class_assignment(p: PatientValues) -> dict[str, Any]:
    """Trace d'attribution destinée au débogage et aux logs."""
    n_total, n_biological = count_cv_risk_factors(p)
    return {
        "assigned_class": assign_class(p).value,
        "n_total_cv_factors": n_total,
        "n_biological_cv_factors": n_biological,
        "is_diabetic": is_diabetic(p),
        "is_dyslipidemic": is_dyslipidemic(p),
        "is_hypertensive": is_hypertensive(p),
        "is_obese": is_obese(p),
        "cv_risk_rule_triggered": (
            n_total >= CV_RISK_MIN_TOTAL_FACTORS
            or n_biological >= CV_RISK_MIN_BIOLOGICAL_FACTORS
        ),
    }
