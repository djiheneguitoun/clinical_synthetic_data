"""Validation épidémiologique des patterns cliniques attendus."""

from __future__ import annotations

import pandas as pd

from ..config.clinical_ranges import (
    DX_BMI,
    DX_GLUCOSE,
    DX_HBA1C,
    DX_LDL,
    DX_SBP,
)
from ..core.patient_schema import ClassLabel


EXPECTED_PATTERNS = {
    ClassLabel.DIABETES: {
        "fasting_glucose": ("min_mean", DX_GLUCOSE),
        "hba1c":           ("min_mean", DX_HBA1C),
    },
    ClassLabel.DYSLIPIDEMIA: {
        "ldl":             ("min_mean", DX_LDL * 0.85),
    },
    ClassLabel.HYPERTENSION: {
        "sbp":             ("min_mean", DX_SBP),
    },
    ClassLabel.OBESITY: {
        "bmi":             ("min_mean", DX_BMI),
    },
    ClassLabel.HEALTHY: {
        "fasting_glucose": ("max_mean", 100.0),
        "bmi":             ("max_mean", 25.0),
        "sbp":             ("max_mean", 120.0),
    },
}


EXPECTED_AGE_ORDERING = (
    (ClassLabel.HEALTHY, ClassLabel.HYPERTENSION),
    (ClassLabel.HEALTHY, ClassLabel.CV_RISK),
    (ClassLabel.HEALTHY, ClassLabel.DIABETES),
)


def check_expected_pattern(
    df: pd.DataFrame,
    class_label: ClassLabel,
    variable: str,
    rule: str,
    threshold: float,
) -> dict:
    """Vérifie qu'un pattern attendu est satisfait dans une classe."""
    subset = df[df["class_label"] == class_label.value]
    if len(subset) == 0:
        return {"passed": False, "reason": f"Classe {class_label.value} absente"}

    mean_value = float(subset[variable].mean())
    if rule == "min_mean":
        passed = mean_value >= threshold
    elif rule == "max_mean":
        passed = mean_value <= threshold
    else:
        raise ValueError(f"rule inconnue : {rule!r}")

    return {
        "class": class_label.value,
        "variable": variable,
        "rule": rule,
        "threshold": threshold,
        "observed_mean": round(mean_value, 2),
        "passed": passed,
    }


def validate_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Lance toutes les vérifications de pattern et retourne un tableau résumé."""
    rows = []
    for class_label, checks in EXPECTED_PATTERNS.items():
        for variable, (rule, threshold) in checks.items():
            rows.append(check_expected_pattern(df, class_label, variable, rule, threshold))
    return pd.DataFrame(rows)


def check_age_ordering(df: pd.DataFrame) -> pd.DataFrame:
    """Vérifie que l'âge moyen suit la hiérarchie attendue."""
    means = df.groupby("class_label")["age"].mean().to_dict()

    rows = []
    for younger, older in EXPECTED_AGE_ORDERING:
        m_young = means.get(younger.value, float("nan"))
        m_old = means.get(older.value, float("nan"))
        rows.append({
            "younger_class": younger.value,
            "older_class": older.value,
            "mean_age_younger": round(m_young, 1),
            "mean_age_older": round(m_old, 1),
            "passed": m_old > m_young,
        })
    return pd.DataFrame(rows)


def build_epidemiological_report(df: pd.DataFrame) -> dict:
    """Synthèse pour intégration au rapport final."""
    patterns = validate_all_patterns(df)
    age_ordering = check_age_ordering(df)
    return {
        "patterns": patterns.to_dict(orient="records"),
        "n_patterns_passed": int(patterns["passed"].sum()),
        "n_patterns_total": int(len(patterns)),
        "age_ordering": age_ordering.to_dict(orient="records"),
        "age_ordering_all_passed": bool(age_ordering["passed"].all()),
    }
