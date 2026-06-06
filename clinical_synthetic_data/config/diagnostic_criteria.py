"""Prédicats diagnostiques par classe."""

from typing import Any, Mapping

from .clinical_ranges import (
    DX_GLUCOSE, DX_HBA1C, RF_GLUCOSE, RF_HBA1C,
    DX_TOTAL_CHOL, DX_LDL, DX_TRIGLYCERIDES,
    RF_TOTAL_CHOL, RF_LDL, RF_TRIGLYCERIDES,
    DX_SBP, DX_DBP, RF_SBP, RF_DBP,
    DX_BMI, RF_BMI,
    RF_HEART_RATE,
    hdl_low_threshold,
)


PatientValues = Mapping[str, Any]


def is_diabetic(p: PatientValues) -> bool:
    """Diabète : glycémie à jeun ≥ 126 mg/dL ou HbA1c ≥ 6,5 %."""
    return p["fasting_glucose"] >= DX_GLUCOSE or p["hba1c"] >= DX_HBA1C


def is_dyslipidemic(p: PatientValues) -> bool:
    """Dyslipidémie : au moins une anomalie lipidique."""
    hdl_threshold = hdl_low_threshold(p["sex"])
    return (
        p["total_chol"] >= DX_TOTAL_CHOL
        or p["ldl"] >= DX_LDL
        or p["hdl"] < hdl_threshold
        or p["triglycerides"] >= DX_TRIGLYCERIDES
    )


def is_hypertensive(p: PatientValues) -> bool:
    """Hypertension : PAS ≥ 140 mmHg ou PAD ≥ 90 mmHg."""
    return p["sbp"] >= DX_SBP or p["dbp"] >= DX_DBP


def is_obese(p: PatientValues) -> bool:
    """Obésité : IMC ≥ 30 kg/m²."""
    return p["bmi"] >= DX_BMI


def _has_glycemic_risk(p: PatientValues) -> bool:
    """Glycémie ≥ 100 ou HbA1c ≥ 5,7 %."""
    return p["fasting_glucose"] >= RF_GLUCOSE or p["hba1c"] >= RF_HBA1C


def _has_lipid_risk(p: PatientValues) -> bool:
    """Anomalie lipidique en zone intermédiaire."""
    hdl_threshold = hdl_low_threshold(p["sex"])
    return (
        p["ldl"] >= RF_LDL
        or p["total_chol"] >= RF_TOTAL_CHOL
        or p["hdl"] < hdl_threshold
        or p["triglycerides"] >= RF_TRIGLYCERIDES
    )


def _has_blood_pressure_risk(p: PatientValues) -> bool:
    """Pression artérielle ≥ 120/80 mmHg."""
    return p["sbp"] >= RF_SBP or p["dbp"] >= RF_DBP


def _has_bmi_risk(p: PatientValues) -> bool:
    """IMC ≥ 25 kg/m²."""
    return p["bmi"] >= RF_BMI


def _has_heart_rate_risk(p: PatientValues) -> bool:
    """Fréquence cardiaque au repos ≥ 80 bpm."""
    return p["heart_rate"] >= RF_HEART_RATE


def _has_smoking_risk(p: PatientValues) -> bool:
    """Tabagisme actif."""
    return p["smoking"] == "Current"


def _has_sedentary_risk(p: PatientValues) -> bool:
    """Sédentarité."""
    return p["physical_activity"] == "Sedentary"


def _has_alcohol_risk(p: PatientValues) -> bool:
    """Consommation d'alcool excessive."""
    return p["alcohol"] == "Excessive"


def _has_diet_risk(p: PatientValues) -> bool:
    """Mauvaise alimentation."""
    return p["diet_quality"] == "Poor"


def count_cv_risk_factors(p: PatientValues) -> tuple[int, int]:
    """Retourne (n_total, n_biological) facteurs de risque cardiovasculaire."""
    biological_checks = (
        _has_glycemic_risk,
        _has_lipid_risk,
        _has_blood_pressure_risk,
        _has_bmi_risk,
    )
    n_biological = sum(1 for check in biological_checks if check(p))

    other_checks = (
        _has_heart_rate_risk,
        _has_smoking_risk,
        _has_sedentary_risk,
        _has_alcohol_risk,
        _has_diet_risk,
    )
    n_other = sum(1 for check in other_checks if check(p))

    return n_biological + n_other, n_biological


def cv_risk_factors_breakdown(p: PatientValues) -> dict[str, bool]:
    """Détail booléen de chaque facteur de risque."""
    return {
        "glycemic": _has_glycemic_risk(p),
        "lipid": _has_lipid_risk(p),
        "blood_pressure": _has_blood_pressure_risk(p),
        "bmi": _has_bmi_risk(p),
        "heart_rate": _has_heart_rate_risk(p),
        "smoking": _has_smoking_risk(p),
        "sedentary": _has_sedentary_risk(p),
        "alcohol": _has_alcohol_risk(p),
        "diet": _has_diet_risk(p),
    }
