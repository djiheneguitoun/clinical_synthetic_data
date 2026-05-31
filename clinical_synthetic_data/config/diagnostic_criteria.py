"""
Prédicats diagnostiques par classe (rapport section 3.3).

Chaque fonction prend une vue plate des valeurs cliniques d'un patient
 et retourne un booléen. Les seuils proviennent exclusivement
de `clinical_ranges` : aucune valeur n'est dupliquée ici.

"""

from typing import Any, Mapping

from .clinical_ranges import (
    # Diabète
    DX_GLUCOSE, DX_HBA1C, RF_GLUCOSE, RF_HBA1C,
    # Dyslipidémie
    DX_TOTAL_CHOL, DX_LDL, DX_TRIGLYCERIDES,
    RF_TOTAL_CHOL, RF_LDL, RF_TRIGLYCERIDES,
    # Hypertension
    DX_SBP, DX_DBP, RF_SBP, RF_DBP,
    # Obésité
    DX_BMI, RF_BMI,
    # Fréquence cardiaque
    RF_HEART_RATE,
    # HDL sexe-dépendant
    hdl_low_threshold,
)


PatientValues = Mapping[str, Any]


# ---------------------------------------------------------------------------
# Prédicats diagnostiques stricts (entrée de classe)
# ---------------------------------------------------------------------------


def is_diabetic(p: PatientValues) -> bool:
    """
    Diabète (rapport 3.3.2, ADA 2025).

    Glycémie à jeun ≥ 126 mg/dL OU HbA1c ≥ 6,5 %.
    """
    return p["fasting_glucose"] >= DX_GLUCOSE or p["hba1c"] >= DX_HBA1C


def is_dyslipidemic(p: PatientValues) -> bool:
    """
    Dyslipidémie (rapport 3.3.3, ESC/EAS 2025).

    Au moins une des anomalies lipidiques :
      - Cholestérol total ≥ 240 mg/dL
      - LDL ≥ 160 mg/dL
      - HDL bas (< 40 H / < 50 F)
      - Triglycérides ≥ 200 mg/dL
    """
    hdl_threshold = hdl_low_threshold(p["sex"])
    return (
        p["total_chol"] >= DX_TOTAL_CHOL
        or p["ldl"] >= DX_LDL
        or p["hdl"] < hdl_threshold
        or p["triglycerides"] >= DX_TRIGLYCERIDES
    )


def is_hypertensive(p: PatientValues) -> bool:
    """
    Hypertension (rapport 3.3.4, ESC 2024).

    PAS ≥ 140 mmHg OU PAD ≥ 90 mmHg.
    """
    return p["sbp"] >= DX_SBP or p["dbp"] >= DX_DBP


def is_obese(p: PatientValues) -> bool:
    """
    Obésité (rapport 3.3.5, WHO 2025).

    IMC ≥ 30 kg/m².
    """
    return p["bmi"] >= DX_BMI


# ---------------------------------------------------------------------------
# Facteurs de risque cardiovasculaire (rapport 3.3.6)
# ---------------------------------------------------------------------------
#
# La classe « risque cardiovasculaire » regroupe les patients cumulant
# plusieurs facteurs de risque parmi un ensemble plus large que les seuls
# critères diagnostiques stricts. Chaque facteur compte pour 1, qu'il
# corresponde à une zone intermédiaire ou à un dépassement franc du seuil
# diagnostique.
#
# Les facteurs sont organisés en trois groupes :
#   - 4 facteurs biologiques (glycémie, lipides, pression, IMC)
#   - 1 facteur lié aux signes vitaux (fréquence cardiaque)
#   - 4 facteurs comportementaux (tabac, sédentarité, alcool, alimentation)
# ---------------------------------------------------------------------------


def _has_glycemic_risk(p: PatientValues) -> bool:
    """Facteur biologique #1 : glycémie ≥ 100 OU HbA1c ≥ 5,7 %."""
    return p["fasting_glucose"] >= RF_GLUCOSE or p["hba1c"] >= RF_HBA1C


def _has_lipid_risk(p: PatientValues) -> bool:
    """
    Facteur biologique #2 : anomalie lipidique (zone intermédiaire incluse).

    LDL ≥ 130, cholestérol total ≥ 200, HDL bas, OU triglycérides ≥ 150.
    """
    hdl_threshold = hdl_low_threshold(p["sex"])
    return (
        p["ldl"] >= RF_LDL
        or p["total_chol"] >= RF_TOTAL_CHOL
        or p["hdl"] < hdl_threshold
        or p["triglycerides"] >= RF_TRIGLYCERIDES
    )


def _has_blood_pressure_risk(p: PatientValues) -> bool:
    """Facteur biologique #3 : pression artérielle ≥ 120/80 mmHg."""
    return p["sbp"] >= RF_SBP or p["dbp"] >= RF_DBP


def _has_bmi_risk(p: PatientValues) -> bool:
    """Facteur biologique #4 : IMC ≥ 25 kg/m² (surpoids ou obésité)."""
    return p["bmi"] >= RF_BMI


def _has_heart_rate_risk(p: PatientValues) -> bool:
    """Facteur signes vitaux : fréquence cardiaque au repos ≥ 80 bpm."""
    return p["heart_rate"] >= RF_HEART_RATE


def _has_smoking_risk(p: PatientValues) -> bool:
    """Facteur comportemental : tabagisme actif."""
    return p["smoking"] == "Current"


def _has_sedentary_risk(p: PatientValues) -> bool:
    """Facteur comportemental : sédentarité."""
    return p["physical_activity"] == "Sedentary"


def _has_alcohol_risk(p: PatientValues) -> bool:
    """Facteur comportemental : consommation d'alcool excessive."""
    return p["alcohol"] == "Excessive"


def _has_diet_risk(p: PatientValues) -> bool:
    """Facteur comportemental : mauvaise alimentation."""
    return p["diet_quality"] == "Poor"


def count_cv_risk_factors(p: PatientValues) -> tuple[int, int]:
    """
    Compte les facteurs de risque cardiovasculaire.

    Retour
    ------
    n_total : int
        Nombre total de facteurs (biologiques + signes vitaux + comportementaux).
    n_biological : int
        Nombre de facteurs biologiques uniquement (parmi les 4).

    Note
    ----
    L'attribution à la classe « risque cardiovasculaire » est définie en
    section 3.4 du rapport :
        n_total >= 3  OU  n_biological >= 2.
    """
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
    """
    Détail booléen de chaque facteur de risque. Utile pour les logs, les
    visualisations et le débogage de la génération.
    """
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
