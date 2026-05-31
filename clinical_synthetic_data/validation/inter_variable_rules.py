"""
Règles de cohérence inter-variables (rapport section 5.3).

Chaque règle est un prédicat retournant True si la cohérence est respectée
et False sinon. Les tolérances sont centralisées dans `clinical_ranges`.

Règles
------
R1  Quetelet      : IMC ≈ poids / taille² à ±0.5 kg/m² près
R2  Friedewald    : Chol_total ≈ LDL + HDL + TG/5 à ±10 mg/dL près,
                    applicable uniquement si TG < 400 mg/dL
R3  Pression      : 20 ≤ PAS − PAD ≤ 100 mmHg
R4  Glycémie/HbA1c: cohérence des trois implications (rapport 5.3 R4)
"""

from __future__ import annotations

from typing import Any, Mapping

from ..config.clinical_ranges import (
    DX_GLUCOSE,
    DX_HBA1C,
    FRIEDEWALD_TG_LIMIT,
    MAX_PULSE_PRESSURE,
    MIN_PULSE_PRESSURE,
    RF_HBA1C,
    TOLERANCE_BMI,
    TOLERANCE_FRIEDEWALD,
)


# Seuils utilisés exclusivement dans R4 (rapport 5.3, encadré R4)
R4_GLUCOSE_HIGH_IF_HBA1C_HIGH: float = 100.0     # mg/dL
R4_GLUCOSE_LOW_IF_HBA1C_LOW: float = 140.0       # mg/dL


# ---------------------------------------------------------------------------
# R1 -- Quetelet
# ---------------------------------------------------------------------------


def check_quetelet(p: Mapping[str, Any]) -> bool:
    """
    IMC cohérent avec taille et poids.

    Tolérance: ±TOLERANCE_BMI kg/m² (rapport 5.3 R1).
    """
    height_m = float(p["height"]) / 100.0
    if height_m <= 0:
        return False
    expected_bmi = float(p["weight"]) / (height_m ** 2)
    return abs(float(p["bmi"]) - expected_bmi) <= TOLERANCE_BMI


# ---------------------------------------------------------------------------
# R2 -- Friedewald
# ---------------------------------------------------------------------------


def check_friedewald(p: Mapping[str, Any]) -> bool:
    """
    Bilan lipidique cohérent.

    Si TG < 400 mg/dL : |Chol_total − (LDL + HDL + TG/5)| ≤ 10 mg/dL.
    Au-delà, la formule de Friedewald n'est pas applicable et la règle est
    considérée comme passante par convention clinique (rapport 5.3 R2).
    """
    tg = float(p["triglycerides"])
    if tg >= FRIEDEWALD_TG_LIMIT:
        return True

    estimated_total = (
        float(p["ldl"]) + float(p["hdl"]) + tg / 5.0
    )
    return abs(float(p["total_chol"]) - estimated_total) <= TOLERANCE_FRIEDEWALD


# ---------------------------------------------------------------------------
# R3 -- Pression artérielle
# ---------------------------------------------------------------------------


def check_pulse_pressure(p: Mapping[str, Any]) -> bool:
    """
    Différence PAS − PAD dans la plage physiologique [20, 100] mmHg.

    La pression différentielle (pulse pressure) reflète l'élasticité
    vasculaire et la fonction cardiaque ; ses valeurs hors de cet
    intervalle sont incompatibles avec un patient ambulatoire stable.
    """
    pulse_pressure = float(p["sbp"]) - float(p["dbp"])
    return MIN_PULSE_PRESSURE <= pulse_pressure <= MAX_PULSE_PRESSURE


# ---------------------------------------------------------------------------
# R4 -- Glycémie / HbA1c
# ---------------------------------------------------------------------------


def check_glucose_hba1c_coherence(p: Mapping[str, Any]) -> bool:
    """
    Cohérence entre glycémie à jeun et HbA1c.

    Trois implications doivent être satisfaites simultanément (rapport
    5.3 R4) :
      (a) HbA1c ≥ 6,5 %   ⇒  glycémie ≥ 100 mg/dL
      (b) HbA1c < 5,7 %   ⇒  glycémie < 140 mg/dL
      (c) Glycémie ≥ 126 mg/dL ⇒ HbA1c ≥ 5,7 %

    L'implication « A ⇒ B » est violée ssi (A ∧ ¬B).
    """
    glucose = float(p["fasting_glucose"])
    hba1c = float(p["hba1c"])

    # (a) HbA1c chronique haute mais glycémie ponctuelle basse : incohérent.
    if hba1c >= DX_HBA1C and glucose < R4_GLUCOSE_HIGH_IF_HBA1C_HIGH:
        return False

    # (b) HbA1c franchement normale mais pic glycémique majeur : incohérent.
    if hba1c < RF_HBA1C and glucose >= R4_GLUCOSE_LOW_IF_HBA1C_LOW:
        return False

    # (c) Glycémie diabétique sans aucune trace sur l'HbA1c : incohérent.
    if glucose >= DX_GLUCOSE and hba1c < RF_HBA1C:
        return False

    return True


# ---------------------------------------------------------------------------
# Registre des règles, dans l'ordre d'exécution de la cascade
# ---------------------------------------------------------------------------

INTER_VARIABLE_RULES: tuple[tuple[str, callable], ...] = (
    ("R1_quetelet",     check_quetelet),
    ("R2_friedewald",   check_friedewald),
    ("R3_pulse_pressure", check_pulse_pressure),
    ("R4_glucose_hba1c", check_glucose_hba1c_coherence),
)
