"""Règles de cohérence inter-variables (rapport section 5.3)."""

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


R4_GLUCOSE_HIGH_IF_HBA1C_HIGH: float = 100.0
R4_GLUCOSE_LOW_IF_HBA1C_LOW: float = 140.0


def check_quetelet(p: Mapping[str, Any]) -> bool:
    """IMC cohérent avec taille et poids (±TOLERANCE_BMI kg/m²)."""
    height_m = float(p["height"]) / 100.0
    if height_m <= 0:
        return False
    expected_bmi = float(p["weight"]) / (height_m ** 2)
    return abs(float(p["bmi"]) - expected_bmi) <= TOLERANCE_BMI


def check_friedewald(p: Mapping[str, Any]) -> bool:
    """Bilan lipidique cohérent (Friedewald, applicable si TG < 400 mg/dL)."""
    tg = float(p["triglycerides"])
    if tg >= FRIEDEWALD_TG_LIMIT:
        return True

    estimated_total = (
        float(p["ldl"]) + float(p["hdl"]) + tg / 5.0
    )
    return abs(float(p["total_chol"]) - estimated_total) <= TOLERANCE_FRIEDEWALD


def check_pulse_pressure(p: Mapping[str, Any]) -> bool:
    """Différence PAS − PAD dans la plage physiologique [20, 100] mmHg."""
    pulse_pressure = float(p["sbp"]) - float(p["dbp"])
    return MIN_PULSE_PRESSURE <= pulse_pressure <= MAX_PULSE_PRESSURE


def check_glucose_hba1c_coherence(p: Mapping[str, Any]) -> bool:
    """Cohérence entre glycémie à jeun et HbA1c (rapport 5.3 R4)."""
    glucose = float(p["fasting_glucose"])
    hba1c = float(p["hba1c"])

    if hba1c >= DX_HBA1C and glucose < R4_GLUCOSE_HIGH_IF_HBA1C_HIGH:
        return False

    if hba1c < RF_HBA1C and glucose >= R4_GLUCOSE_LOW_IF_HBA1C_LOW:
        return False

    if glucose >= DX_GLUCOSE and hba1c < RF_HBA1C:
        return False

    return True


INTER_VARIABLE_RULES: tuple[tuple[str, callable], ...] = (
    ("R1_quetelet",     check_quetelet),
    ("R2_friedewald",   check_friedewald),
    ("R3_pulse_pressure", check_pulse_pressure),
    ("R4_glucose_hba1c", check_glucose_hba1c_coherence),
)
