"""
Plages cliniques de référence et seuils diagnostiques.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class VariableType(str, Enum):
    """Typologie statistique d'une variable (rapport 2.3)."""

    CONTINUOUS = "continuous"
    BINARY_CATEGORICAL = "binary_categorical"
    NOMINAL_CATEGORICAL = "nominal_categorical"
    ORDINAL = "ordinal"


@dataclass(frozen=True)
class ContinuousVariableSpec:
    """
    Spécification d'une variable continue.

    Attributs
    ---------
    name : str
        Identifiant interne (snake_case).
    display_name : str
        Nom à afficher (français, accentué).
    unit : str
        Unité clinique.
    absolute_min, absolute_max : float
        Bornes physiologiques absolues utilisées pour le rejet en
        section 5.2 du rapport.
    normal_min, normal_max : float
        Plage normale clinique (section 3.2 du rapport).
    """

    name: str
    display_name: str
    unit: str
    absolute_min: float
    absolute_max: float
    normal_min: float
    normal_max: float

    def __post_init__(self) -> None:
        if not (self.absolute_min <= self.normal_min <= self.normal_max <= self.absolute_max):
            raise ValueError(
                f"Invariant violé pour '{self.name}' : "
                f"absolute_min={self.absolute_min}, normal_min={self.normal_min}, "
                f"normal_max={self.normal_max}, absolute_max={self.absolute_max}"
            )


@dataclass(frozen=True)
class CategoricalVariableSpec:
    """Spécification d'une variable catégorielle (binaire, nominale ou ordinale)."""

    name: str
    display_name: str
    var_type: VariableType
    modalities: tuple[str, ...]


# ---------------------------------------------------------------------------
# Variables démographiques (rapport 3.2.1)
# ---------------------------------------------------------------------------

AGE = ContinuousVariableSpec(
    name="age",
    display_name="Âge",
    unit="ans",
    absolute_min=18.0,
    absolute_max=90.0,
    normal_min=18.0,
    normal_max=90.0,
)

HEIGHT = ContinuousVariableSpec(
    name="height",
    display_name="Taille",
    unit="cm",
    absolute_min=140.0,
    absolute_max=210.0,
    normal_min=150.0,
    normal_max=195.0,
)

WEIGHT = ContinuousVariableSpec(
    name="weight",
    display_name="Poids",
    unit="kg",
    absolute_min=35.0,
    absolute_max=200.0,
    normal_min=45.0,
    normal_max=150.0,
)

BMI = ContinuousVariableSpec(
    name="bmi",
    display_name="Indice de masse corporelle",
    unit="kg/m²",
    absolute_min=13.0,
    absolute_max=60.0,
    normal_min=18.5,
    normal_max=24.9,
)

# ---------------------------------------------------------------------------
# Signes vitaux (rapport 3.2.2)
# ---------------------------------------------------------------------------

HEART_RATE = ContinuousVariableSpec(
    name="heart_rate",
    display_name="Fréquence cardiaque",
    unit="bpm",
    absolute_min=40.0,
    absolute_max=160.0,
    normal_min=60.0,
    normal_max=79.0,
)

SBP = ContinuousVariableSpec(
    name="sbp",
    display_name="Pression artérielle systolique",
    unit="mmHg",
    absolute_min=70.0,
    absolute_max=220.0,
    normal_min=90.0,   # plancher clinique du normal (< 120 cible)
    normal_max=119.0,
)

DBP = ContinuousVariableSpec(
    name="dbp",
    display_name="Pression artérielle diastolique",
    unit="mmHg",
    absolute_min=40.0,
    absolute_max=130.0,
    normal_min=60.0,
    normal_max=79.0,
)

RESP_RATE = ContinuousVariableSpec(
    name="resp_rate",
    display_name="Fréquence respiratoire",
    unit="cycles/min",
    absolute_min=8.0,
    absolute_max=30.0,
    normal_min=12.0,
    normal_max=20.0,
)

TEMPERATURE = ContinuousVariableSpec(
    name="temp",
    display_name="Température corporelle",
    unit="°C",
    absolute_min=35.0,
    absolute_max=40.0,
    normal_min=36.1,
    normal_max=37.2,
)

# ---------------------------------------------------------------------------
# Résultats de laboratoire (rapport 3.2.3)
# ---------------------------------------------------------------------------

FASTING_GLUCOSE = ContinuousVariableSpec(
    name="fasting_glucose",
    display_name="Glycémie à jeun",
    unit="mg/dL",
    absolute_min=50.0,
    absolute_max=400.0,
    normal_min=70.0,
    normal_max=99.0,
)

HBA1C = ContinuousVariableSpec(
    name="hba1c",
    display_name="Hémoglobine glyquée (HbA1c)",
    unit="%",
    absolute_min=4.0,
    absolute_max=15.0,
    normal_min=4.0,
    normal_max=5.6,
)

TOTAL_CHOL = ContinuousVariableSpec(
    name="total_chol",
    display_name="Cholestérol total",
    unit="mg/dL",
    absolute_min=80.0,
    absolute_max=400.0,
    normal_min=120.0,
    normal_max=199.0,
)

LDL = ContinuousVariableSpec(
    name="ldl",
    display_name="Cholestérol LDL",
    unit="mg/dL",
    absolute_min=30.0,
    absolute_max=300.0,
    normal_min=40.0,
    normal_max=99.0,
)

HDL = ContinuousVariableSpec(
    name="hdl",
    display_name="Cholestérol HDL",
    unit="mg/dL",
    absolute_min=15.0,
    absolute_max=120.0,
    normal_min=40.0,   # plancher H ; pour F le plancher clinique est 50
    normal_max=80.0,
)

TRIGLYCERIDES = ContinuousVariableSpec(
    name="triglycerides",
    display_name="Triglycérides",
    unit="mg/dL",
    absolute_min=30.0,
    absolute_max=1000.0,
    normal_min=50.0,
    normal_max=149.0,
)


# ---------------------------------------------------------------------------
# Variables catégorielles et de mode de vie (rapport 3.2.4)
# ---------------------------------------------------------------------------

SEX = CategoricalVariableSpec(
    name="sex",
    display_name="Sexe",
    var_type=VariableType.BINARY_CATEGORICAL,
    modalities=("Male", "Female"),
)

PHYSICAL_ACTIVITY = CategoricalVariableSpec(
    name="physical_activity",
    display_name="Niveau d'activité physique",
    var_type=VariableType.ORDINAL,
    modalities=("Sedentary", "Moderate", "High"),
)

SMOKING = CategoricalVariableSpec(
    name="smoking",
    display_name="Statut tabagique",
    var_type=VariableType.NOMINAL_CATEGORICAL,
    modalities=("Never", "Former", "Current"),
)

ALCOHOL = CategoricalVariableSpec(
    name="alcohol",
    display_name="Consommation d'alcool",
    var_type=VariableType.ORDINAL,
    modalities=("None", "Moderate", "Excessive"),
)

DIET_QUALITY = CategoricalVariableSpec(
    name="diet_quality",
    display_name="Qualité de l'alimentation",
    var_type=VariableType.ORDINAL,
    modalities=("Poor", "Average", "Good"),
)


# ---------------------------------------------------------------------------
# Registres exposés
# ---------------------------------------------------------------------------

CONTINUOUS_VARIABLES: dict[str, ContinuousVariableSpec] = {
    v.name: v for v in (
        AGE, HEIGHT, WEIGHT, BMI,
        HEART_RATE, SBP, DBP, RESP_RATE, TEMPERATURE,
        FASTING_GLUCOSE, HBA1C, TOTAL_CHOL, LDL, HDL, TRIGLYCERIDES,
    )
}

CATEGORICAL_VARIABLES: dict[str, CategoricalVariableSpec] = {
    v.name: v for v in (SEX, PHYSICAL_ACTIVITY, SMOKING, ALCOHOL, DIET_QUALITY)
}


# ---------------------------------------------------------------------------
# Seuils diagnostiques (rapport section 3.3)
# ---------------------------------------------------------------------------
#
# Convention de nommage :
#   DX_*  → seuil diagnostique pathologique strict (entrée de classe)
#   RF_*  → seuil de facteur de risque (zone intermédiaire pour la classe
#           « risque cardiovasculaire »)
#
# Tous les seuils sont conformes au rapport, qui s'appuie sur :
#   - ADA Standards of Care 2025 (diabète)
#   - ESC/EAS 2025 (dyslipidémies)
#   - ESC 2024 (hypertension)
#   - WHO 2025 (obésité)
# ---------------------------------------------------------------------------


# --- Diabète (ADA 2025) ---
DX_GLUCOSE: float = 126.0           # mg/dL, glycémie à jeun
DX_HBA1C: float = 6.5               # %

# --- Pré-diabète / facteur de risque (ADA 2025) ---
RF_GLUCOSE: float = 100.0           # mg/dL
RF_HBA1C: float = 5.7               # %

# --- Dyslipidémie (ESC/EAS 2025) ---
DX_TOTAL_CHOL: float = 240.0        # mg/dL : cholestérol total élevé
DX_LDL: float = 160.0               # mg/dL
DX_TRIGLYCERIDES: float = 200.0     # mg/dL
DX_HDL_LOW_MALE: float = 40.0       # mg/dL : HDL bas chez l'homme
DX_HDL_LOW_FEMALE: float = 50.0     # mg/dL : HDL bas chez la femme

# --- Facteurs de risque lipidique (zone intermédiaire) ---
RF_TOTAL_CHOL: float = 200.0        # mg/dL : cholestérol total souhaitable
RF_LDL: float = 130.0               # mg/dL : LDL borderline
RF_TRIGLYCERIDES: float = 150.0     # mg/dL : TG borderline

# --- Hypertension (ESC 2024) ---
DX_SBP: float = 140.0               # mmHg
DX_DBP: float = 90.0                # mmHg

# --- Pression artérielle élevée (facteur de risque) ---
RF_SBP: float = 120.0               # mmHg
RF_DBP: float = 80.0                # mmHg

# --- Obésité (WHO 2025) ---
DX_BMI: float = 30.0                # kg/m²
RF_BMI: float = 25.0                # kg/m² : surpoids = facteur de risque

# --- Fréquence cardiaque (facteur de risque CV) ---
RF_HEART_RATE: float = 80.0         # bpm au repos


def hdl_low_threshold(sex: str) -> float:
    """Retourne le seuil HDL bas selon le sexe (rapport 3.2.3 et 3.3.3)."""
    if sex == "Male":
        return DX_HDL_LOW_MALE
    if sex == "Female":
        return DX_HDL_LOW_FEMALE
    raise ValueError(f"Sexe inconnu : {sex!r}. Attendu 'Male' ou 'Female'.")


# ---------------------------------------------------------------------------
# Tolérances pour les contrôles de cohérence (rapport section 5.3)
# ---------------------------------------------------------------------------

TOLERANCE_BMI: float = 0.5          # kg/m² (R1, Quetelet)
TOLERANCE_FRIEDEWALD: float = 10.0  # mg/dL (R2, Friedewald)
MIN_PULSE_PRESSURE: float = 20.0    # mmHg (R3)
MAX_PULSE_PRESSURE: float = 100.0   # mmHg (R3)
FRIEDEWALD_TG_LIMIT: float = 400.0  # mg/dL : Friedewald non applicable au-delà
