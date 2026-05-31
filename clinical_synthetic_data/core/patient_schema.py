"""
Schéma de données patient.

Représentation typée d'un enregistrement clinique synthétique, conforme aux
20 variables définies dans le rapport (section 2) plus l'identifiant et la
classe cible.

"""

from __future__ import annotations

import secrets
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# Enums : modalités strictes
# ---------------------------------------------------------------------------


class ClassLabel(str, Enum):
    """Six classes du jeu de données (rapport 1.2)."""

    HEALTHY = "sain"
    DIABETES = "diabete"
    DYSLIPIDEMIA = "dyslipidemie"
    HYPERTENSION = "hypertension"
    OBESITY = "obesite"
    CV_RISK = "risque_cardiovasculaire"


class Sex(str, Enum):
    MALE = "Male"
    FEMALE = "Female"


class ActivityLevel(str, Enum):
    SEDENTARY = "Sedentary"
    MODERATE = "Moderate"
    HIGH = "High"


class SmokingStatus(str, Enum):
    NEVER = "Never"
    FORMER = "Former"
    CURRENT = "Current"


class AlcoholConsumption(str, Enum):
    NONE = "None"
    MODERATE = "Moderate"
    EXCESSIVE = "Excessive"


class DietQuality(str, Enum):
    POOR = "Poor"
    AVERAGE = "Average"
    GOOD = "Good"


# ---------------------------------------------------------------------------
# Dataclass principale
# ---------------------------------------------------------------------------


@dataclass
class Patient:
    """
    Enregistrement patient complet.

    Variables (rapport section 2) :
        Démographiques   : age, sex, height, weight, bmi
        Signes vitaux    : heart_rate, sbp, dbp, resp_rate, temp
        Laboratoire      : fasting_glucose, hba1c, total_chol, ldl, hdl,
                           triglycerides
        Mode de vie      : physical_activity, smoking, alcohol, diet_quality
        Métadonnées      : patient_id, class_label
    """

    # Identifiant
    patient_id: str

    # Démographiques
    age: float
    sex: Sex
    height: float           # cm
    weight: float           # kg
    bmi: float              # kg/m²

    # Signes vitaux
    heart_rate: float       # bpm
    sbp: float              # mmHg, systolique
    dbp: float              # mmHg, diastolique
    resp_rate: float        # cycles/min
    temp: float             # °C

    # Laboratoire
    fasting_glucose: float  # mg/dL
    hba1c: float            # %
    total_chol: float       # mg/dL
    ldl: float              # mg/dL
    hdl: float              # mg/dL
    triglycerides: float    # mg/dL

    # Mode de vie
    physical_activity: ActivityLevel
    smoking: SmokingStatus
    alcohol: AlcoholConsumption
    diet_quality: DietQuality

    # Cible
    class_label: ClassLabel

    # -----------------------------------------------------------------
    # Conversions
    # -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Conversion en dict pour DataFrame ou JSON (enums → str)."""
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, Enum):
                d[key] = value.value
        return d

    def to_flat_values(self) -> dict[str, Any]:
        """
        Vue plate des valeurs cliniques pour les prédicats diagnostiques.

        Les enums sont aplaties en leurs valeurs string. Utilisée comme
        entrée pour `class_assigner.assign_class` et les règles de
        cohérence.
        """
        return self.to_dict()

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    @staticmethod
    def new_id() -> str:
        """Identifiant patient anonyme, court et unique."""
        return f"P{secrets.token_hex(5).upper()}"

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Noms des champs (utile pour la sérialisation CSV ordonnée)."""
        return tuple(f.name for f in fields(cls))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Patient":
        """
        Construit un Patient depuis un dict (typiquement issu d'un générateur).

        Les chaînes pour les variables catégorielles sont converties en enums.
        Lève une exception si une modalité est inconnue.
        """
        return cls(
            patient_id=str(data.get("patient_id") or cls.new_id()),
            age=float(data["age"]),
            sex=Sex(data["sex"]),
            height=float(data["height"]),
            weight=float(data["weight"]),
            bmi=float(data["bmi"]),
            heart_rate=float(data["heart_rate"]),
            sbp=float(data["sbp"]),
            dbp=float(data["dbp"]),
            resp_rate=float(data["resp_rate"]),
            temp=float(data["temp"]),
            fasting_glucose=float(data["fasting_glucose"]),
            hba1c=float(data["hba1c"]),
            total_chol=float(data["total_chol"]),
            ldl=float(data["ldl"]),
            hdl=float(data["hdl"]),
            triglycerides=float(data["triglycerides"]),
            physical_activity=ActivityLevel(data["physical_activity"]),
            smoking=SmokingStatus(data["smoking"]),
            alcohol=AlcoholConsumption(data["alcohol"]),
            diet_quality=DietQuality(data["diet_quality"]),
            class_label=ClassLabel(data["class_label"]),
        )


# ---------------------------------------------------------------------------
# Métadonnées utiles au reste du pipeline
# ---------------------------------------------------------------------------

CONTINUOUS_FIELDS: tuple[str, ...] = (
    "age", "height", "weight", "bmi",
    "heart_rate", "sbp", "dbp", "resp_rate", "temp",
    "fasting_glucose", "hba1c", "total_chol", "ldl", "hdl", "triglycerides",
)

CATEGORICAL_FIELDS: tuple[str, ...] = (
    "sex", "physical_activity", "smoking", "alcohol", "diet_quality",
)
