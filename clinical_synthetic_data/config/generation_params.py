"""Paramètres de génération : marginales par classe."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..core.patient_schema import ClassLabel


@dataclass(frozen=True)
class ContinuousParam:
    """Paramètres d'une marginale continue."""

    distribution: str
    mean: float
    std: float
    mean_male: Optional[float] = None
    mean_female: Optional[float] = None

    def __post_init__(self) -> None:
        if self.distribution not in ("normal", "lognormal"):
            raise ValueError(f"distribution inconnue : {self.distribution!r}")
        if self.std <= 0:
            raise ValueError(f"std doit être > 0, reçu {self.std}")
        if self.distribution == "lognormal":
            mean = self.mean
            if self.mean_male is not None:
                mean = min(mean, self.mean_male)
            if self.mean_female is not None:
                mean = min(mean, self.mean_female)
            if mean <= 0:
                raise ValueError("lognormal exige une moyenne strictement positive")

    def effective_mean(self, sex: str) -> float:
        """Retourne la moyenne effective compte tenu du sexe."""
        if sex == "Male" and self.mean_male is not None:
            return self.mean_male
        if sex == "Female" and self.mean_female is not None:
            return self.mean_female
        return self.mean

    def transform(self, z: float, sex: str) -> float:
        """Transforme z ~ N(0, 1) en la valeur de cette marginale."""
        mean = self.effective_mean(sex)
        if self.distribution == "normal":
            return mean + self.std * z

        cv_squared = (self.std / mean) ** 2
        sigma_log = math.sqrt(math.log1p(cv_squared))
        mu_log = math.log(mean) - 0.5 * sigma_log * sigma_log
        return math.exp(mu_log + sigma_log * z)


HEIGHT_MEAN_MALE: float = 178.0
HEIGHT_MEAN_FEMALE: float = 164.0
HEIGHT_STD: float = 7.5

RESP_RATE_PARAM = ContinuousParam("normal", mean=15.0, std=2.0)
TEMP_PARAM = ContinuousParam("normal", mean=36.8, std=0.3)


PARAMS_HEALTHY: dict[str, ContinuousParam] = {
    "age":              ContinuousParam("normal", mean=38.0, std=14.0),
    "height":           ContinuousParam("normal", mean=171.0, std=HEIGHT_STD,
                                        mean_male=HEIGHT_MEAN_MALE,
                                        mean_female=HEIGHT_MEAN_FEMALE),
    "bmi":              ContinuousParam("normal", mean=22.0, std=1.7),
    "heart_rate":       ContinuousParam("normal", mean=68.0, std=7.0),
    "sbp":              ContinuousParam("normal", mean=110.0, std=6.0),
    "dbp":              ContinuousParam("normal", mean=72.0, std=5.0),
    "resp_rate":        RESP_RATE_PARAM,
    "temp":             TEMP_PARAM,
    "fasting_glucose":  ContinuousParam("normal", mean=86.0, std=6.0),
    "hba1c":            ContinuousParam("normal", mean=5.1, std=0.22),
    "ldl":              ContinuousParam("normal", mean=85.0, std=14.0),
    "hdl":              ContinuousParam("normal", mean=58.0, std=9.0,
                                        mean_male=55.0, mean_female=64.0),
    "triglycerides":    ContinuousParam("lognormal", mean=95.0, std=22.0),
}


PARAMS_DIABETES: dict[str, ContinuousParam] = {
    "age":              ContinuousParam("normal", mean=55.0, std=12.0),
    "height":           ContinuousParam("normal", mean=171.0, std=HEIGHT_STD,
                                        mean_male=HEIGHT_MEAN_MALE,
                                        mean_female=HEIGHT_MEAN_FEMALE),
    "bmi":              ContinuousParam("normal", mean=22.5, std=1.3),
    "heart_rate":       ContinuousParam("normal", mean=72.0, std=4.5),
    "sbp":              ContinuousParam("normal", mean=115.0, std=3.5),
    "dbp":              ContinuousParam("normal", mean=75.0, std=3.0),
    "resp_rate":        RESP_RATE_PARAM,
    "temp":             TEMP_PARAM,
    "fasting_glucose":  ContinuousParam("normal", mean=158.0, std=22.0),
    "hba1c":            ContinuousParam("normal", mean=7.4, std=0.85),
    "ldl":              ContinuousParam("normal", mean=100.0, std=12.0),
    "hdl":              ContinuousParam("normal", mean=54.0, std=7.0,
                                        mean_male=50.0, mean_female=60.0),
    "triglycerides":    ContinuousParam("lognormal", mean=120.0, std=18.0),
}


PARAMS_DYSLIPIDEMIA: dict[str, ContinuousParam] = {
    "age":              ContinuousParam("normal", mean=52.0, std=12.0),
    "height":           ContinuousParam("normal", mean=171.0, std=HEIGHT_STD,
                                        mean_male=HEIGHT_MEAN_MALE,
                                        mean_female=HEIGHT_MEAN_FEMALE),
    "bmi":              ContinuousParam("normal", mean=22.5, std=1.3),
    "heart_rate":       ContinuousParam("normal", mean=72.0, std=4.5),
    "sbp":              ContinuousParam("normal", mean=115.0, std=3.5),
    "dbp":              ContinuousParam("normal", mean=75.0, std=3.0),
    "resp_rate":        RESP_RATE_PARAM,
    "temp":             TEMP_PARAM,
    "fasting_glucose":  ContinuousParam("normal", mean=90.0, std=5.0),
    "hba1c":            ContinuousParam("normal", mean=5.2, std=0.18),
    "ldl":              ContinuousParam("normal", mean=178.0, std=12.0),
    "hdl":              ContinuousParam("normal", mean=48.0, std=6.0,
                                        mean_male=46.0, mean_female=54.0),
    "triglycerides":    ContinuousParam("lognormal", mean=145.0, std=22.0),
}


PARAMS_HYPERTENSION: dict[str, ContinuousParam] = {
    "age":              ContinuousParam("normal", mean=57.0, std=12.0),
    "height":           ContinuousParam("normal", mean=171.0, std=HEIGHT_STD,
                                        mean_male=HEIGHT_MEAN_MALE,
                                        mean_female=HEIGHT_MEAN_FEMALE),
    "bmi":              ContinuousParam("normal", mean=22.8, std=1.3),
    "heart_rate":       ContinuousParam("normal", mean=73.0, std=4.5),
    "sbp":              ContinuousParam("normal", mean=152.0, std=8.0),
    "dbp":              ContinuousParam("normal", mean=95.0, std=4.0),
    "resp_rate":        RESP_RATE_PARAM,
    "temp":             TEMP_PARAM,
    "fasting_glucose":  ContinuousParam("normal", mean=92.0, std=5.0),
    "hba1c":            ContinuousParam("normal", mean=5.3, std=0.18),
    "ldl":              ContinuousParam("normal", mean=102.0, std=12.0),
    "hdl":              ContinuousParam("normal", mean=54.0, std=7.0,
                                        mean_male=50.0, mean_female=60.0),
    "triglycerides":    ContinuousParam("lognormal", mean=125.0, std=20.0),
}


PARAMS_OBESITY: dict[str, ContinuousParam] = {
    "age":              ContinuousParam("normal", mean=45.0, std=12.0),
    "height":           ContinuousParam("normal", mean=171.0, std=HEIGHT_STD,
                                        mean_male=HEIGHT_MEAN_MALE,
                                        mean_female=HEIGHT_MEAN_FEMALE),
    "bmi":              ContinuousParam("normal", mean=33.0, std=2.3),
    "heart_rate":       ContinuousParam("normal", mean=74.0, std=4.0),
    "sbp":              ContinuousParam("normal", mean=115.0, std=3.5),
    "dbp":              ContinuousParam("normal", mean=76.0, std=3.0),
    "resp_rate":        RESP_RATE_PARAM,
    "temp":             TEMP_PARAM,
    "fasting_glucose":  ContinuousParam("normal", mean=92.0, std=5.0),
    "hba1c":            ContinuousParam("normal", mean=5.3, std=0.18),
    "ldl":              ContinuousParam("normal", mean=105.0, std=12.0),
    "hdl":              ContinuousParam("normal", mean=52.0, std=6.0,
                                        mean_male=48.0, mean_female=58.0),
    "triglycerides":    ContinuousParam("lognormal", mean=130.0, std=22.0),
}


PARAMS_CV_RISK: dict[str, ContinuousParam] = {
    "age":              ContinuousParam("normal", mean=60.0, std=10.0),
    "height":           ContinuousParam("normal", mean=171.0, std=HEIGHT_STD,
                                        mean_male=HEIGHT_MEAN_MALE,
                                        mean_female=HEIGHT_MEAN_FEMALE),
    "bmi":              ContinuousParam("normal", mean=27.5, std=1.4),
    "heart_rate":       ContinuousParam("normal", mean=82.0, std=5.0),
    "sbp":              ContinuousParam("normal", mean=130.0, std=5.0),
    "dbp":              ContinuousParam("normal", mean=84.0, std=3.0),
    "resp_rate":        RESP_RATE_PARAM,
    "temp":             TEMP_PARAM,
    "fasting_glucose":  ContinuousParam("normal", mean=110.0, std=6.0),
    "hba1c":            ContinuousParam("normal", mean=6.0, std=0.20),
    "ldl":              ContinuousParam("normal", mean=145.0, std=8.0),
    "hdl":              ContinuousParam("normal", mean=44.0, std=5.0,
                                        mean_male=42.0, mean_female=52.0),
    "triglycerides":    ContinuousParam("lognormal", mean=170.0, std=18.0),
}


CONTINUOUS_PARAMS_BY_CLASS: dict[ClassLabel, dict[str, ContinuousParam]] = {
    ClassLabel.HEALTHY:       PARAMS_HEALTHY,
    ClassLabel.DIABETES:      PARAMS_DIABETES,
    ClassLabel.DYSLIPIDEMIA:  PARAMS_DYSLIPIDEMIA,
    ClassLabel.HYPERTENSION:  PARAMS_HYPERTENSION,
    ClassLabel.OBESITY:       PARAMS_OBESITY,
    ClassLabel.CV_RISK:       PARAMS_CV_RISK,
}


CategoricalProbs = dict[str, dict[str, float]]


PROBS_HEALTHY: CategoricalProbs = {
    "sex":               {"Male": 0.50, "Female": 0.50},
    "physical_activity": {"Sedentary": 0.15, "Moderate": 0.55, "High": 0.30},
    "smoking":           {"Never": 0.70, "Former": 0.20, "Current": 0.10},
    "alcohol":           {"None": 0.45, "Moderate": 0.50, "Excessive": 0.05},
    "diet_quality":      {"Poor": 0.05, "Average": 0.40, "Good": 0.55},
}

PROBS_DIABETES: CategoricalProbs = {
    "sex":               {"Male": 0.55, "Female": 0.45},
    "physical_activity": {"Sedentary": 0.20, "Moderate": 0.60, "High": 0.20},
    "smoking":           {"Never": 0.65, "Former": 0.25, "Current": 0.10},
    "alcohol":           {"None": 0.55, "Moderate": 0.40, "Excessive": 0.05},
    "diet_quality":      {"Poor": 0.10, "Average": 0.55, "Good": 0.35},
}

PROBS_DYSLIPIDEMIA: CategoricalProbs = {
    "sex":               {"Male": 0.52, "Female": 0.48},
    "physical_activity": {"Sedentary": 0.20, "Moderate": 0.55, "High": 0.25},
    "smoking":           {"Never": 0.55, "Former": 0.30, "Current": 0.15},
    "alcohol":           {"None": 0.45, "Moderate": 0.50, "Excessive": 0.05},
    "diet_quality":      {"Poor": 0.10, "Average": 0.50, "Good": 0.40},
}

PROBS_HYPERTENSION: CategoricalProbs = {
    "sex":               {"Male": 0.50, "Female": 0.50},
    "physical_activity": {"Sedentary": 0.25, "Moderate": 0.55, "High": 0.20},
    "smoking":           {"Never": 0.55, "Former": 0.30, "Current": 0.15},
    "alcohol":           {"None": 0.40, "Moderate": 0.50, "Excessive": 0.10},
    "diet_quality":      {"Poor": 0.15, "Average": 0.55, "Good": 0.30},
}

PROBS_OBESITY: CategoricalProbs = {
    "sex":               {"Male": 0.48, "Female": 0.52},
    "physical_activity": {"Sedentary": 0.40, "Moderate": 0.50, "High": 0.10},
    "smoking":           {"Never": 0.65, "Former": 0.25, "Current": 0.10},
    "alcohol":           {"None": 0.45, "Moderate": 0.45, "Excessive": 0.10},
    "diet_quality":      {"Poor": 0.20, "Average": 0.55, "Good": 0.25},
}

PROBS_CV_RISK: CategoricalProbs = {
    "sex":               {"Male": 0.55, "Female": 0.45},
    "physical_activity": {"Sedentary": 0.55, "Moderate": 0.35, "High": 0.10},
    "smoking":           {"Never": 0.25, "Former": 0.30, "Current": 0.45},
    "alcohol":           {"None": 0.25, "Moderate": 0.45, "Excessive": 0.30},
    "diet_quality":      {"Poor": 0.45, "Average": 0.40, "Good": 0.15},
}


CATEGORICAL_PROBS_BY_CLASS: dict[ClassLabel, CategoricalProbs] = {
    ClassLabel.HEALTHY:       PROBS_HEALTHY,
    ClassLabel.DIABETES:      PROBS_DIABETES,
    ClassLabel.DYSLIPIDEMIA:  PROBS_DYSLIPIDEMIA,
    ClassLabel.HYPERTENSION:  PROBS_HYPERTENSION,
    ClassLabel.OBESITY:       PROBS_OBESITY,
    ClassLabel.CV_RISK:       PROBS_CV_RISK,
}


def _validate_categorical_distributions() -> None:
    """Vérifie à l'import que chaque distribution somme à 1.0 et que ses modalités existent."""
    from .clinical_ranges import CATEGORICAL_VARIABLES

    for class_label, probs_by_var in CATEGORICAL_PROBS_BY_CLASS.items():
        for var_name, probs in probs_by_var.items():
            if var_name not in CATEGORICAL_VARIABLES:
                raise ValueError(
                    f"Variable catégorielle inconnue : {var_name!r} "
                    f"(classe {class_label.value})"
                )
            valid_modalities = set(CATEGORICAL_VARIABLES[var_name].modalities)
            declared_modalities = set(probs)
            if declared_modalities != valid_modalities:
                raise ValueError(
                    f"Modalités incohérentes pour {var_name!r} en classe "
                    f"{class_label.value} : déclarées {declared_modalities}, "
                    f"attendues {valid_modalities}"
                )
            total = sum(probs.values())
            if abs(total - 1.0) > 1e-9:
                raise ValueError(
                    f"Distribution {var_name!r} de la classe "
                    f"{class_label.value} somme à {total} (≠ 1.0)"
                )


_validate_categorical_distributions()
