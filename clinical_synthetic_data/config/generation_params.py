"""
Paramètres de génération : marginales par classe.

Pour chaque (classe, variable continue), définit la loi de probabilité
(normale ou log-normale) et ses paramètres (moyenne, écart-type sur l'échelle
linéaire). Pour chaque (classe, variable catégorielle), définit une
distribution multinomiale.

Conception
----------
Les paramètres sont choisis de sorte qu'un échantillon « moyen » de chaque
classe satisfasse exactement les critères diagnostiques associés ET ne
remplisse pas les conditions des autres classes. Concrètement :

- Pour les classes uni-paramétriques (diabète, dyslipidémie, hypertension,
  obésité), le marqueur diagnostique principal est centré au-dessus du seuil
  pathologique, tandis que les autres variables restent BIEN en dessous des
  seuils de zone intermédiaire (pour ne pas accumuler plus d'un facteur
  biologique et basculer en CV_RISK).
- Pour la classe « risque cardiovasculaire », plusieurs variables sont
  centrées dans la zone intermédiaire (au-dessus du seuil de facteur de
  risque mais sous le seuil pathologique strict), de sorte que l'attribution
  cumule ≥ 2 facteurs biologiques.
- Pour la classe « sain », toutes les variables sont centrées bien dans la
  plage normale, loin des seuils de zone intermédiaire.

Les marqueurs HDL ont des moyennes sexe-conditionnelles, conformément à la
différence physiologique reconnue par les recommandations ESC/EAS 2025.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..core.patient_schema import ClassLabel


# ---------------------------------------------------------------------------
# Type : paramètre d'une variable continue dans une classe donnée
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContinuousParam:
    """
    Paramètres d'une marginale continue.

    Attributs
    ---------
    distribution : str
        "normal" ou "lognormal".
    mean : float
        Moyenne attendue sur l'échelle linéaire (unité clinique).
    std : float
        Écart-type sur l'échelle linéaire (toujours strictement positif).
    mean_male, mean_female : Optional[float]
        Moyennes sexe-conditionnelles, qui priment sur `mean` si fournies.
        Permet de modéliser les différences physiologiques homme/femme
        (taille, HDL).
    """

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
        """
        Transforme une réalisation standard `z` ~ N(0, 1) en la valeur
        correspondante de cette marginale.

        Pour une marginale normale :   X = μ + σ·z
        Pour une log-normale paramétrée en (mean, std) linéaires, la
        conversion vers les paramètres log-scale est :
            σ_log² = ln(1 + (σ / μ)²)
            μ_log  = ln(μ) − σ_log² / 2
        d'où  X = exp(μ_log + σ_log·z).
        """
        mean = self.effective_mean(sex)
        if self.distribution == "normal":
            return mean + self.std * z

        # Log-normale : conversion analytique linéaire ↔ log-scale
        cv_squared = (self.std / mean) ** 2
        sigma_log = math.sqrt(math.log1p(cv_squared))
        mu_log = math.log(mean) - 0.5 * sigma_log * sigma_log
        return math.exp(mu_log + sigma_log * z)


# ---------------------------------------------------------------------------
# Constantes physiologiques sexe-conditionnelles (réutilisées dans toutes les
# classes pour ne pas dupliquer ces valeurs).
# ---------------------------------------------------------------------------

HEIGHT_MEAN_MALE: float = 178.0       # cm
HEIGHT_MEAN_FEMALE: float = 164.0     # cm
HEIGHT_STD: float = 7.5               # cm

# Marginales identiques pour toutes les classes (variables non discriminantes
# entre profils de patients ambulatoires stables)
RESP_RATE_PARAM = ContinuousParam("normal", mean=15.0, std=2.0)
TEMP_PARAM = ContinuousParam("normal", mean=36.8, std=0.3)


# ---------------------------------------------------------------------------
# Paramètres par classe — variables continues
# ---------------------------------------------------------------------------
#
# Ordre des entrées (cohérent avec `core.patient_schema.CONTINUOUS_FIELDS`) :
#   age, height, weight, bmi,
#   heart_rate, sbp, dbp, resp_rate, temp,
#   fasting_glucose, hba1c, total_chol, ldl, hdl, triglycerides
#
# `weight` et `total_chol` sont DÉRIVÉS par construction (et non échantillonnés
# via la copule) : weight = BMI · (height/100)² et total_chol = LDL + HDL +
# TG/5 + bruit. Ils n'apparaissent donc pas ci-dessous.
# ---------------------------------------------------------------------------


# ---- Classe SAIN -----------------------------------------------------------
# Toutes les variables sont confortablement dans la plage normale, loin des
# seuils de zone intermédiaire (100 mg/dL pour la glycémie, 130 pour LDL,
# 120/80 pour la pression, 25 pour l'IMC). Cela garantit qu'aucun facteur
# biologique n'est déclenché.
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


# ---- Classe DIABÈTE --------------------------------------------------------
# Glycémie et HbA1c franchement au-dessus des seuils diagnostiques. Les autres
# variables restent contraintes sous les seuils de zone intermédiaire pour
# éviter le basculement en CV_RISK : BMI < 25, BP < 120/80, lipides normaux,
# HR < 80.
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
    # Marqueurs diabétiques élevés : glycémie ≥ 126, HbA1c ≥ 6.5
    "fasting_glucose":  ContinuousParam("normal", mean=158.0, std=22.0),
    "hba1c":            ContinuousParam("normal", mean=7.4, std=0.85),
    # Lipides contrôlés (sous zone intermédiaire)
    "ldl":              ContinuousParam("normal", mean=100.0, std=12.0),
    "hdl":              ContinuousParam("normal", mean=54.0, std=7.0,
                                        mean_male=50.0, mean_female=60.0),
    "triglycerides":    ContinuousParam("lognormal", mean=120.0, std=18.0),
}


# ---- Classe DYSLIPIDÉMIE ---------------------------------------------------
# Au moins un marqueur lipidique franchement au-dessus de son seuil
# diagnostique (LDL ≥ 160 par défaut ici). Les autres variables restent sous
# les seuils intermédiaires.
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
    # Glycémie contrôlée (sous zone intermédiaire)
    "fasting_glucose":  ContinuousParam("normal", mean=90.0, std=5.0),
    "hba1c":            ContinuousParam("normal", mean=5.2, std=0.18),
    # LDL franchement élevé → critère diagnostique principal
    "ldl":              ContinuousParam("normal", mean=178.0, std=12.0),
    # HDL légèrement bas pour cohérence physiologique
    "hdl":              ContinuousParam("normal", mean=48.0, std=6.0,
                                        mean_male=46.0, mean_female=54.0),
    # Triglycérides légèrement élevés sans franchir 200
    "triglycerides":    ContinuousParam("lognormal", mean=145.0, std=22.0),
}


# ---- Classe HYPERTENSION ---------------------------------------------------
# PAS ≥ 140 et PAD ≥ 90. Les autres variables restent sous les seuils.
PARAMS_HYPERTENSION: dict[str, ContinuousParam] = {
    "age":              ContinuousParam("normal", mean=57.0, std=12.0),
    "height":           ContinuousParam("normal", mean=171.0, std=HEIGHT_STD,
                                        mean_male=HEIGHT_MEAN_MALE,
                                        mean_female=HEIGHT_MEAN_FEMALE),
    "bmi":              ContinuousParam("normal", mean=22.8, std=1.3),
    "heart_rate":       ContinuousParam("normal", mean=73.0, std=4.5),
    # Hypertension : SBP > 140, DBP > 90
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


# ---- Classe OBÉSITÉ --------------------------------------------------------
# IMC ≥ 30. Les autres variables sont contraintes pour éviter le basculement
# en CV_RISK (les obèses « purs » sans comorbidité sont rares en clinique
# réelle, mais notre découpage mono-label l'exige).
PARAMS_OBESITY: dict[str, ContinuousParam] = {
    "age":              ContinuousParam("normal", mean=45.0, std=12.0),
    "height":           ContinuousParam("normal", mean=171.0, std=HEIGHT_STD,
                                        mean_male=HEIGHT_MEAN_MALE,
                                        mean_female=HEIGHT_MEAN_FEMALE),
    # BMI franchement obèse
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


# ---- Classe RISQUE CARDIOVASCULAIRE ----------------------------------------
# Profil multifactoriel : plusieurs variables en zone intermédiaire. La règle
# d'attribution n_total ≥ 3 OU n_biological ≥ 2 sera satisfaite.
# Aucun seuil diagnostique strict n'est franchi en moyenne, mais des
# dépassements occasionnels sont acceptables (le class_assigner les classe
# tout de même en CV_RISK grâce à la priorité de l'étape 1).
PARAMS_CV_RISK: dict[str, ContinuousParam] = {
    "age":              ContinuousParam("normal", mean=60.0, std=10.0),
    "height":           ContinuousParam("normal", mean=171.0, std=HEIGHT_STD,
                                        mean_male=HEIGHT_MEAN_MALE,
                                        mean_female=HEIGHT_MEAN_FEMALE),
    # IMC surpoids
    "bmi":              ContinuousParam("normal", mean=27.5, std=1.4),
    # Fréquence cardiaque ≥ 80 → 1 facteur signes vitaux
    "heart_rate":       ContinuousParam("normal", mean=82.0, std=5.0),
    # PA borderline (≥ 120/80, < 140/90)
    "sbp":              ContinuousParam("normal", mean=130.0, std=5.0),
    "dbp":              ContinuousParam("normal", mean=84.0, std=3.0),
    "resp_rate":        RESP_RATE_PARAM,
    "temp":             TEMP_PARAM,
    # Glycémie borderline (≥ 100, < 126)
    "fasting_glucose":  ContinuousParam("normal", mean=110.0, std=6.0),
    "hba1c":            ContinuousParam("normal", mean=6.0, std=0.20),
    # Lipides borderline (LDL ≥ 130, < 160)
    "ldl":              ContinuousParam("normal", mean=145.0, std=8.0),
    # HDL légèrement bas
    "hdl":              ContinuousParam("normal", mean=44.0, std=5.0,
                                        mean_male=42.0, mean_female=52.0),
    # TG légèrement élevés (≥ 150, < 200)
    "triglycerides":    ContinuousParam("lognormal", mean=170.0, std=18.0),
}


# Registre principal
CONTINUOUS_PARAMS_BY_CLASS: dict[ClassLabel, dict[str, ContinuousParam]] = {
    ClassLabel.HEALTHY:       PARAMS_HEALTHY,
    ClassLabel.DIABETES:      PARAMS_DIABETES,
    ClassLabel.DYSLIPIDEMIA:  PARAMS_DYSLIPIDEMIA,
    ClassLabel.HYPERTENSION:  PARAMS_HYPERTENSION,
    ClassLabel.OBESITY:       PARAMS_OBESITY,
    ClassLabel.CV_RISK:       PARAMS_CV_RISK,
}


# ---------------------------------------------------------------------------
# Paramètres par classe — variables catégorielles
# ---------------------------------------------------------------------------
#
# Pour chaque (classe, variable catégorielle), une distribution multinomiale
# définit P(modalité | classe). Les fréquences sont calibrées pour refléter :
#   - les associations épidémiologiques connues (CV_RISK très associé au
#     tabagisme, à la sédentarité et à la mauvaise alimentation) ;
#   - les contraintes du cadre mono-label : les classes uni-paramétriques
#     doivent éviter d'accumuler trop de facteurs comportementaux, sous
#     peine d'être reclassées en CV_RISK.
#
# Chaque sous-dictionnaire {modalité: probabilité} a une somme = 1.0.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Auto-validation à l'import : invariants sur les distributions catégorielles
# ---------------------------------------------------------------------------


def _validate_categorical_distributions() -> None:
    """
    Vérifie à l'import :
        - chaque distribution somme à 1.0 (à 1e-9 près) ;
        - chaque modalité référencée existe dans `clinical_ranges`.
    """
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
