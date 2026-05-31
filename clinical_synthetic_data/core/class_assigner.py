"""
Hiérarchie d'attribution mono-label (rapport section 3.4).

Implémente la règle de priorité qui garantit qu'un patient n'appartient
qu'à une seule classe :

    1. Si n_total >= 3 ou n_bio >= 2  →  CV_RISK
    2. Sinon, si un seuil diagnostique pathologique est franchi
       (dans l'ordre : DIABETES, DYSLIPIDEMIA, HYPERTENSION, OBESITY) →
       classe correspondante.
    3. Sinon  →  HEALTHY.

Cette même fonction est utilisée à deux endroits du pipeline :
    - dans le `validator` (section 5.4) pour vérifier que la classe
      attribuée par le générateur correspond aux valeurs cliniques ;
    - dans la procédure de rejection sampling, pour rejeter tout
      candidat dont la classe inférée diffère de la classe cible.
"""

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


# Seuils opérationnels de la règle CV_RISK (rapport 3.4, étape 1)
CV_RISK_MIN_TOTAL_FACTORS: int = 3
CV_RISK_MIN_BIOLOGICAL_FACTORS: int = 2


def assign_class(p: PatientValues) -> ClassLabel:
    """
    Détermine la classe d'un patient à partir de ses valeurs cliniques.

    Paramètres
    ----------
    p : Mapping[str, Any]
        Vue plate des valeurs cliniques. Pour un objet `Patient`, utiliser
        `patient.to_flat_values()`.

    Retourne
    --------
    ClassLabel
        La classe unique attribuée selon la hiérarchie de la section 3.4.

    Notes
    -----
    L'ordre des tests reflète strictement le rapport :
        1. Test du cumul → CV_RISK
        2. Tests diagnostiques uni-paramétriques dans l'ordre de la
           section 3.3 (diabète, dyslipidémie, hypertension, obésité)
        3. Cas par défaut → HEALTHY
    """
    n_total, n_biological = count_cv_risk_factors(p)

    # Étape 1 : profil multifactoriel
    if (n_total >= CV_RISK_MIN_TOTAL_FACTORS
            or n_biological >= CV_RISK_MIN_BIOLOGICAL_FACTORS):
        return ClassLabel.CV_RISK

    # Étape 2 : classes pathologiques uni-paramétriques
    if is_diabetic(p):
        return ClassLabel.DIABETES
    if is_dyslipidemic(p):
        return ClassLabel.DYSLIPIDEMIA
    if is_hypertensive(p):
        return ClassLabel.HYPERTENSION
    if is_obese(p):
        return ClassLabel.OBESITY

    # Étape 3 : profil normal
    return ClassLabel.HEALTHY


def matches_expected_class(p: PatientValues, expected: ClassLabel) -> bool:
    """
    Renvoie True si la classe inférée par `assign_class` correspond exactement
    à `expected`. Utilisé par le validateur (section 5.4) et par la procédure
    de rejection sampling.
    """
    return assign_class(p) == expected


def explain_class_assignment(p: PatientValues) -> dict[str, Any]:
    """
    Trace d'attribution destinée au débogage et aux logs.

    Retourne un dict structuré indiquant la classe inférée, les compteurs de
    facteurs CV et l'état de chaque prédicat diagnostique. Cette fonction
    n'est jamais utilisée dans la boucle critique : son but est de fournir un
    rapport lisible quand un patient est rejeté.
    """
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
