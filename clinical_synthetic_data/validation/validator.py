"""
Validateur en cascade (rapport section 5.5).

Combine en un seul point d'entrée :
    1. Bornes physiologiques absolues (section 5.2)
    2. Cohérence inter-variables R1 → R4 (section 5.3)
    3. Cohérence valeurs / étiquette de classe (section 5.4)

L'exécution s'arrête au premier échec ; aucune correction n'est appliquée
sur l'enregistrement. La stratégie utilisée par les générateurs en aval
est le rejection sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..core.patient_schema import ClassLabel
from .class_coherence import check_class_coherence
from .inter_variable_rules import INTER_VARIABLE_RULES
from .physiological_bounds import check_absolute_bounds


# Identifiants stables des étapes de la cascade (utilisés pour la traçabilité
# des rejets dans `RejectionStatistics`).
RULE_BOUNDS: str = "bounds"
RULE_CLASS: str = "class_coherence"


@dataclass(frozen=True)
class ValidationResult:
    """
    Résultat d'une validation.

    Attributs
    ---------
    is_valid : bool
        True si toutes les règles ont passé.
    failed_rule : Optional[str]
        Identifiant de la première règle qui a échoué (None si valide).
    detail : Optional[str]
        Détail complémentaire éventuel (ex. nom de la variable hors borne).
    """

    is_valid: bool
    failed_rule: Optional[str] = None
    detail: Optional[str] = None

    @classmethod
    def ok(cls) -> "ValidationResult":
        return cls(is_valid=True)

    @classmethod
    def failed(cls, rule: str, detail: Optional[str] = None) -> "ValidationResult":
        return cls(is_valid=False, failed_rule=rule, detail=detail)


def validate(
    p: Mapping[str, Any],
    expected_class: ClassLabel,
) -> ValidationResult:
    """
    Pipeline de validation en cascade pour un enregistrement.

    Paramètres
    ----------
    p : Mapping[str, Any]
        Valeurs cliniques du patient (vue plate, telle que produite par
        un générateur). Les variables catégorielles doivent être des
        chaînes correspondant aux modalités déclarées.
    expected_class : ClassLabel
        Classe cible attribuée par le générateur.

    Retourne
    --------
    ValidationResult
        Résultat structuré : valide ou nom de la première règle violée.

    Ordre de la cascade (rapport 5.5) :
        bounds → R1 → R2 → R3 → R4 → class_coherence

    Cet ordre est délibéré :
        - les bornes sont les contrôles les moins coûteux et rejettent
          les candidats manifestement invalides ;
        - R1 à R4 sont des règles de cohérence locale, peu coûteuses ;
        - la cohérence de classe est le contrôle le plus coûteux car il
          réexécute l'ensemble des prédicats diagnostiques.
    """
    # 1. Bornes physiologiques absolues et modalités catégorielles
    bounds_fault = check_absolute_bounds(p)
    if bounds_fault is not None:
        return ValidationResult.failed(RULE_BOUNDS, detail=bounds_fault)

    # 2. Règles inter-variables R1 → R4
    for rule_name, rule_fn in INTER_VARIABLE_RULES:
        if not rule_fn(p):
            return ValidationResult.failed(rule_name)

    # 3. Cohérence valeurs / étiquette
    if not check_class_coherence(p, expected_class):
        return ValidationResult.failed(RULE_CLASS)

    return ValidationResult.ok()


def is_valid(p: Mapping[str, Any], expected_class: ClassLabel) -> bool:
    """Forme booléenne courte de `validate`."""
    return validate(p, expected_class).is_valid
