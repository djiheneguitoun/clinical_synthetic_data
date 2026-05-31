"""
Cohérence valeurs cliniques / étiquette de classe (rapport section 5.4).

Confronte la classe attribuée par le générateur à la classe qui résulterait
de l'application de la hiérarchie d'attribution sur les valeurs cliniques.

Cette règle est la pierre angulaire du cadre mono-label : elle garantit que
le `class_label` stocké dans le dataset correspond exactement à ce qu'un
clinicien ou un classifieur déterministe inférerait à partir des valeurs.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..core.class_assigner import assign_class
from ..core.patient_schema import ClassLabel


def check_class_coherence(
    p: Mapping[str, Any],
    expected_class: ClassLabel,
) -> bool:
    """
    Retourne True ssi la classe inférée par `assign_class` est strictement
    égale à `expected_class`.

    Paramètres
    ----------
    p : Mapping[str, Any]
        Valeurs cliniques du patient (vue plate).
    expected_class : ClassLabel
        Classe stockée dans le dataset / cible de la génération.
    """
    return assign_class(p) == expected_class
