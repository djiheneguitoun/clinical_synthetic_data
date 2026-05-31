"""
Vérification des bornes physiologiques absolues (rapport section 5.2).

Tout enregistrement dont au moins une variable continue sort des bornes
absolues, ou dont une variable catégorielle prend une valeur hors modalités,
est invalide et doit être rejeté.

Les bornes proviennent exclusivement de `clinical_ranges` : ce module n'en
duplique aucune.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from ..config.clinical_ranges import (
    CATEGORICAL_VARIABLES,
    CONTINUOUS_VARIABLES,
)


def violated_continuous_bound(p: Mapping[str, Any]) -> Optional[str]:
    """
    Cherche la première variable continue hors bornes absolues.

    Retourne le nom de la variable fautive, ou None si toutes les valeurs
    sont dans leurs bornes. Une valeur NaN ou infinie est considérée
    comme une violation.
    """
    for name, spec in CONTINUOUS_VARIABLES.items():
        value = p.get(name)
        if value is None:
            return name
        try:
            v = float(value)
        except (TypeError, ValueError):
            return name
        if not math.isfinite(v):
            return name
        if v < spec.absolute_min or v > spec.absolute_max:
            return name
    return None


def violated_categorical_modality(p: Mapping[str, Any]) -> Optional[str]:
    """
    Cherche la première variable catégorielle hors modalités déclarées.

    Retourne le nom de la variable fautive, ou None si toutes les valeurs
    sont valides.
    """
    for name, spec in CATEGORICAL_VARIABLES.items():
        value = p.get(name)
        if value is None:
            return name
        if value not in spec.modalities:
            return name
    return None


def check_absolute_bounds(p: Mapping[str, Any]) -> Optional[str]:
    """
    Vérifie l'ensemble des bornes (continues puis catégorielles).

    Retourne None si tout est valide, sinon le nom de la première variable
    fautive (préfixée par 'continuous:' ou 'categorical:' pour
    distinguer le type d'échec).
    """
    fault = violated_continuous_bound(p)
    if fault is not None:
        return f"continuous:{fault}"
    fault = violated_categorical_modality(p)
    if fault is not None:
        return f"categorical:{fault}"
    return None
