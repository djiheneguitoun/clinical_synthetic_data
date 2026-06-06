"""Vérification des bornes physiologiques absolues (rapport section 5.2)."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from ..config.clinical_ranges import (
    CATEGORICAL_VARIABLES,
    CONTINUOUS_VARIABLES,
)


def violated_continuous_bound(p: Mapping[str, Any]) -> Optional[str]:
    """Première variable continue hors bornes (NaN/inf compris), sinon None."""
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
    """Première variable catégorielle hors modalités déclarées, sinon None."""
    for name, spec in CATEGORICAL_VARIABLES.items():
        value = p.get(name)
        if value is None:
            return name
        if value not in spec.modalities:
            return name
    return None


def check_absolute_bounds(p: Mapping[str, Any]) -> Optional[str]:
    """Vérifie les bornes continues puis catégorielles, retourne la 1re faute."""
    fault = violated_continuous_bound(p)
    if fault is not None:
        return f"continuous:{fault}"
    fault = violated_categorical_modality(p)
    if fault is not None:
        return f"categorical:{fault}"
    return None
