"""Cohérence valeurs cliniques / étiquette de classe (rapport section 5.4)."""

from __future__ import annotations

from typing import Any, Mapping

from ..core.class_assigner import assign_class
from ..core.patient_schema import ClassLabel


def check_class_coherence(
    p: Mapping[str, Any],
    expected_class: ClassLabel,
) -> bool:
    """Retourne True ssi `assign_class(p)` est égal à `expected_class`."""
    return assign_class(p) == expected_class
