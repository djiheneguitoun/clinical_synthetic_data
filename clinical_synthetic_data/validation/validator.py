"""Validateur en cascade (rapport section 5.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..core.patient_schema import ClassLabel
from .class_coherence import check_class_coherence
from .inter_variable_rules import INTER_VARIABLE_RULES
from .physiological_bounds import check_absolute_bounds


RULE_BOUNDS: str = "bounds"
RULE_CLASS: str = "class_coherence"


@dataclass(frozen=True)
class ValidationResult:
    """Résultat d'une validation."""

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
    """Cascade bounds → R1 → R2 → R3 → R4 → class_coherence (arrêt au 1er échec)."""
    bounds_fault = check_absolute_bounds(p)
    if bounds_fault is not None:
        return ValidationResult.failed(RULE_BOUNDS, detail=bounds_fault)

    for rule_name, rule_fn in INTER_VARIABLE_RULES:
        if not rule_fn(p):
            return ValidationResult.failed(rule_name)

    if not check_class_coherence(p, expected_class):
        return ValidationResult.failed(RULE_CLASS)

    return ValidationResult.ok()


def is_valid(p: Mapping[str, Any], expected_class: ClassLabel) -> bool:
    """Forme booléenne courte de `validate`."""
    return validate(p, expected_class).is_valid
