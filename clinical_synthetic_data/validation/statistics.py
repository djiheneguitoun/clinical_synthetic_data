"""Collecteur de statistiques de rejet (rapport section 5.6)."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from ..core.patient_schema import ClassLabel


@dataclass
class RejectionStatistics:
    """Compteurs de tentatives et de rejets."""

    attempts_per_class: Counter = field(default_factory=Counter)
    accepted_per_class: Counter = field(default_factory=Counter)
    rejections_per_rule: Counter = field(default_factory=Counter)
    rejections_per_rule_per_class: dict = field(
        default_factory=lambda: defaultdict(Counter)
    )

    def record_attempt(self, target_class: ClassLabel) -> None:
        self.attempts_per_class[target_class] += 1

    def record_acceptance(self, target_class: ClassLabel) -> None:
        self.accepted_per_class[target_class] += 1

    def record_rejection(
        self,
        target_class: ClassLabel,
        failed_rule: str,
    ) -> None:
        self.rejections_per_rule[failed_rule] += 1
        self.rejections_per_rule_per_class[target_class][failed_rule] += 1

    def total_attempts(self) -> int:
        return sum(self.attempts_per_class.values())

    def total_accepted(self) -> int:
        return sum(self.accepted_per_class.values())

    def total_rejected(self) -> int:
        return self.total_attempts() - self.total_accepted()

    def global_rejection_rate(self) -> float:
        attempts = self.total_attempts()
        return self.total_rejected() / attempts if attempts > 0 else 0.0

    def rejection_rate_by_class(self) -> dict[ClassLabel, float]:
        return {
            cls: (
                (self.attempts_per_class[cls] - self.accepted_per_class[cls])
                / self.attempts_per_class[cls]
                if self.attempts_per_class[cls] > 0
                else 0.0
            )
            for cls in self.attempts_per_class
        }

    def average_attempts_per_accepted(self) -> Optional[float]:
        accepted = self.total_accepted()
        if accepted == 0:
            return None
        return self.total_attempts() / accepted

    def to_report(self) -> dict:
        """Représentation sérialisable (JSON) du rapport de rejet."""
        return {
            "total_attempts": self.total_attempts(),
            "total_accepted": self.total_accepted(),
            "total_rejected": self.total_rejected(),
            "global_rejection_rate": self.global_rejection_rate(),
            "average_attempts_per_accepted": self.average_attempts_per_accepted(),
            "rejection_rate_by_class": {
                cls.value: rate for cls, rate in self.rejection_rate_by_class().items()
            },
            "rejections_per_rule": dict(self.rejections_per_rule),
            "rejections_per_rule_per_class": {
                cls.value: dict(counter)
                for cls, counter in self.rejections_per_rule_per_class.items()
            },
        }
