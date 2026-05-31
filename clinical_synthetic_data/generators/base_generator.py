"""
Interface abstraite des générateurs de données cliniques synthétiques.

"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from ..core.patient_schema import ClassLabel, Patient
from ..validation import RejectionStatistics

if TYPE_CHECKING:
    import pandas as pd


_logger = logging.getLogger("pipeline")


class BaseGenerator(ABC):
    """Interface commune des générateurs."""

    stats: RejectionStatistics

    @abstractmethod
    def fit(self, training_data: Optional["pd.DataFrame"] = None) -> None:
        """Entraîne ou paramètre le générateur."""

    @abstractmethod
    def sample_class(
        self,
        class_label: ClassLabel,
        n: int,
    ) -> list[Patient]:
        """Échantillonne `n` patients valides de la classe demandée."""

    def generate_balanced_dataset(
        self,
        m_per_class: int,
        log_progress: bool = True,
    ) -> "pd.DataFrame":
        """
        Produit un jeu équilibré : `m_per_class` patients par classe.

        Si `log_progress=True`, affiche pour chaque classe :
            [i/6] classe : ✓ N patients (rejet X%, durée s)
        """
        import pandas as pd

        all_patients: list[Patient] = []
        n_classes = len(ClassLabel)

        for i, class_label in enumerate(ClassLabel, start=1):
            t0 = time.time()
            attempts_before = self.stats.attempts_per_class.get(class_label, 0)

            patients = self.sample_class(class_label, m_per_class)
            all_patients.extend(patients)

            if log_progress:
                dt = time.time() - t0
                attempts_after = self.stats.attempts_per_class[class_label]
                attempts = attempts_after - attempts_before
                rejects = attempts - m_per_class
                rate = rejects / attempts if attempts > 0 else 0.0
                _logger.info(
                    f"  [{i}/{n_classes}] {class_label.value:<25s} "
                    f"✓ {m_per_class} patients (rejet {rate:.0%}, {dt:.1f}s)"
                )

        df = pd.DataFrame([p.to_dict() for p in all_patients])
        return df[list(Patient.field_names())]
