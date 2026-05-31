
from __future__ import annotations

import logging
import warnings


# Loggers tiers qui inondent stdout d'informations non actionnables.
# Mis au niveau WARNING : on garde les erreurs vraies, on cache le bruit.
NOISY_THIRD_PARTY_LOGGERS: tuple[str, ...] = (
    "rdt",
    "rdt.transformers",
    "rdt.transformers.utils",
    "rdt.transformers.null",
    "sdv",
    "sdv.data_processing",
    "sdv.data_processing.data_processor",
    "sdv.single_table",
    "sdv.single_table.base",
    "sdv.single_table.SingleTableSynthesizer",
    "SingleTableSynthesizer",
)


def setup_logging(
    level: int = logging.INFO,
    quiet_third_party: bool = True,
) -> None:
    """
    Configure le logging racine avec un format propre et silencie les
    bibliothèques tierces verbeuses.

    Paramètres
    ----------
    level : int
        Niveau du logger racine (défaut : INFO).
    quiet_third_party : bool
        Si True, met les loggers SDV/rdt à WARNING et ignore les
        FutureWarning / UserWarning courants.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,           # remplace toute config déjà installée
    )

    if quiet_third_party:
        for logger_name in NOISY_THIRD_PARTY_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
