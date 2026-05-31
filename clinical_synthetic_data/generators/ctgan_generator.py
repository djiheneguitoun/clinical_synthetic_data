"""
Génération par CTGAN (Conditional Tabular GAN, rapport section 4.2).

Méthode 2 du pipeline : un GAN tabulaire conditionnel apprend la distribution
jointe des variables cliniques à partir d'un jeu d'entraînement, puis génère
de nouveaux échantillons. Le conditionnement s'effectue sur la variable de
classe pour produire un jeu équilibré.

"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..io_utils import read_dataset_csv

from ..core.patient_schema import (
    CATEGORICAL_FIELDS,
    CONTINUOUS_FIELDS,
    ClassLabel,
    Patient,
)
from ..validation import RejectionStatistics, validate
from .base_generator import BaseGenerator


# Bruit ajouté à la reconstruction Friedewald (cf. gaussian_copula.py)
FRIEDEWALD_NOISE_STD: float = 3.0


class CTGANGenerator(BaseGenerator):
    """
    Générateur appris par GAN tabulaire conditionnel.

    Paramètres
    ----------
    epochs : int
        Nombre d'epochs d'entraînement. Par défaut 300, conforme aux
        recommandations de SDV pour des datasets de quelques milliers de
        lignes. Réduire pour un développement rapide.
    batch_size : int
        Taille du batch d'entraînement. Par défaut 500.
    enforce_derived_variables : bool
        Si True (défaut), recalcule `weight` et `total_chol` à partir des
        autres variables après échantillonnage, ce qui garantit R1 et R2
        sans dépendre de la précision d'apprentissage du GAN.
    seed : int, optionnel
        Graine pour reproductibilité. Propagée à numpy, torch et SDV.
    max_total_attempts_factor : int
        Garde-fou anti-boucle infinie (cf. GaussianCopulaGenerator).
    verbose : bool
        Active les logs d'entraînement SDV.
    """

    def __init__(
        self,
        epochs: int = 300,
        batch_size: int = 500,
        enforce_derived_variables: bool = True,
        seed: Optional[int] = None,
        max_total_attempts_factor: int = 500,
        verbose: bool = True,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.enforce_derived_variables = enforce_derived_variables
        self.seed = seed
        self.max_total_attempts_factor = max_total_attempts_factor
        self.verbose = verbose

        self.stats = RejectionStatistics()
        self._synthesizer = None
        self._metadata = None
        self._is_fitted = False

        if seed is not None:
            self._set_seeds(seed)

    # -----------------------------------------------------------------
    # Initialisation déterministe
    # -----------------------------------------------------------------

    @staticmethod
    def _set_seeds(seed: int) -> None:
        """Propage une graine à numpy et torch pour la reproductibilité."""
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            # torch n'est pas installé : SDV ne devrait pas s'importer non plus
            pass

    # -----------------------------------------------------------------
    # Métadonnée SDV
    # -----------------------------------------------------------------

    def _build_metadata(self, df: pd.DataFrame):
        """
        Construit la métadonnée SDV à partir du schéma typé du projet.

        Variables continues  → sdtype='numerical'
        Variables catégorielles + classe → sdtype='categorical'
        """
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()

        for col in CONTINUOUS_FIELDS:
            if col in df.columns:
                metadata.add_column(col, sdtype="numerical")

        for col in CATEGORICAL_FIELDS:
            if col in df.columns:
                metadata.add_column(col, sdtype="categorical")

        if "class_label" in df.columns:
            metadata.add_column("class_label", sdtype="categorical")

        return metadata

    # -----------------------------------------------------------------
    # Entraînement
    # -----------------------------------------------------------------

    def fit(self, training_data: Optional[pd.DataFrame] = None) -> None:
        """
        Entraîne CTGAN sur `training_data` (typiquement le dataset produit
        par la copule).
        """
        from sdv.single_table import CTGANSynthesizer

        if training_data is None or len(training_data) == 0:
            raise ValueError(
                "CTGAN exige un dataset d'entraînement non vide. "
                "Générer d'abord un dataset avec la copule (Méthode 1)."
            )

        # `patient_id` est un identifiant unique sans valeur statistique :
        # on le retire avant entraînement.
        train_df = training_data.drop(columns=["patient_id"], errors="ignore")

        self._metadata = self._build_metadata(train_df)
        self._synthesizer = CTGANSynthesizer(
            metadata=self._metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        self._synthesizer.fit(train_df)
        self._is_fitted = True

    # -----------------------------------------------------------------
    # Échantillonnage avec validation
    # -----------------------------------------------------------------

    def _sample_raw_batch(
        self,
        class_label: ClassLabel,
        n_rows: int,
    ) -> pd.DataFrame:
        """
        Échantillonne `n_rows` lignes brutes de CTGAN conditionnellement à
        la classe demandée. SDV peut occasionnellement retourner moins de
        lignes que demandé pour des classes très rares — on accepte cela
        et la boucle d'échantillonnage compensera au prochain batch.
        """
        from sdv.sampling import Condition

        condition = Condition(
            num_rows=n_rows,
            column_values={"class_label": class_label.value},
        )
        return self._synthesizer.sample_from_conditions(conditions=[condition])

    def _enforce_constructive_constraints(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Recalcule les variables dérivées pour garantir R1 et R2 exactement
        (rapport 5.3). Strictement identique au post-traitement de la
        copule, ce qui rend les deux méthodes comparables pour la
        validation.
        """
        if not self.enforce_derived_variables:
            return df

        df = df.copy()

        # R1 — Quetelet : weight = BMI · (height/100)²
        df["weight"] = df["bmi"] * (df["height"] / 100.0) ** 2

        # R2 — Friedewald : total_chol = LDL + HDL + TG/5 + ε
        noise = np.random.normal(0.0, FRIEDEWALD_NOISE_STD, size=len(df))
        df["total_chol"] = (
            df["ldl"] + df["hdl"] + df["triglycerides"] / 5.0 + noise
        )
        return df

    def sample_class(
        self,
        class_label: ClassLabel,
        n: int,
    ) -> list[Patient]:
        """
        Échantillonne exactement `n` patients valides de la classe demandée.

        Procédure :
            1. Tirer un batch conditionnel via CTGAN.
            2. Re-imposer R1 et R2 par construction.
            3. Valider chaque candidat (bornes + R3 + R4 + classe).
            4. Accepter / rejeter, accumuler les statistiques.
            5. Itérer jusqu'à atteindre `n`, en sécurisant contre les
               boucles infinies via `max_total_attempts_factor`.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "CTGAN n'est pas encore entraîné : appeler `fit()` avant "
                "`sample_class()`."
            )
        if n <= 0:
            return []

        accepted: list[Patient] = []
        total_attempts = 0
        max_attempts = max(n * self.max_total_attempts_factor, 1000)

        while len(accepted) < n:
            remaining = n - len(accepted)
            batch_size = max(remaining * 2, 200)
            batch_size = min(batch_size, 2000)

            df_batch = self._sample_raw_batch(class_label, batch_size)
            if len(df_batch) == 0:
                # SDV n'a produit aucune ligne pour cette classe ; on
                # retente avec un batch suivant. Si cela persiste, le
                # garde-fou `max_attempts` lèvera une RuntimeError.
                continue

            df_batch = self._enforce_constructive_constraints(df_batch)

            for _, row in df_batch.iterrows():
                self.stats.record_attempt(class_label)
                total_attempts += 1

                if total_attempts > max_attempts:
                    raise RuntimeError(
                        f"Génération CTGAN impossible pour la classe "
                        f"{class_label.value} : {total_attempts} tentatives, "
                        f"seulement {len(accepted)}/{n} acceptés. "
                        f"Le modèle est probablement sous-entraîné. "
                        f"Augmenter `epochs` (recommandation : ≥ 300) "
                        f"ou `max_total_attempts_factor`."
                    )

                candidate = row.to_dict()
                # Force la classe attendue (SDV peut occasionnellement
                # dévier malgré la condition)
                candidate["class_label"] = class_label.value

                result = validate(candidate, class_label)
                if result.is_valid:
                    candidate["patient_id"] = Patient.new_id()
                    accepted.append(Patient.from_mapping(candidate))
                    self.stats.record_acceptance(class_label)
                    if len(accepted) >= n:
                        return accepted
                else:
                    self.stats.record_rejection(class_label, result.failed_rule)

        return accepted


# ===========================================================================
# CLI — exécution autonome
# ===========================================================================


def _build_cli_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="Génération par CTGAN (Méthode 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--training", required=True,
                        help="CSV du dataset d'entraînement (ex: dataset_copula.csv).")
    parser.add_argument("--m", "--m-per-class", type=int, default=1000,
                        dest="m_per_class",
                        help="Patients par classe à générer.")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Nombre d'epochs d'entraînement.")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Taille de batch CTGAN.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Graine reproductible.")
    parser.add_argument("--output", "-o", default="dataset_ctgan.csv",
                        help="Chemin du fichier CSV de sortie.")
    parser.add_argument("--max-attempts-factor", type=int, default=500,
                        help="Facteur de tolérance au rejet (n × factor max).")
    return parser


def main(argv=None) -> int:
    """
    Point d'entrée CLI.

        python -m clinical_synthetic_data.generators.ctgan_generator \\
            --training dataset_copula.csv --m 1000
    """
    import logging
    import time
    import pandas as pd
    from ..io_utils import read_dataset_csv
    from ..logging_setup import setup_logging

    setup_logging()
    log = logging.getLogger("pipeline")

    args = _build_cli_parser().parse_args(argv)

    log.info("=" * 60)
    log.info("Méthode 2 — CTGAN")
    log.info(f"  Training       : {args.training}")
    log.info(f"  M par classe   : {args.m_per_class}  →  total {6 * args.m_per_class}")
    log.info(f"  Epochs         : {args.epochs}")
    log.info(f"  Batch size     : {args.batch_size}")
    log.info(f"  Seed           : {args.seed}")
    log.info("=" * 60)

    log.info(f"Chargement du dataset d'entraînement…")
    training = read_dataset_csv(args.training)
    log.info(f"  → {len(training)} patients chargés")

    generator = CTGANGenerator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_total_attempts_factor=args.max_attempts_factor,
        seed=args.seed,
    )

    log.info(f"Entraînement CTGAN ({args.epochs} epochs)…")
    log.info("  (Une barre de progression va s'afficher pour le suivi en temps réel)")
    t0 = time.time()
    generator.fit(training)
    log.info(f"  ✓ Entraînement terminé en {time.time() - t0:.0f}s")

    log.info("Génération par classe…")
    t0 = time.time()
    df = generator.generate_balanced_dataset(m_per_class=args.m_per_class)
    dt = time.time() - t0

    stats = generator.stats.to_report()
    log.info("-" * 60)
    log.info(f"Terminé en {dt:.0f}s")
    log.info(f"  Patients générés : {len(df)}")
    log.info(f"  Taux rejet global : {stats['global_rejection_rate']:.1%}")
    log.info(f"  Tentatives moyennes/accepté : {stats['average_attempts_per_accepted']:.2f}")

    df.to_csv(args.output, index=False)
    log.info(f"  ✓ Sauvegardé : {args.output}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
