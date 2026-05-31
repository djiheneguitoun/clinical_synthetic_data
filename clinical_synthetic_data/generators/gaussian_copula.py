"""
Génération de patients par copule gaussienne (Méthode 1, rapport 4.2).

Principe mathématique
---------------------
La copule gaussienne sépare la structure de dépendance (matrice de
corrélation R) des distributions marginales (lois cliniques par variable).

Pour un vecteur Z ~ N(0, R), chaque composante Z_i est standard normale, et
l'application X_i = F_i^{-1}(Φ(Z_i)) produit une variable de loi marginale
F_i tout en préservant les corrélations encodées par R.

Spécialisation pour ce projet
-----------------------------
1. Les marginales F_i sont normales ou log-normales paramétrées en
   (mean, std) sur l'échelle linéaire, et dépendent éventuellement du sexe
   (HDL, taille).
2. Pour les marginales normales, la transformation se simplifie en
   X_i = μ_i + σ_i · Z_i.
3. Pour les marginales log-normales, on applique la formule classique
   X_i = exp(μ_log_i + σ_log_i · Z_i) avec conversion analytique.
4. Le poids et le cholestérol total sont DÉRIVÉS (pas échantillonnés),
   ce qui garantit R1 (Quetelet) et R2 (Friedewald) par construction.
5. Le rejection sampling rejette les candidats qui violent les autres
   contraintes (R3, R4, bornes, cohérence de classe).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config.correlations import (
    BASE_CORRELATION_MATRIX,
    COPULA_VARIABLES,
)
from ..config.generation_params import (
    CATEGORICAL_PROBS_BY_CLASS,
    CONTINUOUS_PARAMS_BY_CLASS,
    ContinuousParam,
)
from ..core.patient_schema import ClassLabel, Patient
from ..validation import RejectionStatistics, validate
from .base_generator import BaseGenerator


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Bruit gaussien ajouté à total_chol = LDL + HDL + TG/5 + ε. σ = 3 mg/dL
# garantit |ε| ≤ 9 mg/dL avec ~99 % de probabilité, soit sous la tolérance
# de 10 mg/dL de R2 (rapport 5.3).
FRIEDEWALD_NOISE_STD: float = 3.0


# ---------------------------------------------------------------------------
# Transformation marginale vectorisée
# ---------------------------------------------------------------------------


def _vectorized_marginal_transform(
    param: ContinuousParam,
    z: np.ndarray,
    sexes: np.ndarray,
) -> np.ndarray:
    """
    Applique la transformation marginale F_i^{-1} ∘ Φ sur un batch.

    Équivalent vectorisé de `ContinuousParam.transform()` ; les moyennes
    dépendantes du sexe sont calculées vectoriellement sans boucle Python.

    Paramètres
    ----------
    param : ContinuousParam
        Paramètres de la marginale.
    z : np.ndarray
        Composante du vecteur Z ~ N(0, R) correspondant à cette variable
        (déjà standard normale par marginal).
    sexes : np.ndarray
        Sexes des n patients du batch (array de strings 'Male' / 'Female').

    Retourne
    --------
    np.ndarray
        Valeurs cliniques sur l'échelle linéaire, taille `len(z)`.
    """
    n = len(z)

    # Moyennes effectives, sexe-conditionnelles si applicable
    if param.mean_male is not None or param.mean_female is not None:
        m_male = param.mean_male if param.mean_male is not None else param.mean
        m_female = param.mean_female if param.mean_female is not None else param.mean
        means = np.where(sexes == "Male", m_male, m_female)
    else:
        means = np.full(n, param.mean)

    if param.distribution == "normal":
        return means + param.std * z

    # Log-normale : conversion (mean, std) linéaires → (μ_log, σ_log)
    cv_squared = (param.std / means) ** 2
    sigma_log = np.sqrt(np.log1p(cv_squared))
    mu_log = np.log(means) - 0.5 * sigma_log * sigma_log
    return np.exp(mu_log + sigma_log * z)


# ---------------------------------------------------------------------------
# Générateur principal
# ---------------------------------------------------------------------------


class GaussianCopulaGenerator(BaseGenerator):
    """
    Générateur paramétrique par copule gaussienne (Méthode 1).

    Paramètres
    ----------
    correlation_matrix : np.ndarray, optionnel
        Matrice 13×13 SDP utilisée par la copule. Par défaut, la matrice de
        référence du pipeline (`BASE_CORRELATION_MATRIX`).
    seed : int, optionnel
        Graine de reproductibilité du PRNG. Si None, tirage non reproductible.
    max_total_attempts_factor : int
        Garde-fou contre les boucles infinies : la génération échoue si le
        nombre total de tentatives dépasse `n × max_total_attempts_factor`
        pour une classe donnée. La valeur par défaut (200) tolère un taux
        de rejet supérieur à 99,5 % avant d'abandonner.
    """

    def __init__(
        self,
        correlation_matrix: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        max_total_attempts_factor: int = 200,
    ) -> None:
        self.correlation_matrix = (
            BASE_CORRELATION_MATRIX
            if correlation_matrix is None
            else correlation_matrix
        )
        self.rng = np.random.default_rng(seed)
        self.max_total_attempts_factor = max_total_attempts_factor
        self.stats = RejectionStatistics()

        # Cholesky pré-calculée : L tel que L · L^T = R.
        # Pour échantillonner X ~ N(0, R), on utilise X = Y · L^T avec
        # Y ~ N(0, I) (forme adaptée aux batchs en lignes).
        self._cholesky_factor = np.linalg.cholesky(self.correlation_matrix)

    # -----------------------------------------------------------------
    # Interface
    # -----------------------------------------------------------------

    def fit(self, training_data=None) -> None:
        """
        Générateur paramétrique : pas d'apprentissage. Conservé pour
        respecter l'interface `BaseGenerator`.
        """
        return None

    def sample_class(
        self,
        class_label: ClassLabel,
        n: int,
    ) -> list[Patient]:
        """
        Échantillonne exactement `n` patients valides de la classe demandée.

        Procédure :
            - boucle de génération par batchs sur-dimensionnés ;
            - validation de chaque candidat par le pipeline en cascade ;
            - acceptation ou rejet, traçabilité via `self.stats` ;
            - arrêt et erreur si le quota de tentatives est dépassé.
        """
        if n <= 0:
            return []

        accepted: list[Patient] = []
        total_attempts = 0
        max_attempts = max(n * self.max_total_attempts_factor, 1000)

        while len(accepted) < n:
            remaining = n - len(accepted)
            # Dimension du batch : suffisamment large pour amortir le coût
            # de l'algèbre matricielle, mais bornée pour éviter le gaspillage.
            batch_size = max(remaining * 2, 200)
            batch_size = min(batch_size, 2000)

            for candidate in self._generate_batch(class_label, batch_size):
                self.stats.record_attempt(class_label)
                total_attempts += 1

                if total_attempts > max_attempts:
                    raise RuntimeError(
                        f"Génération impossible pour la classe "
                        f"{class_label.value} : {total_attempts} tentatives, "
                        f"seulement {len(accepted)}/{n} acceptés. "
                        f"Vérifier les paramètres de la classe."
                    )

                result = validate(candidate, class_label)
                if result.is_valid:
                    candidate["patient_id"] = Patient.new_id()
                    candidate["class_label"] = class_label.value
                    accepted.append(Patient.from_mapping(candidate))
                    self.stats.record_acceptance(class_label)
                    if len(accepted) >= n:
                        return accepted
                else:
                    self.stats.record_rejection(class_label, result.failed_rule)

        return accepted

    # -----------------------------------------------------------------
    # Internes
    # -----------------------------------------------------------------

    def _sample_correlated_z(self, batch_size: int) -> np.ndarray:
        """
        Tire `batch_size` vecteurs ~ N(0, R) à partir de Y ~ N(0, I_p).

        Si L = cholesky(R) (matrice triangulaire inférieure), alors
        X = Y · L^T  vérifie  Cov(X) = L · L^T = R.
        """
        p = len(COPULA_VARIABLES)
        Y = self.rng.standard_normal((batch_size, p))
        return Y @ self._cholesky_factor.T

    def _sample_categorical(
        self,
        class_label: ClassLabel,
        var_name: str,
        n: int,
    ) -> np.ndarray:
        """Tirage multinomial conditionnel à la classe."""
        dist = CATEGORICAL_PROBS_BY_CLASS[class_label][var_name]
        modalities = np.array(list(dist.keys()))
        probabilities = np.array(list(dist.values()))
        return self.rng.choice(modalities, size=n, p=probabilities)

    def _generate_batch(
        self,
        class_label: ClassLabel,
        batch_size: int,
    ) -> list[dict]:
        """
        Construit un batch de candidats (non encore validés).

        Étapes :
            1. Tirage du sexe (multinomial).
            2. Tirage de Z ~ N(0, R) (cf. _sample_correlated_z).
            3. Transformation marginale par variable (sexe-conditionnelle).
            4. Tirage des autres variables catégorielles (indépendantes
               de Z, multinomial conditionnel à la classe).
            5. Dérivation déterministe de `weight` et `total_chol` pour
               satisfaire R1 et R2 par construction.
            6. Assemblage en dicts de valeurs cliniques plates.
        """
        params = CONTINUOUS_PARAMS_BY_CLASS[class_label]

        # 1. Sexe d'abord (pilote les moyennes sexe-conditionnelles)
        sexes = self._sample_categorical(class_label, "sex", batch_size)

        # 2. Vecteurs corrélés en colonne par variable
        Z = self._sample_correlated_z(batch_size)

        # 3. Transformation marginale variable par variable
        continuous: dict[str, np.ndarray] = {}
        for j, var_name in enumerate(COPULA_VARIABLES):
            continuous[var_name] = _vectorized_marginal_transform(
                params[var_name], Z[:, j], sexes
            )

        # 4. Variables catégorielles restantes (indépendantes de Z)
        categorical: dict[str, np.ndarray] = {"sex": sexes}
        for var_name in ("physical_activity", "smoking", "alcohol", "diet_quality"):
            categorical[var_name] = self._sample_categorical(
                class_label, var_name, batch_size
            )

        # 5. Variables dérivées : R1 (exact) et R2 (à ε près)
        weight = continuous["bmi"] * (continuous["height"] / 100.0) ** 2
        friedewald_noise = self.rng.normal(
            0.0, FRIEDEWALD_NOISE_STD, size=batch_size
        )
        total_chol = (
            continuous["ldl"]
            + continuous["hdl"]
            + continuous["triglycerides"] / 5.0
            + friedewald_noise
        )

        # 6. Assemblage des dicts (boucle Python compatible avec
        # `Patient.from_mapping` ; coût négligeable face à la validation).
        candidates: list[dict] = []
        for i in range(batch_size):
            c: dict = {var: float(continuous[var][i]) for var in COPULA_VARIABLES}
            c["weight"] = float(weight[i])
            c["total_chol"] = float(total_chol[i])
            for var, arr in categorical.items():
                c[var] = str(arr[i])
            candidates.append(c)

        return candidates


# ===========================================================================
# CLI — exécution autonome
# ===========================================================================


def _build_cli_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="Génération par copule gaussienne (Méthode 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--m", "--m-per-class", type=int, default=1000,
                        dest="m_per_class",
                        help="Patients par classe (6 × M au total).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Graine reproductible.")
    parser.add_argument("--output", "-o", default="dataset_copula.csv",
                        help="Chemin du fichier CSV de sortie.")
    return parser


def main(argv=None) -> int:
    """
    Point d'entrée CLI.

        python -m clinical_synthetic_data.generators.gaussian_copula --m 1000
    """
    import logging
    from ..logging_setup import setup_logging
    setup_logging()
    log = logging.getLogger("pipeline")

    args = _build_cli_parser().parse_args(argv)

    log.info("=" * 60)
    log.info(f"Méthode 1 — Copule gaussienne")
    log.info(f"  M par classe : {args.m_per_class}  →  total {6 * args.m_per_class}")
    log.info(f"  Seed         : {args.seed}")
    log.info("=" * 60)

    generator = GaussianCopulaGenerator(seed=args.seed)

    log.info("Génération en cours…")
    import time
    t0 = time.time()
    df = generator.generate_balanced_dataset(m_per_class=args.m_per_class)
    dt = time.time() - t0

    stats = generator.stats.to_report()
    log.info("-" * 60)
    log.info(f"Terminé en {dt:.1f}s")
    log.info(f"  Patients générés : {len(df)}")
    log.info(f"  Taux rejet global : {stats['global_rejection_rate']:.1%}")
    log.info(f"  Tentatives moyennes/accepté : {stats['average_attempts_per_accepted']:.2f}")

    df.to_csv(args.output, index=False)
    log.info(f"  ✓ Sauvegardé : {args.output}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
