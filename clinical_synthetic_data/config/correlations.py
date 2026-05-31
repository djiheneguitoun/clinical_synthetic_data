"""
Matrice de corrélation pour la copule gaussienne.

Définit la structure de dépendance entre les variables continues
échantillonnées par le copuleur. Les corrélations reflètent les associations
physiopathologiques bien établies, communes à toutes les classes :
seules les marginales (moyennes, écarts-types) varient selon la classe.

Variables incluses dans la copule (13)
--------------------------------------
    age, height, bmi,
    heart_rate, sbp, dbp, resp_rate, temp,
    fasting_glucose, hba1c, ldl, hdl, triglycerides

Les variables `weight` et `total_chol` ne sont PAS échantillonnées par la
copule : elles sont dérivées de manière déterministe pour satisfaire
exactement les règles R1 (Quetelet) et R2 (Friedewald) :
    weight     = bmi · (height / 100)²
    total_chol = ldl + hdl + tg / 5 + ε   où ε ~ N(0, 3) mg/dL
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Variables de la copule (ordre canonique)
# ---------------------------------------------------------------------------

COPULA_VARIABLES: tuple[str, ...] = (
    "age",
    "height",
    "bmi",
    "heart_rate",
    "sbp",
    "dbp",
    "resp_rate",
    "temp",
    "fasting_glucose",
    "hba1c",
    "ldl",
    "hdl",
    "triglycerides",
)


# ---------------------------------------------------------------------------
# Paires de corrélation
# ---------------------------------------------------------------------------
#
# Justification clinique de chaque paire :
#
# Liées à l'âge :
#   - age ↔ sbp        : 0.30   Rigidité artérielle augmente avec l'âge.
#   - age ↔ dbp        : 0.10   Liée mais plus faiblement.
#   - age ↔ glucose    : 0.20   Résistance à l'insuline progressive.
#   - age ↔ hba1c      : 0.18   Idem, marqueur chronique.
#   - age ↔ ldl        : 0.20   Augmentation graduelle au cours de la vie.
#   - age ↔ hdl        : -0.10  Légère baisse avec l'âge.
#
# Liées à l'IMC (axe cardio-métabolique) :
#   - bmi ↔ sbp        : 0.30   Surcharge pondérale → élévation tensionnelle.
#   - bmi ↔ dbp        : 0.25
#   - bmi ↔ glucose    : 0.30   Insulinorésistance.
#   - bmi ↔ hba1c      : 0.25
#   - bmi ↔ ldl        : 0.18
#   - bmi ↔ hdl        : -0.30  Adiposité → baisse du HDL.
#   - bmi ↔ tg         : 0.40   Forte association lipidique.
#
# Cohérence physiologique forte :
#   - sbp ↔ dbp                : 0.70   Pression systolique/diastolique.
#   - fasting_glucose ↔ hba1c  : 0.85   Glycémie aiguë / chronique.
#
# Bilan lipidique :
#   - ldl ↔ hdl   : -0.15   Légère opposition.
#   - ldl ↔ tg    :  0.30   Co-élévation fréquente.
#   - hdl ↔ tg    : -0.35   Pattern athérogène classique.
#
# Tonus sympathique :
#   - heart_rate ↔ sbp : 0.15   Couplage autonomique léger.
#
# Toutes les autres paires sont supposées indépendantes (corrélation 0).
# ---------------------------------------------------------------------------


CORRELATION_PAIRS: tuple[tuple[str, str, float], ...] = (
    # Axe âge
    ("age", "sbp", 0.30),
    ("age", "dbp", 0.10),
    ("age", "fasting_glucose", 0.20),
    ("age", "hba1c", 0.18),
    ("age", "ldl", 0.20),
    ("age", "hdl", -0.10),
    # Axe IMC
    ("bmi", "sbp", 0.30),
    ("bmi", "dbp", 0.25),
    ("bmi", "fasting_glucose", 0.30),
    ("bmi", "hba1c", 0.25),
    ("bmi", "ldl", 0.18),
    ("bmi", "hdl", -0.30),
    ("bmi", "triglycerides", 0.40),
    # Cohérences physiologiques fortes
    ("sbp", "dbp", 0.70),
    ("fasting_glucose", "hba1c", 0.85),
    # Bilan lipidique
    ("ldl", "hdl", -0.15),
    ("ldl", "triglycerides", 0.30),
    ("hdl", "triglycerides", -0.35),
    # Tonus sympathique
    ("heart_rate", "sbp", 0.15),
)


# ---------------------------------------------------------------------------
# Construction et validation de la matrice
# ---------------------------------------------------------------------------


def build_correlation_matrix(
    variables: tuple[str, ...],
    pairs: tuple[tuple[str, str, float], ...],
) -> np.ndarray:
    """
    Construit une matrice de corrélation symétrique à partir d'une liste de
    paires (var1, var2, ρ). Les variables non listées en paire reçoivent une
    corrélation 0.

    Lève ValueError en cas de nom inconnu, de doublon ou de ρ hors [-1, 1].
    """
    name_to_index = {name: i for i, name in enumerate(variables)}
    n = len(variables)
    R = np.eye(n, dtype=float)

    seen: set[frozenset[str]] = set()
    for var1, var2, rho in pairs:
        if var1 not in name_to_index:
            raise ValueError(f"Variable inconnue : {var1!r}")
        if var2 not in name_to_index:
            raise ValueError(f"Variable inconnue : {var2!r}")
        if var1 == var2:
            raise ValueError(f"Auto-corrélation interdite pour {var1!r}")
        if not -1.0 <= rho <= 1.0:
            raise ValueError(f"Corrélation hors [-1, 1] : {rho}")
        key = frozenset({var1, var2})
        if key in seen:
            raise ValueError(f"Paire dupliquée : ({var1}, {var2})")
        seen.add(key)
        i, j = name_to_index[var1], name_to_index[var2]
        R[i, j] = rho
        R[j, i] = rho
    return R


def is_positive_definite(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Vérifie qu'une matrice est définie positive (donc factorisable par
    Cholesky), pré-requis pour échantillonner une N(0, R).
    """
    try:
        np.linalg.cholesky(matrix + tol * np.eye(matrix.shape[0]))
        return True
    except np.linalg.LinAlgError:
        return False


def project_to_nearest_correlation(
    matrix: np.ndarray,
    min_eigenvalue: float = 1e-6,
) -> np.ndarray:
    """
    Projection d'une matrice symétrique sur le cône des matrices de
    corrélation SDP.

    Méthode : décomposition spectrale, écrêtage des valeurs propres
    strictement positives, ré-écriture, puis renormalisation à diagonale
    unitaire (propriété d'une matrice de corrélation).

    C'est l'approche standard quand une matrice empirique ou paramétrique
    n'est pas strictement SDP en raison d'arrondis ou d'inconsistances
    locales. Pour des cas pathologiques, l'algorithme de Higham (« nearest
    correlation matrix ») serait préférable, mais cette projection simple
    suffit pour nos corrélations modérées.
    """
    # Symétrisation défensive
    matrix = 0.5 * (matrix + matrix.T)

    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, min_eigenvalue)
    psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Renormalisation à diagonale unitaire
    d = np.sqrt(np.diag(psd))
    return psd / np.outer(d, d)


def ensure_valid_correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Retourne la matrice telle quelle si elle est SDP, sinon sa projection
    sur le cône SDP.
    """
    if is_positive_definite(matrix):
        return matrix
    projected = project_to_nearest_correlation(matrix)
    if not is_positive_definite(projected):
        raise RuntimeError(
            "Échec de la projection SDP : la matrice projetée reste invalide."
        )
    return projected


# ---------------------------------------------------------------------------
# Matrice de référence du pipeline
# ---------------------------------------------------------------------------

BASE_CORRELATION_MATRIX: np.ndarray = ensure_valid_correlation_matrix(
    build_correlation_matrix(COPULA_VARIABLES, CORRELATION_PAIRS)
)


# Index inverse pour l'usage par le générateur
COPULA_INDEX: dict[str, int] = {
    name: i for i, name in enumerate(COPULA_VARIABLES)
}
