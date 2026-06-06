"""Matrice de corrélation pour la copule gaussienne."""

from __future__ import annotations

import numpy as np


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


CORRELATION_PAIRS: tuple[tuple[str, str, float], ...] = (
    ("age", "sbp", 0.30),
    ("age", "dbp", 0.10),
    ("age", "fasting_glucose", 0.20),
    ("age", "hba1c", 0.18),
    ("age", "ldl", 0.20),
    ("age", "hdl", -0.10),
    ("bmi", "sbp", 0.30),
    ("bmi", "dbp", 0.25),
    ("bmi", "fasting_glucose", 0.30),
    ("bmi", "hba1c", 0.25),
    ("bmi", "ldl", 0.18),
    ("bmi", "hdl", -0.30),
    ("bmi", "triglycerides", 0.40),
    ("sbp", "dbp", 0.70),
    ("fasting_glucose", "hba1c", 0.85),
    ("ldl", "hdl", -0.15),
    ("ldl", "triglycerides", 0.30),
    ("hdl", "triglycerides", -0.35),
    ("heart_rate", "sbp", 0.15),
)


def build_correlation_matrix(
    variables: tuple[str, ...],
    pairs: tuple[tuple[str, str, float], ...],
) -> np.ndarray:
    """Construit une matrice de corrélation symétrique à partir de paires."""
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
    """Vérifie qu'une matrice est définie positive (factorisable par Cholesky)."""
    try:
        np.linalg.cholesky(matrix + tol * np.eye(matrix.shape[0]))
        return True
    except np.linalg.LinAlgError:
        return False


def project_to_nearest_correlation(
    matrix: np.ndarray,
    min_eigenvalue: float = 1e-6,
) -> np.ndarray:
    """Projette une matrice symétrique sur le cône des matrices de corrélation SDP."""
    matrix = 0.5 * (matrix + matrix.T)

    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, min_eigenvalue)
    psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    d = np.sqrt(np.diag(psd))
    return psd / np.outer(d, d)


def ensure_valid_correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Retourne la matrice si SDP, sinon sa projection sur le cône SDP."""
    if is_positive_definite(matrix):
        return matrix
    projected = project_to_nearest_correlation(matrix)
    if not is_positive_definite(projected):
        raise RuntimeError(
            "Échec de la projection SDP : la matrice projetée reste invalide."
        )
    return projected


BASE_CORRELATION_MATRIX: np.ndarray = ensure_valid_correlation_matrix(
    build_correlation_matrix(COPULA_VARIABLES, CORRELATION_PAIRS)
)


COPULA_INDEX: dict[str, int] = {
    name: i for i, name in enumerate(COPULA_VARIABLES)
}
