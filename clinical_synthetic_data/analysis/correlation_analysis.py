"""
Analyse de corrélation entre variables cliniques (rapport section 6).

Calcule deux types de corrélations :
    - **Pearson** : capture les relations linéaires (référence classique).
    - **Spearman** : capture les relations monotones, robuste aux outliers
      et aux distributions non-normales (cas de TG, log-normal).

Le rapport (section 4.2) indique que la copule encode des corrélations
physiologiques connues ; ces fonctions permettent de **vérifier** que les
corrélations empiriques du jeu généré correspondent à celles attendues, et
de **comparer** les corrélations apprises par CTGAN à celles imposées par
la copule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config.correlations import BASE_CORRELATION_MATRIX, COPULA_VARIABLES
from ..core.patient_schema import CONTINUOUS_FIELDS


# ---------------------------------------------------------------------------
# Matrices de corrélation
# ---------------------------------------------------------------------------


def pearson_correlation_matrix(
    df: pd.DataFrame,
    variables: tuple[str, ...] = CONTINUOUS_FIELDS,
) -> pd.DataFrame:
    """Matrice de corrélation de Pearson sur les variables continues."""
    cols = [v for v in variables if v in df.columns]
    return df[cols].corr(method="pearson")


def spearman_correlation_matrix(
    df: pd.DataFrame,
    variables: tuple[str, ...] = CONTINUOUS_FIELDS,
) -> pd.DataFrame:
    """Matrice de corrélation de Spearman sur les variables continues."""
    cols = [v for v in variables if v in df.columns]
    return df[cols].corr(method="spearman")


def correlation_matrix_by_class(
    df: pd.DataFrame,
    method: str = "pearson",
    variables: tuple[str, ...] = CONTINUOUS_FIELDS,
) -> dict[str, pd.DataFrame]:
    """
    Matrice de corrélation par classe.

    Retourne un dict {class_label: DataFrame de corrélations}. Utile pour
    voir si certaines corrélations s'expriment plus fortement dans une
    classe (ex : glucose ↔ HbA1c dans la classe diabète).
    """
    cols = [v for v in variables if v in df.columns]
    return {
        cls: g[cols].corr(method=method)
        for cls, g in df.groupby("class_label")
    }


# ---------------------------------------------------------------------------
# Comparaison entre matrices
# ---------------------------------------------------------------------------


def correlation_difference(
    matrix_a: pd.DataFrame,
    matrix_b: pd.DataFrame,
) -> pd.DataFrame:
    """
    Différence terme à terme entre deux matrices de corrélation (A − B).

    Utile pour visualiser où CTGAN diverge de la copule.
    """
    common = matrix_a.columns.intersection(matrix_b.columns)
    return matrix_a.loc[common, common] - matrix_b.loc[common, common]


def frobenius_distance(
    matrix_a: pd.DataFrame,
    matrix_b: pd.DataFrame,
) -> float:
    """
    Distance de Frobenius entre deux matrices : ||A − B||_F.

    Métrique globale unique pour résumer la similarité de deux structures
    de dépendance. Plus c'est proche de 0, plus les deux méthodes ont
    appris la même structure de corrélation.
    """
    diff = correlation_difference(matrix_a, matrix_b)
    return float(np.linalg.norm(diff.values, ord="fro"))


# ---------------------------------------------------------------------------
# Comparaison à la matrice cible de la copule
# ---------------------------------------------------------------------------


def empirical_vs_target_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare les corrélations empiriques du dataset à la matrice cible
    `BASE_CORRELATION_MATRIX` imposée à la copule.

    Retourne un DataFrame indexé par paire (var1, var2) avec colonnes :
        target  : valeur ρ imposée par la copule
        empirical : valeur ρ effectivement observée
        delta : empirical - target
    """
    target = pd.DataFrame(
        BASE_CORRELATION_MATRIX,
        index=COPULA_VARIABLES,
        columns=COPULA_VARIABLES,
    )
    empirical = pearson_correlation_matrix(df, variables=COPULA_VARIABLES)

    rows = []
    for i, var1 in enumerate(COPULA_VARIABLES):
        for j, var2 in enumerate(COPULA_VARIABLES):
            if i < j:   # paires uniques (triangle supérieur strict)
                rows.append({
                    "var1": var1,
                    "var2": var2,
                    "target": float(target.loc[var1, var2]),
                    "empirical": float(empirical.loc[var1, var2]),
                    "delta": float(empirical.loc[var1, var2] - target.loc[var1, var2]),
                })

    result = pd.DataFrame(rows)
    # Tri par |delta| décroissant pour mettre en évidence les paires
    # où l'empirique diverge le plus de la cible.
    return result.reindex(result["delta"].abs().sort_values(ascending=False).index)
