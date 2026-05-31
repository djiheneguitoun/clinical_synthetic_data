"""
Comparaison des deux méthodes de génération (rapport section 6).

Confronte un dataset produit par la copule (Méthode 1) à un dataset produit
par CTGAN (Méthode 2) sur trois axes :
    1. Marginales : moyennes et écarts-types par variable et par classe.
    2. Structure de corrélation : matrices et distance de Frobenius.
    3. Distributions catégorielles : différences de proportions.

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.patient_schema import CATEGORICAL_FIELDS, CONTINUOUS_FIELDS
from .correlation_analysis import (
    frobenius_distance,
    pearson_correlation_matrix,
)


# ---------------------------------------------------------------------------
# Comparaison des marginales
# ---------------------------------------------------------------------------


def compare_continuous_means(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str = "method_A",
    label_b: str = "method_B",
) -> pd.DataFrame:
    """
    Compare les moyennes et écarts-types par classe pour chaque variable
    continue. Retourne un DataFrame indexé par (class_label, variable).
    """
    cols = [c for c in CONTINUOUS_FIELDS if c in df_a.columns and c in df_b.columns]

    means_a = df_a.groupby("class_label")[cols].mean()
    means_b = df_b.groupby("class_label")[cols].mean()
    stds_a = df_a.groupby("class_label")[cols].std()
    stds_b = df_b.groupby("class_label")[cols].std()

    result = pd.DataFrame({
        f"mean_{label_a}": means_a.stack(),
        f"mean_{label_b}": means_b.stack(),
        f"std_{label_a}": stds_a.stack(),
        f"std_{label_b}": stds_b.stack(),
    })
    result["mean_delta"] = result[f"mean_{label_a}"] - result[f"mean_{label_b}"]
    result["mean_relative_delta"] = (
        result["mean_delta"] / result[f"mean_{label_a}"].replace(0, np.nan)
    )
    result.index.names = ["class_label", "variable"]
    return result


# ---------------------------------------------------------------------------
# Comparaison des distributions catégorielles
# ---------------------------------------------------------------------------


def compare_categorical_distributions(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str = "method_A",
    label_b: str = "method_B",
) -> dict[str, pd.DataFrame]:
    """
    Pour chaque variable catégorielle, compare les proportions (par classe)
    entre les deux datasets.

    Retourne un dict {variable: DataFrame avec colonnes pivotées par méthode}.
    """
    cols = [c for c in CATEGORICAL_FIELDS if c in df_a.columns and c in df_b.columns]

    result: dict[str, pd.DataFrame] = {}
    for col in cols:
        props_a = pd.crosstab(df_a["class_label"], df_a[col], normalize="index")
        props_b = pd.crosstab(df_b["class_label"], df_b[col], normalize="index")
        # Stack pour aligner sur (class_label, modalité)
        df = pd.DataFrame({
            label_a: props_a.stack(),
            label_b: props_b.stack(),
        })
        df["delta"] = df[label_a] - df[label_b]
        df.index.names = ["class_label", "modality"]
        result[col] = df
    return result


# ---------------------------------------------------------------------------
# Comparaison de la structure de corrélation
# ---------------------------------------------------------------------------


def compare_correlation_structures(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> dict:
    """
    Compare les matrices de corrélation Pearson des deux datasets.

    Retourne :
        - matrices A et B
        - différence terme à terme
        - distance de Frobenius
        - les 10 paires de variables où la corrélation diffère le plus
    """
    corr_a = pearson_correlation_matrix(df_a)
    corr_b = pearson_correlation_matrix(df_b)
    diff = corr_a - corr_b

    # Top divergences (triangle supérieur strict)
    diff_long_rows = []
    cols = list(diff.columns)
    for i, v1 in enumerate(cols):
        for j, v2 in enumerate(cols):
            if i < j:
                diff_long_rows.append({
                    "var1": v1, "var2": v2,
                    "corr_a": float(corr_a.loc[v1, v2]),
                    "corr_b": float(corr_b.loc[v1, v2]),
                    "delta": float(diff.loc[v1, v2]),
                })
    diff_long = pd.DataFrame(diff_long_rows)
    top_divergences = diff_long.reindex(
        diff_long["delta"].abs().sort_values(ascending=False).index
    ).head(10)

    return {
        "frobenius_distance": frobenius_distance(corr_a, corr_b),
        "correlation_a": corr_a,
        "correlation_b": corr_b,
        "difference": diff,
        "top_divergences": top_divergences,
    }


# ---------------------------------------------------------------------------
# Rapport agrégé
# ---------------------------------------------------------------------------


def build_method_comparison_report(
    df_copula: pd.DataFrame,
    df_ctgan: pd.DataFrame,
) -> dict:
    """
    Synthèse de la comparaison Méthode 1 vs Méthode 2 (rapport section 6).
    """
    corr_comparison = compare_correlation_structures(df_copula, df_ctgan)

    # Aplatissement explicite du DataFrame à MultiIndex pour rester
    # JSON-sérialisable (tuple keys interdites en JSON).
    means_df = compare_continuous_means(
        df_copula, df_ctgan, label_a="copula", label_b="ctgan"
    ).round(3)
    nested_means: dict = {}
    for (cls, var), row in means_df.iterrows():
        nested_means.setdefault(cls, {})[var] = row.to_dict()

    return {
        "n_copula": int(len(df_copula)),
        "n_ctgan": int(len(df_ctgan)),
        "continuous_means": nested_means,
        "correlation_frobenius_distance": corr_comparison["frobenius_distance"],
        "correlation_top_divergences": corr_comparison["top_divergences"].round(3).to_dict(orient="records"),
    }
