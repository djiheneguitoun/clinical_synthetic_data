"""
Statistiques descriptives du jeu de données (rapport section 6).

Pour chaque jeu généré, calcule :
    - moyenne, médiane, écart-type, IQR par classe et globalement (variables
      continues) ;
    - fréquences (counts et proportions) par classe (variables catégorielles).

"""

from __future__ import annotations

import pandas as pd

from ..core.patient_schema import CATEGORICAL_FIELDS, CONTINUOUS_FIELDS


# ---------------------------------------------------------------------------
# Statistiques sur les variables continues
# ---------------------------------------------------------------------------


def describe_continuous_global(df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistiques descriptives globales (toutes classes confondues) pour les
    variables continues.

    Retourne un DataFrame indexé par variable, colonnes :
        count, mean, std, min, q25, median, q75, max, iqr
    """
    cols = [c for c in CONTINUOUS_FIELDS if c in df.columns]
    desc = df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc = desc.rename(columns={
        "50%": "median", "25%": "q25", "75%": "q75",
    })
    desc["iqr"] = desc["q75"] - desc["q25"]
    return desc[["count", "mean", "std", "min", "q25", "median", "q75", "max", "iqr"]]


def describe_continuous_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistiques descriptives stratifiées par classe.

    Retourne un DataFrame avec MultiIndex (class_label, variable) et
    colonnes : count, mean, std, min, median, max, iqr.
    """
    cols = [c for c in CONTINUOUS_FIELDS if c in df.columns]
    grouped = df.groupby("class_label")[cols]

    pieces = []
    for stat_name, stat_fn in (
        ("count", "count"),
        ("mean", "mean"),
        ("std", "std"),
        ("min", "min"),
        ("median", "median"),
        ("max", "max"),
    ):
        pieces.append(grouped.agg(stat_fn).stack().rename(stat_name))

    # IQR via groupby + quantile
    q25 = grouped.quantile(0.25).stack().rename("q25")
    q75 = grouped.quantile(0.75).stack().rename("q75")
    iqr = (q75 - q25).rename("iqr")

    result = pd.concat([*pieces, q25, q75, iqr], axis=1)
    result.index.names = ["class_label", "variable"]
    return result


# ---------------------------------------------------------------------------
# Statistiques sur les variables catégorielles
# ---------------------------------------------------------------------------


def describe_categorical_global(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Proportions globales pour chaque variable catégorielle.

    Retourne un dict {variable: Series(modalité -> proportion)}.
    """
    cols = [c for c in CATEGORICAL_FIELDS if c in df.columns]
    return {
        col: df[col].value_counts(normalize=True).sort_index()
        for col in cols
    }


def describe_categorical_by_class(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Proportions stratifiées par classe.

    Retourne un dict {variable: DataFrame(class_label × modalité, valeurs en
    proportions ligne par ligne)}.
    """
    cols = [c for c in CATEGORICAL_FIELDS if c in df.columns]
    return {
        col: pd.crosstab(df["class_label"], df[col], normalize="index")
        for col in cols
    }


# ---------------------------------------------------------------------------
# Rapport agrégé
# ---------------------------------------------------------------------------


def build_descriptive_report(df: pd.DataFrame) -> dict:
    """
    Construit un rapport descriptif structuré, sérialisable en JSON.

    Utile pour la production automatique du tableau « Caractéristiques de
    l'échantillon » dans le rapport scientifique final.
    """
    # Aplatissement explicite : les MultiIndex doivent être convertis en
    # dicts imbriqués pour rester JSON-sérialisables.
    by_class_continuous = describe_continuous_by_class(df).round(2)
    nested_continuous: dict = {}
    for (cls, var), row in by_class_continuous.iterrows():
        nested_continuous.setdefault(cls, {})[var] = row.to_dict()

    nested_categorical: dict = {}
    for var, df_var in describe_categorical_by_class(df).items():
        nested_categorical[var] = {
            cls: row.round(3).to_dict() for cls, row in df_var.iterrows()
        }

    return {
        "n_total": int(len(df)),
        "n_by_class": df["class_label"].value_counts().to_dict(),
        "continuous_global": describe_continuous_global(df).round(2).to_dict(orient="index"),
        "continuous_by_class": nested_continuous,
        "categorical_global": {
            var: s.round(3).to_dict()
            for var, s in describe_categorical_global(df).items()
        },
        "categorical_by_class": nested_categorical,
    }
