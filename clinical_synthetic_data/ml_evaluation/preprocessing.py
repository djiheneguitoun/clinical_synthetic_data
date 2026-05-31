"""
Pré-traitement des données pour l'évaluation ML (rapport section 8).

Transforme un DataFrame patient en :
    - matrice de features (X) : continues standardisées + catégorielles
      encodées
    - vecteur cible (y) : classe à prédire

Encodage par type, conformément à la typologie de la section 2 du rapport :
    - continues       → StandardScaler (centrage + réduction)
    - nominales       → OneHotEncoder (sexe, statut tabagique)
    - ordinales       → OrdinalEncoder avec ordre cliniquement significatif
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from ..core.patient_schema import CONTINUOUS_FIELDS


# ---------------------------------------------------------------------------
# Catégorisation des features (rapport 2.3)
# ---------------------------------------------------------------------------

# Variables catégorielles nominales : encodage one-hot
NOMINAL_FIELDS: tuple[str, ...] = ("sex", "smoking")

# Variables ordinales : encodage par rang croissant cliniquement significatif
ORDINAL_FIELDS: tuple[str, ...] = (
    "physical_activity", "alcohol", "diet_quality",
)
ORDINAL_CATEGORIES: dict[str, list[str]] = {
    "physical_activity": ["Sedentary", "Moderate", "High"],
    "alcohol":           ["None", "Moderate", "Excessive"],
    "diet_quality":      ["Poor", "Average", "Good"],
}

# Colonnes à exclure des features (non explicatives)
EXCLUDED_FROM_FEATURES: tuple[str, ...] = ("patient_id", "class_label")


# ---------------------------------------------------------------------------
# Construction du pipeline
# ---------------------------------------------------------------------------


def build_preprocessor() -> ColumnTransformer:
    """
    Pipeline de prétraitement sklearn (ColumnTransformer).

    À utiliser dans un Pipeline avec le classifieur en aval pour éviter
    les fuites entre train et test :
        Pipeline([("preprocessor", build_preprocessor()),
                  ("classifier", ...)])
    """
    return ColumnTransformer(
        transformers=[
            ("continuous", StandardScaler(), list(CONTINUOUS_FIELDS)),
            (
                "nominal",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                list(NOMINAL_FIELDS),
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[ORDINAL_CATEGORIES[v] for v in ORDINAL_FIELDS],
                    handle_unknown="use_encoded_value", unknown_value=-1,
                ),
                list(ORDINAL_FIELDS),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Sépare un DataFrame patient en features X et cible y.

    Retourne
    --------
    X : pd.DataFrame
        Features brutes (continues + catégorielles, non encodées).
        L'encodage est fait par `build_preprocessor()` dans le Pipeline.
    y : pd.Series
        Étiquette de classe (chaîne, à passer telle quelle aux classifieurs
        sklearn qui gèrent les chaînes nativement).
    """
    feature_cols = (
        list(CONTINUOUS_FIELDS) + list(NOMINAL_FIELDS) + list(ORDINAL_FIELDS)
    )
    available = [c for c in feature_cols if c in df.columns]

    X = df[available].copy()
    y = df["class_label"].copy()
    return X, y
