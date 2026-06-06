"""Pré-traitement des données pour l'évaluation ML (rapport section 8)."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from ..core.patient_schema import CONTINUOUS_FIELDS


NOMINAL_FIELDS: tuple[str, ...] = ("sex", "smoking")

ORDINAL_FIELDS: tuple[str, ...] = (
    "physical_activity", "alcohol", "diet_quality",
)
ORDINAL_CATEGORIES: dict[str, list[str]] = {
    "physical_activity": ["Sedentary", "Moderate", "High"],
    "alcohol":           ["None", "Moderate", "Excessive"],
    "diet_quality":      ["Poor", "Average", "Good"],
}

EXCLUDED_FROM_FEATURES: tuple[str, ...] = ("patient_id", "class_label")


def build_preprocessor() -> ColumnTransformer:
    """Pipeline de prétraitement sklearn (ColumnTransformer)."""
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
    """Sépare un DataFrame patient en features X et cible y."""
    feature_cols = (
        list(CONTINUOUS_FIELDS) + list(NOMINAL_FIELDS) + list(ORDINAL_FIELDS)
    )
    available = [c for c in feature_cols if c in df.columns]

    X = df[available].copy()
    y = df["class_label"].copy()
    return X, y
