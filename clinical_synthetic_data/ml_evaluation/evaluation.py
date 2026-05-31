"""
Évaluation des modèles ML (rapport section 8).

Trois protocoles :
    1. **Split unique** : train/test stratifié, métriques sur le test set.
    2. **Validation croisée** : 5-fold stratifié pour estimer la variance.
    3. **Évaluation croisée entre méthodes** : train sur Méthode 1, test sur
       Méthode 2 (et inversement). Mesure la transférabilité de la
       classification.

Métriques rapportées :
    - accuracy (score brut)
    - F1 macro (donne le même poids à chaque classe ; pertinent en
      classification équilibrée)
    - F1 par classe (mise en évidence des classes les plus difficiles)
    - ROC AUC one-vs-rest macro (utilise les probabilités prédites)
    - matrice de confusion (interprétation détaillée)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from .models import MODEL_FACTORIES
from .preprocessing import prepare_features_and_target


# ---------------------------------------------------------------------------
# Évaluation d'un modèle unique
# ---------------------------------------------------------------------------


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Entraîne `model` sur (X_train, y_train) et l'évalue sur (X_test, y_test).

    Retourne un dict de métriques sérialisable JSON.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = list(model.classes_)

    # ROC AUC OvR — nécessite que toutes les classes soient présentes
    try:
        auc = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro",
            labels=classes,
        )
    except ValueError:
        # Une classe absente du test set : on rapporte NaN plutôt que
        # de faire échouer l'évaluation.
        auc = float("nan")

    f1_per_class = f1_score(y_test, y_pred, average=None, labels=classes)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    return {
        "accuracy":            float(accuracy_score(y_test, y_pred)),
        "f1_macro":            float(f1_score(y_test, y_pred, average="macro")),
        "f1_per_class":        {cls: float(s) for cls, s in zip(classes, f1_per_class)},
        "roc_auc_ovr_macro":   float(auc) if auc == auc else None,  # NaN-safe
        "confusion_matrix":    cm.tolist(),
        "classes":             classes,
    }


# ---------------------------------------------------------------------------
# Protocole 1 : split train/test unique
# ---------------------------------------------------------------------------


def evaluate_all_models_on_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Entraîne et évalue les 3 modèles sur un split stratifié unique.

    Retourne {model_name: métriques}.
    """
    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed,
    )

    results: dict[str, dict] = {}
    for name, factory in MODEL_FACTORIES.items():
        model = factory(seed=seed)
        results[name] = evaluate_model(model, X_train, y_train, X_test, y_test)
    return results


# ---------------------------------------------------------------------------
# Protocole 2 : validation croisée stratifiée
# ---------------------------------------------------------------------------


def cross_validate_models(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    scoring: str = "f1_macro",
) -> dict[str, dict]:
    """
    Validation croisée k-fold stratifiée pour chaque modèle.

    Estime moyenne et écart-type du score sur k folds. Permet de vérifier
    que les performances ne dépendent pas d'un split favorable.
    """
    X, y = prepare_features_and_target(df)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    results: dict[str, dict] = {}
    for name, factory in MODEL_FACTORIES.items():
        model = factory(seed=seed)
        scores = cross_val_score(
            model, X, y, cv=cv, scoring=scoring, n_jobs=-1,
        )
        results[name] = {
            "scoring":          scoring,
            "mean":             float(scores.mean()),
            "std":              float(scores.std()),
            "fold_scores":      scores.tolist(),
            "n_splits":         n_splits,
        }
    return results


# ---------------------------------------------------------------------------
# Protocole 3 : évaluation croisée entre méthodes
# ---------------------------------------------------------------------------


def cross_method_evaluation(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Entraîne sur `df_train` (typiquement Méthode 1), teste sur `df_test`
    (typiquement Méthode 2).

    Mesure la **transférabilité** : un modèle entraîné sur les données de
    la copule peut-il classifier correctement les données du CTGAN ?
    """
    X_train, y_train = prepare_features_and_target(df_train)
    X_test, y_test = prepare_features_and_target(df_test)

    results: dict[str, dict] = {}
    for name, factory in MODEL_FACTORIES.items():
        model = factory(seed=seed)
        results[name] = evaluate_model(
            model, X_train, y_train, X_test, y_test,
        )
    return results


# ---------------------------------------------------------------------------
# Rapport agrégé
# ---------------------------------------------------------------------------


def build_ml_evaluation_report(
    df: pd.DataFrame,
    df_other: Optional[pd.DataFrame] = None,
    seed: int = 42,
) -> dict:
    """
    Rapport ML complet, sérialisable JSON.

    Toujours inclut : split unique + validation croisée.
    Si `df_other` est fourni : ajoute l'évaluation croisée entre méthodes.
    """
    report: dict = {
        "n_total":           int(len(df)),
        "single_split":      evaluate_all_models_on_split(df, seed=seed),
        "cross_validation":  cross_validate_models(df, seed=seed),
    }
    if df_other is not None:
        report["cross_method"] = {
            "train_on_main_test_on_other": cross_method_evaluation(df, df_other, seed=seed),
            "train_on_other_test_on_main": cross_method_evaluation(df_other, df, seed=seed),
        }
    return report
