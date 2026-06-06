"""Évaluation des modèles ML (rapport section 8)."""

from __future__ import annotations

import logging
from typing import Callable, Optional

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


_LOG = logging.getLogger("pipeline")

_MODEL_LABELS = {
    "logistic_regression": "Régression logistique",
    "random_forest": "Forêt aléatoire",
    "mlp": "Réseau de neurones (MLP)",
}

_STAGE_LABELS = {
    "split": "Split unique (80/20)",
    "cv": "Validation croisée 5-fold",
    "cross_main": "Test croisé — entraîné sur Méthode 1",
    "cross_other": "Test croisé — entraîné sur Méthode 2",
}


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Entraîne `model` et l'évalue, retourne un dict de métriques."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = list(model.classes_)

    try:
        auc = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro",
            labels=classes,
        )
    except ValueError:
        auc = float("nan")

    f1_per_class = f1_score(y_test, y_pred, average=None, labels=classes)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    return {
        "accuracy":            float(accuracy_score(y_test, y_pred)),
        "f1_macro":            float(f1_score(y_test, y_pred, average="macro")),
        "f1_per_class":        {cls: float(s) for cls, s in zip(classes, f1_per_class)},
        "roc_auc_ovr_macro":   float(auc) if auc == auc else None,
        "confusion_matrix":    cm.tolist(),
        "classes":             classes,
    }


def evaluate_all_models_on_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    seed: int = 42,
    on_model: Optional[Callable[[str], None]] = None,
) -> dict[str, dict]:
    """Entraîne et évalue les 3 modèles sur un split stratifié unique."""
    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed,
    )

    results: dict[str, dict] = {}
    for name, factory in MODEL_FACTORIES.items():
        if on_model is not None:
            on_model(name)
        model = factory(seed=seed)
        results[name] = evaluate_model(model, X_train, y_train, X_test, y_test)
    return results


def cross_validate_models(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    scoring: str = "f1_macro",
    on_model: Optional[Callable[[str], None]] = None,
) -> dict[str, dict]:
    """Validation croisée k-fold stratifiée pour chaque modèle."""
    X, y = prepare_features_and_target(df)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    results: dict[str, dict] = {}
    for name, factory in MODEL_FACTORIES.items():
        if on_model is not None:
            on_model(name)
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


def cross_method_evaluation(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    seed: int = 42,
    on_model: Optional[Callable[[str], None]] = None,
) -> dict[str, dict]:
    """Entraîne sur `df_train`, teste sur `df_test` (transférabilité)."""
    X_train, y_train = prepare_features_and_target(df_train)
    X_test, y_test = prepare_features_and_target(df_test)

    results: dict[str, dict] = {}
    for name, factory in MODEL_FACTORIES.items():
        if on_model is not None:
            on_model(name)
        model = factory(seed=seed)
        results[name] = evaluate_model(
            model, X_train, y_train, X_test, y_test,
        )
    return results


def build_ml_evaluation_report(
    df: pd.DataFrame,
    df_other: Optional[pd.DataFrame] = None,
    seed: int = 42,
    progress: Optional[Callable[[float], None]] = None,
) -> dict:
    """Rapport ML complet, sérialisable JSON.

    `progress`, si fourni, est appelé avec une fraction d'avancement (0→1)
    avant l'entraînement de chaque modèle ; chaque étape est aussi journalisée.
    """
    n_models = len(MODEL_FACTORIES)
    total = n_models * (2 + (2 if df_other is not None else 0))
    done = {"n": 0}

    def hook(stage: str, name: str) -> None:
        done["n"] += 1
        _LOG.info(
            f"{_STAGE_LABELS.get(stage, stage)} — "
            f"{_MODEL_LABELS.get(name, name)}… ({done['n']}/{total})"
        )
        if progress is not None:
            progress(min(done["n"] / total, 1.0))

    report: dict = {
        "n_total": int(len(df)),
        "single_split": evaluate_all_models_on_split(
            df, seed=seed, on_model=lambda n: hook("split", n)),
        "cross_validation": cross_validate_models(
            df, seed=seed, on_model=lambda n: hook("cv", n)),
    }
    if df_other is not None:
        report["cross_method"] = {
            "train_on_main_test_on_other": cross_method_evaluation(
                df, df_other, seed=seed, on_model=lambda n: hook("cross_main", n)),
            "train_on_other_test_on_main": cross_method_evaluation(
                df_other, df, seed=seed, on_model=lambda n: hook("cross_other", n)),
        }
    return report
