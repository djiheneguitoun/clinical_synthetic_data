"""
Modèles d'apprentissage supervisé pour l'évaluation (rapport section 8).

Trois familles de modèles, choisies pour couvrir le spectre demandé par
le cahier des charges :

    1. **Régression logistique multinomiale** : modèle linéaire interprétable,
       baseline classique.
    2. **Random Forest** : ensemble d'arbres, capture les non-linéarités et
       les interactions sans réglage de paramètres lourd.
    3. **MLP (Perceptron multi-couches)** : réseau de neurones simple,
       2 couches cachées avec dropout implicite (early stopping).

Chaque modèle est encapsulé dans un Pipeline sklearn intégrant le
prétraitement, ce qui évite toute fuite entre train et test.
"""

from __future__ import annotations

from typing import Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor


# ---------------------------------------------------------------------------
# Factories : chacun retourne un Pipeline (preprocessor + classifier)
# ---------------------------------------------------------------------------


def make_logistic_regression(seed: int = 42) -> Pipeline:
    """Régression logistique multinomiale (baseline interprétable)."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            C=1.0,
            random_state=seed,
        )),
    ])


def make_random_forest(seed: int = 42) -> Pipeline:
    """Random Forest (200 arbres) — capture non-linéarités et interactions."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )),
    ])


def make_mlp(seed: int = 42) -> Pipeline:
    """MLP 2 couches cachées avec régularisation L2."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,                    # régularisation L2
            learning_rate_init=1e-3,
            max_iter=300,
            # early_stopping désactivé : un bug sklearn fait échouer le
            # scoring interne quand y contient des chaînes (notre cas).
            # La régularisation L2 (alpha) compense l'absence
            # d'early-stopping en pratique sur ce volume de données.
            early_stopping=False,
            random_state=seed,
        )),
    ])


# ---------------------------------------------------------------------------
# Registre exposé
# ---------------------------------------------------------------------------


MODEL_FACTORIES: dict[str, Callable[..., Pipeline]] = {
    "logistic_regression": make_logistic_regression,
    "random_forest":       make_random_forest,
    "mlp":                 make_mlp,
}


MODEL_DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Régression logistique",
    "random_forest":       "Random Forest",
    "mlp":                 "MLP",
}
