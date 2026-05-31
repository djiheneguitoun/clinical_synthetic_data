"""
Visualisations des résultats ML (rapport section 8).

- Matrices de confusion (une par modèle).
- Comparaison des métriques entre modèles (bar chart).
- Comparaison croisée des méthodes de génération sur la qualité de
  classification.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..ml_evaluation.models import MODEL_DISPLAY_NAMES


# ---------------------------------------------------------------------------
# Matrice de confusion
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    ax: Optional[plt.Axes] = None,
    normalize: bool = True,
    title: str = "",
) -> plt.Axes:
    """Heatmap d'une matrice de confusion normalisée par ligne."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    cm = np.array(cm)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(
            cm, row_sums, where=row_sums != 0,
            out=np.zeros_like(cm, dtype=float),
        )
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    sns.heatmap(
        cm_display, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        ax=ax, square=True, cbar=False, vmin=0, vmax=1,
        annot_kws={"fontsize": 9},
    )
    ax.set_xlabel("Prédit", fontsize=10)
    ax.set_ylabel("Vrai",   fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0,  labelsize=8)
    return ax


def plot_all_confusion_matrices(
    results: dict,
    output_path: Optional[str] = None,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Grille de matrices de confusion (une par modèle), normalisées par ligne.
    """
    models = list(results.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, models):
        cm = results[name]["confusion_matrix"]
        classes = results[name]["classes"]
        display = MODEL_DISPLAY_NAMES.get(name, name)
        plot_confusion_matrix(cm, classes, ax=ax, normalize=True,
                              title=display)
    fig.suptitle(f"Matrices de confusion (normalisées){title_suffix}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Comparaison des modèles
# ---------------------------------------------------------------------------


def plot_model_metrics_comparison(
    results: dict,
    metrics: tuple[str, ...] = ("accuracy", "f1_macro", "roc_auc_ovr_macro"),
    output_path: Optional[str] = None,
    title: str = "Comparaison des modèles ML",
) -> plt.Figure:
    """
    Bar chart : pour chaque métrique (axe X), barres juxtaposées par modèle.
    """
    models = list(results.keys())
    display_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(models)))

    for i, (model_name, color) in enumerate(zip(models, colors)):
        values = [
            results[model_name].get(m, np.nan) or 0.0
            for m in metrics
        ]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width,
                      label=display_names[i], color=color, edgecolor="black", linewidth=0.5)
        # Annotations
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_cross_validation_results(
    cv_results: dict,
    output_path: Optional[str] = None,
    title: str = "Validation croisée 5-fold (F1 macro)",
) -> plt.Figure:
    """
    Bar chart avec barres d'erreur : moyenne ± écart-type sur les folds.
    """
    models = list(cv_results.keys())
    means = [cv_results[m]["mean"] for m in models]
    stds = [cv_results[m]["std"] for m in models]
    display_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(models)))
    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors,
                  edgecolor="black", linewidth=0.5)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.02,
                f"{mean:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("F1 macro", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig
