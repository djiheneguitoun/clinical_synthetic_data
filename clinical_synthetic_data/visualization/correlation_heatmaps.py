"""Heatmaps de matrices de corrélation."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Matrice de corrélation (Pearson)",
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    annot: bool = True,
) -> plt.Figure:
    """Heatmap d'une matrice de corrélation, échelle bipolaire centrée sur 0."""
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        correlation_matrix,
        cmap=cmap, vmin=vmin, vmax=vmax, center=0,
        annot=annot, fmt=".2f", annot_kws={"fontsize": 7},
        square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.7, "label": "ρ"},
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_correlation_comparison(
    matrix_a: pd.DataFrame,
    matrix_b: pd.DataFrame,
    label_a: str = "Copule",
    label_b: str = "CTGAN",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Comparaison côte-à-côte de deux matrices, plus la différence (A − B)."""
    common = matrix_a.columns.intersection(matrix_b.columns)
    m_a = matrix_a.loc[common, common]
    m_b = matrix_b.loc[common, common]
    diff = m_a - m_b

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    sns.heatmap(m_a, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", annot_kws={"fontsize": 6},
                square=True, linewidths=0.4, ax=axes[0],
                cbar_kws={"shrink": 0.6})
    axes[0].set_title(label_a, fontsize=11, fontweight="bold")

    sns.heatmap(m_b, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", annot_kws={"fontsize": 6},
                square=True, linewidths=0.4, ax=axes[1],
                cbar_kws={"shrink": 0.6})
    axes[1].set_title(label_b, fontsize=11, fontweight="bold")

    sns.heatmap(diff, cmap="RdBu_r", vmin=-0.3, vmax=0.3, center=0,
                annot=True, fmt=".2f", annot_kws={"fontsize": 6},
                square=True, linewidths=0.4, ax=axes[2],
                cbar_kws={"shrink": 0.6, "label": "Δρ"})
    axes[2].set_title(f"Différence ({label_a} − {label_b})",
                      fontsize=11, fontweight="bold")

    for ax in axes:
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelrotation=0, labelsize=7)

    fig.suptitle("Comparaison des structures de corrélation",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig
