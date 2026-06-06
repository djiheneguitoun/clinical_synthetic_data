"""Nuages de points colorés par classe."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .distributions import CLASS_COLORS


KEY_SCATTER_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("fasting_glucose", "hba1c", "Glycémie vs HbA1c (diabète)"),
    ("ldl", "hdl", "LDL vs HDL (dyslipidémie)"),
    ("sbp", "dbp", "PAS vs PAD (hypertension)"),
    ("bmi", "fasting_glucose", "IMC vs glycémie (axe cardio-métabolique)"),
    ("age", "sbp", "Âge vs PAS (rigidité artérielle)"),
    ("bmi", "triglycerides", "IMC vs triglycérides (lipides/adiposité)"),
)


def plot_scatter_pairs(
    df: pd.DataFrame,
    pairs: tuple[tuple[str, str, str], ...] = KEY_SCATTER_PAIRS,
    output_path: Optional[str] = None,
    title_suffix: str = "",
    alpha: float = 0.45,
    point_size: int = 12,
) -> plt.Figure:
    """Grille de nuages de points colorés par classe."""
    n_pairs = len(pairs)
    n_cols = 3
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    samples = []
    rng = np.random.default_rng(seed=0)
    for cls in CLASS_COLORS:
        subset = df[df["class_label"] == cls]
        if len(subset) > 250:
            idx = rng.choice(subset.index, size=250, replace=False)
            subset = subset.loc[idx]
        samples.append(subset)
    df_plot = pd.concat(samples)

    for ax, (var_x, var_y, title) in zip(axes, pairs):
        if var_x not in df.columns or var_y not in df.columns:
            ax.set_visible(False)
            continue
        for cls, color in CLASS_COLORS.items():
            subset = df_plot[df_plot["class_label"] == cls]
            ax.scatter(
                subset[var_x], subset[var_y],
                c=color, s=point_size, alpha=alpha,
                label=cls, edgecolors="none",
            )
        ax.set_xlabel(var_x, fontsize=9)
        ax.set_ylabel(var_y, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)

    for ax in axes[n_pairs:]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=8, alpha=0.6, label=cls)
        for cls, c in CLASS_COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Nuages de points par classe{title_suffix}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig
