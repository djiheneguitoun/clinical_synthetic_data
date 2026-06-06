"""Visualisations de distributions (histogrammes et boxplots)."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import clinical_ranges as cr
from ..core.patient_schema import ClassLabel


CLASS_COLORS: dict[str, str] = {
    ClassLabel.HEALTHY.value:      "#2ca02c",
    ClassLabel.DIABETES.value:     "#d62728",
    ClassLabel.DYSLIPIDEMIA.value: "#ff7f0e",
    ClassLabel.HYPERTENSION.value: "#9467bd",
    ClassLabel.OBESITY.value:      "#8c564b",
    ClassLabel.CV_RISK.value:      "#1f77b4",
}


DIAGNOSTIC_THRESHOLDS: dict[str, list[tuple[float, str]]] = {
    "fasting_glucose": [
        (cr.RF_GLUCOSE, "RF=100"),
        (cr.DX_GLUCOSE, "DX=126"),
    ],
    "hba1c": [
        (cr.RF_HBA1C, "RF=5.7"),
        (cr.DX_HBA1C, "DX=6.5"),
    ],
    "bmi": [
        (cr.RF_BMI, "RF=25"),
        (cr.DX_BMI, "DX=30"),
    ],
    "sbp": [
        (cr.RF_SBP, "RF=120"),
        (cr.DX_SBP, "DX=140"),
    ],
    "dbp": [
        (cr.RF_DBP, "RF=80"),
        (cr.DX_DBP, "DX=90"),
    ],
    "ldl": [
        (cr.RF_LDL, "RF=130"),
        (cr.DX_LDL, "DX=160"),
    ],
    "triglycerides": [
        (cr.RF_TRIGLYCERIDES, "RF=150"),
        (cr.DX_TRIGLYCERIDES, "DX=200"),
    ],
    "total_chol": [
        (cr.RF_TOTAL_CHOL, "RF=200"),
        (cr.DX_TOTAL_CHOL, "DX=240"),
    ],
    "heart_rate": [
        (cr.RF_HEART_RATE, "RF=80"),
    ],
}


KEY_VARIABLES: tuple[str, ...] = (
    "age", "bmi", "sbp", "dbp",
    "fasting_glucose", "hba1c",
    "ldl", "hdl", "triglycerides", "total_chol",
    "heart_rate", "weight",
)


def plot_histograms_by_class(
    df: pd.DataFrame,
    variables: tuple[str, ...] = KEY_VARIABLES,
    output_path: Optional[str] = None,
    title_suffix: str = "",
    bins: int = 30,
) -> plt.Figure:
    """Grille d'histogrammes par variable, courbes superposées par classe."""
    variables = tuple(v for v in variables if v in df.columns)
    n_vars = len(variables)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows))
    axes = np.array(axes).flatten()

    for ax, var in zip(axes, variables):
        for cls_name, color in CLASS_COLORS.items():
            subset = df[df["class_label"] == cls_name][var]
            if len(subset) == 0:
                continue
            ax.hist(
                subset, bins=bins, density=True, alpha=0.35,
                color=color, label=cls_name, edgecolor="none",
            )

        for threshold, label in DIAGNOSTIC_THRESHOLDS.get(var, []):
            linestyle = "--" if label.startswith("RF") else "-."
            color = "gray" if label.startswith("RF") else "red"
            ax.axvline(threshold, color=color, linestyle=linestyle,
                       linewidth=1, alpha=0.7)
            ax.text(threshold, ax.get_ylim()[1] * 0.95, label,
                    rotation=90, fontsize=7, color=color,
                    verticalalignment="top")

        unit = cr.CONTINUOUS_VARIABLES[var].unit if var in cr.CONTINUOUS_VARIABLES else ""
        display = cr.CONTINUOUS_VARIABLES[var].display_name if var in cr.CONTINUOUS_VARIABLES else var
        ax.set_xlabel(f"{display} ({unit})", fontsize=9)
        ax.set_ylabel("Densité", fontsize=9)
        ax.tick_params(labelsize=8)

    for ax in axes[n_vars:]:
        ax.set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6)
        for c in CLASS_COLORS.values()
    ]
    fig.legend(handles, list(CLASS_COLORS), loc="lower center",
               ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Distribution des variables cliniques par classe{title_suffix}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_boxplots_by_class(
    df: pd.DataFrame,
    variables: tuple[str, ...] = KEY_VARIABLES,
    output_path: Optional[str] = None,
    title_suffix: str = "",
) -> plt.Figure:
    """Grille de boxplots : une cellule par variable, classes en abscisse."""
    variables = tuple(v for v in variables if v in df.columns)
    n_vars = len(variables)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows))
    axes = np.array(axes).flatten()

    class_order = list(CLASS_COLORS)
    palette = [CLASS_COLORS[c] for c in class_order]

    for ax, var in zip(axes, variables):
        sns.boxplot(
            data=df, x="class_label", y=var,
            order=class_order, palette=palette, ax=ax,
            showfliers=True, fliersize=2, linewidth=0.8,
        )
        for threshold, label in DIAGNOSTIC_THRESHOLDS.get(var, []):
            color = "gray" if label.startswith("RF") else "red"
            ax.axhline(threshold, color=color, linestyle="--",
                       linewidth=0.8, alpha=0.7)

        unit = cr.CONTINUOUS_VARIABLES[var].unit if var in cr.CONTINUOUS_VARIABLES else ""
        display = cr.CONTINUOUS_VARIABLES[var].display_name if var in cr.CONTINUOUS_VARIABLES else var
        ax.set_ylabel(f"{display} ({unit})", fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=35, labelsize=7)
        ax.tick_params(axis="y", labelsize=8)

    for ax in axes[n_vars:]:
        ax.set_visible(False)

    fig.suptitle(f"Boxplots des variables cliniques par classe{title_suffix}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig
