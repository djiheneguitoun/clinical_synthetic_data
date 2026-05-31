"""Visualisations (rapport sections 7 et 8)."""

from .correlation_heatmaps import (
    plot_correlation_comparison,
    plot_correlation_heatmap,
)
from .distributions import (
    CLASS_COLORS,
    DIAGNOSTIC_THRESHOLDS,
    KEY_VARIABLES,
    plot_boxplots_by_class,
    plot_histograms_by_class,
)
from .ml_visualizations import (
    plot_all_confusion_matrices,
    plot_confusion_matrix,
    plot_cross_validation_results,
    plot_model_metrics_comparison,
)
from .scatter_plots import KEY_SCATTER_PAIRS, plot_scatter_pairs

__all__ = [
    "CLASS_COLORS",
    "DIAGNOSTIC_THRESHOLDS",
    "KEY_VARIABLES",
    "KEY_SCATTER_PAIRS",
    "plot_histograms_by_class",
    "plot_boxplots_by_class",
    "plot_scatter_pairs",
    "plot_correlation_heatmap",
    "plot_correlation_comparison",
    "plot_confusion_matrix",
    "plot_all_confusion_matrices",
    "plot_model_metrics_comparison",
    "plot_cross_validation_results",
]
