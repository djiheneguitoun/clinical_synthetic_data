"""Analyses statistiques du jeu généré (rapport section 6)."""

from .correlation_analysis import (
    correlation_difference,
    correlation_matrix_by_class,
    empirical_vs_target_correlation,
    frobenius_distance,
    pearson_correlation_matrix,
    spearman_correlation_matrix,
)
from .descriptive_stats import (
    build_descriptive_report,
    describe_categorical_by_class,
    describe_categorical_global,
    describe_continuous_by_class,
    describe_continuous_global,
)
from .epidemiological_validation import (
    build_epidemiological_report,
    check_age_ordering,
    validate_all_patterns,
)
from .method_comparison import (
    build_method_comparison_report,
    compare_categorical_distributions,
    compare_continuous_means,
    compare_correlation_structures,
)

__all__ = [
    "describe_continuous_global",
    "describe_continuous_by_class",
    "describe_categorical_global",
    "describe_categorical_by_class",
    "build_descriptive_report",
    "pearson_correlation_matrix",
    "spearman_correlation_matrix",
    "correlation_matrix_by_class",
    "correlation_difference",
    "frobenius_distance",
    "empirical_vs_target_correlation",
    "compare_continuous_means",
    "compare_categorical_distributions",
    "compare_correlation_structures",
    "build_method_comparison_report",
    "validate_all_patterns",
    "check_age_ordering",
    "build_epidemiological_report",
]
