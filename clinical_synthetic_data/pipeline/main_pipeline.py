"""Pipeline end-to-end de génération et évaluation de données cliniques."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..analysis import (
    build_descriptive_report,
    build_epidemiological_report,
    build_method_comparison_report,
    empirical_vs_target_correlation,
    pearson_correlation_matrix,
)
from ..generators import CTGANGenerator, GaussianCopulaGenerator
from ..ml_evaluation import build_ml_evaluation_report
from ..visualization import (
    plot_all_confusion_matrices,
    plot_boxplots_by_class,
    plot_correlation_comparison,
    plot_correlation_heatmap,
    plot_cross_validation_results,
    plot_histograms_by_class,
    plot_model_metrics_comparison,
    plot_scatter_pairs,
)


@dataclass
class PipelineConfig:
    """Configuration centrale du pipeline."""

    m_per_class: int = 1000
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    seed: int = 42
    ctgan_epochs: int = 300
    ctgan_batch_size: int = 500
    ctgan_max_attempts_factor: int = 500
    skip_ctgan: bool = False
    skip_ml: bool = False

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)


class SyntheticDataPipeline:
    """Pipeline orchestrant la génération, l'analyse et l'évaluation."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.logger = logging.getLogger("pipeline")

        self._dirs = {
            "datasets":  self.config.output_dir / "datasets",
            "figures":   self.config.output_dir / "figures",
            "reports":   self.config.output_dir / "reports",
        }

        self.df_copula: Optional[pd.DataFrame] = None
        self.df_ctgan: Optional[pd.DataFrame] = None
        self.copula_generator: Optional[GaussianCopulaGenerator] = None
        self.ctgan_generator: Optional[CTGANGenerator] = None
        self.timings: dict[str, float] = {}

    def run(self) -> dict:
        """Exécute l'ensemble du pipeline et retourne le rapport consolidé."""
        self._setup_directories()
        self._log_header()

        self._step_generate_copula()
        if not self.config.skip_ctgan:
            self._step_generate_ctgan()

        self._step_save_datasets()
        self._step_run_analysis()
        self._step_run_visualizations()

        if not self.config.skip_ml:
            self._step_run_ml_evaluation()

        summary = self._step_save_summary()
        self._log_footer()
        return summary

    def _setup_directories(self) -> None:
        for path in self._dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    def _log_header(self) -> None:
        cfg = self.config
        self.logger.info("=" * 70)
        self.logger.info("PIPELINE — Génération de données cliniques synthétiques")
        self.logger.info("=" * 70)
        self.logger.info(f"M par classe   : {cfg.m_per_class}  →  total {6 * cfg.m_per_class} patients")
        self.logger.info(f"Seed           : {cfg.seed}")
        self.logger.info(f"Output         : {cfg.output_dir.absolute()}")
        self.logger.info(f"CTGAN epochs   : {cfg.ctgan_epochs}"
                         + ("  (sera SKIP)" if cfg.skip_ctgan else ""))
        self.logger.info(f"ML evaluation  : {'SKIP' if cfg.skip_ml else 'activée'}")
        self.logger.info("")

    def _log_footer(self) -> None:
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("Récapitulatif des durées")
        self.logger.info("-" * 70)
        for step, duration in self.timings.items():
            self.logger.info(f"  {step:<40s} {duration:>7.1f}s")
        total = sum(self.timings.values())
        self.logger.info("-" * 70)
        self.logger.info(f"  {'TOTAL':<40s} {total:>7.1f}s")
        self.logger.info("=" * 70)

    def _timed(self, step_name: str):
        """Context manager mesurant le temps écoulé dans self.timings."""

        class _Timer:
            def __init__(self_inner, owner, name):
                self_inner.owner = owner
                self_inner.name = name
                self_inner.t0 = 0.0

            def __enter__(self_inner):
                self_inner.owner.logger.info("")
                self_inner.owner.logger.info(f"▶ {self_inner.name}")
                self_inner.t0 = time.time()
                return self_inner

            def __exit__(self_inner, *args):
                dt = time.time() - self_inner.t0
                self_inner.owner.timings[self_inner.name] = dt
                self_inner.owner.logger.info(f"✓ {self_inner.name} ({dt:.1f}s)")

        return _Timer(self, step_name)

    def _step_generate_copula(self) -> None:
        with self._timed("1. Génération copule (Méthode 1)"):
            self.copula_generator = GaussianCopulaGenerator(seed=self.config.seed)
            self.df_copula = self.copula_generator.generate_balanced_dataset(
                m_per_class=self.config.m_per_class,
            )
            stats = self.copula_generator.stats.to_report()
            self.logger.info(
                f"  → {len(self.df_copula)} patients valides "
                f"(rejet global {stats['global_rejection_rate']:.1%})"
            )

    def _step_generate_ctgan(self) -> None:
        with self._timed("2. CTGAN (Méthode 2) — entraînement + génération"):
            self.ctgan_generator = CTGANGenerator(
                epochs=self.config.ctgan_epochs,
                batch_size=self.config.ctgan_batch_size,
                max_total_attempts_factor=self.config.ctgan_max_attempts_factor,
                seed=self.config.seed,
            )

            self.logger.info(f"  Entraînement CTGAN ({self.config.ctgan_epochs} epochs)…")
            self.logger.info(f"  (une barre de progression s'affiche en temps réel)")
            t0 = time.time()
            self.ctgan_generator.fit(self.df_copula)
            self.logger.info(f"  ✓ Entraînement terminé ({time.time() - t0:.0f}s)")

            self.logger.info(f"  Génération par classe :")
            self.df_ctgan = self.ctgan_generator.generate_balanced_dataset(
                m_per_class=self.config.m_per_class,
            )
            stats = self.ctgan_generator.stats.to_report()
            self.logger.info(
                f"  → {len(self.df_ctgan)} patients valides "
                f"(rejet global {stats['global_rejection_rate']:.1%})"
            )

    def _step_save_datasets(self) -> None:
        with self._timed("3. Sauvegarde des datasets CSV"):
            path_copula = self._dirs["datasets"] / "dataset_copula.csv"
            self.df_copula.to_csv(path_copula, index=False)
            self.logger.info(f"  ✓ {path_copula}")

            if self.df_ctgan is not None:
                path_ctgan = self._dirs["datasets"] / "dataset_ctgan.csv"
                self.df_ctgan.to_csv(path_ctgan, index=False)
                self.logger.info(f"  ✓ {path_ctgan}")

    def _step_run_analysis(self) -> None:
        with self._timed("4. Analyses statistiques"):
            report: dict = {}

            self.logger.info("  Statistiques descriptives (copule)…")
            report["copula"] = {
                "descriptive": build_descriptive_report(self.df_copula),
                "epidemiological": build_epidemiological_report(self.df_copula),
                "empirical_vs_target_correlation": (
                    empirical_vs_target_correlation(self.df_copula)
                    .round(3).to_dict(orient="records")
                ),
                "generation_stats": self.copula_generator.stats.to_report(),
            }

            if self.df_ctgan is not None:
                self.logger.info("  Statistiques descriptives (CTGAN)…")
                report["ctgan"] = {
                    "descriptive": build_descriptive_report(self.df_ctgan),
                    "epidemiological": build_epidemiological_report(self.df_ctgan),
                    "generation_stats": self.ctgan_generator.stats.to_report(),
                }
                self.logger.info("  Comparaison Méthode 1 vs Méthode 2…")
                report["method_comparison"] = build_method_comparison_report(
                    self.df_copula, self.df_ctgan,
                )

            output = self._dirs["reports"] / "analysis_report.json"
            with open(output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"  ✓ {output}")

    def _step_run_visualizations(self) -> None:
        with self._timed("5. Visualisations"):
            fig_dir = self._dirs["figures"]

            self.logger.info("  Distributions et scatter (copule)…")
            self._save_fig(
                plot_histograms_by_class(self.df_copula, title_suffix=" — Copule"),
                fig_dir / "histograms_copula.png",
            )
            self._save_fig(
                plot_boxplots_by_class(self.df_copula, title_suffix=" — Copule"),
                fig_dir / "boxplots_copula.png",
            )
            self._save_fig(
                plot_scatter_pairs(self.df_copula, title_suffix=" — Copule"),
                fig_dir / "scatter_copula.png",
            )

            self.logger.info("  Heatmap de corrélation (copule)…")
            corr_copula = pearson_correlation_matrix(self.df_copula)
            self._save_fig(
                plot_correlation_heatmap(corr_copula,
                                         title="Corrélations — Copule (Pearson)"),
                fig_dir / "heatmap_copula.png",
            )

            if self.df_ctgan is not None:
                self.logger.info("  Distributions et scatter (CTGAN)…")
                self._save_fig(
                    plot_histograms_by_class(self.df_ctgan, title_suffix=" — CTGAN"),
                    fig_dir / "histograms_ctgan.png",
                )
                self._save_fig(
                    plot_scatter_pairs(self.df_ctgan, title_suffix=" — CTGAN"),
                    fig_dir / "scatter_ctgan.png",
                )
                self.logger.info("  Heatmaps de corrélation (CTGAN + comparaison)…")
                corr_ctgan = pearson_correlation_matrix(self.df_ctgan)
                self._save_fig(
                    plot_correlation_heatmap(corr_ctgan,
                                             title="Corrélations — CTGAN (Pearson)"),
                    fig_dir / "heatmap_ctgan.png",
                )
                self._save_fig(
                    plot_correlation_comparison(corr_copula, corr_ctgan),
                    fig_dir / "heatmap_comparison.png",
                )
            self.logger.info(f"  ✓ {fig_dir}/ ({len(list(fig_dir.glob('*.png')))} figures)")

    @staticmethod
    def _save_fig(fig: plt.Figure, path: Path) -> None:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _step_run_ml_evaluation(self) -> None:
        with self._timed("6. Évaluation ML"):
            self.logger.info("  Entraînement et évaluation des 3 modèles…")
            ml_report = build_ml_evaluation_report(
                self.df_copula,
                df_other=self.df_ctgan,
                seed=self.config.seed,
            )

            self.logger.info("  Résultats (split unique sur copule) :")
            for model_name, metrics in ml_report["single_split"].items():
                auc = metrics.get("roc_auc_ovr_macro") or 0
                self.logger.info(
                    f"    {model_name:<25s} "
                    f"acc={metrics['accuracy']:.3f}  "
                    f"f1={metrics['f1_macro']:.3f}  "
                    f"auc={auc:.3f}"
                )

            output = self._dirs["reports"] / "ml_report.json"
            with open(output, "w", encoding="utf-8") as f:
                json.dump(ml_report, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"  ✓ {output}")

            fig_dir = self._dirs["figures"]
            self.logger.info("  Figures ML (confusion, métriques, validation croisée)…")
            single = ml_report["single_split"]
            self._save_fig(
                plot_all_confusion_matrices(single, title_suffix=" — Copule"),
                fig_dir / "confusion_copula.png",
            )
            self._save_fig(
                plot_model_metrics_comparison(
                    single, title="Performances ML — dataset copule"),
                fig_dir / "metrics_copula.png",
            )
            self._save_fig(
                plot_cross_validation_results(
                    ml_report["cross_validation"],
                    title="Validation croisée 5-fold — dataset copule"),
                fig_dir / "cv_copula.png",
            )

    def _step_save_summary(self) -> dict:
        cfg = self.config
        summary = {
            "config": {
                "m_per_class":      cfg.m_per_class,
                "seed":             cfg.seed,
                "ctgan_epochs":     cfg.ctgan_epochs,
                "ctgan_batch_size": cfg.ctgan_batch_size,
                "output_dir":       str(cfg.output_dir.absolute()),
            },
            "timings": self.timings,
            "outputs": {
                "datasets":  [str(p.name) for p in self._dirs["datasets"].glob("*.csv")],
                "figures":   [str(p.name) for p in self._dirs["figures"].glob("*.png")],
                "reports":   [str(p.name) for p in self._dirs["reports"].glob("*.json")],
            },
            "generation": {
                "copula": (
                    self.copula_generator.stats.to_report()
                    if self.copula_generator else None
                ),
                "ctgan": (
                    self.ctgan_generator.stats.to_report()
                    if self.ctgan_generator else None
                ),
            },
        }
        output = self._dirs["reports"] / "summary.json"
        with open(output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        return summary
