"""CLI : produit les figures à partir d'un (ou deux) dataset(s) CSV."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..io_utils import read_dataset_csv

from ..analysis import pearson_correlation_matrix
from ..logging_setup import setup_logging
from .correlation_heatmaps import (
    plot_correlation_comparison,
    plot_correlation_heatmap,
)
from .distributions import plot_boxplots_by_class, plot_histograms_by_class
from .scatter_plots import plot_scatter_pairs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Génère les figures d'analyse d'un dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Dataset à visualiser.")
    parser.add_argument("--compare", default=None,
                        help="Second dataset pour les comparaisons (optionnel).")
    parser.add_argument("--output", "-o", default="figures",
                        help="Dossier de sortie des figures PNG.")
    parser.add_argument("--label", default="dataset",
                        help="Suffixe à ajouter aux titres et noms de fichiers.")
    return parser


def _save_fig(fig, path: Path, log) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  ✓ {path.name}")


def main(argv=None) -> int:
    setup_logging()
    log = logging.getLogger("pipeline")
    args = _build_parser().parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Visualisation")
    log.info(f"  Input  : {args.input}")
    log.info(f"  Output : {output_dir.absolute()}")
    log.info("=" * 60)

    df = read_dataset_csv(args.input)
    log.info(f"Chargement : {len(df)} patients")

    suffix = f" — {args.label}"
    label = args.label.lower().replace(" ", "_")

    t0 = time.time()
    log.info("Figures principales :")
    _save_fig(plot_histograms_by_class(df, title_suffix=suffix),
              output_dir / f"histograms_{label}.png", log)
    _save_fig(plot_boxplots_by_class(df, title_suffix=suffix),
              output_dir / f"boxplots_{label}.png", log)
    _save_fig(plot_scatter_pairs(df, title_suffix=suffix),
              output_dir / f"scatter_{label}.png", log)

    corr = pearson_correlation_matrix(df)
    _save_fig(plot_correlation_heatmap(corr, title=f"Corrélations — {args.label}"),
              output_dir / f"heatmap_{label}.png", log)

    if args.compare:
        log.info("Comparaison de méthodes :")
        df2 = read_dataset_csv(args.compare)
        corr2 = pearson_correlation_matrix(df2)
        _save_fig(plot_correlation_comparison(corr, corr2,
                                               label_a=args.label,
                                               label_b="autre"),
                  output_dir / "heatmap_comparison.png", log)

    log.info("-" * 60)
    log.info(f"Terminé en {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
