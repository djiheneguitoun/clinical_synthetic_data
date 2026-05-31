"""
CLI : analyse statistique d'un (ou deux) dataset(s) CSV.

    # Analyse d'un dataset
    python -m clinical_synthetic_data.analysis --input dataset.csv --output reports/

    # Comparaison de deux datasets (méthode 1 vs méthode 2)
    python -m clinical_synthetic_data.analysis \\
        --input dataset_copula.csv --compare dataset_ctgan.csv --output reports/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from ..io_utils import read_dataset_csv

from ..logging_setup import setup_logging
from .descriptive_stats import build_descriptive_report
from .epidemiological_validation import build_epidemiological_report
from .method_comparison import build_method_comparison_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse statistique d'un dataset clinique synthétique.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Dataset principal à analyser.")
    parser.add_argument("--compare", default=None,
                        help="Second dataset à comparer (optionnel).")
    parser.add_argument("--output", "-o", default="analysis_output",
                        help="Dossier où sauvegarder les rapports JSON.")
    return parser


def main(argv=None) -> int:
    setup_logging()
    log = logging.getLogger("pipeline")
    args = _build_parser().parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Analyse statistique")
    log.info(f"  Input    : {args.input}")
    if args.compare:
        log.info(f"  Compare  : {args.compare}")
    log.info(f"  Output   : {output_dir.absolute()}")
    log.info("=" * 60)

    df = read_dataset_csv(args.input)
    log.info(f"Chargement : {len(df)} patients")

    log.info("Calcul des statistiques descriptives…")
    desc = build_descriptive_report(df)

    log.info("Validation épidémiologique…")
    epi = build_epidemiological_report(df)
    log.info(f"  → {epi['n_patterns_passed']}/{epi['n_patterns_total']} patterns satisfaits")
    log.info(f"  → Hiérarchie d'âges : {'OK' if epi['age_ordering_all_passed'] else 'ÉCHEC'}")

    report = {"descriptive": desc, "epidemiological": epi}

    if args.compare:
        log.info("Comparaison de méthodes…")
        df2 = read_dataset_csv(args.compare)
        comp = build_method_comparison_report(df, df2)
        report["comparison"] = comp
        log.info(f"  → Distance Frobenius corrélations : {comp['correlation_frobenius_distance']:.3f}")

    output_path = output_dir / "analysis_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"  ✓ Rapport sauvegardé : {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
