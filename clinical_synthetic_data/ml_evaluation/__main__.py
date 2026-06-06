"""CLI : évalue les modèles ML (3 modèles × 3 protocoles) sur un dataset."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from ..io_utils import read_dataset_csv

from ..logging_setup import setup_logging
from .evaluation import build_ml_evaluation_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Évaluation ML (LogReg + Random Forest + MLP).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Dataset principal.")
    parser.add_argument("--compare", default=None,
                        help="Second dataset pour test de transférabilité.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Graine reproductible.")
    parser.add_argument("--output", "-o", default="ml_report.json",
                        help="Chemin du rapport JSON.")
    return parser


def main(argv=None) -> int:
    setup_logging()
    log = logging.getLogger("pipeline")
    args = _build_parser().parse_args(argv)

    log.info("=" * 60)
    log.info("Évaluation ML")
    log.info(f"  Input    : {args.input}")
    if args.compare:
        log.info(f"  Compare  : {args.compare}")
    log.info(f"  Seed     : {args.seed}")
    log.info("=" * 60)

    df = read_dataset_csv(args.input)
    log.info(f"Chargement : {len(df)} patients")

    df_other = None
    if args.compare:
        df_other = read_dataset_csv(args.compare)
        log.info(f"Chargement comparaison : {len(df_other)} patients")

    log.info("Entraînement et évaluation des 3 modèles…")
    t0 = time.time()
    report = build_ml_evaluation_report(df, df_other=df_other, seed=args.seed)
    dt = time.time() - t0
    log.info(f"  ✓ Terminé en {dt:.1f}s")

    log.info("")
    log.info("Résultats split unique :")
    for model_name, metrics in report["single_split"].items():
        auc = metrics.get("roc_auc_ovr_macro") or 0
        log.info(
            f"  {model_name:<25s} "
            f"acc={metrics['accuracy']:.3f}  "
            f"f1={metrics['f1_macro']:.3f}  "
            f"auc={auc:.3f}"
        )

    log.info("")
    log.info("Validation croisée 5-fold (F1 macro) :")
    for model_name, m in report["cross_validation"].items():
        log.info(f"  {model_name:<25s} {m['mean']:.3f} ± {m['std']:.3f}")

    if "cross_method" in report:
        log.info("")
        log.info("Transférabilité (train input → test compare) :")
        for model_name, metrics in report["cross_method"]["train_on_main_test_on_other"].items():
            log.info(
                f"  {model_name:<25s} "
                f"acc={metrics['accuracy']:.3f}  f1={metrics['f1_macro']:.3f}"
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    log.info("")
    log.info(f"  ✓ Rapport sauvegardé : {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
