"""Point d'entrée du pipeline complet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clinical_synthetic_data.logging_setup import setup_logging
from clinical_synthetic_data.pipeline import PipelineConfig, SyntheticDataPipeline


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Pipeline de génération de données cliniques synthétiques.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--m", "--m-per-class", type=int, default=1000,
                        dest="m_per_class",
                        help="Patients par classe (6 × M au total).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Graine de reproductibilité.")
    parser.add_argument("--output", "-o", type=Path, default=Path("outputs"),
                        help="Dossier de sortie.")
    parser.add_argument("--ctgan-epochs", type=int, default=600,
                        help="Epochs CTGAN.")
    parser.add_argument("--ctgan-batch-size", type=int, default=500,
                        help="Batch size CTGAN.")
    parser.add_argument("--skip-ctgan", action="store_true",
                        help="Saute la génération CTGAN.")
    parser.add_argument("--skip-ml", action="store_true",
                        help="Saute l'évaluation ML.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    setup_logging()

    config = PipelineConfig(
        m_per_class=args.m_per_class,
        seed=args.seed,
        output_dir=args.output,
        ctgan_epochs=args.ctgan_epochs,
        ctgan_batch_size=args.ctgan_batch_size,
        skip_ctgan=args.skip_ctgan,
        skip_ml=args.skip_ml,
    )

    pipeline = SyntheticDataPipeline(config)
    summary = pipeline.run()

    print()
    print(f"✓ Pipeline terminé. Sorties dans : {config.output_dir.absolute()}")
    n_datasets = len(summary["outputs"]["datasets"])
    n_figures = len(summary["outputs"]["figures"])
    n_reports = len(summary["outputs"]["reports"])
    print(f"  {n_datasets} dataset(s), {n_figures} figure(s), {n_reports} rapport(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
