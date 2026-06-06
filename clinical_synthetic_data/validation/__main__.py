"""CLI : valide un dataset CSV existant."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter

import pandas as pd

from ..io_utils import read_dataset_csv

from ..core.patient_schema import ClassLabel
from ..logging_setup import setup_logging
from .validator import validate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Valide un dataset CSV ligne par ligne.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="CSV à valider (doit contenir une colonne class_label).")
    parser.add_argument("--show-first-failures", type=int, default=5,
                        help="Affiche les N premières lignes invalides en détail.")
    return parser


def main(argv=None) -> int:
    setup_logging()
    log = logging.getLogger("pipeline")
    args = _build_parser().parse_args(argv)

    log.info("=" * 60)
    log.info(f"Validation : {args.input}")
    log.info("=" * 60)

    df = read_dataset_csv(args.input)
    log.info(f"Chargement : {len(df)} patients à valider")

    t0 = time.time()
    valid_count = 0
    failures: list[tuple[int, str, str]] = []
    rule_counter: Counter = Counter()
    class_rule_counter: dict[str, Counter] = {}

    for idx, row in df.iterrows():
        values = row.to_dict()
        cls_str = values.get("class_label")
        try:
            cls = ClassLabel(cls_str)
        except ValueError:
            failures.append((idx, str(cls_str), "class_label_invalide"))
            rule_counter["class_label_invalide"] += 1
            continue

        result = validate(values, cls)
        if result.is_valid:
            valid_count += 1
        else:
            failures.append((idx, cls.value, result.failed_rule))
            rule_counter[result.failed_rule] += 1
            class_rule_counter.setdefault(cls.value, Counter())[result.failed_rule] += 1

    dt = time.time() - t0

    log.info("-" * 60)
    log.info(f"Terminé en {dt:.1f}s")
    log.info(f"  Patients valides   : {valid_count} / {len(df)}  ({valid_count / len(df):.1%})")
    log.info(f"  Patients invalides : {len(failures)}")

    if rule_counter:
        log.info("")
        log.info("Rejets par règle :")
        for rule, count in rule_counter.most_common():
            log.info(f"  {rule:<25s} {count}")

    if class_rule_counter:
        log.info("")
        log.info("Rejets par classe :")
        for cls, counter in sorted(class_rule_counter.items()):
            top = ", ".join(f"{r}={n}" for r, n in counter.most_common(3))
            log.info(f"  {cls:<30s} {top}")

    if args.show_first_failures > 0 and failures:
        log.info("")
        log.info(f"Premières {min(args.show_first_failures, len(failures))} lignes invalides :")
        for idx, cls, rule in failures[: args.show_first_failures]:
            log.info(f"  ligne {idx}  classe={cls:<22s} règle échouée={rule}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
