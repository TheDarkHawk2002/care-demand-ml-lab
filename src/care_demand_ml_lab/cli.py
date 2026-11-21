from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .data import load_records
from .model import evaluate
from .synthetic import write_csv


def main() -> None:
    parser = argparse.ArgumentParser(prog="care-demand-lab", description="Evaluate a model on synthetic operational service data.")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Create reproducible synthetic CSV data")
    gen.add_argument("--output", default="examples/requests.csv")
    gen.add_argument("--rows", type=int, default=360)
    gen.add_argument("--seed", type=int, default=41)
    run = sub.add_parser("evaluate", help="Compare baseline and text-plus-tabular model")
    run.add_argument("input")
    run.add_argument("--report")
    run.add_argument("--predictions")
    args = parser.parse_args()
    if args.command == "generate":
        print(write_csv(args.output, args.rows, args.seed))
        return
    rows = load_records(args.input)
    report, predictions = evaluate(rows)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if args.predictions:
        target = Path(args.predictions)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)

if __name__ == "__main__":
    main()
