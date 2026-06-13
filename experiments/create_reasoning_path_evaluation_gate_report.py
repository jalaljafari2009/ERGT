"""Create the Phase 11 Reasoning Path Evaluation readiness report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.reasoning_path_evaluation_gate import (  # noqa: E402
    build_reasoning_path_evaluation_gate_report,
)
from experiments.data_utils import load_json, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ERGT Reasoning Path Evaluation gate report."
    )
    parser.add_argument(
        "--complete-architecture-report",
        type=Path,
        default=Path(
            "runs/complete_ergt_architecture/complete_ergt_architecture_gate_report.json"
        ),
        help="Input Complete ERGT Architecture gate report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/reasoning_path_evaluation/reasoning_path_gate_report.json"),
        help="Output path for reasoning_path_gate_report.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    complete_report = (
        load_json(args.complete_architecture_report)
        if args.complete_architecture_report.exists()
        else None
    )
    report = build_reasoning_path_evaluation_gate_report(complete_report)
    if complete_report is not None:
        report["source_report"] = args.complete_architecture_report.as_posix()
    save_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
