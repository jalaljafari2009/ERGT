"""Create the Phase 12 Intelligence Space Evaluation readiness report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.intelligence_space_evaluation_gate import (  # noqa: E402
    build_intelligence_space_evaluation_gate_report,
)
from experiments.data_utils import load_json, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ERGT Intelligence Space Evaluation gate report."
    )
    parser.add_argument(
        "--reasoning-path-report",
        type=Path,
        default=Path("runs/reasoning_path_evaluation/reasoning_path_gate_report.json"),
        help="Input Reasoning Path Evaluation gate report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/intelligence_space_evaluation/intelligence_space_gate_report.json"
        ),
        help="Output path for intelligence_space_gate_report.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reasoning_report = (
        load_json(args.reasoning_path_report)
        if args.reasoning_path_report.exists()
        else None
    )
    report = build_intelligence_space_evaluation_gate_report(reasoning_report)
    if reasoning_report is not None:
        report["source_report"] = args.reasoning_path_report.as_posix()
    save_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
