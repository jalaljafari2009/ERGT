"""Write the stage-22 late-window and post-1000 analysis report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.late_window_post1000_analysis import (  # noqa: E402
    build_late_window_post1000_analysis_report,
)
from experiments.data_utils import save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ERGT stage-22 late-window and post-1000 analysis report."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/contracts/late_window_post1000_analysis.json"),
        help="Output path for late_window_post1000_analysis.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_late_window_post1000_analysis_report()
    save_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
