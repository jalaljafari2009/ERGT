"""Write the stage-26 longer-run or multi-seed confirmation report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.longer_run_multi_seed_confirmation import (  # noqa: E402
    build_longer_run_multi_seed_confirmation_report,
)
from experiments.data_utils import save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ERGT stage-26 longer-run/multi-seed confirmation report."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/contracts/longer_run_multi_seed_confirmation.json"),
        help="Output path for longer_run_multi_seed_confirmation.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_longer_run_multi_seed_confirmation_report()
    save_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
