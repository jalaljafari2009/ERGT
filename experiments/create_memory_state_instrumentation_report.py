"""Create the Memory State Instrumentation report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.memory_state_instrumentation import (  # noqa: E402
    build_memory_state_instrumentation_report,
)
from experiments.data_utils import save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create memory-state instrumentation report.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/contracts/memory_state_instrumentation.json"),
        help="Output path for memory_state_instrumentation.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
        help="Synthetic smoke seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_memory_state_instrumentation_report(seed=args.seed)
    save_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
