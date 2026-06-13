"""Create the Phase 9 Auxiliary Physics-Inspired Loss readiness report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.auxiliary_physics_loss_gate import (  # noqa: E402
    build_auxiliary_physics_loss_gate_report,
)
from experiments.data_utils import load_json, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ERGT Auxiliary Physics-Inspired Loss gate report."
    )
    parser.add_argument(
        "--geoattention-v2-report",
        type=Path,
        default=Path("runs/geoattention_v2/geoattention_v2_report.json"),
        help="Input GeoAttention v2 report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/auxiliary_physics_loss/auxiliary_physics_loss_gate_report.json"),
        help="Output path for auxiliary_physics_loss_gate_report.json.",
    )
    parser.add_argument("--lambda", dest="lambda_value", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geoattention_v2_report = load_json(args.geoattention_v2_report)
    report = build_auxiliary_physics_loss_gate_report(
        geoattention_v2_report,
        requested_lambda=args.lambda_value,
    )
    report["source_report"] = args.geoattention_v2_report.as_posix()
    save_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
