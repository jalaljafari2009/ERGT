"""Create the Phase 10 Complete ERGT Architecture readiness report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.complete_ergt_architecture_gate import (  # noqa: E402
    build_complete_ergt_architecture_gate_report,
)
from experiments.data_utils import load_json, save_json  # noqa: E402

DEFAULT_REPORT_PATHS = {
    "measurement_contracts": Path("runs/contracts/measurement_contract_report.json"),
    "strict_w_controls": Path("runs/contracts/strict_w_controls_report.json"),
    "relational_field_observer": Path("runs/observers/relational_field_observer_report.json"),
    "resonant_response_observer": Path("runs/observers/resonant_response_observer_report.json"),
    "information_potential_phi": Path("runs/observers/information_potential_report.json"),
    "reconstruction_gate": Path("runs/observers/reconstruction_gate_report.json"),
    "relational_memory_observer": Path("runs/observers/relational_memory_observer_report.json"),
    "causal_shortest_path_geometry": Path(
        "runs/observers/causal_shortest_path_geometry_report.json"
    ),
    "geoattention_v2": Path("runs/geoattention_v2/geoattention_v2_report.json"),
    "auxiliary_physics_loss_gate": Path(
        "runs/auxiliary_physics_loss/auxiliary_physics_loss_gate_report.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ERGT Complete Architecture gate report."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/complete_ergt_architecture/complete_ergt_architecture_gate_report.json"),
        help="Output path for complete_ergt_architecture_gate_report.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = {
        name: load_json(path)
        for name, path in DEFAULT_REPORT_PATHS.items()
        if path.exists()
    }
    report = build_complete_ergt_architecture_gate_report(reports)
    report["source_reports"] = {
        name: path.as_posix()
        for name, path in DEFAULT_REPORT_PATHS.items()
        if path.exists()
    }
    save_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
