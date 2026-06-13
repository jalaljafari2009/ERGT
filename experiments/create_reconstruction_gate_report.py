"""Create the Phase 5 Reconstruction Gate report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.reconstruction_gate import build_reconstruction_gate_report  # noqa: E402
from experiments.data_utils import load_json, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ERGT Reconstruction Gate report.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional ERGT config used to derive graph policy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/observers/reconstruction_gate_report.json"),
        help="Output path for reconstruction_gate_report.json.",
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--min-context-edges", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_config = _load_graph_config(args.config)
    report = build_reconstruction_gate_report(
        graph_config=graph_config,
        seed=args.seed,
        min_context_edges=args.min_context_edges,
    )
    if args.config is not None:
        report["source_config"] = args.config.as_posix()
    save_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _load_graph_config(config_path: Path | None) -> dict[str, Any] | None:
    if config_path is None:
        return None
    config = load_json(config_path)
    return config.get("relational_graph")


if __name__ == "__main__":
    main()
