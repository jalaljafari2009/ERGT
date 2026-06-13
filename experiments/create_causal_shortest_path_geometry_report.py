"""Create the Phase 7 Causal Shortest-Path Geometry report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.causal_shortest_path_geometry import (  # noqa: E402
    build_causal_shortest_path_geometry_report,
)
from evaluation.relational_memory_observer import MemoryConfig  # noqa: E402
from experiments.data_utils import load_json, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ERGT Causal Shortest-Path Geometry report."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional ERGT config used to derive graph policy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/observers/causal_shortest_path_geometry_report.json"),
        help="Output path for causal_shortest_path_geometry_report.json.",
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--decay", type=float, default=0.7)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--gate-floor", type=float, default=0.05)
    parser.add_argument("--min-context-edges", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--max-causal-step",
        type=int,
        default=1,
        help="Largest direct causal step before shortest-path reachability.",
    )
    parser.add_argument("--score-margin", type=float, default=1e-4)
    parser.add_argument("--contextuality-margin", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_config = _load_graph_config(args.config)
    memory_config = MemoryConfig(
        decay=args.decay,
        eta=args.eta,
        gate_floor=args.gate_floor,
        min_context_edges=args.min_context_edges,
    )
    report = build_causal_shortest_path_geometry_report(
        graph_config=graph_config,
        seed=args.seed,
        memory_config=memory_config,
        epsilon=args.epsilon,
        max_causal_step=args.max_causal_step,
        score_margin=args.score_margin,
        contextuality_margin=args.contextuality_margin,
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
