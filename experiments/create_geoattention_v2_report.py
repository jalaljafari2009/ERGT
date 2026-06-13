"""Create the Phase 8 GeoAttention v2 smoke report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.geoattention_v2_report import build_geoattention_v2_report  # noqa: E402
from evaluation.relational_memory_observer import MemoryConfig  # noqa: E402
from experiments.data_utils import load_json, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ERGT GeoAttention v2 report.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional ERGT config used to derive graph/distance policy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/geoattention_v2/geoattention_v2_report.json"),
        help="Output path for geoattention_v2_report.json.",
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--decay", type=float, default=0.7)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--gate-floor", type=float, default=0.05)
    parser.add_argument("--min-context-edges", type=int, default=2)
    parser.add_argument("--max-causal-step", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config) if args.config is not None else {}
    memory_config = MemoryConfig(
        decay=args.decay,
        eta=args.eta,
        gate_floor=args.gate_floor,
        min_context_edges=args.min_context_edges,
    )
    report = build_geoattention_v2_report(
        graph_config=_load_graph_config(config),
        distance_config=_load_distance_config(config),
        seed=args.seed,
        alpha=args.alpha,
        max_causal_step=args.max_causal_step,
        memory_config=memory_config,
    )
    if args.config is not None:
        report["source_config"] = args.config.as_posix()
    save_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _load_graph_config(config: dict[str, Any]) -> dict[str, Any] | None:
    return config.get("relational_graph")


def _load_distance_config(config: dict[str, Any]) -> dict[str, Any] | None:
    return config.get("distance")


if __name__ == "__main__":
    main()
