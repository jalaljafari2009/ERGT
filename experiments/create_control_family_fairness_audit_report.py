"""Create the Control-Family Fairness Audit v2 report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.control_family_fairness_audit import (  # noqa: E402
    build_control_family_fairness_audit_report,
)
from evaluation.relational_memory_observer import MemoryConfig  # noqa: E402
from experiments.data_utils import load_json, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ERGT stage-6 control-family fairness audit report."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional ERGT config used to derive graph/distance policy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/contracts/control_family_fairness_audit_v2.json"),
        help="Output path for control_family_fairness_audit_v2.json.",
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--decay", type=float, default=0.7)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--gate-floor", type=float, default=0.05)
    parser.add_argument("--min-context-edges", type=int, default=2)
    parser.add_argument("--max-causal-step", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.1)
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
    report = build_control_family_fairness_audit_report(
        graph_config=_load_graph_config(config),
        distance_config=_load_distance_config(config),
        seed=args.seed,
        memory_config=memory_config,
        max_causal_step=args.max_causal_step,
        alpha=args.alpha,
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
