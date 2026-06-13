"""Create the Phase 4 Information Potential Phi report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.information_potential_phi import (  # noqa: E402
    build_information_potential_report,
)
from experiments.data_utils import load_json, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ERGT Information Potential Phi report.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional ERGT config used to derive graph and distance policies.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/observers/information_potential_report.json"),
        help="Output path for information_potential_report.json.",
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--neighborhood-k", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_config, distance_config = _load_config_parts(args.config)
    report = build_information_potential_report(
        graph_config=graph_config,
        distance_config=distance_config,
        seed=args.seed,
        neighborhood_k=args.neighborhood_k,
    )
    if args.config is not None:
        report["source_config"] = args.config.as_posix()
    save_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _load_config_parts(
    config_path: Path | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if config_path is None:
        return None, None
    config = load_json(config_path)
    return config.get("relational_graph"), {
        **config.get("distance", {}),
        "causal_runtime_distance": config.get("attention", {}).get(
            "causal_runtime_distance",
            False,
        ),
    }


if __name__ == "__main__":
    main()
