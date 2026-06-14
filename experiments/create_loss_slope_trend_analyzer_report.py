"""Create the Loss-Slope and Trend Analyzer report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.loss_slope_trend_analyzer import (  # noqa: E402
    TrendAnalyzerConfig,
    build_loss_slope_trend_analyzer_report,
)
from experiments.data_utils import save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ERGT stage-7 loss-slope and trend analyzer report."
    )
    parser.add_argument(
        "--condition-progress",
        type=Path,
        default=None,
        help="Optional condition progress_log.jsonl.",
    )
    parser.add_argument(
        "--baseline-progress",
        type=Path,
        default=None,
        help="Optional baseline progress_log.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/contracts/loss_slope_trend_analyzer.json"),
        help="Output path for loss_slope_trend_analyzer.json.",
    )
    parser.add_argument("--min-points-for-slope", type=int, default=3)
    parser.add_argument("--rolling-window-points", type=int, default=4)
    parser.add_argument("--ema-beta", type=float, default=0.7)
    parser.add_argument("--late-window-start", type=int, default=1000)
    parser.add_argument("--post-1000-start", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrendAnalyzerConfig(
        min_points_for_slope=args.min_points_for_slope,
        rolling_window_points=args.rolling_window_points,
        ema_beta=args.ema_beta,
        late_window_start=args.late_window_start,
        post_1000_start=args.post_1000_start,
    )
    condition_progress = _read_jsonl(args.condition_progress)
    baseline_progress = _read_jsonl(args.baseline_progress)
    report = build_loss_slope_trend_analyzer_report(
        condition_progress=condition_progress,
        baseline_progress=baseline_progress,
        config=config,
    )
    save_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _read_jsonl(path: Path | None) -> list[dict] | None:
    if path is None:
        return None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
