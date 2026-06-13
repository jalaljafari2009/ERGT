"""Create a Run-02 adaptive-alpha evidence consolidation report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.run02_evidence_consolidation import (  # noqa: E402
    build_run02_evidence_consolidation_report,
)
from experiments.data_utils import save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Run-02 evidence report.")
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Run-02 root directory containing condition subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <run-root>/run02_evidence_consolidation_report.json.",
    )
    parser.add_argument("--late-window-start", type=int, default=1000)
    parser.add_argument("--late-window-end", type=int, default=2000)
    parser.add_argument("--alpha-zero-tolerance", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.run_root / "run02_evidence_consolidation_report.json"
    report = build_run02_evidence_consolidation_report(
        args.run_root,
        late_window=(args.late_window_start, args.late_window_end),
        alpha_zero_tolerance=args.alpha_zero_tolerance,
    )
    save_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
