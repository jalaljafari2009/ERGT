"""Write the stage-19 adaptive notebook contract report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.adaptive_notebook_ergt_03 import (  # noqa: E402
    NOTEBOOK_PATH,
    build_adaptive_notebook_ergt_03_report,
)
from experiments.data_utils import save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook-path", type=Path, default=NOTEBOOK_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/contracts/adaptive_notebook_ergt_03.json"),
    )
    args = parser.parse_args()

    report = build_adaptive_notebook_ergt_03_report(args.notebook_path)
    save_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
