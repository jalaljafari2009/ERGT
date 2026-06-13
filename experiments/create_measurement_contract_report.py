"""Create the Phase 0 measurement contract report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.measurement_contracts import (  # noqa: E402
    MeasurementContract,
    build_measurement_contract_report,
    contract_from_project_config,
)
from experiments.data_utils import load_json, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ERGT measurement contract report.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional ERGT project config to derive contract fields from.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/contracts/measurement_contract_report.json"),
        help="Output path for measurement_contract_report.json.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="Small tensor sequence length used for valid-edge checks.",
    )
    parser.add_argument(
        "--attention-mask",
        type=str,
        default=None,
        help="Optional JSON list of lists, e.g. '[[1,1,1,0,0]]'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = _load_contract(args.config)
    attention_mask = _load_attention_mask(args.attention_mask, args.sequence_length)
    report = build_measurement_contract_report(
        contract,
        sequence_length=args.sequence_length,
        attention_mask=attention_mask,
    )
    if args.config is not None:
        report["source_config"] = args.config.as_posix()
    save_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _load_contract(config_path: Path | None) -> MeasurementContract:
    if config_path is None:
        return MeasurementContract()
    project_config: dict[str, Any] = load_json(config_path)
    return contract_from_project_config(project_config)


def _load_attention_mask(mask_json: str | None, sequence_length: int) -> torch.Tensor:
    if mask_json is None:
        return torch.ones(1, sequence_length, dtype=torch.long)
    payload = json.loads(mask_json)
    mask = torch.tensor(payload, dtype=torch.long)
    if mask.dim() != 2:
        raise ValueError("--attention-mask must be a JSON list of lists")
    if mask.size(1) != sequence_length:
        raise ValueError("--attention-mask width must match --sequence-length")
    return mask


if __name__ == "__main__":
    main()
