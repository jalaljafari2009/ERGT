"""Evaluate a trained from-scratch transformer baseline checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_utils import load_json, load_prepared_datasets, save_json  # noqa: E402
from experiments.train_baseline import build_model_config, default_data_dir, evaluate  # noqa: E402
from models.transformer_baseline import TransformerBaseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ERGT Phase 0 baseline.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the saved run config or original baseline config.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint to evaluate.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Prepared data directory. Defaults from config.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to evaluation_results.json next to checkpoint run.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Evaluation device. Defaults to cuda if available, otherwise cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    checkpoint_path = Path(args.checkpoint)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir(config)
    _, validation_dataset, metadata = load_prepared_or_raise(data_dir)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        drop_last=False,
    )

    model_config = build_model_config(config, metadata)
    model = TransformerBaseline(model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    validation_loss = evaluate(model, validation_loader, device)
    results: dict[str, Any] = {
        "condition": config["run"]["condition"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "data_dir": str(data_dir),
        "validation_loss": validation_loss,
        "perplexity": math.exp(validation_loss) if validation_loss < 50 else math.inf,
        "device": str(device),
    }

    output_path = Path(args.output) if args.output else default_output_path(checkpoint_path)
    save_json(output_path, results)
    print(json.dumps(results, indent=2, sort_keys=True))


def load_prepared_or_raise(data_dir: Path):
    if not (data_dir / "metadata.json").exists():
        raise FileNotFoundError(
            f"prepared dataset not found at {data_dir}. "
            "Run scripts/prepare_wikitext2.py first, or pass --data-dir."
        )
    return load_prepared_datasets(data_dir)


def default_output_path(checkpoint_path: Path) -> Path:
    run_dir = (
        checkpoint_path.parent.parent
        if checkpoint_path.parent.name == "checkpoints"
        else checkpoint_path.parent
    )
    return run_dir / "evaluation_results.json"


if __name__ == "__main__":
    main()
