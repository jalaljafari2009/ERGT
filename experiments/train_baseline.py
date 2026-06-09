"""Train the controlled from-scratch transformer baseline."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_utils import (  # noqa: E402
    load_json,
    load_prepared_datasets,
    load_tokenizer,
    save_json,
)
from models.transformer_baseline import TransformerBaseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ERGT Phase 0 baseline.")
    parser.add_argument(
        "--config",
        default="configs/baseline/debug_wikitext2.json",
        help="Path to baseline training config.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Prepared data directory. Defaults from dataset, tokenizer, and context.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Training device. Defaults to cuda if available, otherwise cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_json(config_path)

    seed = int(config["run"].get("seed", 1337))
    set_seed(seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    shutil.copyfile(config_path, output_dir / "config.json")

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir(config)
    train_dataset, validation_dataset, metadata = load_prepared_or_raise(data_dir)

    model_config = build_model_config(config, metadata)
    model = TransformerBaseline(model_config).to(device)
    optimizer = build_optimizer(model, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        drop_last=False,
    )
    if len(train_loader) == 0:
        raise ValueError("train loader produced no batches; reduce batch_size or provide more data")
    if len(validation_loader) == 0:
        raise ValueError(
            "validation loader produced no batches; reduce batch_size or provide more data"
        )

    save_json(output_dir / "model_summary.json", model.model_summary())

    max_steps = int(config["training"]["max_steps"])
    eval_interval = int(config["training"]["eval_interval"])
    checkpoint_interval = int(config["training"]["checkpoint_interval"])
    grad_clip = float(config["training"].get("grad_clip", 0.0))

    train_log_path = output_dir / config["logging"].get("train_log", "train_log.jsonl")
    best_val_loss = math.inf
    final_train_loss = math.inf
    start_time = time.perf_counter()
    tokens_processed = 0
    step = 0

    while step < max_steps:
        for input_ids, targets in train_loader:
            if step >= max_steps:
                break
            step += 1

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            lr = learning_rate_for_step(step, config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            model.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids, targets=targets)
            loss = outputs["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            batch_tokens = input_ids.numel()
            tokens_processed += batch_tokens
            final_train_loss = float(loss.item())

            should_eval = step == 1 or step % eval_interval == 0 or step == max_steps
            log_record: dict[str, Any] = {
                "step": step,
                "train_loss": final_train_loss,
                "learning_rate": lr,
                "tokens_processed": tokens_processed,
                "tokens_per_second": tokens_processed / max(time.perf_counter() - start_time, 1e-9),
                "seed": int(config["run"].get("seed", 1337)),
                "device": str(device),
            }

            if should_eval:
                val_loss = evaluate(model, validation_loader, device)
                if not math.isfinite(val_loss):
                    raise RuntimeError(f"non-finite validation loss at step {step}: {val_loss}")
                perplexity = math.exp(val_loss) if val_loss < 50 else math.inf
                log_record["validation_loss"] = val_loss
                log_record["perplexity"] = perplexity
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        output_dir / "checkpoints" / "best.pt", model, optimizer, step, config
                    )

            append_jsonl(train_log_path, log_record)

            if step % checkpoint_interval == 0 or step == max_steps:
                save_checkpoint(
                    output_dir / "checkpoints" / "last.pt", model, optimizer, step, config
                )

    wall_time = time.perf_counter() - start_time
    final_val_loss = evaluate(model, validation_loader, device)
    results = {
        "condition": config["run"]["condition"],
        "final_training_loss": final_train_loss,
        "best_validation_loss": best_val_loss,
        "final_validation_loss": final_val_loss,
        "perplexity": math.exp(final_val_loss) if final_val_loss < 50 else math.inf,
        "total_training_tokens": tokens_processed,
        "total_wall_clock_seconds": wall_time,
        "average_tokens_per_second": tokens_processed / max(wall_time, 1e-9),
        "device": str(device),
        "seed": seed,
        "data_dir": str(data_dir),
    }
    if torch.cuda.is_available() and device.type == "cuda":
        results["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)

    save_json(output_dir / config["logging"].get("results", "baseline_results.json"), results)
    save_checkpoint(output_dir / "checkpoints" / "last.pt", model, optimizer, step, config)
    print(json.dumps(results, indent=2, sort_keys=True))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_data_dir(config: dict[str, Any]) -> Path:
    dataset = str(config["dataset"]["name"]).replace("-", "")
    tokenizer = str(config["dataset"]["tokenizer"]).replace("/", "_")
    context = int(config["dataset"]["context_length"])
    return Path("data") / "processed" / f"{dataset}_{tokenizer}_ctx{context}"


def load_prepared_or_raise(data_dir: Path):
    if not (data_dir / "metadata.json").exists():
        raise FileNotFoundError(
            f"prepared dataset not found at {data_dir}. "
            "Run scripts/prepare_wikitext2.py first, or pass --data-dir."
        )
    return load_prepared_datasets(data_dir)


def build_model_config(config: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    model_config = dict(config["model"])
    model_config.pop("type", None)
    model_config.pop("positional_encoding", None)

    if model_config.get("vocab_size") is None:
        if metadata.get("vocab_size") is not None:
            model_config["vocab_size"] = int(metadata["vocab_size"])
        else:
            tokenizer = load_tokenizer(config["dataset"]["tokenizer"])
            model_config["vocab_size"] = len(tokenizer)

    model_config["context_length"] = int(config["dataset"]["context_length"])
    return model_config


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    training = config["training"]
    if training["optimizer"].lower() != "adamw":
        raise ValueError("only AdamW is supported for the baseline trainer")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training.get("betas", [0.9, 0.95])),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )


def learning_rate_for_step(step: int, config: dict[str, Any]) -> float:
    base_lr = float(config["training"]["learning_rate"])
    warmup_steps = int(config["training"].get("warmup_steps", 0))
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr


@torch.no_grad()
def evaluate(model: TransformerBaseline, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for input_ids, targets in data_loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        outputs = model(input_ids, targets=targets)
        loss = outputs["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite validation loss: {loss.item()}")
        tokens = targets.numel()
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens
    if total_tokens == 0:
        raise ValueError("validation loader produced no tokens")
    return total_loss / total_tokens


def save_checkpoint(
    path: Path,
    model: TransformerBaseline,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": config,
        },
        path,
    )


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
