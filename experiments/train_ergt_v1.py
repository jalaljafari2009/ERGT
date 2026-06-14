"""Train ERGT-v1 under Phase 3 comparison conditions."""

from __future__ import annotations

import argparse
import json
import math
import random
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

from evaluation.attention_metrics import aggregate_attention_diagnostics  # noqa: E402
from experiments.data_utils import (  # noqa: E402
    load_json,
    load_prepared_datasets,
    load_tokenizer,
    save_json,
)
from experiments.progress_logging import (  # noqa: E402
    append_jsonl,
    format_progress_line,
    geometry_progress_fields,
    gpu_memory_snapshot,
)
from experiments.train_baseline import default_data_dir, learning_rate_for_step  # noqa: E402
from models.ergt_v1 import ERGTV1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ERGT-v1 Phase 3 condition.")
    parser.add_argument(
        "--config",
        default="configs/ergt_v1/real_d.json",
        help="Path to ERGT-v1 training config.",
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
    ensure_attention_control_seed(config, seed)
    set_seed(seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_torch_runtime(config, device)
    output_dir = Path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    save_json(output_dir / "config.json", config)

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir(config)
    train_dataset, validation_dataset, metadata = load_prepared_or_raise(data_dir)

    model_config = build_ergt_model_config(config, metadata)
    model = ERGTV1(model_config).to(device)
    model_summary = model.model_summary()
    model = maybe_compile_model(model, config, device)
    optimizer = build_optimizer(model, config)
    precision = precision_config(config, device)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=precision["use_grad_scaler"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        drop_last=True,
        **dataloader_runtime_kwargs(config, device),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        **dataloader_runtime_kwargs(config, device),
    )
    if len(train_loader) == 0:
        raise ValueError("train loader produced no batches; reduce batch_size or provide more data")
    if len(validation_loader) == 0:
        raise ValueError(
            "validation loader produced no batches; reduce batch_size or provide more data"
        )

    save_json(output_dir / "model_summary.json", model_summary)

    max_steps = int(config["training"]["max_steps"])
    eval_interval = int(config["training"]["eval_interval"])
    checkpoint_interval = int(config["training"]["checkpoint_interval"])
    grad_clip = float(config["training"].get("grad_clip", 0.0))
    max_eval_batches = config["training"].get("max_eval_batches")
    max_eval_batches = int(max_eval_batches) if max_eval_batches is not None else None
    log_geometry = bool(config.get("logging", {}).get("log_geometry_diagnostics", True))
    train_geometry_interval = int(
        config.get("logging", {}).get("train_geometry_diagnostics_interval", eval_interval)
    )
    if train_geometry_interval <= 0:
        train_geometry_interval = eval_interval

    train_log_path = output_dir / config["logging"].get("train_log", "train_log.jsonl")
    progress_log_path = output_dir / config["logging"].get(
        "progress_log", "progress_log.jsonl"
    )
    best_val_loss = math.inf
    final_train_loss = math.inf
    start_time = time.perf_counter()
    tokens_processed = 0
    step = 0

    if train_log_path.exists():
        train_log_path.unlink()
    if progress_log_path.exists():
        progress_log_path.unlink()

    while step < max_steps:
        for input_ids, targets in train_loader:
            if step >= max_steps:
                break
            step += 1

            input_ids = input_ids.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")

            lr = learning_rate_for_step(step, config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            model.train()
            model.set_training_step(step)
            optimizer.zero_grad(set_to_none=True)
            should_eval = step == 1 or step % eval_interval == 0 or step == max_steps
            should_log_train_geometry = (
                log_geometry
                and (step == 1 or step % train_geometry_interval == 0 or step == max_steps)
            )
            with torch.autocast(
                device_type=device.type,
                dtype=precision["autocast_dtype"],
                enabled=precision["use_autocast"],
            ):
                outputs = model(
                    input_ids,
                    targets=targets,
                    return_geometry_diagnostics=should_log_train_geometry,
                )
            loss = outputs["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

            scaler.scale(loss).backward()
            grad_norm = None
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                    error_if_nonfinite=True,
                )
                grad_norm = float(grad_norm_tensor.detach().cpu().item())
            scaler.step(optimizer)
            scaler.update()

            batch_tokens = input_ids.numel()
            tokens_processed += batch_tokens
            final_train_loss = float(loss.item())
            elapsed_seconds = time.perf_counter() - start_time

            log_record: dict[str, Any] = {
                "step": step,
                "train_loss": final_train_loss,
                "learning_rate": lr,
                "tokens_processed": tokens_processed,
                "tokens_per_second": tokens_processed / max(elapsed_seconds, 1e-9),
                "elapsed_seconds": elapsed_seconds,
                "elapsed_minutes": elapsed_seconds / 60.0,
                "grad_norm": grad_norm,
                "seed": seed,
                "device": str(device),
                "condition": config["run"]["condition"],
                "distance_mode": config.get("attention", {}).get("distance_mode"),
                "gradient_mode": config.get("attention", {}).get("gradient_mode"),
                "kernel": config.get("relational_graph", {}).get("kernel"),
                "normalize_hidden": config.get("relational_graph", {}).get("normalize_hidden"),
                "nan_or_inf_detected": False,
            }

            if should_log_train_geometry:
                log_record["geometry"] = aggregate_attention_diagnostics(
                    outputs["geometry_diagnostics"]
                )

            if should_eval:
                val_result = evaluate(
                    model,
                    validation_loader,
                    device,
                    log_geometry=log_geometry,
                    precision=precision,
                    max_batches=max_eval_batches,
                )
                val_loss = val_result["validation_loss"]
                if not math.isfinite(val_loss):
                    raise RuntimeError(f"non-finite validation loss at step {step}: {val_loss}")
                perplexity = math.exp(val_loss) if val_loss < 50 else math.inf
                log_record["validation_loss"] = val_loss
                log_record["perplexity"] = perplexity
                if log_geometry:
                    log_record["validation_geometry"] = val_result.get("geometry")
                    log_record.update(geometry_progress_fields(val_result.get("geometry")))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        output_dir / "checkpoints" / "best.pt", model, optimizer, step, config
                    )
                log_record["best_validation_loss"] = best_val_loss
                log_record.update(gpu_memory_snapshot(device))
                append_jsonl(progress_log_path, log_record)
                print(format_progress_line(log_record), flush=True)

            append_jsonl(train_log_path, log_record)

            if step % checkpoint_interval == 0 or step == max_steps:
                save_checkpoint(
                    output_dir / "checkpoints" / "last.pt", model, optimizer, step, config
                )

    wall_time = time.perf_counter() - start_time
    final_eval = evaluate(
        model,
        validation_loader,
        device,
        log_geometry=log_geometry,
        precision=precision,
        max_batches=max_eval_batches,
    )
    final_val_loss = final_eval["validation_loss"]
    results: dict[str, Any] = {
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
        "max_eval_batches": max_eval_batches,
        "runtime": runtime_summary(config, device, precision),
        "attention": config.get("attention", {}),
        "distance": config.get("distance", {}),
    }
    if log_geometry:
        results["geometry"] = final_eval.get("geometry")
    if torch.cuda.is_available() and device.type == "cuda":
        results["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)

    save_json(output_dir / config["logging"].get("results", "metrics.json"), results)
    save_checkpoint(output_dir / "checkpoints" / "last.pt", model, optimizer, step, config)
    print(json.dumps(results, indent=2, sort_keys=True))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_attention_control_seed(config: dict[str, Any], seed: int) -> None:
    """Persist the W-control RNG seed used by random/shuffled conditions."""

    attention = config.setdefault("attention", {})
    attention.setdefault("control_seed", int(seed))


def load_prepared_or_raise(data_dir: Path):
    if not (data_dir / "metadata.json").exists():
        raise FileNotFoundError(
            f"prepared dataset not found at {data_dir}. "
            "Run scripts/prepare_wikitext2.py first, or pass --data-dir."
        )
    return load_prepared_datasets(data_dir)


def build_ergt_model_config(config: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    model_config = dict(config["model"])
    model_config.pop("positional_encoding", None)
    if model_config.get("vocab_size") is None:
        if metadata.get("vocab_size") is not None:
            model_config["vocab_size"] = int(metadata["vocab_size"])
        else:
            tokenizer = load_tokenizer(config["dataset"]["tokenizer"])
            model_config["vocab_size"] = len(tokenizer)
    model_config["context_length"] = int(config["dataset"]["context_length"])
    return {
        "model": model_config,
        "attention": config.get("attention", {}),
        "relational_graph": config.get("relational_graph", {}),
        "distance": config.get("distance", {}),
    }


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    training = config["training"]
    if training["optimizer"].lower() != "adamw":
        raise ValueError("only AdamW is supported for ERGT-v1 trainer")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training.get("betas", [0.9, 0.95])),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )


def configure_torch_runtime(config: dict[str, Any], device: torch.device) -> None:
    runtime = config.get("runtime", {})
    training = config.get("training", {})
    matmul_precision = str(
        runtime.get("float32_matmul_precision", training.get("float32_matmul_precision", "high"))
    )
    if matmul_precision in {"highest", "high", "medium"}:
        torch.set_float32_matmul_precision(matmul_precision)
    if device.type == "cuda":
        allow_tf32 = bool(runtime.get("allow_tf32", training.get("allow_tf32", True)))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        cudnn_benchmark = bool(
            runtime.get("cudnn_benchmark", training.get("cudnn_benchmark", True))
        )
        torch.backends.cudnn.benchmark = cudnn_benchmark


def precision_config(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    training = config.get("training", {})
    requested = str(training.get("precision", "tf32")).lower()
    if device.type != "cuda" or requested in {"fp32", "float32", "tf32"}:
        return {
            "mode": requested,
            "use_autocast": False,
            "autocast_dtype": torch.float32,
            "use_grad_scaler": False,
        }
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested in {"bf16", "bfloat16"} and torch.cuda.is_bf16_supported():
        return {
            "mode": "bf16",
            "use_autocast": True,
            "autocast_dtype": torch.bfloat16,
            "use_grad_scaler": False,
        }
    if requested in {"bf16", "bfloat16"}:
        requested = "fp16"
    if requested in {"fp16", "float16"}:
        return {
            "mode": "fp16",
            "use_autocast": True,
            "autocast_dtype": torch.float16,
            "use_grad_scaler": True,
        }
    raise ValueError(
        "training.precision must be one of fp32, tf32, bf16/bfloat16, or fp16/float16"
    )


def maybe_compile_model(
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    compile_config = config.get("runtime", {}).get(
        "torch_compile",
    ) or config.get("training", {}).get(
        "torch_compile",
        False,
    )
    if device.type != "cuda" or not compile_config:
        return model
    if not hasattr(torch, "compile"):
        return model
    mode = "default"
    fullgraph = False
    if isinstance(compile_config, dict):
        mode = str(compile_config.get("mode", mode))
        fullgraph = bool(compile_config.get("fullgraph", fullgraph))
    return torch.compile(model, mode=mode, fullgraph=fullgraph)


def dataloader_runtime_kwargs(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    training = config.get("training", {})
    num_workers = int(training.get("num_workers", training.get("dataloader_num_workers", 0)))
    pin_memory = bool(training.get("pin_memory", device.type == "cuda" and num_workers > 0))
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(training.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(training.get("prefetch_factor", 2))
    return kwargs


def runtime_summary(
    config: dict[str, Any],
    device: torch.device,
    precision: dict[str, Any],
) -> dict[str, Any]:
    training = config.get("training", {})
    runtime = config.get("runtime", {})
    return {
        "precision": precision["mode"],
        "use_autocast": bool(precision["use_autocast"]),
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32)
        if device.type == "cuda"
        else False,
        "float32_matmul_precision": str(
            runtime.get(
                "float32_matmul_precision",
                training.get("float32_matmul_precision", "high"),
            )
        ),
        "torch_compile": bool(
            runtime.get("torch_compile") or training.get("torch_compile", False)
        ),
        "num_workers": int(
            training.get("num_workers", training.get("dataloader_num_workers", 0))
        ),
        "pin_memory": bool(training.get("pin_memory", device.type == "cuda")),
    }


@torch.no_grad()
def evaluate(
    model: ERGTV1,
    data_loader: DataLoader,
    device: torch.device,
    *,
    log_geometry: bool,
    precision: dict[str, Any] | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    precision = precision or precision_config({}, device)
    total_loss = 0.0
    total_tokens = 0
    geometry_records: list[dict[str, Any]] = []
    for batch_index, (input_ids, targets) in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        input_ids = input_ids.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        with torch.autocast(
            device_type=device.type,
            dtype=precision["autocast_dtype"],
            enabled=precision["use_autocast"],
        ):
            outputs = model(
                input_ids,
                targets=targets,
                return_geometry_diagnostics=log_geometry,
            )
        loss = outputs["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite validation loss: {loss.item()}")
        tokens = targets.numel()
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens
        if log_geometry:
            geometry_records.append(
                aggregate_attention_diagnostics(outputs["geometry_diagnostics"])
            )

    if total_tokens == 0:
        raise ValueError("validation loader produced no tokens")

    result: dict[str, Any] = {"validation_loss": total_loss / total_tokens}
    if log_geometry:
        result["geometry"] = average_nested_metrics(geometry_records)
    return result


def save_checkpoint(
    path: Path,
    model: ERGTV1,
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


def average_nested_metrics(items: list[Any]) -> Any:
    if not items:
        return {}
    return _average_values(items)


def _average_values(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, dict):
        keys = sorted({key for value in values for key in value.keys()})
        return {
            key: _average_values([value[key] for value in values if key in value]) for key in keys
        }
    if isinstance(first, list):
        return first
    if isinstance(first, str | bool) or first is None:
        return first
    if isinstance(first, int | float):
        finite_values = [float(value) for value in values if math.isfinite(float(value))]
        if not finite_values:
            return math.nan
        return sum(finite_values) / len(finite_values)
    return first


if __name__ == "__main__":
    main()
