"""Train ERGT-v1 with adaptive competitive alpha control."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.attention_metrics import aggregate_attention_diagnostics  # noqa: E402
from experiments.adaptive_alpha import (  # noqa: E402
    AdaptiveAlphaConfig,
    AdaptiveAlphaController,
    AlphaObservation,
    load_reference_progress,
    reference_loss_for_step,
    set_model_fixed_alpha,
)
from experiments.data_utils import load_json, save_json  # noqa: E402
from experiments.progress_logging import (  # noqa: E402
    append_jsonl,
    format_progress_line,
    geometry_progress_fields,
    gpu_memory_snapshot,
)
from experiments.train_baseline import default_data_dir, learning_rate_for_step  # noqa: E402
from experiments.train_ergt_v1 import (  # noqa: E402
    build_ergt_model_config,
    build_optimizer,
    evaluate,
    load_prepared_or_raise,
    save_checkpoint,
    set_seed,
)
from models.ergt_v1 import ERGTV1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ERGT-v1 with adaptive alpha.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to ERGT-v1 adaptive-alpha training config.",
    )
    parser.add_argument(
        "--reference-progress",
        default=None,
        help="Optional baseline progress_log.jsonl used for slope/advantage feedback.",
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

    model_config = build_ergt_model_config(config, metadata)
    model = ERGTV1(model_config).to(device)
    optimizer = build_optimizer(model, config)

    adaptive_config = AdaptiveAlphaConfig(**config.get("adaptive_alpha", {}))
    controller = AdaptiveAlphaController(adaptive_config)
    set_model_fixed_alpha(model, controller.current_alpha)

    reference_progress_path = args.reference_progress or config.get("adaptive_alpha", {}).get(
        "reference_progress"
    )
    reference_progress = load_reference_progress(reference_progress_path)

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

    print(
        "Live adaptive telemetry fields: "
        "step train val best alpha a_next d_alpha decision score slope adv "
        "geo/qk gRisk ent eRisk maxp pRisk grad tok/s gpu min",
        flush=True,
    )

    save_json(output_dir / "model_summary.json", model.model_summary())

    max_steps = int(config["training"]["max_steps"])
    eval_interval = int(config["training"]["eval_interval"])
    checkpoint_interval = int(config["training"]["checkpoint_interval"])
    grad_clip = float(config["training"].get("grad_clip", 0.0))
    max_eval_batches = config["training"].get("max_eval_batches")
    max_eval_batches = int(max_eval_batches) if max_eval_batches is not None else None
    log_geometry = bool(config.get("logging", {}).get("log_geometry_diagnostics", True))

    train_log_path = output_dir / config["logging"].get("train_log", "train_log.jsonl")
    progress_log_path = output_dir / config["logging"].get(
        "progress_log", "progress_log.jsonl"
    )
    alpha_history_path = output_dir / config["logging"].get(
        "adaptive_alpha_log", "adaptive_alpha_log.jsonl"
    )
    best_val_loss = math.inf
    final_train_loss = math.inf
    start_time = time.perf_counter()
    tokens_processed = 0
    step = 0

    for path in [train_log_path, progress_log_path, alpha_history_path]:
        if path.exists():
            path.unlink()

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
            model.set_training_step(step)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids,
                targets=targets,
                return_geometry_diagnostics=log_geometry,
            )
            loss = outputs["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

            loss.backward()
            grad_norm = None
            if grad_clip > 0:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                    error_if_nonfinite=True,
                )
                grad_norm = float(grad_norm_tensor.detach().cpu().item())
            optimizer.step()

            batch_tokens = input_ids.numel()
            tokens_processed += batch_tokens
            final_train_loss = float(loss.item())
            elapsed_seconds = time.perf_counter() - start_time

            should_eval = step == 1 or step % eval_interval == 0 or step == max_steps
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

            if log_geometry:
                log_record["geometry"] = aggregate_attention_diagnostics(
                    outputs["geometry_diagnostics"]
                )

            if should_eval:
                val_result = evaluate(
                    model,
                    validation_loader,
                    device,
                    log_geometry=log_geometry,
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

                alpha_decision = controller.update(
                    AlphaObservation(
                        step=step,
                        validation_loss=val_loss,
                        reference_validation_loss=reference_loss_for_step(
                            reference_progress, step
                        ),
                        geo_to_qk_ratio=_optional_float(log_record.get("geo_to_qk_ratio")),
                        attention_entropy=_optional_float(log_record.get("attention_entropy")),
                        mean_max_probability=_optional_float(
                            log_record.get("mean_max_probability")
                        ),
                    )
                )
                set_model_fixed_alpha(model, alpha_decision.next_alpha)
                decision_row = asdict(alpha_decision)
                log_record["adaptive_alpha"] = decision_row
                log_record["alpha_previous"] = alpha_decision.previous_alpha
                log_record["alpha_next"] = alpha_decision.next_alpha
                log_record["alpha_delta"] = alpha_decision.alpha_delta
                log_record["adaptive_score"] = alpha_decision.score
                log_record["adaptive_slope_gain"] = alpha_decision.slope_gain
                log_record["adaptive_advantage"] = alpha_decision.advantage
                log_record["alpha_decision"] = alpha_decision.decision
                log_record["geo_qk_risk"] = alpha_decision.geo_qk_risk
                log_record["entropy_risk"] = alpha_decision.entropy_risk
                log_record["max_probability_risk"] = alpha_decision.max_probability_risk
                log_record["alpha_points_used"] = alpha_decision.points_used
                append_jsonl(alpha_history_path, decision_row)

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
        "attention": config.get("attention", {}),
        "distance": config.get("distance", {}),
        "adaptive_alpha": controller.summary(),
    }
    if log_geometry:
        results["geometry"] = final_eval.get("geometry")
    if torch.cuda.is_available() and device.type == "cuda":
        results["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)

    save_json(output_dir / config["logging"].get("results", "metrics.json"), results)
    save_json(output_dir / "adaptive_alpha_summary.json", controller.summary())
    save_checkpoint(output_dir / "checkpoints" / "last.pt", model, optimizer, step, config)
    print(json.dumps(results, indent=2, sort_keys=True))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


if __name__ == "__main__":
    main()
