"""Lightweight progress logging for long ERGT experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_for_json(record), sort_keys=True) + "\n")


def gpu_memory_snapshot(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"gpu_memory_gb": None, "gpu_peak_memory_gb": None}
    return {
        "gpu_memory_gb": torch.cuda.memory_allocated(device) / 1024**3,
        "gpu_peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
    }


def geometry_progress_fields(geometry: dict[str, Any] | None) -> dict[str, Any]:
    if not geometry:
        return {}
    summary = geometry.get("summary", geometry)
    if not isinstance(summary, dict):
        return {}

    fields = {
        "alpha_effective": summary.get("alpha"),
        "alpha_warmup_factor": summary.get("alpha_warmup_factor"),
        "target_alpha": summary.get("target_alpha"),
        "geo_to_qk_ratio": summary.get("geo_to_qk_ratio"),
        "distance_mean": summary.get("distance_mean"),
        "distance_std": summary.get("distance_std"),
        "attention_entropy": summary.get("attention_entropy"),
        "mean_max_probability": summary.get("mean_max_probability"),
    }
    sparsity = summary.get("attention_sparsity")
    if isinstance(sparsity, dict):
        fields["attention_sparsity_0_01"] = sparsity.get("0.01")
        fields["attention_sparsity_0_001"] = sparsity.get("0.001")
    return {key: value for key, value in fields.items() if value is not None}


def format_progress_line(record: dict[str, Any]) -> str:
    condition = str(record.get("condition", "run"))
    parts = [
        f"[{condition}]",
        f"step={record.get('step')}",
    ]
    decision = record.get("alpha_decision")
    if decision:
        parts.append(f"decision={decision}")
    for key, label, precision in [
        ("train_loss", "train", 4),
        ("validation_loss", "val", 4),
        ("best_validation_loss", "best", 4),
        ("perplexity", "ppl", 1),
        ("alpha_effective", "alpha", 4),
        ("alpha_next", "a_next", 4),
        ("alpha_delta", "d_alpha", 4),
        ("adaptive_score", "score", 6),
        ("adaptive_slope_gain", "slope", 6),
        ("adaptive_advantage", "adv", 6),
        ("geo_to_qk_ratio", "geo/qk", 3),
        ("geo_qk_risk", "gRisk", 3),
        ("attention_entropy", "ent", 3),
        ("entropy_risk", "eRisk", 3),
        ("mean_max_probability", "maxp", 3),
        ("max_probability_risk", "pRisk", 3),
        ("grad_norm", "grad", 3),
        ("tokens_per_second", "tok/s", 0),
        ("gpu_memory_gb", "gpu", 2),
        ("elapsed_minutes", "min", 1),
    ]:
        value = as_float_or_none(record.get(key))
        if value is not None:
            parts.append(f"{label}={value:.{precision}f}")
    status = record.get("status")
    if status:
        parts.append(f"status={status}")
    return " ".join(parts)


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return sanitize_for_json(float(value.detach().cpu().item()))
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
