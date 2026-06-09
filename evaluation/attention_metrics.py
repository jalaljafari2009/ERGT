"""Attention and geometry diagnostics for ERGT Phase 3."""

from __future__ import annotations

import math
from typing import Any

import torch


def attention_metrics(
    attention_weights: torch.Tensor,
    *,
    qk_logits: torch.Tensor | None = None,
    distance: torch.Tensor | None = None,
    alpha: float | torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute JSON-compatible metrics for attention and optional geometry terms."""
    _validate_attention(attention_weights)
    finite_weights = attention_weights[torch.isfinite(attention_weights)]
    if finite_weights.numel() == 0:
        raise ValueError("attention_weights contains no finite values")

    metrics: dict[str, Any] = {
        "shape": list(attention_weights.shape),
        "attention_entropy": attention_entropy(attention_weights),
        "mean_max_probability": mean_max_probability(attention_weights),
        "attention_sparsity": {
            "0.01": attention_sparsity(attention_weights, 0.01),
            "0.001": attention_sparsity(attention_weights, 0.001),
        },
        "mean": _to_float(finite_weights.mean()),
        "std": _to_float(finite_weights.std(unbiased=False)),
        "min": _to_float(finite_weights.min()),
        "max": _to_float(finite_weights.max()),
    }

    if qk_logits is not None and distance is not None and alpha is not None:
        metrics["geometry"] = geometry_contribution_metrics(qk_logits, distance, alpha)

    return metrics


def geometry_contribution_metrics(
    qk_logits: torch.Tensor,
    distance: torch.Tensor,
    alpha: float | torch.Tensor,
) -> dict[str, float]:
    if qk_logits.shape != distance.shape:
        if distance.dim() == 4 and distance.size(1) == 1:
            distance = distance.expand(qk_logits.shape)
        else:
            raise ValueError("qk_logits and distance must match or distance must be head-shared")

    alpha_tensor = torch.as_tensor(alpha, dtype=qk_logits.dtype, device=qk_logits.device)
    finite_qk = qk_logits[torch.isfinite(qk_logits)]
    finite_distance = distance[torch.isfinite(distance)]
    mean_abs_qk = finite_qk.abs().mean() if finite_qk.numel() else torch.tensor(float("nan"))

    if finite_distance.numel() == 0 or torch.all(alpha_tensor == 0):
        mean_abs_geo = torch.tensor(0.0, dtype=qk_logits.dtype, device=qk_logits.device)
    else:
        mean_abs_geo = (alpha_tensor * finite_distance).abs().mean()

    return {
        "alpha": _to_float(alpha_tensor),
        "mean_abs_qk": _to_float(mean_abs_qk),
        "mean_abs_geo": _to_float(mean_abs_geo),
        "geo_to_qk_ratio": _to_float(mean_abs_geo / (mean_abs_qk.abs() + 1e-12)),
        "distance_mean": _to_float(finite_distance.mean()) if finite_distance.numel() else math.nan,
        "distance_std": _to_float(finite_distance.std(unbiased=False))
        if finite_distance.numel()
        else math.nan,
    }


def aggregate_attention_diagnostics(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-layer GeoAttention diagnostics returned by `ERGTV1`."""
    if not diagnostics:
        return {}

    layer_metrics: dict[str, Any] = {}
    for layer_index, item in enumerate(diagnostics):
        diag = item.get("diagnostics", {})
        attention_weights = item.get("attention_weights")
        layer_record = dict(diag)
        if attention_weights is not None:
            layer_record.update(attention_metrics(attention_weights))
        layer_metrics[f"layer_{layer_index}"] = _jsonify(layer_record)

    numeric_keys = sorted(
        {
            key
            for layer in layer_metrics.values()
            for key, value in layer.items()
            if isinstance(value, int | float) and math.isfinite(float(value))
        }
    )
    summary = {
        key: sum(float(layer[key]) for layer in layer_metrics.values() if key in layer)
        / max(sum(1 for layer in layer_metrics.values() if key in layer), 1)
        for key in numeric_keys
    }
    return {"layers": layer_metrics, "summary": summary}


def attention_entropy(attention_weights: torch.Tensor) -> float:
    _validate_attention(attention_weights)
    weights = attention_weights.clamp_min(0)
    entropy = -(weights * torch.log(weights + 1e-12)).sum(dim=-1)
    return _to_float(entropy.mean())


def mean_max_probability(attention_weights: torch.Tensor) -> float:
    _validate_attention(attention_weights)
    return _to_float(attention_weights.max(dim=-1).values.mean())


def attention_sparsity(attention_weights: torch.Tensor, threshold: float) -> float:
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    _validate_attention(attention_weights)
    finite_weights = attention_weights[torch.isfinite(attention_weights)]
    if finite_weights.numel() == 0:
        return math.nan
    return _to_float((finite_weights < threshold).float().mean())


def _validate_attention(attention_weights: torch.Tensor) -> None:
    if attention_weights.dim() != 4:
        raise ValueError("attention_weights must have shape [batch, heads, sequence, sequence]")
    if attention_weights.size(-1) != attention_weights.size(-2):
        raise ValueError("attention_weights must be square in the last two dimensions")


def _jsonify(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _to_float(value)
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    return value


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
