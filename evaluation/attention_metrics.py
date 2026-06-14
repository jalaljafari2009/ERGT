"""Attention and geometry diagnostics for ERGT Phase 3."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

ATTENTION_RIGIDITY_FIELDS = [
    "attention_entropy_normalized",
    "attention_entropy_drop",
    "valid_mean_max_probability",
    "valid_attention_sparsity_0_01",
    "valid_attention_sparsity_0_001",
    "head_attention_diversity",
    "head_collapse_risk",
    "geo_qk_risk",
    "geometry_takeover_score",
    "entropy_risk",
    "max_probability_risk",
    "valid_sparsity_risk",
    "rigidity_risk",
    "collapse_risk",
    "severe_attention_collapse_detected",
]


@dataclass(frozen=True)
class AttentionRigidityConfig:
    min_normalized_entropy: float = 0.35
    max_probability_warning: float = 0.85
    valid_sparsity_warning: float = 0.9
    min_head_diversity: float = 0.05
    geometry_takeover_warning_ratio: float = 0.15
    geometry_takeover_saturation_ratio: float = 0.4
    severe_collapse_risk: float = 0.95

    def validate(self) -> None:
        if not 0 < self.min_normalized_entropy <= 1:
            raise ValueError("min_normalized_entropy must be in (0, 1]")
        if not 0 < self.max_probability_warning <= 1:
            raise ValueError("max_probability_warning must be in (0, 1]")
        if not 0 <= self.valid_sparsity_warning <= 1:
            raise ValueError("valid_sparsity_warning must be in [0, 1]")
        if not 0 <= self.min_head_diversity <= 1:
            raise ValueError("min_head_diversity must be in [0, 1]")
        if self.geometry_takeover_warning_ratio < 0:
            raise ValueError("geometry_takeover_warning_ratio must be non-negative")
        if self.geometry_takeover_saturation_ratio <= self.geometry_takeover_warning_ratio:
            raise ValueError(
                "geometry_takeover_saturation_ratio must be greater than warning ratio"
            )
        if not 0 < self.severe_collapse_risk <= 1:
            raise ValueError("severe_collapse_risk must be in (0, 1]")


def attention_metrics(
    attention_weights: torch.Tensor,
    *,
    qk_logits: torch.Tensor | None = None,
    distance: torch.Tensor | None = None,
    alpha: float | torch.Tensor | None = None,
    geo_to_qk_ratio: float | None = None,
    rigidity_config: AttentionRigidityConfig | None = None,
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
    metrics.update(
        attention_rigidity_metrics(
            attention_weights,
            geo_to_qk_ratio=geo_to_qk_ratio,
            config=rigidity_config,
        )
    )

    if qk_logits is not None and distance is not None and alpha is not None:
        metrics["geometry"] = geometry_contribution_metrics(qk_logits, distance, alpha)

    return metrics


def attention_rigidity_metrics(
    attention_weights: torch.Tensor,
    *,
    geo_to_qk_ratio: float | None = None,
    config: AttentionRigidityConfig | None = None,
) -> dict[str, float | bool]:
    """Measure pressure toward rigid or collapsed attention.

    The risk scores are diagnostic-only. They do not change attention, loss, or
    gradients. Scores are bounded in [0, 1], where larger values mean stronger
    restraint evidence for later adaptive controllers.
    """

    _validate_attention(attention_weights)
    config = config or AttentionRigidityConfig()
    config.validate()
    probabilities, valid_mask = _causal_row_probabilities(attention_weights)
    valid_counts = valid_mask.sum(dim=-1)
    entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum(dim=-1)
    normalizable_rows = valid_counts > 1
    if bool(normalizable_rows.any().item()):
        entropy_denominator = torch.log(valid_counts[normalizable_rows].to(probabilities.dtype))
        normalized_entropy = entropy[normalizable_rows] / entropy_denominator.clamp_min(1e-12)
        normalized_entropy_value = _to_float(normalized_entropy.mean().clamp(0.0, 1.0))
    else:
        normalized_entropy_value = 1.0

    valid_mean_max = _to_float(probabilities.max(dim=-1).values[valid_mask.any(dim=-1)].mean())
    valid_sparsity_0_01 = _valid_attention_sparsity(probabilities, valid_mask, 0.01)
    valid_sparsity_0_001 = _valid_attention_sparsity(probabilities, valid_mask, 0.001)
    head_diversity = head_attention_diversity(attention_weights)
    entropy_drop = _bounded(1.0 - normalized_entropy_value)
    entropy_risk = _bounded(
        (config.min_normalized_entropy - normalized_entropy_value)
        / max(config.min_normalized_entropy, 1e-8)
    )
    max_probability_risk = _bounded(
        (valid_mean_max - config.max_probability_warning)
        / max(1.0 - config.max_probability_warning, 1e-8)
    )
    valid_sparsity_risk = _bounded(
        (valid_sparsity_0_01 - config.valid_sparsity_warning)
        / max(1.0 - config.valid_sparsity_warning, 1e-8)
    )
    head_collapse_risk = _bounded(
        (config.min_head_diversity - head_diversity)
        / max(config.min_head_diversity, 1e-8)
    )
    geo_ratio = max(float(geo_to_qk_ratio or 0.0), 0.0)
    geometry_takeover_score = _bounded(geo_ratio / config.geometry_takeover_saturation_ratio)
    geo_qk_risk = _bounded(
        (geo_ratio - config.geometry_takeover_warning_ratio)
        / (
            config.geometry_takeover_saturation_ratio
            - config.geometry_takeover_warning_ratio
        )
    )
    rigidity_risk = max(
        entropy_risk,
        max_probability_risk,
        valid_sparsity_risk,
        head_collapse_risk,
    )
    collapse_risk = max(rigidity_risk, geo_qk_risk)

    return {
        "attention_entropy_normalized": normalized_entropy_value,
        "attention_entropy_drop": entropy_drop,
        "valid_mean_max_probability": valid_mean_max,
        "valid_attention_sparsity_0_01": valid_sparsity_0_01,
        "valid_attention_sparsity_0_001": valid_sparsity_0_001,
        "head_attention_diversity": head_diversity,
        "head_collapse_risk": head_collapse_risk,
        "geo_qk_risk": geo_qk_risk,
        "geometry_takeover_score": geometry_takeover_score,
        "entropy_risk": entropy_risk,
        "max_probability_risk": max_probability_risk,
        "valid_sparsity_risk": valid_sparsity_risk,
        "rigidity_risk": rigidity_risk,
        "collapse_risk": collapse_risk,
        "severe_attention_collapse_detected": (
            collapse_risk >= config.severe_collapse_risk
        ),
    }


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
    attention_layers = []
    for layer_index, item in enumerate(diagnostics):
        diag = item.get("diagnostics", {})
        attention_weights = item.get("attention_weights")
        layer_record = dict(diag)
        if attention_weights is not None:
            layer_record.update(
                attention_metrics(
                    attention_weights,
                    geo_to_qk_ratio=_optional_float(diag.get("geo_to_qk_ratio")),
                )
            )
            attention_layers.append(attention_weights)
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
    if attention_layers:
        layer_diversity = layer_attention_diversity(attention_layers)
        layer_collapse_risk = _bounded((0.05 - layer_diversity) / 0.05)
        summary["layer_attention_diversity"] = layer_diversity
        summary["layer_collapse_risk"] = layer_collapse_risk
        summary["rigidity_risk"] = max(
            float(summary.get("rigidity_risk", 0.0)),
            layer_collapse_risk,
        )
        summary["collapse_risk"] = max(
            float(summary.get("collapse_risk", 0.0)),
            layer_collapse_risk,
        )
        summary["severe_attention_collapse_detected"] = float(
            summary.get("collapse_risk", 0.0)
        ) >= AttentionRigidityConfig().severe_collapse_risk
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


def head_attention_diversity(attention_weights: torch.Tensor) -> float:
    """Return mean pairwise head dissimilarity in [0, 1]."""

    _validate_attention(attention_weights)
    if attention_weights.size(1) < 2:
        return 1.0
    vectors = attention_weights.clamp_min(0).mean(dim=0).reshape(attention_weights.size(1), -1)
    return _mean_pairwise_dissimilarity(vectors)


def layer_attention_diversity(attention_layers: list[torch.Tensor]) -> float:
    """Return mean pairwise layer dissimilarity in [0, 1]."""

    if len(attention_layers) < 2:
        return 1.0
    first_shape = attention_layers[0].shape
    if any(layer.shape != first_shape for layer in attention_layers):
        return math.nan
    vectors = torch.stack([layer.clamp_min(0).reshape(-1) for layer in attention_layers])
    return _mean_pairwise_dissimilarity(vectors)


def _validate_attention(attention_weights: torch.Tensor) -> None:
    if attention_weights.dim() != 4:
        raise ValueError("attention_weights must have shape [batch, heads, sequence, sequence]")
    if attention_weights.size(-1) != attention_weights.size(-2):
        raise ValueError("attention_weights must be square in the last two dimensions")


def _causal_row_probabilities(
    attention_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_length = attention_weights.size(-1)
    valid_mask = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=attention_weights.device,
    ).tril()
    valid_mask = valid_mask.view(1, 1, sequence_length, sequence_length).expand_as(
        attention_weights
    )
    weights = torch.where(
        valid_mask,
        attention_weights.clamp_min(0),
        torch.zeros_like(attention_weights),
    )
    row_sum = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return weights / row_sum, valid_mask


def _valid_attention_sparsity(
    probabilities: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float,
) -> float:
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    values = probabilities[valid_mask & torch.isfinite(probabilities)]
    if values.numel() == 0:
        return math.nan
    return _to_float((values < threshold).float().mean())


def _mean_pairwise_dissimilarity(vectors: torch.Tensor) -> float:
    if vectors.size(0) < 2:
        return 1.0
    normalized = torch.nn.functional.normalize(vectors, p=2, dim=-1, eps=1e-12)
    similarities = normalized @ normalized.transpose(0, 1)
    pair_mask = torch.ones(
        similarities.size(0),
        similarities.size(1),
        dtype=torch.bool,
        device=similarities.device,
    ).triu(diagonal=1)
    pair_values = similarities[pair_mask]
    if pair_values.numel() == 0:
        return 1.0
    return _to_float((1.0 - pair_values).clamp(0.0, 1.0).mean())


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bounded(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return min(max(float(value), 0.0), 1.0)


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


def attention_rigidity_config_dict(
    config: AttentionRigidityConfig | None = None,
) -> dict[str, float]:
    return asdict(config or AttentionRigidityConfig())
