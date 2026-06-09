"""Distance and geometry metrics for ERGT Phase 2."""

from __future__ import annotations

import math
from typing import Any

import torch

from geometry.emergent_distance import make_random_distance_like, make_shuffled_distance


def distance_metrics(
    distance: torch.Tensor,
    *,
    attention_logits: torch.Tensor | None = None,
    neighborhood_k: list[int] | tuple[int, ...] = (4, 8, 16),
) -> dict[str, Any]:
    """Compute JSON-compatible metrics for a distance tensor."""
    _validate_distance(distance)
    finite_values = distance[torch.isfinite(distance)]
    if finite_values.numel() == 0:
        raise ValueError("distance contains no finite values for metric computation")

    metrics: dict[str, Any] = {
        "shape": list(distance.shape),
        "mean": _to_float(finite_values.mean()),
        "std": _to_float(finite_values.std(unbiased=False)),
        "variance": _to_float(finite_values.var(unbiased=False)),
        "min": _to_float(finite_values.min()),
        "max": _to_float(finite_values.max()),
        "entropy": distance_entropy(distance),
        "diagonal": diagonal_statistics(distance),
        "off_diagonal": off_diagonal_statistics(distance),
        "neighborhood": {},
    }

    if attention_logits is not None:
        metrics["attention_logit_correlation"] = tensor_correlation(distance, attention_logits)

    for k in neighborhood_k:
        metrics["neighborhood"][str(k)] = neighborhood_statistics(distance, k)

    return metrics


def distance_metrics_with_controls(
    distance: torch.Tensor,
    *,
    attention_logits: torch.Tensor | None = None,
    neighborhood_k: list[int] | tuple[int, ...] = (4, 8, 16),
    seed: int = 1337,
) -> dict[str, Any]:
    generator = torch.Generator(device=distance.device)
    generator.manual_seed(seed)

    random_distance = make_random_distance_like(distance, generator=generator)
    shuffled_distance = make_shuffled_distance(distance, generator=generator)

    return {
        "real_D": distance_metrics(
            distance,
            attention_logits=attention_logits,
            neighborhood_k=neighborhood_k,
        ),
        "controls": {
            "random_D": distance_metrics(
                random_distance,
                attention_logits=attention_logits,
                neighborhood_k=neighborhood_k,
            ),
            "shuffled_D": distance_metrics(
                shuffled_distance,
                attention_logits=attention_logits,
                neighborhood_k=neighborhood_k,
            ),
        },
    }


def distance_entropy(distance: torch.Tensor) -> float:
    """Entropy over `exp(-D)` affinities."""
    finite_values = distance[torch.isfinite(distance)]
    if finite_values.numel() == 0:
        return math.nan
    affinities = torch.exp(-finite_values)
    total = affinities.sum()
    if total <= 0:
        return 0.0
    probabilities = affinities / total
    entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum()
    return _to_float(entropy)


def diagonal_statistics(distance: torch.Tensor) -> dict[str, float]:
    diagonal = torch.diagonal(distance, dim1=-2, dim2=-1)
    return _basic_stats(diagonal)


def off_diagonal_statistics(distance: torch.Tensor) -> dict[str, float]:
    off_diagonal = distance[_offdiag_mask(distance)]
    return _basic_stats(off_diagonal)


def neighborhood_statistics(distance: torch.Tensor, k: int) -> dict[str, float]:
    if k <= 0:
        raise ValueError("k must be positive")
    _validate_distance(distance)
    sequence_length = distance.size(-1)
    effective_k = min(k, max(sequence_length - 1, 1))

    masked = distance.masked_fill(_diagonal_mask(distance), torch.inf)
    finite_neighbor_counts = torch.isfinite(masked).sum(dim=-1).to(dtype=torch.float32)
    topk = torch.topk(masked, k=effective_k, dim=-1, largest=False).values
    finite_topk = torch.isfinite(topk).to(dtype=torch.float32)
    return {
        "k": float(effective_k),
        "mean_available_neighbors": _to_float(finite_neighbor_counts.mean()),
        "mean_finite_topk": _to_float(finite_topk.sum(dim=-1).mean()),
        "mean_topk_distance": _nan_safe_mean(topk),
    }


def neighborhood_overlap(distance_a: torch.Tensor, distance_b: torch.Tensor, k: int) -> float:
    _validate_distance(distance_a)
    _validate_distance(distance_b)
    if distance_a.shape != distance_b.shape:
        raise ValueError("distance_a and distance_b must have the same shape")
    if k <= 0:
        raise ValueError("k must be positive")

    sequence_length = distance_a.size(-1)
    effective_k = min(k, max(sequence_length - 1, 1))
    a = distance_a.masked_fill(_diagonal_mask(distance_a), torch.inf)
    b = distance_b.masked_fill(_diagonal_mask(distance_b), torch.inf)
    idx_a = torch.topk(a, k=effective_k, dim=-1, largest=False).indices
    idx_b = torch.topk(b, k=effective_k, dim=-1, largest=False).indices

    overlaps = []
    for neighbor_a, neighbor_b in zip(
        idx_a.reshape(-1, effective_k), idx_b.reshape(-1, effective_k), strict=True
    ):
        set_a = set(int(value) for value in neighbor_a.detach().cpu().tolist())
        set_b = set(int(value) for value in neighbor_b.detach().cpu().tolist())
        overlaps.append(len(set_a & set_b) / effective_k)
    return sum(overlaps) / len(overlaps)


def tensor_correlation(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> float:
    if tensor_a.shape != tensor_b.shape:
        if tensor_a.size(1) == 1 and tensor_b.dim() == 4:
            tensor_a = tensor_a.expand(tensor_b.shape)
        elif tensor_b.size(1) == 1 and tensor_a.dim() == 4:
            tensor_b = tensor_b.expand(tensor_a.shape)
        else:
            raise ValueError("tensor shapes must match or be head-broadcastable")

    mask = torch.isfinite(tensor_a) & torch.isfinite(tensor_b)
    a = tensor_a[mask]
    b = tensor_b[mask]
    if a.numel() < 2:
        return math.nan

    a = a - a.mean()
    b = b - b.mean()
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if denominator <= 0:
        return math.nan
    return _to_float((a * b).sum() / denominator)


def _basic_stats(values: torch.Tensor) -> dict[str, float]:
    finite_values = values[torch.isfinite(values)]
    if finite_values.numel() == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": _to_float(finite_values.mean()),
        "std": _to_float(finite_values.std(unbiased=False)),
        "min": _to_float(finite_values.min()),
        "max": _to_float(finite_values.max()),
    }


def _nan_safe_mean(values: torch.Tensor) -> float:
    finite_values = values[torch.isfinite(values)]
    if finite_values.numel() == 0:
        return math.nan
    return _to_float(finite_values.mean())


def _validate_distance(distance: torch.Tensor) -> None:
    if distance.dim() != 4:
        raise ValueError("distance must have shape [batch, heads, sequence, sequence]")
    if distance.size(-1) != distance.size(-2):
        raise ValueError("distance must be square in the last two dimensions")


def _diagonal_mask(tensor: torch.Tensor) -> torch.Tensor:
    sequence_length = tensor.size(-1)
    return torch.eye(sequence_length, dtype=torch.bool, device=tensor.device).view(
        1,
        1,
        sequence_length,
        sequence_length,
    )


def _offdiag_mask(tensor: torch.Tensor) -> torch.Tensor:
    return ~_diagonal_mask(tensor).expand_as(tensor)


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
