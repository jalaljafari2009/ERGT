"""Graph metrics for ERGT relational graph analysis."""

from __future__ import annotations

import math
from typing import Any

import torch

from layers.relational_graph import make_random_graph_like, make_shuffled_graph


def graph_metrics(
    graph: torch.Tensor,
    *,
    exclude_diagonal: bool = True,
    sparsity_thresholds: list[float] | tuple[float, ...] = (0.5, 0.75, 0.9, 0.95),
) -> dict[str, Any]:
    """Compute JSON-compatible metrics for `graph`.

    Args:
        graph: Tensor with shape `[batch, heads, sequence, sequence]`.
        exclude_diagonal: Whether off-diagonal metrics should ignore self edges.
        sparsity_thresholds: Thresholds for effective sparsity and degree metrics.
    """
    _validate_graph(graph)

    values = _selected_values(graph, exclude_diagonal=exclude_diagonal)
    finite_values = values[torch.isfinite(values)]
    if finite_values.numel() == 0:
        raise ValueError("graph contains no finite values for metric computation")

    metrics: dict[str, Any] = {
        "shape": list(graph.shape),
        "exclude_diagonal": exclude_diagonal,
        "mean": _to_float(finite_values.mean()),
        "variance": _to_float(finite_values.var(unbiased=False)),
        "min": _to_float(finite_values.min()),
        "max": _to_float(finite_values.max()),
        "entropy": _entropy(finite_values),
        "diagonal_dominance": diagonal_dominance(graph),
        "sparsity": {},
        "degree_distribution": {},
    }

    for threshold in sparsity_thresholds:
        key = _threshold_key(threshold)
        metrics["sparsity"][key] = sparsity(graph, threshold, exclude_diagonal=exclude_diagonal)
        metrics["degree_distribution"][key] = degree_distribution(graph, threshold)

    return metrics


def graph_metrics_with_controls(
    graph: torch.Tensor,
    *,
    exclude_diagonal: bool = True,
    sparsity_thresholds: list[float] | tuple[float, ...] = (0.5, 0.75, 0.9, 0.95),
    seed: int = 1337,
) -> dict[str, Any]:
    """Compute real graph metrics plus random and shuffled controls."""
    generator = torch.Generator(device=graph.device)
    generator.manual_seed(seed)

    random_graph = make_random_graph_like(graph, generator=generator)
    shuffled_graph = make_shuffled_graph(graph, generator=generator)

    return {
        "real_W": graph_metrics(
            graph,
            exclude_diagonal=exclude_diagonal,
            sparsity_thresholds=sparsity_thresholds,
        ),
        "controls": {
            "random_W": graph_metrics(
                random_graph,
                exclude_diagonal=exclude_diagonal,
                sparsity_thresholds=sparsity_thresholds,
            ),
            "shuffled_W": graph_metrics(
                shuffled_graph,
                exclude_diagonal=exclude_diagonal,
                sparsity_thresholds=sparsity_thresholds,
            ),
        },
    }


def sparsity(graph: torch.Tensor, threshold: float, *, exclude_diagonal: bool = True) -> float:
    values = _selected_values(graph, exclude_diagonal=exclude_diagonal)
    finite_values = values[torch.isfinite(values)]
    if finite_values.numel() == 0:
        return math.nan
    return _to_float((finite_values < threshold).float().mean())


def degree_distribution(graph: torch.Tensor, threshold: float) -> dict[str, float]:
    _validate_graph(graph)
    finite_graph = torch.nan_to_num(graph, nan=-math.inf)
    adjacency = finite_graph >= threshold
    degrees = adjacency.sum(dim=-1).to(dtype=torch.float32)
    return {
        "mean": _to_float(degrees.mean()),
        "variance": _to_float(degrees.var(unbiased=False)),
        "min": _to_float(degrees.min()),
        "max": _to_float(degrees.max()),
    }


def diagonal_dominance(graph: torch.Tensor) -> float:
    _validate_graph(graph)
    sequence_length = graph.size(-1)
    diagonal_mask = torch.eye(sequence_length, dtype=torch.bool, device=graph.device).view(
        1,
        1,
        sequence_length,
        sequence_length,
    )
    diagonal = graph[diagonal_mask.expand_as(graph)]
    off_diagonal = graph[~diagonal_mask.expand_as(graph)]

    diagonal = diagonal[torch.isfinite(diagonal)]
    off_diagonal = off_diagonal[torch.isfinite(off_diagonal)]
    if diagonal.numel() == 0 or off_diagonal.numel() == 0:
        return math.nan
    return _to_float(diagonal.mean() / (off_diagonal.mean().abs() + 1e-12))


def layer_to_layer_similarity(graph_a: torch.Tensor, graph_b: torch.Tensor) -> dict[str, float]:
    _validate_graph(graph_a)
    _validate_graph(graph_b)
    if graph_a.shape != graph_b.shape:
        raise ValueError("graph_a and graph_b must have the same shape")

    a = torch.nan_to_num(graph_a.reshape(graph_a.size(0), graph_a.size(1), -1), nan=0.0)
    b = torch.nan_to_num(graph_b.reshape(graph_b.size(0), graph_b.size(1), -1), nan=0.0)
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    frobenius = torch.linalg.vector_norm(a - b, dim=-1)
    return {
        "cosine_mean": _to_float(cosine.mean()),
        "cosine_min": _to_float(cosine.min()),
        "frobenius_mean": _to_float(frobenius.mean()),
    }


def _selected_values(graph: torch.Tensor, *, exclude_diagonal: bool) -> torch.Tensor:
    _validate_graph(graph)
    if not exclude_diagonal:
        return graph.reshape(-1)

    sequence_length = graph.size(-1)
    diagonal_mask = torch.eye(sequence_length, dtype=torch.bool, device=graph.device).view(
        1,
        1,
        sequence_length,
        sequence_length,
    )
    return graph[~diagonal_mask.expand_as(graph)]


def _entropy(values: torch.Tensor) -> float:
    positive_values = values[values > 0]
    if positive_values.numel() == 0:
        return 0.0
    probabilities = positive_values / positive_values.sum()
    entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum()
    return _to_float(entropy)


def _validate_graph(graph: torch.Tensor) -> None:
    if graph.dim() != 4:
        raise ValueError("graph must have shape [batch, heads, sequence, sequence]")
    if graph.size(-1) != graph.size(-2):
        raise ValueError("graph must be square in the last two dimensions")


def _threshold_key(threshold: float) -> str:
    return f"{threshold:g}"


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
