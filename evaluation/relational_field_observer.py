"""Relational Field Observer for post-Phase-3 ERGT.

This module observes `H -> W -> D` without changing model computation. It is
the Phase 2 observer in the strengthened program, after measurement contracts
and strict W-level controls.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
import torch.nn.functional as F

from evaluation.distance_metrics import distance_entropy, neighborhood_overlap
from evaluation.graph_metrics import layer_to_layer_similarity
from geometry.emergent_distance import EmergentDistance
from layers.relational_graph import (
    RelationalGraph,
    make_random_graph_like,
    make_shuffled_graph,
    make_valid_edge_mask_like,
)

ObserverStatus = Literal["pass", "fail"]

FIELD_METRIC_KEYS: tuple[str, ...] = (
    "relational_entropy_mean",
    "local_relational_entropy_mean",
    "spectral_entropy_mean",
    "effective_rank_mean",
    "coherence_mean",
    "boundary_sharpness_mean",
    "distance_entropy",
)


def build_relational_field_observer_report(
    *,
    hidden_layers: list[torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    graph_config: dict[str, Any] | None = None,
    distance_config: dict[str, Any] | None = None,
    seed: int = 2027,
    neighborhood_k: int = 2,
    separation_threshold: float = 1e-3,
    stability_threshold: float = 0.75,
) -> dict[str, Any]:
    """Build a Relational Field Observer report.

    If `hidden_layers` is omitted, a deterministic structured hidden field is
    used as a smoke input. Real model runs should pass actual hidden states.
    """

    input_source = "provided_hidden_layers"
    if hidden_layers is None:
        hidden_layers, attention_mask = synthetic_structured_hidden_layers(seed=seed)
        input_source = "synthetic_structured_smoke"
    if not hidden_layers:
        raise ValueError("hidden_layers must not be empty")
    attention_mask = _attention_mask_or_ones(hidden_layers[0], attention_mask)

    graph_builder = RelationalGraph(_normalize_graph_config(graph_config))
    distance_builder = EmergentDistance(_distance_config(distance_config))
    generator = torch.Generator(device=hidden_layers[0].device)
    generator.manual_seed(seed)

    layer_reports: dict[str, Any] = {}
    real_graphs: list[torch.Tensor] = []
    real_distances: list[torch.Tensor] = []

    for layer_index, hidden_states in enumerate(hidden_layers):
        graph = graph_builder(hidden_states, attention_mask=attention_mask)
        valid_edge_mask = make_valid_edge_mask_like(graph, attention_mask=attention_mask)
        controls = {
            "random": make_random_graph_like(
                graph,
                generator=generator,
                valid_edge_mask=valid_edge_mask,
            ),
            "shuffled": make_shuffled_graph(
                graph,
                generator=generator,
                valid_edge_mask=valid_edge_mask,
            ),
        }

        graphs = {"real": graph, **controls}
        distances = {
            name: distance_builder(control_graph, attention_mask=attention_mask)
            for name, control_graph in graphs.items()
        }
        metrics = {
            name: field_metrics(
                hidden_states,
                control_graph,
                distances[name],
                valid_edge_mask,
            )
            for name, control_graph in graphs.items()
        }
        separation = separation_metrics(metrics["real"], metrics["random"], metrics["shuffled"])
        layer_reports[f"layer_{layer_index}"] = {
            "metrics": metrics,
            "separation": separation,
            "checks": {
                "real_separates_from_random": separation["real_vs_random"][
                    "aggregate_abs_delta"
                ]
                > separation_threshold,
                "real_separates_from_shuffled": separation["real_vs_shuffled"][
                    "aggregate_abs_delta"
                ]
                > separation_threshold,
                "real_not_uniform": metrics["real"]["valid_weight_variance"] > 0.0,
                "real_not_saturated": metrics["real"]["saturation_fraction"] < 0.95,
            },
        }
        real_graphs.append(graph)
        real_distances.append(distances["real"])

    stability = stability_metrics(real_graphs, real_distances, neighborhood_k=neighborhood_k)
    checks = {
        "all_layers_separate_from_random": all(
            layer["checks"]["real_separates_from_random"] for layer in layer_reports.values()
        ),
        "all_layers_separate_from_shuffled": all(
            layer["checks"]["real_separates_from_shuffled"] for layer in layer_reports.values()
        ),
        "all_layers_non_uniform": all(
            layer["checks"]["real_not_uniform"] for layer in layer_reports.values()
        ),
        "all_layers_not_saturated": all(
            layer["checks"]["real_not_saturated"] for layer in layer_reports.values()
        ),
        "nearby_layers_stable": stability["mean_graph_cosine"] >= stability_threshold,
    }
    status: ObserverStatus = "pass" if all(checks.values()) else "fail"

    return {
        "phase": "phase2_relational_field_observer",
        "status": status,
        "seed": seed,
        "input_source": input_source,
        "observer_pipeline": "H -> W -> D",
        "model_intervention": "none",
        "scientific_scope": (
            "smoke_validates_observer_mechanics"
            if input_source == "synthetic_structured_smoke"
            else "observes_provided_hidden_states"
        ),
        "checks": checks,
        "layers": layer_reports,
        "stability": stability,
        "controls": {
            "control_generation_level": "W_level_before_distance_normalization",
            "families": ["real", "random", "shuffled"],
        },
        "next_required_step": "resonant_response_observer" if status == "pass" else "fix_field",
    }


def field_metrics(
    hidden_states: torch.Tensor,
    graph: torch.Tensor,
    distance: torch.Tensor,
    valid_edge_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute JSON-compatible relational field metrics for one layer."""

    _validate_hidden_and_graph(hidden_states, graph)
    valid_edge_mask = _prepare_mask(valid_edge_mask, graph)
    valid_weights = graph[valid_edge_mask & torch.isfinite(graph)]
    if valid_weights.numel() == 0:
        raise ValueError("graph has no valid finite edges")

    node_entropy = _row_entropy(graph, valid_edge_mask)
    spectral = _spectral_summary(graph, valid_edge_mask)
    coherence = _coherence_summary(hidden_states, graph, valid_edge_mask)
    finite_distance = distance[torch.isfinite(distance)]

    return {
        "relational_entropy_mean": _finite_mean(node_entropy),
        "relational_entropy_std": _finite_std(node_entropy),
        "local_relational_entropy_mean": _finite_mean(
            _normalize_entropy_by_degree(node_entropy, valid_edge_mask)
        ),
        "spectral_entropy_mean": spectral["spectral_entropy_mean"],
        "effective_rank_mean": spectral["effective_rank_mean"],
        "coherence_mean": coherence["coherence_mean"],
        "boundary_sharpness_mean": coherence["boundary_sharpness_mean"],
        "distance_entropy": float(distance_entropy(distance)),
        "valid_weight_mean": _to_float(valid_weights.mean()),
        "valid_weight_variance": _to_float(valid_weights.var(unbiased=False)),
        "distance_variance": _to_float(finite_distance.var(unbiased=False))
        if finite_distance.numel()
        else math.nan,
        "saturation_fraction": _to_float(
            ((valid_weights <= 1e-6) | (valid_weights >= 1 - 1e-6)).float().mean()
        ),
    }


def separation_metrics(
    real_metrics: dict[str, float],
    random_metrics: dict[str, float],
    shuffled_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "real_vs_random": _metric_delta(real_metrics, random_metrics),
        "real_vs_shuffled": _metric_delta(real_metrics, shuffled_metrics),
    }


def stability_metrics(
    graphs: list[torch.Tensor],
    distances: list[torch.Tensor],
    *,
    neighborhood_k: int,
) -> dict[str, Any]:
    graph_cosines: list[float] = []
    graph_frobenius: list[float] = []
    neighborhood_overlaps: list[float] = []

    for layer_index in range(len(graphs) - 1):
        similarity = layer_to_layer_similarity(graphs[layer_index], graphs[layer_index + 1])
        graph_cosines.append(similarity["cosine_mean"])
        graph_frobenius.append(similarity["frobenius_mean"])
        neighborhood_overlaps.append(
            neighborhood_overlap(
                distances[layer_index],
                distances[layer_index + 1],
                neighborhood_k,
            )
        )

    return {
        "pairs": max(len(graphs) - 1, 0),
        "mean_graph_cosine": _mean_or_nan(graph_cosines),
        "mean_graph_frobenius": _mean_or_nan(graph_frobenius),
        "mean_neighborhood_overlap": _mean_or_nan(neighborhood_overlaps),
        "neighborhood_k": neighborhood_k,
    }


def synthetic_structured_hidden_layers(
    *,
    seed: int = 2027,
    batch_size: int = 2,
    sequence_length: int = 6,
    hidden_dim: int = 8,
    layers: int = 3,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Create a deterministic structured hidden field for smoke reports."""

    if hidden_dim < 4:
        raise ValueError("hidden_dim must be at least 4")
    generator = torch.Generator()
    generator.manual_seed(seed)

    prototypes = torch.zeros(3, hidden_dim)
    prototypes[0, 0] = 1.0
    prototypes[1, 1] = 1.0
    prototypes[2, 2] = 1.0
    assignments = torch.tensor([0, 0, 1, 1, 2, 2])[:sequence_length]
    base = prototypes[assignments].unsqueeze(0).repeat(batch_size, 1, 1)
    position_signal = torch.linspace(0.0, 0.3, sequence_length).view(1, sequence_length, 1)
    base = base + position_signal

    hidden_layers = []
    for layer_index in range(layers):
        noise = torch.randn(base.shape, generator=generator) * 0.015
        drift = 0.01 * layer_index
        hidden_layers.append(base + drift + noise)

    attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long)
    attention_mask[0, -1] = 0
    return hidden_layers, attention_mask


def _metric_delta(
    real_metrics: dict[str, float],
    control_metrics: dict[str, float],
) -> dict[str, Any]:
    deltas = {
        key: real_metrics[key] - control_metrics[key]
        for key in FIELD_METRIC_KEYS
        if _is_finite(real_metrics.get(key)) and _is_finite(control_metrics.get(key))
    }
    aggregate = sum(abs(value) for value in deltas.values()) / max(len(deltas), 1)
    return {
        "deltas": deltas,
        "aggregate_abs_delta": aggregate,
    }


def _row_entropy(graph: torch.Tensor, valid_edge_mask: torch.Tensor) -> torch.Tensor:
    weights = torch.where(valid_edge_mask, graph, torch.zeros_like(graph))
    row_sum = weights.sum(dim=-1, keepdim=True)
    probabilities = weights / row_sum.clamp_min(1e-12)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=-1)
    has_edges = valid_edge_mask.any(dim=-1)
    return torch.where(has_edges, entropy, torch.full_like(entropy, torch.nan))


def _normalize_entropy_by_degree(
    entropy: torch.Tensor,
    valid_edge_mask: torch.Tensor,
) -> torch.Tensor:
    degree = valid_edge_mask.sum(dim=-1).to(dtype=entropy.dtype)
    denominator = torch.log(degree.clamp_min(2))
    normalized = entropy / denominator.clamp_min(1e-12)
    return torch.where(degree > 1, normalized, torch.full_like(normalized, torch.nan))


def _spectral_summary(graph: torch.Tensor, valid_edge_mask: torch.Tensor) -> dict[str, float]:
    masked_graph = torch.where(valid_edge_mask, graph, torch.zeros_like(graph))
    sym_graph = 0.5 * (masked_graph + masked_graph.transpose(-2, -1))
    sequence_length = graph.size(-1)
    eye = torch.eye(sequence_length, dtype=sym_graph.dtype, device=sym_graph.device).view(
        1,
        1,
        sequence_length,
        sequence_length,
    )
    degree = sym_graph.sum(dim=-1)
    inv_sqrt_degree = torch.where(degree > 0, degree.rsqrt(), torch.zeros_like(degree))
    normalized_affinity = (
        inv_sqrt_degree.unsqueeze(-1) * sym_graph * inv_sqrt_degree.unsqueeze(-2)
    )
    laplacian = eye - normalized_affinity
    eigenvalues = torch.linalg.eigvalsh(laplacian).clamp_min(0.0)
    total = eigenvalues.sum(dim=-1, keepdim=True)
    probabilities = eigenvalues / total.clamp_min(1e-12)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=-1)
    entropy = torch.where(total.squeeze(-1) > 0, entropy, torch.zeros_like(entropy))
    effective_rank = torch.exp(entropy)
    return {
        "spectral_entropy_mean": _finite_mean(entropy),
        "effective_rank_mean": _finite_mean(effective_rank),
    }


def _coherence_summary(
    hidden_states: torch.Tensor,
    graph: torch.Tensor,
    valid_edge_mask: torch.Tensor,
) -> dict[str, float]:
    normalized_hidden = F.normalize(hidden_states, p=2, dim=-1)
    similarity = normalized_hidden @ normalized_hidden.transpose(-2, -1)
    similarity = similarity.unsqueeze(1).expand_as(graph)
    weights = torch.where(valid_edge_mask, graph, torch.zeros_like(graph))
    row_weight_sum = weights.sum(dim=-1)
    node_coherence = (weights * similarity).sum(dim=-1) / row_weight_sum.clamp_min(1e-12)
    node_coherence = torch.where(
        row_weight_sum > 0,
        node_coherence,
        torch.full_like(node_coherence, torch.nan),
    )
    neighbor_coherence = torch.nan_to_num(node_coherence, nan=0.0)
    neighbor_coherence = (
        weights * neighbor_coherence.unsqueeze(-2)
    ).sum(dim=-1) / row_weight_sum.clamp_min(1e-12)
    boundary = (node_coherence - neighbor_coherence).abs()
    boundary = torch.where(
        row_weight_sum > 0,
        boundary,
        torch.full_like(boundary, torch.nan),
    )
    return {
        "coherence_mean": _finite_mean(node_coherence),
        "boundary_sharpness_mean": _finite_mean(boundary),
    }


def _attention_mask_or_ones(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if attention_mask is not None:
        return attention_mask
    return torch.ones(hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device)


def _normalize_graph_config(graph_config: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(graph_config or {"kernel": "sigmoid_cosine", "normalize_hidden": True})
    if normalized.get("diagonal_policy", "keep_for_distance") == "keep_for_distance":
        normalized["diagonal_policy"] = "keep"
    return normalized


def _distance_config(distance_config: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "normalization": "offdiag_zscore_clamp",
        "clip_value": 5.0,
        "diagonal_policy": "zero",
        "causal_runtime_distance": True,
        **(distance_config or {}),
    }


def _prepare_mask(valid_edge_mask: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
    valid_edge_mask = valid_edge_mask.to(dtype=torch.bool, device=graph.device)
    if valid_edge_mask.shape == graph.shape:
        return valid_edge_mask
    if valid_edge_mask.size(1) == 1:
        return valid_edge_mask.expand_as(graph)
    raise ValueError("valid_edge_mask must match graph shape or be head-shared")


def _validate_hidden_and_graph(hidden_states: torch.Tensor, graph: torch.Tensor) -> None:
    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden_dim]")
    if graph.dim() != 4:
        raise ValueError("graph must have shape [batch, heads, sequence, sequence]")
    if hidden_states.shape[:2] != graph.shape[0:1] + graph.shape[-2:-1]:
        raise ValueError("hidden_states and graph batch/sequence dimensions must match")


def _finite_mean(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return math.nan
    return _to_float(finite.mean())


def _finite_std(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return math.nan
    return _to_float(finite.std(unbiased=False))


def _mean_or_nan(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def _is_finite(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
