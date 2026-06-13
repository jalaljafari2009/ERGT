"""Strict W-level control diagnostics for post-Phase-3 ERGT."""

from __future__ import annotations

from typing import Any, Literal

import torch

from geometry.emergent_distance import EmergentDistance
from layers.relational_graph import (
    RelationalGraph,
    make_random_graph_like,
    make_shuffled_graph,
    make_valid_edge_mask_like,
)

StrictControlStatus = Literal["pass", "fail"]


def build_strict_w_controls_report(
    *,
    graph_config: dict[str, Any] | None = None,
    distance_config: dict[str, Any] | None = None,
    seed: int = 2027,
    batch_size: int = 2,
    sequence_length: int = 5,
    hidden_dim: int = 8,
) -> dict[str, Any]:
    """Build a small-tensor report proving controls are generated at W level."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    hidden_states = torch.randn(batch_size, sequence_length, hidden_dim, generator=generator)
    attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long)
    attention_mask[0, -1] = 0

    graph_config = _normalize_graph_config(graph_config or {"diagonal_policy": "keep_for_distance"})
    graph_builder = RelationalGraph(graph_config)
    distance_builder = EmergentDistance(
        {
            "normalization": "offdiag_zscore_clamp",
            "clip_value": 5.0,
            "diagonal_policy": "zero",
            "causal_runtime_distance": True,
            **(distance_config or {}),
        }
    )

    real_w = graph_builder(hidden_states, attention_mask=attention_mask)
    valid_edge_mask = make_valid_edge_mask_like(real_w, attention_mask=attention_mask)
    random_w = make_random_graph_like(real_w, generator=generator, valid_edge_mask=valid_edge_mask)
    shuffled_w = make_shuffled_graph(real_w, generator=generator, valid_edge_mask=valid_edge_mask)

    distances = {
        "real_D": distance_builder(real_w, attention_mask=attention_mask),
        "random_D": distance_builder(random_w, attention_mask=attention_mask),
        "shuffled_D": distance_builder(shuffled_w, attention_mask=attention_mask),
    }
    checks = {
        "controls_built_at_w_level": True,
        "same_valid_region_random": _same_invalid_values(real_w, random_w, valid_edge_mask),
        "same_valid_region_shuffled": _same_invalid_values(real_w, shuffled_w, valid_edge_mask),
        "random_changes_valid_edges": _changes_valid_edges(real_w, random_w, valid_edge_mask),
        "shuffled_changes_valid_edges": _changes_valid_edges(real_w, shuffled_w, valid_edge_mask),
        "shuffled_preserves_valid_multiset": _preserves_valid_multiset(
            real_w, shuffled_w, valid_edge_mask
        ),
        "random_preserves_valid_range": _preserves_valid_range(real_w, random_w, valid_edge_mask),
        "distance_finite_regions_match": _distance_finite_regions_match(distances),
        "distance_diagonals_match": _distance_diagonals_match(distances),
        "future_distances_masked": _future_distances_masked(distances["real_D"]),
    }
    status: StrictControlStatus = "pass" if all(checks.values()) else "fail"

    return {
        "phase": "phase1_strict_w_level_controls",
        "status": status,
        "seed": seed,
        "control_pipeline": (
            "H -> W_family -> valid_edge_mask -> D_family -> normalization -> clipping"
        ),
        "invalid_pipeline": "H -> real D normalized -> random/shuffled D",
        "checks": checks,
        "graph": {
            "shape": list(real_w.shape),
            "valid_edges": int(valid_edge_mask.sum().item()),
            "invalid_edges": int((~valid_edge_mask).sum().item()),
        },
        "distance": {
            "normalization": distance_builder.config.normalization,
            "clip_value": distance_builder.config.clip_value,
            "diagonal_policy": distance_builder.config.diagonal_policy,
            "causal_runtime_distance": distance_builder.config.causal_runtime_distance,
        },
        "next_required_step": "relational_field_observer" if status == "pass" else "fix_controls",
    }


def _same_invalid_values(
    real_w: torch.Tensor,
    control_w: torch.Tensor,
    valid_edge_mask: torch.Tensor,
) -> bool:
    invalid = ~valid_edge_mask
    return bool(torch.allclose(real_w[invalid], control_w[invalid], equal_nan=True))


def _changes_valid_edges(
    real_w: torch.Tensor,
    control_w: torch.Tensor,
    valid_edge_mask: torch.Tensor,
) -> bool:
    valid = valid_edge_mask & torch.isfinite(real_w) & torch.isfinite(control_w)
    if int(valid.sum().item()) <= 1:
        return False
    return not bool(torch.allclose(real_w[valid], control_w[valid]))


def _preserves_valid_multiset(
    real_w: torch.Tensor,
    control_w: torch.Tensor,
    valid_edge_mask: torch.Tensor,
) -> bool:
    for batch_idx in range(real_w.size(0)):
        for head_idx in range(real_w.size(1)):
            valid = valid_edge_mask[batch_idx, head_idx]
            real_values = real_w[batch_idx, head_idx][valid].sort().values
            control_values = control_w[batch_idx, head_idx][valid].sort().values
            if not torch.allclose(real_values, control_values):
                return False
    return True


def _preserves_valid_range(
    real_w: torch.Tensor,
    control_w: torch.Tensor,
    valid_edge_mask: torch.Tensor,
) -> bool:
    valid = valid_edge_mask & torch.isfinite(real_w) & torch.isfinite(control_w)
    if int(valid.sum().item()) == 0:
        return False
    real_values = real_w[valid]
    control_values = control_w[valid]
    return bool(
        control_values.min() >= real_values.min()
        and control_values.max() <= real_values.max()
    )


def _distance_finite_regions_match(distances: dict[str, torch.Tensor]) -> bool:
    finite_masks = [torch.isfinite(distance) for distance in distances.values()]
    first = finite_masks[0]
    return all(bool(torch.equal(first, mask)) for mask in finite_masks[1:])


def _distance_diagonals_match(distances: dict[str, torch.Tensor]) -> bool:
    diagonals = [torch.diagonal(distance, dim1=-2, dim2=-1) for distance in distances.values()]
    first = diagonals[0]
    return all(bool(torch.allclose(first, diagonal, equal_nan=True)) for diagonal in diagonals[1:])


def _future_distances_masked(distance: torch.Tensor) -> bool:
    sequence_length = distance.size(-1)
    future = torch.ones(sequence_length, sequence_length, dtype=torch.bool).triu(diagonal=1)
    future = future.to(device=distance.device).view(1, 1, sequence_length, sequence_length)
    return bool(torch.isinf(distance[future.expand_as(distance)]).all())


def _normalize_graph_config(graph_config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(graph_config)
    if normalized.get("diagonal_policy") == "keep_for_distance":
        normalized["diagonal_policy"] = "keep"
    return normalized
