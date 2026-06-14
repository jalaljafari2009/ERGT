"""Causal shortest-path geometry observer for the strengthened ERGT program."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Literal

import torch

from evaluation.relational_memory_observer import (
    MemoryConfig,
    relational_memory_sequence,
    stable_memory_update,
    synthetic_memory_hidden_layers,
)
from layers.relational_graph import (
    RelationalGraph,
    make_random_graph_like,
    make_shuffled_graph,
    make_valid_edge_mask_like,
)

GeometryStatus = Literal["pass", "fail"]


def build_causal_shortest_path_geometry_report(
    *,
    hidden_layers: list[torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    graph_config: dict[str, Any] | None = None,
    seed: int = 2027,
    memory_config: MemoryConfig | None = None,
    epsilon: float = 1e-6,
    max_causal_step: int | None = 1,
    score_margin: float = 1e-4,
    contextuality_margin: float = 1e-4,
) -> dict[str, Any]:
    """Build the Phase 7 Causal Shortest-Path Geometry report."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_causal_step is not None and max_causal_step <= 0:
        raise ValueError("max_causal_step must be positive or None")
    memory_config = memory_config or MemoryConfig()
    memory_config.validate()

    input_source = "provided_hidden_layers"
    if hidden_layers is None:
        hidden_layers, attention_mask = synthetic_memory_hidden_layers(seed=seed)
        input_source = "synthetic_memory_smoke"
    if len(hidden_layers or []) < 2:
        raise ValueError("hidden_layers must contain at least two layers/steps")

    attention_mask = _attention_mask_or_ones(hidden_layers[0], attention_mask)
    normalized_graph_config = _normalize_graph_config(graph_config)
    graph_builder = RelationalGraph(normalized_graph_config)
    family_graphs = _build_family_graphs(
        hidden_layers,
        attention_mask,
        graph_builder,
        seed=seed,
    )
    family_updates: dict[str, list[dict[str, torch.Tensor]]] = {
        family: []
        for family in ("real", "random", "shuffled")
    }
    leakage_checks: list[dict[str, bool]] = []
    for layer_index, hidden_states in enumerate(hidden_layers):
        valid_edge_mask = family_graphs["valid_edge_masks"][layer_index]
        for family in ("real", "random", "shuffled"):
            update = stable_memory_update(
                hidden_states,
                family_graphs[family][layer_index],
                valid_edge_mask,
                attention_mask=attention_mask,
                graph_config=normalized_graph_config,
                memory_config=memory_config,
            )
            family_updates[family].append(update)
            leakage_checks.append(update["leakage_checks"])

    memory_sequences = {
        family: relational_memory_sequence(
            [update["stable_update"] for update in updates],
            memory_config=memory_config,
        )
        for family, updates in family_updates.items()
    }
    direct_edge_masks = [
        finite_speed_edge_mask(mask, max_causal_step=max_causal_step)
        for mask in family_graphs["valid_edge_masks"]
    ]
    target_sequence = [update["stable_update"] for update in family_updates["real"]]
    instantaneous_sequence = [update["stable_update"] for update in family_updates["real"]]
    no_memory_sequence = [
        _valid_only(graph, family_graphs["valid_edge_masks"][layer_index])
        for layer_index, graph in enumerate(family_graphs["real"])
    ]

    distance_sequences = {
        "real_causal_path": [
            causal_shortest_path_distance(memory, direct_mask, epsilon=epsilon)
            for memory, direct_mask in zip(
                memory_sequences["real"],
                direct_edge_masks,
                strict=True,
            )
        ],
        "random_causal_path": [
            causal_shortest_path_distance(memory, direct_mask, epsilon=epsilon)
            for memory, direct_mask in zip(
                memory_sequences["random"],
                direct_edge_masks,
                strict=True,
            )
        ],
        "shuffled_causal_path": [
            causal_shortest_path_distance(memory, direct_mask, epsilon=epsilon)
            for memory, direct_mask in zip(
                memory_sequences["shuffled"],
                direct_edge_masks,
                strict=True,
            )
        ],
        "pairwise_real": [
            pairwise_edge_distance(memory, direct_mask, epsilon=epsilon)
            for memory, direct_mask in zip(
                memory_sequences["real"],
                direct_edge_masks,
                strict=True,
            )
        ],
        "instantaneous_pairwise": [
            pairwise_edge_distance(update, direct_mask, epsilon=epsilon)
            for update, direct_mask in zip(
                instantaneous_sequence,
                direct_edge_masks,
                strict=True,
            )
        ],
        "no_memory_pairwise": [
            pairwise_edge_distance(graph, direct_mask, epsilon=epsilon)
            for graph, direct_mask in zip(
                no_memory_sequence,
                direct_edge_masks,
                strict=True,
            )
        ],
    }
    metrics = {
        name: geometry_quality_metrics(
            distances,
            target_sequence,
            family_graphs["valid_edge_masks"],
            pairwise_reference=distance_sequences["pairwise_real"]
            if name == "real_causal_path"
            else None,
        )
        for name, distances in distance_sequences.items()
    }
    layer_reports = _layer_reports(
        distance_sequences,
        target_sequence,
        family_graphs["valid_edge_masks"],
    )
    real_score = metrics["real_causal_path"]["geometry_quality_score"]

    checks = {
        "model_intervention_none": True,
        "w_level_controls_used": True,
        "memory_observer_used": True,
        "future_inputs_forbidden": all(
            check["future_sources_forbidden"] for check in leakage_checks
        ),
        "future_edges_forbidden": all(
            check["future_edges_forbidden"] for check in leakage_checks
        ) and all(
            future_distances_are_infinite(distance)
            for distances in distance_sequences.values()
            for distance in distances
        ),
        "diagonal_zero": all(
            distance_diagonal_is_zero(distance)
            for distances in distance_sequences.values()
            for distance in distances
        ),
        "real_causal_path_separates_from_random": _beats(
            real_score,
            metrics["random_causal_path"]["geometry_quality_score"],
            margin=score_margin,
        ),
        "real_causal_path_separates_from_shuffled": _beats(
            real_score,
            metrics["shuffled_causal_path"]["geometry_quality_score"],
            margin=score_margin,
        ),
        "causal_path_beats_pairwise": _beats(
            real_score,
            metrics["pairwise_real"]["geometry_quality_score"],
            margin=score_margin,
        ),
        "causal_path_beats_no_memory": _beats(
            real_score,
            metrics["no_memory_pairwise"]["geometry_quality_score"],
            margin=score_margin,
        ),
        "causal_path_adds_contextual_signal": (
            metrics["real_causal_path"]["path_improvement_fraction"]
            > contextuality_margin
        ),
    }
    status: GeometryStatus = "pass" if all(checks.values()) else "fail"

    return {
        "phase": "phase7_causal_shortest_path_geometry",
        "status": status,
        "seed": seed,
        "input_source": input_source,
        "scientific_scope": (
            "smoke_validates_causal_shortest_path_geometry_mechanics"
            if input_source == "synthetic_memory_smoke"
            else "observes_provided_hidden_states"
        ),
        "observer_pipeline": "H -> W_t memory -> edge_cost -> D_causal shortest path",
        "model_intervention": "none",
        "geometry_formula": {
            "edge_cost": "edge_cost_ij = -log(W_t_ij + epsilon)",
            "distance": "D_causal(i,j) = shortest path over valid causal edges",
            "epsilon": epsilon,
            "direct_edge_policy": (
                "valid causal edges with finite-speed max_causal_step; "
                "multi-step reachability is created only by shortest path"
            ),
            "max_causal_step": max_causal_step,
            "memory_config": asdict(memory_config),
        },
        "checks": checks,
        "metrics": metrics,
        "layers": layer_reports,
        "controls": {
            "families": [
                "real_causal_path",
                "random_causal_path",
                "shuffled_causal_path",
                "pairwise_real",
                "instantaneous_pairwise",
                "no_memory_pairwise",
            ],
            "control_generation_level": "W_level_before_distance_normalization",
        },
        "next_required_step": "geoattention_v2" if status == "pass" else "fix_causal_geometry",
    }


def causal_shortest_path_distance(
    memory_graph: torch.Tensor,
    valid_edge_mask: torch.Tensor,
    *,
    epsilon: float = 1e-6,
    unit_step_causal: bool | None = None,
) -> torch.Tensor:
    """Compute directed causal shortest-path distance over `memory_graph`."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    _validate_graph(memory_graph)
    valid_edge_mask = _prepare_mask(valid_edge_mask, memory_graph)
    if unit_step_causal is True or (
        unit_step_causal is None and _is_unit_step_causal_mask(valid_edge_mask)
    ):
        return _unit_step_causal_path_distance(
            memory_graph,
            valid_edge_mask,
            epsilon=epsilon,
        )

    direct = pairwise_edge_distance(memory_graph, valid_edge_mask, epsilon=epsilon)
    distance = direct.clone()
    sequence_length = distance.size(-1)
    diagonal = torch.eye(sequence_length, dtype=torch.bool, device=distance.device).view(
        1,
        1,
        sequence_length,
        sequence_length,
    )
    distance = distance.masked_fill(diagonal, 0.0)

    for bridge in range(sequence_length):
        via_bridge = (
            distance[..., :, bridge].unsqueeze(-1)
            + distance[..., bridge, :].unsqueeze(-2)
        )
        distance = torch.minimum(distance, via_bridge)

    future = _future_mask(distance)
    distance = distance.masked_fill(future, torch.inf)
    distance = distance.masked_fill(diagonal, 0.0)
    return distance


def pairwise_edge_distance(
    graph: torch.Tensor,
    valid_edge_mask: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Convert direct causal edges into pairwise edge costs."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    _validate_graph(graph)
    valid_edge_mask = _prepare_mask(valid_edge_mask, graph)
    safe_graph = graph.clamp_min(epsilon)
    distance = -torch.log(safe_graph)
    distance = torch.where(valid_edge_mask, distance, torch.full_like(distance, torch.inf))
    sequence_length = distance.size(-1)
    diagonal = torch.eye(sequence_length, dtype=torch.bool, device=distance.device).view(
        1,
        1,
        sequence_length,
        sequence_length,
    )
    return distance.masked_fill(diagonal, 0.0)


def finite_speed_edge_mask(
    valid_edge_mask: torch.Tensor,
    *,
    max_causal_step: int | None,
) -> torch.Tensor:
    """Restrict direct edges to a finite causal speed before shortest paths."""

    if max_causal_step is None:
        return valid_edge_mask.to(dtype=torch.bool)
    if max_causal_step <= 0:
        raise ValueError("max_causal_step must be positive or None")

    sequence_length = valid_edge_mask.size(-1)
    positions = torch.arange(sequence_length, device=valid_edge_mask.device)
    current = positions.view(sequence_length, 1)
    context = positions.view(1, sequence_length)
    local = (current - context) <= max_causal_step
    local = local.view(1, 1, sequence_length, sequence_length)
    return valid_edge_mask.to(dtype=torch.bool) & local.expand_as(valid_edge_mask)


def _is_unit_step_causal_mask(valid_edge_mask: torch.Tensor) -> bool:
    """Return true when direct causal edges are only adjacent past edges."""

    sequence_length = valid_edge_mask.size(-1)
    positions = torch.arange(sequence_length, device=valid_edge_mask.device)
    current = positions.view(sequence_length, 1)
    context = positions.view(1, sequence_length)
    unit_step = (current - context) == 1
    diagonal = current == context
    non_unit_edges = valid_edge_mask & ~(unit_step | diagonal).view(
        1,
        1,
        sequence_length,
        sequence_length,
    )
    return not bool(non_unit_edges.any().item())


def _unit_step_causal_path_distance(
    memory_graph: torch.Tensor,
    valid_edge_mask: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Fast shortest-path distance for max_causal_step=1 chain geometry."""

    sequence_length = memory_graph.size(-1)
    diagonal = torch.eye(sequence_length, dtype=torch.bool, device=memory_graph.device).view(
        1,
        1,
        sequence_length,
        sequence_length,
    )
    empty = torch.full_like(memory_graph, torch.inf).masked_fill(diagonal, 0.0)
    if sequence_length <= 1:
        return empty

    indices = torch.arange(1, sequence_length, device=memory_graph.device)
    adjacent_valid = valid_edge_mask[..., indices, indices - 1]
    adjacent_graph = memory_graph[..., indices, indices - 1].clamp_min(epsilon)
    adjacent_cost = -torch.log(adjacent_graph)
    adjacent_finite = adjacent_valid & torch.isfinite(adjacent_cost)
    safe_adjacent_cost = torch.where(
        adjacent_finite,
        adjacent_cost,
        torch.zeros_like(adjacent_cost),
    )
    zero_cost = torch.zeros(
        *adjacent_cost.shape[:-1],
        1,
        dtype=adjacent_cost.dtype,
        device=adjacent_cost.device,
    )
    prefix_cost = torch.cat(
        [zero_cost, torch.cumsum(safe_adjacent_cost, dim=-1)],
        dim=-1,
    )

    missing_edge = (~adjacent_finite).to(dtype=torch.int64)
    zero_missing = torch.zeros(
        *missing_edge.shape[:-1],
        1,
        dtype=missing_edge.dtype,
        device=missing_edge.device,
    )
    prefix_missing = torch.cat(
        [zero_missing, torch.cumsum(missing_edge, dim=-1)],
        dim=-1,
    )

    distance = prefix_cost.unsqueeze(-1) - prefix_cost.unsqueeze(-2)
    missing = prefix_missing.unsqueeze(-1) - prefix_missing.unsqueeze(-2)
    positions = torch.arange(sequence_length, device=memory_graph.device)
    causal = positions.view(sequence_length, 1) >= positions.view(1, sequence_length)
    reachable = (missing == 0) & causal.view(1, 1, sequence_length, sequence_length)
    distance = torch.where(reachable, distance, torch.full_like(distance, torch.inf))
    future = _future_mask(distance)
    distance = distance.masked_fill(future, torch.inf)
    return distance.masked_fill(diagonal, 0.0)


def geometry_quality_metrics(
    distance_sequence: list[torch.Tensor],
    target_sequence: list[torch.Tensor],
    valid_edge_masks: list[torch.Tensor],
    *,
    pairwise_reference: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    """Score causal geometry as next-update affinity prediction plus path signal."""

    if len(distance_sequence) != len(target_sequence):
        raise ValueError("distance and target sequences must have the same length")
    if len(distance_sequence) < 2:
        raise ValueError("distance_sequence must contain at least two items")

    prediction_errors = []
    prediction_scores = []
    prediction_cosines = []
    finite_fractions = []
    for index in range(len(distance_sequence) - 1):
        mask = valid_edge_masks[index + 1]
        predicted_affinity = distance_to_affinity(distance_sequence[index])
        target = target_sequence[index + 1]
        mse = _masked_mse(predicted_affinity, target, mask)
        target_variance = _masked_variance(target, mask)
        normalized_error = mse / max(target_variance, 1e-8)
        prediction_errors.append(normalized_error)
        prediction_scores.append(1.0 / (1.0 + normalized_error))
        prediction_cosines.append(_masked_cosine(predicted_affinity, target, mask))
        finite_fractions.append(_finite_fraction(distance_sequence[index], mask))

    path_improvement = 0.0
    if pairwise_reference is not None:
        path_improvement = _path_improvement_fraction(
            distance_sequence,
            pairwise_reference,
            valid_edge_masks,
        )

    prediction_score = _mean(prediction_scores)
    cosine_to_next = _mean(prediction_cosines)
    finite_fraction = _mean(finite_fractions)
    quality = (
        0.55 * prediction_score
        + 0.25 * max(_finite_or_zero(cosine_to_next), 0.0)
        + 0.1 * finite_fraction
        + 0.1 * path_improvement
    )
    return {
        "geometry_quality_score": quality,
        "next_update_prediction_score": prediction_score,
        "next_update_normalized_mse": _mean(prediction_errors),
        "next_update_cosine": cosine_to_next,
        "finite_valid_fraction": finite_fraction,
        "path_improvement_fraction": path_improvement,
    }


def distance_to_affinity(distance: torch.Tensor) -> torch.Tensor:
    """Convert distance to affinity with unreachable pairs mapped to zero."""

    affinity = torch.exp(-torch.where(torch.isfinite(distance), distance, torch.inf))
    return torch.where(torch.isfinite(affinity), affinity, torch.zeros_like(affinity))


def future_distances_are_infinite(distance: torch.Tensor) -> bool:
    future = _future_mask(distance)
    return bool(torch.isinf(distance[future]).all().item())


def distance_diagonal_is_zero(distance: torch.Tensor) -> bool:
    diagonal = torch.diagonal(distance, dim1=-2, dim2=-1)
    return bool(torch.allclose(diagonal, torch.zeros_like(diagonal)))


def _build_family_graphs(
    hidden_layers: list[torch.Tensor],
    attention_mask: torch.Tensor,
    graph_builder: RelationalGraph,
    *,
    seed: int,
) -> dict[str, list[torch.Tensor]]:
    family_graphs: dict[str, list[torch.Tensor]] = {
        "real": [],
        "random": [],
        "shuffled": [],
        "valid_edge_masks": [],
    }
    for layer_index, hidden_states in enumerate(hidden_layers):
        graph = graph_builder(hidden_states, attention_mask=attention_mask)
        valid_edge_mask = make_valid_edge_mask_like(graph, attention_mask=attention_mask)
        generator = torch.Generator(device=graph.device)
        generator.manual_seed(seed + layer_index)
        family_graphs["real"].append(_valid_only(graph, valid_edge_mask))
        family_graphs["random"].append(
            _valid_only(
                make_random_graph_like(
                    graph,
                    generator=generator,
                    valid_edge_mask=valid_edge_mask,
                ),
                valid_edge_mask,
            )
        )
        family_graphs["shuffled"].append(
            _valid_only(
                make_shuffled_graph(
                    graph,
                    generator=generator,
                    valid_edge_mask=valid_edge_mask,
                ),
                valid_edge_mask,
            )
        )
        family_graphs["valid_edge_masks"].append(valid_edge_mask)
    return family_graphs


def _layer_reports(
    distance_sequences: dict[str, list[torch.Tensor]],
    target_sequence: list[torch.Tensor],
    valid_edge_masks: list[torch.Tensor],
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for layer_index in range(len(target_sequence) - 1):
        mask = valid_edge_masks[layer_index + 1]
        target = target_sequence[layer_index + 1]
        reports[f"transition_{layer_index}_to_{layer_index + 1}"] = {
            name: {
                "normalized_mse_to_next_update": _masked_mse(
                    distance_to_affinity(distances[layer_index]),
                    target,
                    mask,
                )
                / max(_masked_variance(target, mask), 1e-8),
                "cosine_to_next_update": _masked_cosine(
                    distance_to_affinity(distances[layer_index]),
                    target,
                    mask,
                ),
            }
            for name, distances in distance_sequences.items()
        }
    return reports


def _path_improvement_fraction(
    distance_sequence: list[torch.Tensor],
    pairwise_reference: list[torch.Tensor],
    valid_edge_masks: list[torch.Tensor],
    *,
    margin: float = 1e-6,
) -> float:
    fractions = []
    for distance, pairwise, mask in zip(
        distance_sequence,
        pairwise_reference,
        valid_edge_masks,
        strict=True,
    ):
        mask = _prepare_mask(mask, distance)
        reachable = torch.isfinite(distance)
        pairwise_finite = torch.isfinite(pairwise)
        improved = reachable & (~pairwise_finite | (distance + margin < pairwise)) & mask
        valid = mask
        if bool(valid.any().item()):
            fractions.append(_to_float(improved.float().sum() / valid.float().sum()))
    return _mean(fractions)


def _finite_fraction(distance: torch.Tensor, mask: torch.Tensor) -> float:
    mask = _prepare_mask(mask, distance)
    if not bool(mask.any().item()):
        return math.nan
    return _to_float((torch.isfinite(distance) & mask).float().sum() / mask.float().sum())


def _valid_only(graph: torch.Tensor, valid_edge_mask: torch.Tensor) -> torch.Tensor:
    valid_edge_mask = _prepare_mask(valid_edge_mask, graph)
    return torch.where(valid_edge_mask, graph, torch.zeros_like(graph))


def _masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    mask = _prepare_mask(mask, a) & torch.isfinite(a) & torch.isfinite(b)
    if not bool(mask.any().item()):
        return math.nan
    return _to_float(((a - b) ** 2)[mask].mean())


def _masked_variance(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask = _prepare_mask(mask, values) & torch.isfinite(values)
    if not bool(mask.any().item()):
        return math.nan
    selected = values[mask]
    return _to_float(selected.var(unbiased=False))


def _masked_cosine(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    mask = _prepare_mask(mask, a) & torch.isfinite(a) & torch.isfinite(b)
    if int(mask.sum().item()) < 2:
        return math.nan
    a_values = a[mask]
    b_values = b[mask]
    denominator = torch.linalg.vector_norm(a_values) * torch.linalg.vector_norm(b_values)
    if denominator <= 0:
        return math.nan
    return _to_float((a_values * b_values).sum() / denominator)


def _future_mask(tensor: torch.Tensor) -> torch.Tensor:
    sequence_length = tensor.size(-1)
    positions = torch.arange(sequence_length, device=tensor.device)
    future = positions.view(1, sequence_length) > positions.view(sequence_length, 1)
    return future.view(1, 1, sequence_length, sequence_length).expand_as(tensor)


def _beats(real_score: float, control_score: float, *, margin: float) -> bool:
    if not math.isfinite(real_score) or not math.isfinite(control_score):
        return False
    return real_score > control_score + margin


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def _finite_or_zero(value: float) -> float:
    return value if math.isfinite(value) else 0.0


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


def _prepare_mask(valid_edge_mask: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
    valid_edge_mask = valid_edge_mask.to(dtype=torch.bool, device=graph.device)
    if valid_edge_mask.shape == graph.shape:
        return valid_edge_mask
    if valid_edge_mask.size(1) == 1:
        return valid_edge_mask.expand_as(graph)
    raise ValueError("valid_edge_mask must match graph shape or be head-shared")


def _validate_graph(graph: torch.Tensor) -> None:
    if graph.dim() != 4:
        raise ValueError("graph must have shape [batch, heads, sequence, sequence]")
    if graph.size(-1) != graph.size(-2):
        raise ValueError("graph must be square in the last two dimensions")


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
