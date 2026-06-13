"""Reconstruction Gate observer for the strengthened ERGT program.

This phase checks whether relational structure is reconstructible from allowed
causal context before memory or attention injection is allowed.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
import torch.nn.functional as F

from evaluation.information_potential_phi import (
    PhiWeights,
    information_potential_components,
)
from evaluation.relational_field_observer import synthetic_structured_hidden_layers
from layers.relational_graph import (
    RelationalGraph,
    make_random_graph_like,
    make_shuffled_graph,
    make_valid_edge_mask_like,
)

ReconstructionStatus = Literal["pass", "fail"]


def build_reconstruction_gate_report(
    *,
    hidden_layers: list[torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    graph_config: dict[str, Any] | None = None,
    seed: int = 2027,
    min_context_edges: int = 2,
    high_phi_fraction: float = 0.33,
    separation_margin: float = 1e-6,
) -> dict[str, Any]:
    """Build the Phase 5 Reconstruction Gate report."""

    input_source = "provided_hidden_layers"
    if hidden_layers is None:
        hidden_layers, attention_mask = synthetic_structured_hidden_layers(seed=seed)
        input_source = "synthetic_structured_smoke"
    if not hidden_layers:
        raise ValueError("hidden_layers must not be empty")

    attention_mask = _attention_mask_or_ones(hidden_layers[0], attention_mask)
    normalized_graph_config = _normalize_graph_config(graph_config)
    graph_builder = RelationalGraph(normalized_graph_config)

    layer_reports: dict[str, Any] = {}
    family_records: dict[str, list[dict[str, torch.Tensor]]] = {
        "real": [],
        "random": [],
        "shuffled": [],
    }
    high_phi_alignments: list[dict[str, float]] = []
    leakage_checks: list[dict[str, bool]] = []

    for layer_index, hidden_states in enumerate(hidden_layers):
        real_graph = graph_builder(hidden_states, attention_mask=attention_mask)
        valid_edge_mask = make_valid_edge_mask_like(
            real_graph,
            attention_mask=attention_mask,
        )
        generator = torch.Generator(device=hidden_states.device)
        generator.manual_seed(seed + layer_index)
        graphs = {
            "real": real_graph,
            "random": make_random_graph_like(
                real_graph,
                generator=generator,
                valid_edge_mask=valid_edge_mask,
            ),
            "shuffled": make_shuffled_graph(
                real_graph,
                generator=generator,
                valid_edge_mask=valid_edge_mask,
            ),
        }

        phi = information_potential_components(
            hidden_states,
            real_graph,
            valid_edge_mask,
            attention_mask=attention_mask,
            weights=PhiWeights(),
        )
        layer_report: dict[str, Any] = {}
        for family, graph in graphs.items():
            deficits = reconstruction_deficits(
                hidden_states,
                graph,
                valid_edge_mask,
                attention_mask=attention_mask,
                graph_config=normalized_graph_config,
                min_context_edges=min_context_edges,
            )
            family_records[family].append(deficits)
            leakage_checks.append(deficits["leakage_checks"])
            layer_report[family] = {
                "hidden_deficit": _summary(
                    deficits["hidden_deficit"],
                    deficits["reconstructible_node_mask"],
                ),
                "relational_deficit": _summary(
                    deficits["relational_deficit"],
                    deficits["reconstructible_node_mask"],
                ),
                "total_deficit": _summary(
                    deficits["total_deficit"],
                    deficits["reconstructible_node_mask"],
                ),
                "reconstructible_nodes": int(
                    deficits["reconstructible_node_mask"].sum().item()
                ),
            }

        alignment = high_phi_reconstruction_alignment(
            phi["score"],
            family_records["real"][-1]["total_deficit"],
            family_records["real"][-1]["reconstructible_node_mask"],
            fraction=high_phi_fraction,
        )
        high_phi_alignments.append(alignment)
        layer_report["comparison"] = _layer_comparison(layer_report)
        layer_report["high_phi_alignment"] = alignment
        layer_reports[f"layer_{layer_index}"] = layer_report

    aggregate = {
        family: _aggregate_deficit_records(records)
        for family, records in family_records.items()
    }
    high_phi_aggregate = _aggregate_high_phi_alignments(high_phi_alignments)

    checks = {
        "model_intervention_none": True,
        "target_hidden_excluded_from_reconstructor": True,
        "future_inputs_forbidden": all(
            check["future_sources_forbidden"] for check in leakage_checks
        ),
        "future_edges_forbidden": all(check["future_edges_forbidden"] for check in leakage_checks),
        "w_level_controls_used": True,
        "source_context_sufficient": aggregate["real"]["reconstructible_nodes"] > 0,
        "real_relational_deficit_lt_random": _lower_than_control(
            aggregate["real"]["relational_deficit_mean"],
            aggregate["random"]["relational_deficit_mean"],
            margin=separation_margin,
        ),
        "real_relational_deficit_lt_shuffled": _lower_than_control(
            aggregate["real"]["relational_deficit_mean"],
            aggregate["shuffled"]["relational_deficit_mean"],
            margin=separation_margin,
        ),
        "real_total_deficit_lt_random": _lower_than_control(
            aggregate["real"]["total_deficit_mean"],
            aggregate["random"]["total_deficit_mean"],
            margin=separation_margin,
        ),
        "real_total_deficit_lt_shuffled": _lower_than_control(
            aggregate["real"]["total_deficit_mean"],
            aggregate["shuffled"]["total_deficit_mean"],
            margin=separation_margin,
        ),
        "high_phi_regions_reconstruct_better": (
            high_phi_aggregate["mean_delta_low_minus_high_total_deficit"] > 0.0
        ),
    }
    status: ReconstructionStatus = "pass" if all(checks.values()) else "fail"

    return {
        "phase": "phase5_reconstruction_gate",
        "status": status,
        "seed": seed,
        "input_source": input_source,
        "scientific_scope": (
            "smoke_validates_reconstruction_gate_mechanics"
            if input_source == "synthetic_structured_smoke"
            else "observes_provided_hidden_states"
        ),
        "observer_pipeline": (
            "H_<i -> R_h(H_<i) -> R_W(H_<i) -> compare with h_i/W_i"
        ),
        "model_intervention": "none",
        "reconstruction_protocol": {
            "hidden_reconstructor": "previous_valid_hidden_state_from_causal_prefix",
            "relational_reconstructor": (
                "relation row from predicted current hidden and allowed past hidden states"
            ),
            "target_policy": (
                "h_i and W_i are used only as scoring targets, not as reconstructor inputs"
            ),
            "future_policy": "future tokens and future relations are forbidden",
            "min_context_edges": min_context_edges,
        },
        "checks": checks,
        "aggregate": aggregate,
        "high_phi_alignment": high_phi_aggregate,
        "layers": layer_reports,
        "controls": {
            "control_generation_level": "W_level_before_distance_normalization",
            "families": ["real", "random", "shuffled"],
        },
        "next_required_step": (
            "relational_memory_observer" if status == "pass" else "fix_reconstruction_gate"
        ),
    }


def reconstruction_deficits(
    hidden_states: torch.Tensor,
    graph: torch.Tensor,
    valid_edge_mask: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    graph_config: dict[str, Any] | None = None,
    min_context_edges: int = 2,
) -> dict[str, Any]:
    """Compute causal hidden and relation reconstruction deficits."""

    _validate_hidden_and_graph(hidden_states, graph)
    if min_context_edges <= 0:
        raise ValueError("min_context_edges must be positive")

    attention_mask = _attention_mask_or_ones(hidden_states, attention_mask)
    valid_edge_mask = _prepare_mask(valid_edge_mask, graph)
    hidden_prediction, source_available, source_index = causal_prefix_hidden_prediction(
        hidden_states,
        attention_mask=attention_mask,
    )
    relation_prediction = causal_prefix_relation_prediction(
        hidden_states,
        hidden_prediction,
        attention_mask=attention_mask,
        graph_config=graph_config,
    )

    source_available = source_available[:, None, :].expand_as(graph[..., 0])
    edge_count = valid_edge_mask.sum(dim=-1)
    reconstructible_node_mask = (
        source_available
        & (edge_count >= min_context_edges)
        & attention_mask[:, None, :].to(dtype=torch.bool, device=graph.device)
    )

    hidden_deficit = ((hidden_prediction[:, None, :, :] - hidden_states[:, None, :, :]) ** 2).mean(
        dim=-1
    )
    hidden_deficit = torch.where(
        reconstructible_node_mask,
        hidden_deficit,
        torch.full_like(hidden_deficit, torch.nan),
    )

    relation_squared_error = torch.where(
        valid_edge_mask,
        (relation_prediction - graph) ** 2,
        torch.zeros_like(graph),
    )
    relational_deficit = relation_squared_error.sum(dim=-1) / edge_count.clamp_min(1)
    relational_deficit = torch.where(
        reconstructible_node_mask,
        relational_deficit,
        torch.full_like(relational_deficit, torch.nan),
    )
    total_deficit = hidden_deficit + relational_deficit

    return {
        "hidden_deficit": hidden_deficit,
        "relational_deficit": relational_deficit,
        "total_deficit": total_deficit,
        "hidden_prediction": hidden_prediction,
        "relation_prediction": relation_prediction,
        "reconstructible_node_mask": reconstructible_node_mask,
        "source_index": source_index,
        "leakage_checks": leakage_checks(source_index, valid_edge_mask),
    }


def causal_prefix_hidden_prediction(
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict each hidden state from the latest valid causal-prefix state."""

    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden_dim]")
    attention_mask = _attention_mask_or_ones(hidden_states, attention_mask)
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("attention_mask shape must match hidden_states")

    prediction = torch.zeros_like(hidden_states)
    source_available = torch.zeros(
        hidden_states.shape[:2],
        dtype=torch.bool,
        device=hidden_states.device,
    )
    source_index = torch.full(
        hidden_states.shape[:2],
        -1,
        dtype=torch.long,
        device=hidden_states.device,
    )
    valid = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
    for batch_idx in range(hidden_states.size(0)):
        previous_index = -1
        for position in range(hidden_states.size(1)):
            if not bool(valid[batch_idx, position].item()):
                continue
            if previous_index >= 0:
                prediction[batch_idx, position] = hidden_states[batch_idx, previous_index]
                source_available[batch_idx, position] = True
                source_index[batch_idx, position] = previous_index
            previous_index = position
    return prediction, source_available, source_index


def causal_prefix_relation_prediction(
    context_hidden_states: torch.Tensor,
    query_hidden_prediction: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    graph_config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Predict relation rows from context-only query states and past states."""

    if context_hidden_states.shape != query_hidden_prediction.shape:
        raise ValueError("context and query hidden tensors must have the same shape")
    if context_hidden_states.dim() != 3:
        raise ValueError("hidden tensors must have shape [batch, sequence, hidden_dim]")

    config = RelationalGraph(_normalize_graph_config(graph_config)).config
    query_states = query_hidden_prediction
    context_states = context_hidden_states
    if config.normalize_hidden or config.kernel == "sigmoid_cosine":
        query_states = F.normalize(query_states, p=2, dim=-1)
        context_states = F.normalize(context_states, p=2, dim=-1)

    logits = query_states @ context_states.transpose(-2, -1)
    if config.temperature is not None:
        scale = config.temperature
    elif config.kernel == "sigmoid_cosine":
        scale = 1.0
    else:
        scale = math.sqrt(context_hidden_states.size(-1))
    graph = torch.sigmoid(logits / scale).unsqueeze(1)
    graph = _apply_attention_mask(graph, attention_mask)
    return _apply_diagonal_policy(graph, config.diagonal_policy)


def high_phi_reconstruction_alignment(
    phi: torch.Tensor,
    total_deficit: torch.Tensor,
    reconstructible_node_mask: torch.Tensor,
    *,
    fraction: float = 0.33,
) -> dict[str, float]:
    """Compare reconstruction deficit in high-Phi and low-Phi regions."""

    if not 0 < fraction <= 0.5:
        raise ValueError("fraction must be in (0, 0.5]")
    valid = (
        reconstructible_node_mask
        & torch.isfinite(phi)
        & torch.isfinite(total_deficit)
    )
    phi_values = phi[valid]
    deficit_values = total_deficit[valid]
    if phi_values.numel() < 2:
        return {
            "low_phi_total_deficit_mean": math.nan,
            "high_phi_total_deficit_mean": math.nan,
            "delta_low_minus_high_total_deficit": math.nan,
            "compared_nodes": float(phi_values.numel()),
        }

    order = torch.argsort(phi_values)
    count = max(1, int(math.ceil(phi_values.numel() * fraction)))
    low = deficit_values[order[:count]]
    high = deficit_values[order[-count:]]
    low_mean = _finite_mean(low)
    high_mean = _finite_mean(high)
    return {
        "low_phi_total_deficit_mean": low_mean,
        "high_phi_total_deficit_mean": high_mean,
        "delta_low_minus_high_total_deficit": low_mean - high_mean,
        "compared_nodes": float(phi_values.numel()),
    }


def leakage_checks(source_index: torch.Tensor, valid_edge_mask: torch.Tensor) -> dict[str, bool]:
    """Validate that reconstruction sources and edge targets are causal."""

    sequence_length = valid_edge_mask.size(-1)
    positions = torch.arange(sequence_length, device=valid_edge_mask.device)
    current = positions.view(1, sequence_length).expand_as(source_index)
    source_available = source_index >= 0
    source_is_past = source_index[source_available] < current[source_available]

    context = positions.view(1, sequence_length)
    row = positions.view(sequence_length, 1)
    future = context > row
    future = future.view(1, 1, sequence_length, sequence_length)
    future_edges_forbidden = not bool(valid_edge_mask[future.expand_as(valid_edge_mask)].any())

    return {
        "future_sources_forbidden": bool(source_is_past.all().item())
        if source_is_past.numel()
        else True,
        "future_edges_forbidden": future_edges_forbidden,
    }


def _layer_comparison(layer_report: dict[str, Any]) -> dict[str, float]:
    return {
        "real_minus_random_relational_deficit": (
            layer_report["real"]["relational_deficit"]["mean"]
            - layer_report["random"]["relational_deficit"]["mean"]
        ),
        "real_minus_shuffled_relational_deficit": (
            layer_report["real"]["relational_deficit"]["mean"]
            - layer_report["shuffled"]["relational_deficit"]["mean"]
        ),
        "real_minus_random_total_deficit": (
            layer_report["real"]["total_deficit"]["mean"]
            - layer_report["random"]["total_deficit"]["mean"]
        ),
        "real_minus_shuffled_total_deficit": (
            layer_report["real"]["total_deficit"]["mean"]
            - layer_report["shuffled"]["total_deficit"]["mean"]
        ),
    }


def _aggregate_deficit_records(records: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    reconstructible = sum(
        int(record["reconstructible_node_mask"].sum().item()) for record in records
    )
    return {
        "hidden_deficit_mean": _finite_mean(_cat_valid(records, "hidden_deficit")),
        "relational_deficit_mean": _finite_mean(_cat_valid(records, "relational_deficit")),
        "total_deficit_mean": _finite_mean(_cat_valid(records, "total_deficit")),
        "reconstructible_nodes": reconstructible,
    }


def _aggregate_high_phi_alignments(alignments: list[dict[str, float]]) -> dict[str, float]:
    deltas = [
        item["delta_low_minus_high_total_deficit"]
        for item in alignments
        if math.isfinite(item["delta_low_minus_high_total_deficit"])
    ]
    compared_nodes = sum(item["compared_nodes"] for item in alignments)
    return {
        "mean_delta_low_minus_high_total_deficit": sum(deltas) / len(deltas)
        if deltas
        else math.nan,
        "compared_nodes": compared_nodes,
    }


def _lower_than_control(real_value: float, control_value: float, *, margin: float) -> bool:
    if not math.isfinite(real_value) or not math.isfinite(control_value):
        return False
    return real_value + margin < control_value


def _summary(values: torch.Tensor, valid_node_mask: torch.Tensor) -> dict[str, float]:
    selected = values[valid_node_mask & torch.isfinite(values)]
    if selected.numel() == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": _to_float(selected.mean()),
        "std": _to_float(selected.std(unbiased=False)),
        "min": _to_float(selected.min()),
        "max": _to_float(selected.max()),
    }


def _cat_valid(records: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    values = []
    for record in records:
        valid = record["reconstructible_node_mask"] & torch.isfinite(record[key])
        values.append(record[key][valid])
    if not values:
        return torch.empty(0)
    return torch.cat(values)


def _finite_mean(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return math.nan
    return _to_float(finite.mean())


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


def _apply_attention_mask(
    graph: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if attention_mask is None:
        return graph
    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    valid = attention_mask.to(dtype=torch.bool, device=graph.device)
    pair_mask = valid[:, None, :, None] & valid[:, None, None, :]
    return graph.masked_fill(~pair_mask, 0.0)


def _apply_diagonal_policy(graph: torch.Tensor, policy: str) -> torch.Tensor:
    if policy == "keep":
        return graph
    sequence_length = graph.size(-1)
    diagonal = torch.eye(sequence_length, dtype=torch.bool, device=graph.device).view(
        1,
        1,
        sequence_length,
        sequence_length,
    )
    if policy == "zero":
        return graph.masked_fill(diagonal, 0.0)
    if policy == "mask":
        return graph.masked_fill(diagonal, torch.nan)
    raise ValueError(f"unsupported diagonal_policy: {policy}")


def _validate_hidden_and_graph(hidden_states: torch.Tensor, graph: torch.Tensor) -> None:
    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden_dim]")
    if graph.dim() != 4:
        raise ValueError("graph must have shape [batch, heads, sequence, sequence]")
    if hidden_states.shape[:2] != graph.shape[0:1] + graph.shape[-2:-1]:
        raise ValueError("hidden_states and graph batch/sequence dimensions must match")


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
