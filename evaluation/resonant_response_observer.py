"""Resonant-response observer for the strengthened ERGT program."""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
import torch.nn.functional as F

from evaluation.distance_metrics import neighborhood_overlap
from evaluation.graph_metrics import layer_to_layer_similarity
from evaluation.relational_field_observer import (
    FIELD_METRIC_KEYS,
    field_metrics,
    synthetic_structured_hidden_layers,
)
from geometry.emergent_distance import EmergentDistance
from layers.relational_graph import (
    RelationalGraph,
    make_random_graph_like,
    make_shuffled_graph,
    make_valid_edge_mask_like,
)

ResponseStatus = Literal["pass", "fail"]


def build_resonant_response_observer_report(
    *,
    hidden_layers: list[torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    graph_config: dict[str, Any] | None = None,
    distance_config: dict[str, Any] | None = None,
    seed: int = 2027,
    perturbation_scale: float = 0.08,
    neighborhood_k: int = 2,
    stability_margin: float = 0.05,
    response_margin: float = 1e-3,
    max_norm_ratio: float = 1.2,
) -> dict[str, Any]:
    """Probe `H -> W/D/Phi-proxy` response without changing training state."""

    input_source = "provided_hidden_layers"
    if hidden_layers is None:
        hidden_layers, attention_mask = synthetic_structured_hidden_layers(seed=seed)
        input_source = "synthetic_structured_smoke"
    if not hidden_layers:
        raise ValueError("hidden_layers must not be empty")

    attention_mask = _attention_mask_or_ones(hidden_layers[0], attention_mask)
    before_snapshot = [layer.clone() for layer in hidden_layers]
    after_layers = controlled_resonant_perturbation(
        hidden_layers,
        attention_mask=attention_mask,
        scale=perturbation_scale,
    )
    reset_max_abs_diff = max(
        _max_abs_diff(before, current)
        for before, current in zip(before_snapshot, hidden_layers, strict=True)
    )
    norm_ratio = _hidden_norm_ratio(before_snapshot, after_layers)

    graph_builder = RelationalGraph(_normalize_graph_config(graph_config))
    distance_builder = EmergentDistance(_distance_config(distance_config))
    layer_reports: dict[str, Any] = {}

    for layer_index, (before_hidden, after_hidden) in enumerate(
        zip(hidden_layers, after_layers, strict=True)
    ):
        family_before = _family_state(
            before_hidden,
            attention_mask,
            graph_builder,
            distance_builder,
            seed=seed + layer_index,
        )
        family_after = _family_state(
            after_hidden,
            attention_mask,
            graph_builder,
            distance_builder,
            seed=seed + layer_index + 100_000,
        )
        responses = {
            family: _response_metrics(
                before_hidden,
                after_hidden,
                family_before[family],
                family_after[family],
                neighborhood_k=neighborhood_k,
            )
            for family in ("real", "random", "shuffled")
        }
        comparison = _compare_real_to_controls(
            responses,
            stability_margin=stability_margin,
            response_margin=response_margin,
        )
        layer_reports[f"layer_{layer_index}"] = {
            "responses": responses,
            "comparison": comparison,
        }

    checks = {
        "model_intervention_none": True,
        "probe_reset_to_before_supported": reset_max_abs_diff == 0.0,
        "response_not_scale_artifact": norm_ratio <= max_norm_ratio,
        "w_level_controls_used": True,
        "all_layers_real_response_beats_or_stabilizes_vs_controls": all(
            layer["comparison"]["real_beats_or_stabilizes_vs_controls"]
            for layer in layer_reports.values()
        ),
    }
    status: ResponseStatus = "pass" if all(checks.values()) else "fail"

    return {
        "phase": "phase3_resonant_response_observer",
        "status": status,
        "seed": seed,
        "input_source": input_source,
        "scientific_scope": (
            "smoke_validates_response_observer_mechanics"
            if input_source == "synthetic_structured_smoke"
            else "observes_provided_hidden_states"
        ),
        "observer_pipeline": (
            "H_before -> W/D/Phi_proxy_before -> perturb -> "
            "H_after -> W/D/Phi_proxy_after -> reset"
        ),
        "model_intervention": "none",
        "perturbation": {
            "type": "resonant_similarity_boost",
            "scale": perturbation_scale,
            "max_hidden_norm_ratio": norm_ratio,
            "reset_max_abs_diff": reset_max_abs_diff,
        },
        "response_control_policy": (
            "W-level controls are regenerated after perturbation with an independent "
            "seed so fixed control permutations cannot explain response stability."
        ),
        "checks": checks,
        "layers": layer_reports,
        "controls": {
            "control_generation_level": "W_level_before_distance_normalization",
            "families": ["real", "random", "shuffled"],
        },
        "next_required_step": (
            "information_potential_phi" if status == "pass" else "fix_response_probe"
        ),
    }


def controlled_resonant_perturbation(
    hidden_layers: list[torch.Tensor],
    *,
    attention_mask: torch.Tensor | None = None,
    scale: float = 0.08,
) -> list[torch.Tensor]:
    """Return perturbed hidden states without mutating the input layers."""

    if scale < 0:
        raise ValueError("scale must be non-negative")
    if not hidden_layers:
        raise ValueError("hidden_layers must not be empty")

    attention_mask = _attention_mask_or_ones(hidden_layers[0], attention_mask)
    perturbed_layers: list[torch.Tensor] = []
    for layer in hidden_layers:
        if layer.dim() != 3:
            raise ValueError("hidden layer must have shape [batch, sequence, hidden_dim]")
        anchor = _first_valid_anchor(layer, attention_mask)
        normalized = F.normalize(layer, p=2, dim=-1)
        anchor_normalized = F.normalize(anchor, p=2, dim=-1)
        similarity = (normalized * anchor_normalized[:, None, :]).sum(dim=-1)
        valid = attention_mask.to(dtype=torch.bool, device=layer.device)
        gate = torch.relu(similarity).unsqueeze(-1) * valid.unsqueeze(-1)
        update = scale * gate * anchor_normalized[:, None, :]
        perturbed_layers.append(layer + update)
    return perturbed_layers


def phi_proxy(metrics: dict[str, float], *, stability: float = 1.0) -> float:
    """Small operational Phi proxy used only for response observation."""

    coherence = max(metrics["coherence_mean"], 0.0)
    local_order = max(1.0 - metrics["local_relational_entropy_mean"], 0.0)
    anti_collapse = max(1.0 - metrics["saturation_fraction"], 0.0)
    variance = max(metrics["valid_weight_variance"], 0.0)
    return coherence * local_order * anti_collapse * (1.0 + variance) * max(stability, 0.0)


def _family_state(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    graph_builder: RelationalGraph,
    distance_builder: EmergentDistance,
    *,
    seed: int,
) -> dict[str, dict[str, Any]]:
    graph = graph_builder(hidden_states, attention_mask=attention_mask)
    valid_edge_mask = make_valid_edge_mask_like(graph, attention_mask=attention_mask)
    controls = {
        "random": make_random_graph_like(
            graph,
            generator=torch.Generator(device=graph.device).manual_seed(seed),
            valid_edge_mask=valid_edge_mask,
        ),
        "shuffled": make_shuffled_graph(
            graph,
            generator=torch.Generator(device=graph.device).manual_seed(seed),
            valid_edge_mask=valid_edge_mask,
        ),
    }
    graphs = {"real": graph, **controls}
    return {
        family: {
            "graph": family_graph,
            "distance": distance_builder(family_graph, attention_mask=attention_mask),
            "valid_edge_mask": valid_edge_mask,
            "metrics": field_metrics(
                hidden_states,
                family_graph,
                distance_builder(family_graph, attention_mask=attention_mask),
                valid_edge_mask,
            ),
        }
        for family, family_graph in graphs.items()
    }


def _response_metrics(
    before_hidden: torch.Tensor,
    after_hidden: torch.Tensor,
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    neighborhood_k: int,
) -> dict[str, Any]:
    graph_similarity = layer_to_layer_similarity(before["graph"], after["graph"])
    neighborhood_stability = neighborhood_overlap(
        before["distance"],
        after["distance"],
        neighborhood_k,
    )
    before_metrics = before["metrics"]
    after_metrics = after["metrics"]
    metric_deltas = {
        key: after_metrics[key] - before_metrics[key]
        for key in FIELD_METRIC_KEYS
        if _is_finite(after_metrics.get(key)) and _is_finite(before_metrics.get(key))
    }
    metric_response_magnitude = sum(abs(delta) for delta in metric_deltas.values()) / max(
        len(metric_deltas),
        1,
    )
    before_phi = phi_proxy(before_metrics, stability=1.0)
    after_phi = phi_proxy(after_metrics, stability=neighborhood_stability)
    return {
        "graph_cosine": graph_similarity["cosine_mean"],
        "graph_frobenius": graph_similarity["frobenius_mean"],
        "neighborhood_stability": neighborhood_stability,
        "metric_deltas": metric_deltas,
        "metric_response_magnitude": metric_response_magnitude,
        "phi_proxy_before": before_phi,
        "phi_proxy_after": after_phi,
        "delta_phi_proxy": after_phi - before_phi,
        "hidden_delta_norm": _max_abs_diff(before_hidden, after_hidden),
    }


def _compare_real_to_controls(
    responses: dict[str, dict[str, Any]],
    *,
    stability_margin: float,
    response_margin: float,
) -> dict[str, Any]:
    real = responses["real"]
    random = responses["random"]
    shuffled = responses["shuffled"]
    best_control_stability = max(
        random["neighborhood_stability"],
        shuffled["neighborhood_stability"],
    )
    best_control_response = max(
        random["metric_response_magnitude"],
        shuffled["metric_response_magnitude"],
    )
    real_more_stable = (
        real["neighborhood_stability"] >= best_control_stability + stability_margin
    )
    real_stronger = (
        real["metric_response_magnitude"] >= best_control_response + response_margin
    )
    return {
        "real_more_stable_than_controls": real_more_stable,
        "real_stronger_than_controls": real_stronger,
        "real_beats_or_stabilizes_vs_controls": real_more_stable or real_stronger,
        "real_minus_best_control_neighborhood_stability": (
            real["neighborhood_stability"] - best_control_stability
        ),
        "real_minus_best_control_response_magnitude": (
            real["metric_response_magnitude"] - best_control_response
        ),
    }


def _first_valid_anchor(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    valid = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
    anchor_indices = valid.to(dtype=torch.long).argmax(dim=1)
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, anchor_indices]


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


def _hidden_norm_ratio(before: list[torch.Tensor], after: list[torch.Tensor]) -> float:
    before_norm = torch.stack([layer.norm() for layer in before]).mean()
    after_norm = torch.stack([layer.norm() for layer in after]).mean()
    return float((after_norm / before_norm.clamp_min(1e-12)).detach().cpu().item())


def _max_abs_diff(before: torch.Tensor, after: torch.Tensor) -> float:
    return float((after - before).abs().max().detach().cpu().item())


def _is_finite(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))
