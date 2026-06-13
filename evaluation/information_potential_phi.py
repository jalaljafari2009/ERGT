"""Information Potential Phi observer for the strengthened ERGT program.

Phi is treated here as an operational stability selector over hidden-state
relations. It is not a loss, an attention bias, or a claim about consciousness.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F

from evaluation.relational_field_observer import (
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

PhiStatus = Literal["pass", "fail"]


@dataclass(frozen=True)
class PhiWeights:
    """Fixed log-linear Phi exponents for `fixed_log_linear_anti_collapse_v1`."""

    coherence: float = 1.0
    coherence_gradient: float = 0.5
    low_local_entropy: float = 0.5
    salience: float = 0.5
    stability: float = 1.0
    reconstruction_score: float = 1.0
    anti_collapse: float = 1.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"Phi weight {name} must be non-negative")


def build_information_potential_report(
    *,
    hidden_layers: list[torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    graph_config: dict[str, Any] | None = None,
    distance_config: dict[str, Any] | None = None,
    seed: int = 2027,
    neighborhood_k: int = 2,
    weights: PhiWeights | None = None,
    separation_margin: float = 1e-7,
    low_entropy_correlation_limit: float = 0.995,
    anti_collapse_threshold: float = 0.2,
) -> dict[str, Any]:
    """Build the Phase 4 Information Potential Phi observer report."""

    weights = weights or PhiWeights()
    weights.validate()

    input_source = "provided_hidden_layers"
    if hidden_layers is None:
        hidden_layers, attention_mask = synthetic_structured_hidden_layers(seed=seed)
        input_source = "synthetic_structured_smoke"
    if not hidden_layers:
        raise ValueError("hidden_layers must not be empty")

    attention_mask = _attention_mask_or_ones(hidden_layers[0], attention_mask)
    graph_builder = RelationalGraph(_normalize_graph_config(graph_config))
    distance_builder = EmergentDistance(_distance_config(distance_config))

    family_states = _build_family_states(
        hidden_layers,
        attention_mask,
        graph_builder,
        distance_builder,
        seed=seed,
    )

    layer_reports: dict[str, Any] = {}
    phi_records: dict[str, list[dict[str, torch.Tensor]]] = {
        "real": [],
        "random": [],
        "shuffled": [],
    }

    for layer_index, hidden_states in enumerate(hidden_layers):
        layer_report: dict[str, Any] = {}
        for family in ("real", "random", "shuffled"):
            state = family_states[family][layer_index]
            previous_stability = None
            if layer_index > 0:
                previous_stability = node_neighborhood_overlap(
                    family_states[family][layer_index - 1]["distance"],
                    state["distance"],
                    neighborhood_k,
                )

            next_stability = None
            if layer_index < len(hidden_layers) - 1:
                next_stability = node_neighborhood_overlap(
                    state["distance"],
                    family_states[family][layer_index + 1]["distance"],
                    neighborhood_k,
                )

            phi = information_potential_components(
                hidden_states,
                state["graph"],
                state["valid_edge_mask"],
                attention_mask=attention_mask,
                stability=previous_stability,
                weights=weights,
            )
            phi["next_neighborhood_stability"] = (
                next_stability
                if next_stability is not None
                else torch.full_like(phi["score"], torch.nan)
            )
            phi["attention_order_proxy"] = (
                phi["coherence"] * phi["low_local_entropy"] * phi["anti_collapse"]
            )
            phi_records[family].append(phi)

            layer_report[family] = {
                "phi": _summary(phi["score"], phi["valid_node_mask"]),
                "components": {
                    key: _summary(phi[key], phi["valid_node_mask"])
                    for key in _component_keys()
                },
                "collapse_components": {
                    key: _summary(phi[key], phi["valid_node_mask"])
                    for key in _collapse_component_keys()
                },
                "field_metrics": state["field_metrics"],
            }

        layer_report["comparison"] = _layer_phi_comparison(layer_report)
        layer_reports[f"layer_{layer_index}"] = layer_report

    aggregate = {
        family: _aggregate_phi_records(records)
        for family, records in phi_records.items()
    }
    real_alignment = _high_phi_alignment(phi_records["real"])
    entropy_correlation = _component_correlation(
        phi_records["real"],
        "score",
        "low_local_entropy",
    )
    component_diversity = _component_diversity(phi_records["real"])

    checks = {
        "formula_registered": True,
        "model_intervention_none": True,
        "w_level_controls_used": True,
        "causal_validity_enforced": aggregate["real"]["causal_validity_mean"] == 1.0,
        "reconstruction_score_present": aggregate["real"]["reconstruction_score_mean"] > 0.0,
        "anti_collapse_present": aggregate["real"]["anti_collapse_mean"] > 0.0,
        "real_phi_separates_from_random": _separates_from_control(
            aggregate["real"]["phi_mean"],
            aggregate["random"]["phi_mean"],
            absolute_margin=separation_margin,
        ),
        "real_phi_separates_from_shuffled": _separates_from_control(
            aggregate["real"]["phi_mean"],
            aggregate["shuffled"]["phi_mean"],
            absolute_margin=separation_margin,
        ),
        "phi_not_low_entropy_only": (
            math.isfinite(entropy_correlation)
            and abs(entropy_correlation) < low_entropy_correlation_limit
            and component_diversity["nontrivial_component_count"] >= 4
        ),
        "high_phi_predicts_stability_or_order": (
            real_alignment["best_target_delta_high_minus_low"] > 0.0
        ),
        "high_phi_not_from_collapse": (
            real_alignment["top_phi_anti_collapse_mean"] >= anti_collapse_threshold
        ),
    }
    status: PhiStatus = "pass" if all(checks.values()) else "fail"

    return {
        "phase": "phase4_information_potential_phi",
        "status": status,
        "seed": seed,
        "input_source": input_source,
        "scientific_scope": (
            "smoke_validates_phi_observer_mechanics"
            if input_source == "synthetic_structured_smoke"
            else "observes_provided_hidden_states"
        ),
        "observer_pipeline": (
            "H -> W_family -> Phi components -> fixed log-linear Phi score"
        ),
        "model_intervention": "none",
        "phi_formula": {
            "name": "fixed_log_linear_anti_collapse_v1",
            "score": (
                "exp(sum_k weight_k * log(component_k + eps)) "
                "* causal_validity"
            ),
            "weights": asdict(weights),
            "components": _component_keys(),
            "collapse_penalties": _collapse_component_keys(),
            "reconstruction_score_policy": (
                "causal_neighbor_hidden_reconstruction_proxy; target hidden is "
                "used only for scoring error, never as reconstruction input"
            ),
            "salience_definition": "hidden_norm",
        },
        "checks": checks,
        "aggregate": aggregate,
        "high_phi_alignment": real_alignment,
        "low_entropy_correlation": entropy_correlation,
        "component_diversity": component_diversity,
        "layers": layer_reports,
        "controls": {
            "control_generation_level": "W_level_before_distance_normalization",
            "families": ["real", "random", "shuffled"],
        },
        "next_required_step": "reconstruction_gate" if status == "pass" else "fix_phi",
    }


def information_potential_components(
    hidden_states: torch.Tensor,
    graph: torch.Tensor,
    valid_edge_mask: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    stability: torch.Tensor | None = None,
    weights: PhiWeights | None = None,
) -> dict[str, torch.Tensor]:
    """Compute node-level Phi components and score for one graph family."""

    _validate_hidden_and_graph(hidden_states, graph)
    weights = weights or PhiWeights()
    weights.validate()

    valid_edge_mask = _prepare_mask(valid_edge_mask, graph)
    attention_mask = _attention_mask_or_ones(hidden_states, attention_mask)
    valid_node_mask = valid_edge_mask.any(dim=-1)
    if attention_mask is not None:
        valid_node_mask &= attention_mask[:, None, :].to(dtype=torch.bool, device=graph.device)

    probability = _row_probability(graph, valid_edge_mask)
    coherence = _node_coherence(hidden_states, probability, valid_node_mask)
    coherence_gradient = _coherence_gradient(coherence, probability, valid_node_mask)
    low_local_entropy, normalized_entropy = _low_local_entropy(
        probability,
        valid_edge_mask,
        valid_node_mask,
    )
    salience = _hidden_norm_salience(hidden_states, valid_node_mask)
    stability_component = _stability_or_neutral(stability, valid_node_mask, graph)
    reconstruction_score = causal_neighbor_reconstruction_score(
        hidden_states,
        probability,
        valid_node_mask,
    )
    collapse = anti_collapse_components(
        graph,
        probability,
        valid_edge_mask,
        normalized_entropy,
        valid_node_mask,
    )
    causal_validity = _causal_validity(valid_edge_mask, valid_node_mask)

    components = {
        "coherence": coherence,
        "coherence_gradient": coherence_gradient,
        "low_local_entropy": low_local_entropy,
        "salience": salience,
        "stability": stability_component,
        "reconstruction_score": reconstruction_score,
        "causal_validity": causal_validity,
        "anti_collapse": collapse["anti_collapse"],
        **collapse,
        "valid_node_mask": valid_node_mask,
    }
    components["score"] = phi_score_from_components(components, weights=weights)
    return components


def phi_score_from_components(
    components: dict[str, torch.Tensor],
    *,
    weights: PhiWeights | None = None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute fixed log-linear Phi from bounded component tensors."""

    weights = weights or PhiWeights()
    weights.validate()
    valid_node_mask = components["valid_node_mask"]
    score = torch.zeros_like(components["coherence"])
    for key, weight in asdict(weights).items():
        if weight == 0:
            continue
        component = components[key].clamp(min=epsilon, max=1.0)
        score = score + weight * torch.log(component)
    score = torch.exp(score) * components["causal_validity"].clamp(min=0.0, max=1.0)
    return torch.where(valid_node_mask, score, torch.full_like(score, torch.nan))


def causal_neighbor_reconstruction_score(
    hidden_states: torch.Tensor,
    probability: torch.Tensor,
    valid_node_mask: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Score causal-prefix reconstruction by W-weighted past hidden states."""

    reconstruction = torch.einsum("bhij,bjd->bhid", probability, hidden_states)
    target = hidden_states[:, None, :, :]
    mse = ((reconstruction - target) ** 2).mean(dim=-1)
    valid_mse = mse[valid_node_mask & torch.isfinite(mse)]
    if valid_mse.numel() == 0:
        return torch.full_like(mse, torch.nan)
    scale = valid_mse.mean().clamp_min(epsilon)
    score = torch.exp(-mse / scale)
    return torch.where(valid_node_mask, score, torch.full_like(score, torch.nan))


def anti_collapse_components(
    graph: torch.Tensor,
    probability: torch.Tensor,
    valid_edge_mask: torch.Tensor,
    normalized_entropy: torch.Tensor,
    valid_node_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return anti-collapse components required by the Phi contract."""

    valid_weights = torch.where(valid_edge_mask, graph, torch.zeros_like(graph))
    degree = valid_edge_mask.sum(dim=-1).to(dtype=graph.dtype)
    row_mean = valid_weights.sum(dim=-1) / degree.clamp_min(1.0)
    row_variance = (
        torch.where(valid_edge_mask, (graph - row_mean.unsqueeze(-1)) ** 2, torch.zeros_like(graph))
    ).sum(dim=-1) / degree.clamp_min(1.0)
    row_cv = torch.sqrt(row_variance.clamp_min(0.0)) / row_mean.abs().clamp_min(1e-8)
    non_uniform = (0.25 + 0.75 * (row_cv / 0.03).clamp(0.0, 1.0)).clamp(0.0, 1.0)

    participation = 1.0 / (probability.square().sum(dim=-1).clamp_min(1e-8))
    participation_denominator = (degree - 1.0).clamp_min(1.0)
    spread = ((participation - 1.0) / participation_denominator).clamp(0.0, 1.0)
    spread = torch.where(degree > 1, spread, torch.zeros_like(spread))

    max_probability = probability.max(dim=-1).values
    no_single_lock = (1.0 - ((max_probability - 0.85) / 0.15).clamp(0.0, 1.0)).clamp(0.0, 1.0)

    not_uniform_entropy = 1.0 - ((normalized_entropy - 0.995) / 0.005).clamp(0.0, 1.0)
    not_entropy_collapsed = (normalized_entropy / 0.02).clamp(0.0, 1.0)
    entropy_band = 0.2 + 0.8 * torch.minimum(not_uniform_entropy, not_entropy_collapsed)
    entropy_band = torch.where(degree > 1, entropy_band, torch.full_like(entropy_band, 0.1))

    diagonal = torch.diagonal(graph, dim1=-2, dim2=-1)
    diagonal_ratio = diagonal / row_mean.abs().clamp_min(1e-8)
    not_diagonal_dominated = (
        1.0 - ((diagonal_ratio - 3.0) / 3.0).clamp(0.0, 1.0)
    ).clamp(0.0, 1.0)

    pieces = {
        "anti_uniform": non_uniform,
        "anti_over_sparse": spread,
        "anti_single_token_lock": no_single_lock,
        "anti_entropy_collapse": entropy_band,
        "anti_diagonal_domination": not_diagonal_dominated,
    }
    stacked = torch.stack([value.clamp_min(1e-8) for value in pieces.values()])
    anti_collapse = torch.exp(torch.log(stacked).mean(dim=0))
    pieces["anti_collapse"] = anti_collapse
    return {
        key: torch.where(valid_node_mask, value, torch.full_like(value, torch.nan))
        for key, value in pieces.items()
    }


def node_neighborhood_overlap(
    distance_a: torch.Tensor,
    distance_b: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Return per-node finite-neighborhood overlap between two distances."""

    if k <= 0:
        raise ValueError("k must be positive")
    if distance_a.shape != distance_b.shape:
        raise ValueError("distance_a and distance_b must have the same shape")
    if distance_a.dim() != 4:
        raise ValueError("distance tensors must have shape [batch, heads, sequence, sequence]")

    sequence_length = distance_a.size(-1)
    diagonal = torch.eye(sequence_length, dtype=torch.bool, device=distance_a.device).view(
        1,
        1,
        sequence_length,
        sequence_length,
    )
    a = distance_a.masked_fill(diagonal, torch.inf)
    b = distance_b.masked_fill(diagonal, torch.inf)
    output = torch.full(distance_a.shape[:-1], torch.nan, dtype=distance_a.dtype)

    flat_a = a.reshape(-1, sequence_length)
    flat_b = b.reshape(-1, sequence_length)
    flat_output = output.reshape(-1)
    for row_index, (row_a, row_b) in enumerate(zip(flat_a, flat_b, strict=True)):
        neighbors_a = _finite_nearest_indices(row_a, k)
        neighbors_b = _finite_nearest_indices(row_b, k)
        if not neighbors_a or not neighbors_b:
            continue
        denominator = max(len(neighbors_a), len(neighbors_b), 1)
        flat_output[row_index] = len(set(neighbors_a) & set(neighbors_b)) / denominator
    return output


def _build_family_states(
    hidden_layers: list[torch.Tensor],
    attention_mask: torch.Tensor,
    graph_builder: RelationalGraph,
    distance_builder: EmergentDistance,
    *,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    family_states: dict[str, list[dict[str, Any]]] = {
        "real": [],
        "random": [],
        "shuffled": [],
    }
    for layer_index, hidden_states in enumerate(hidden_layers):
        graph = graph_builder(hidden_states, attention_mask=attention_mask)
        valid_edge_mask = make_valid_edge_mask_like(graph, attention_mask=attention_mask)
        generator = torch.Generator(device=graph.device)
        generator.manual_seed(seed + layer_index)
        graphs = {
            "real": graph,
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
        for family, family_graph in graphs.items():
            distance = distance_builder(family_graph, attention_mask=attention_mask)
            family_states[family].append(
                {
                    "graph": family_graph,
                    "distance": distance,
                    "valid_edge_mask": valid_edge_mask,
                    "field_metrics": field_metrics(
                        hidden_states,
                        family_graph,
                        distance,
                        valid_edge_mask,
                    ),
                }
            )
    return family_states


def _node_coherence(
    hidden_states: torch.Tensor,
    probability: torch.Tensor,
    valid_node_mask: torch.Tensor,
) -> torch.Tensor:
    normalized = F.normalize(hidden_states, p=2, dim=-1)
    similarity = normalized @ normalized.transpose(-2, -1)
    similarity = similarity[:, None, :, :].expand_as(probability)
    raw = (probability * similarity).sum(dim=-1)
    bounded = ((raw + 1.0) / 2.0).clamp(0.0, 1.0)
    return torch.where(valid_node_mask, bounded, torch.full_like(bounded, torch.nan))


def _coherence_gradient(
    coherence: torch.Tensor,
    probability: torch.Tensor,
    valid_node_mask: torch.Tensor,
) -> torch.Tensor:
    context_coherence = torch.nan_to_num(coherence, nan=0.0)
    neighbor_coherence = (probability * context_coherence.unsqueeze(-2)).sum(dim=-1)
    gradient = (coherence - neighbor_coherence).abs().clamp(0.0, 1.0).sqrt()
    return torch.where(valid_node_mask, gradient, torch.full_like(gradient, torch.nan))


def _low_local_entropy(
    probability: torch.Tensor,
    valid_edge_mask: torch.Tensor,
    valid_node_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    entropy = -(probability * torch.log(probability.clamp_min(1e-12))).sum(dim=-1)
    degree = valid_edge_mask.sum(dim=-1).to(dtype=probability.dtype)
    normalized = torch.where(
        degree > 1,
        entropy / torch.log(degree.clamp_min(2.0)),
        torch.ones_like(entropy),
    ).clamp(0.0, 1.0)
    low_local_entropy = torch.where(
        degree > 1,
        1.0 - normalized,
        torch.zeros_like(normalized),
    )
    nan = torch.full_like(low_local_entropy, torch.nan)
    return (
        torch.where(valid_node_mask, low_local_entropy, nan),
        torch.where(valid_node_mask, normalized, nan),
    )


def _hidden_norm_salience(
    hidden_states: torch.Tensor,
    valid_node_mask: torch.Tensor,
) -> torch.Tensor:
    hidden_norm = hidden_states.norm(dim=-1)
    expanded = hidden_norm[:, None, :].expand_as(valid_node_mask).to(dtype=hidden_states.dtype)
    valid = expanded[valid_node_mask & torch.isfinite(expanded)]
    if valid.numel() == 0:
        return torch.full_like(expanded, torch.nan)
    z = (expanded - valid.mean()) / valid.std(unbiased=False).clamp_min(1e-8)
    salience = torch.sigmoid(z)
    return torch.where(valid_node_mask, salience, torch.full_like(salience, torch.nan))


def _stability_or_neutral(
    stability: torch.Tensor | None,
    valid_node_mask: torch.Tensor,
    graph: torch.Tensor,
) -> torch.Tensor:
    if stability is None:
        neutral = torch.ones(graph.shape[:-1], dtype=graph.dtype, device=graph.device)
        return torch.where(valid_node_mask, neutral, torch.full_like(neutral, torch.nan))
    stability = stability.to(dtype=graph.dtype, device=graph.device)
    if stability.shape != graph.shape[:-1]:
        raise ValueError("stability must have shape [batch, heads, sequence]")
    stability = stability.clamp(0.0, 1.0)
    return torch.where(valid_node_mask, stability, torch.full_like(stability, torch.nan))


def _causal_validity(
    valid_edge_mask: torch.Tensor,
    valid_node_mask: torch.Tensor,
) -> torch.Tensor:
    sequence_length = valid_edge_mask.size(-1)
    positions = torch.arange(sequence_length, device=valid_edge_mask.device)
    future = positions.view(1, sequence_length) > positions.view(sequence_length, 1)
    future = future.view(1, 1, sequence_length, sequence_length)
    future_is_excluded = ~valid_edge_mask[future.expand_as(valid_edge_mask)].any()
    value = 1.0 if bool(future_is_excluded.item()) else 0.0
    causal = torch.full(
        valid_node_mask.shape,
        value,
        dtype=torch.float32,
        device=valid_node_mask.device,
    )
    return torch.where(valid_node_mask, causal, torch.full_like(causal, torch.nan))


def _row_probability(graph: torch.Tensor, valid_edge_mask: torch.Tensor) -> torch.Tensor:
    weights = torch.where(
        valid_edge_mask & torch.isfinite(graph),
        graph.clamp_min(0.0),
        torch.zeros_like(graph),
    )
    row_sum = weights.sum(dim=-1, keepdim=True)
    return weights / row_sum.clamp_min(1e-12)


def _layer_phi_comparison(layer_report: dict[str, Any]) -> dict[str, float]:
    real = layer_report["real"]["phi"]["mean"]
    random = layer_report["random"]["phi"]["mean"]
    shuffled = layer_report["shuffled"]["phi"]["mean"]
    return {
        "real_minus_random_phi_mean": real - random,
        "real_minus_shuffled_phi_mean": real - shuffled,
    }


def _separates_from_control(
    real_value: float,
    control_value: float,
    *,
    absolute_margin: float,
    relative_margin: float = 0.01,
) -> bool:
    if not math.isfinite(real_value) or not math.isfinite(control_value):
        return False
    absolute_delta = real_value - control_value
    relative_delta = absolute_delta / max(abs(control_value), 1e-12)
    return absolute_delta > absolute_margin and relative_delta > relative_margin


def _aggregate_phi_records(records: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    keys = ["score", *_component_keys()]
    return {
        _summary_key(key): _finite_mean(_cat_valid(records, key))
        for key in keys
    }


def _high_phi_alignment(
    records: list[dict[str, torch.Tensor]],
    fraction: float = 0.33,
) -> dict[str, Any]:
    phi = _cat_valid(records, "score")
    anti_collapse = _cat_valid(records, "anti_collapse")
    if phi.numel() == 0:
        return {
            "top_phi_anti_collapse_mean": math.nan,
            "best_target_delta_high_minus_low": math.nan,
            "targets": {},
        }

    order = torch.argsort(phi)
    count = max(1, int(math.ceil(phi.numel() * fraction)))
    high_idx = order[-count:]
    targets = {}
    best_delta = -math.inf
    for key in (
        "next_neighborhood_stability",
        "reconstruction_score",
        "attention_order_proxy",
    ):
        paired_phi, target = _cat_paired(records, key)
        if target.numel() < 2:
            continue
        paired_order = torch.argsort(paired_phi)
        paired_count = max(1, int(math.ceil(paired_phi.numel() * fraction)))
        paired_low_idx = paired_order[:paired_count]
        paired_high_idx = paired_order[-paired_count:]
        low_values = target[paired_low_idx]
        high_values = target[paired_high_idx]
        low_mean = _finite_mean(low_values)
        high_mean = _finite_mean(high_values)
        delta = high_mean - low_mean
        targets[key] = {
            "low_phi_mean": low_mean,
            "high_phi_mean": high_mean,
            "delta_high_minus_low": delta,
        }
        if math.isfinite(delta):
            best_delta = max(best_delta, delta)

    return {
        "top_phi_anti_collapse_mean": _finite_mean(anti_collapse[high_idx]),
        "best_target_delta_high_minus_low": best_delta,
        "targets": targets,
    }


def _component_correlation(
    records: list[dict[str, torch.Tensor]],
    key_a: str,
    key_b: str,
) -> float:
    a = _cat_valid(records, key_a)
    b = _cat_valid(records, key_b)
    if a.numel() != b.numel() or a.numel() < 2:
        return math.nan
    finite = torch.isfinite(a) & torch.isfinite(b)
    if int(finite.sum().item()) < 2:
        return math.nan
    a = a[finite] - a[finite].mean()
    b = b[finite] - b[finite].mean()
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if denominator <= 0:
        return math.nan
    return _to_float((a * b).sum() / denominator)


def _component_diversity(records: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    variances = {}
    for key in _component_keys():
        values = _cat_valid(records, key)
        if values.numel() <= 1:
            variance = math.nan
        else:
            variance = _to_float(values.var(unbiased=False))
        variances[key] = variance
    return {
        "component_variance": variances,
        "nontrivial_component_count": sum(
            1 for value in variances.values() if math.isfinite(value) and value > 1e-8
        ),
    }


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
        valid = record["valid_node_mask"] & torch.isfinite(record[key])
        values.append(record[key][valid])
    if not values:
        return torch.empty(0)
    return torch.cat(values)


def _cat_paired(
    records: list[dict[str, torch.Tensor]],
    target_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    phi_values = []
    target_values = []
    for record in records:
        valid = (
            record["valid_node_mask"]
            & torch.isfinite(record["score"])
            & torch.isfinite(record[target_key])
        )
        phi_values.append(record["score"][valid])
        target_values.append(record[target_key][valid])
    if not phi_values:
        return torch.empty(0), torch.empty(0)
    return torch.cat(phi_values), torch.cat(target_values)


def _finite_mean(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return math.nan
    return _to_float(finite.mean())


def _summary_key(component_key: str) -> str:
    return "phi_mean" if component_key == "score" else f"{component_key}_mean"


def _component_keys() -> list[str]:
    return [
        "coherence",
        "coherence_gradient",
        "low_local_entropy",
        "salience",
        "stability",
        "reconstruction_score",
        "causal_validity",
        "anti_collapse",
    ]


def _collapse_component_keys() -> list[str]:
    return [
        "anti_uniform",
        "anti_over_sparse",
        "anti_single_token_lock",
        "anti_entropy_collapse",
        "anti_diagonal_domination",
    ]


def _finite_nearest_indices(row: torch.Tensor, k: int) -> list[int]:
    finite_indices = torch.where(torch.isfinite(row))[0]
    if finite_indices.numel() == 0:
        return []
    finite_values = row[finite_indices]
    order = torch.argsort(finite_values)[:k]
    return [int(value) for value in finite_indices[order].detach().cpu().tolist()]


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


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
