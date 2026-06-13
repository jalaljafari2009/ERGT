"""GeoAttention v2 injection smoke report for the strengthened ERGT program."""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
import torch.nn.functional as F

from attention.geo_attention import GeoAttention, GeoAttentionConfig
from evaluation.relational_memory_observer import (
    MemoryConfig,
    relational_memory_sequence,
    stable_memory_update,
    synthetic_memory_hidden_layers,
)
from layers.relational_graph import RelationalGraph, make_valid_edge_mask_like
from models.transformer_baseline import CausalSelfAttention, TransformerBaselineConfig

GeoAttentionV2Status = Literal["pass", "fail"]

GEOATTENTION_V2_MODES = [
    "real_stable_causal_d",
    "random_stable_causal_d",
    "shuffled_stable_causal_d",
    "instantaneous_real_d",
    "pairwise_real_d",
    "no_memory_real_d",
]


def build_geoattention_v2_report(
    *,
    hidden_layers: list[torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    graph_config: dict[str, Any] | None = None,
    distance_config: dict[str, Any] | None = None,
    seed: int = 2027,
    alpha: float = 2.0,
    max_causal_step: int = 1,
    memory_config: MemoryConfig | None = None,
    score_margin: float = 1e-4,
    alpha_zero_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Build a falsifiable GeoAttention v2 mechanics report."""

    memory_config = memory_config or MemoryConfig()
    memory_config.validate()
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    input_source = "provided_hidden_layers"
    if hidden_layers is None:
        hidden_layers, attention_mask = synthetic_memory_hidden_layers(seed=seed)
        input_source = "synthetic_memory_smoke"
    if len(hidden_layers or []) < 2:
        raise ValueError("hidden_layers must contain at least two layers/steps")

    attention_mask = _attention_mask_or_ones(hidden_layers[0], attention_mask)
    graph_config = _normalize_graph_config(graph_config)
    distance_config = _normalize_distance_config(distance_config)
    target_memory, valid_edge_masks = _real_memory_targets(
        hidden_layers,
        attention_mask,
        graph_config=graph_config,
        memory_config=memory_config,
    )

    mode_records = {
        mode: _geometry_attention_sequence(
            hidden_layers,
            attention_mask,
            mode=mode,
            graph_config=graph_config,
            distance_config=distance_config,
            memory_config=memory_config,
            alpha=alpha,
            max_causal_step=max_causal_step,
        )
        for mode in GEOATTENTION_V2_MODES
    }
    metrics = {
        mode: attention_alignment_metrics(
            record["attention_weights"],
            target_memory,
            valid_edge_masks,
        )
        for mode, record in mode_records.items()
    }
    real_score = metrics["real_stable_causal_d"]["attention_alignment_score"]
    alpha_zero = alpha_zero_neutrality_check(
        hidden_layers[0],
        graph_config=graph_config,
        distance_config=distance_config,
        memory_config=memory_config,
        max_causal_step=max_causal_step,
    )

    checks = {
        "stable_causal_geometry_injected": True,
        "required_controls_present": sorted(mode_records) == sorted(GEOATTENTION_V2_MODES),
        "alpha_zero_matches_baseline": alpha_zero["max_abs_delta"] <= alpha_zero_tolerance,
        "future_attention_forbidden": all(
            future_attention_forbidden(weights)
            for record in mode_records.values()
            for weights in record["attention_weights"]
        ),
        "real_uses_memory_after_first_layer": any(
            item["diagnostics"]["geometry_memory_used"]
            for item in mode_records["real_stable_causal_d"]["diagnostics"][1:]
        ),
    }
    strict_gate_checks = {
        "real_stable_beats_random": _beats(
            real_score,
            metrics["random_stable_causal_d"]["attention_alignment_score"],
            margin=score_margin,
        ),
        "real_stable_beats_shuffled": _beats(
            real_score,
            metrics["shuffled_stable_causal_d"]["attention_alignment_score"],
            margin=score_margin,
        ),
        "real_stable_beats_instantaneous": _beats(
            real_score,
            metrics["instantaneous_real_d"]["attention_alignment_score"],
            margin=score_margin,
        ),
        "real_stable_beats_pairwise": _beats(
            real_score,
            metrics["pairwise_real_d"]["attention_alignment_score"],
            margin=score_margin,
        ),
        "real_stable_beats_no_memory": _beats(
            real_score,
            metrics["no_memory_real_d"]["attention_alignment_score"],
            margin=score_margin,
        ),
    }
    status: GeoAttentionV2Status = "pass" if all(checks.values()) else "fail"
    strict_gate_status = "pass" if all(strict_gate_checks.values()) else "needs_training_run"

    return {
        "phase": "phase8_geoattention_v2",
        "status": status,
        "seed": seed,
        "input_source": input_source,
        "scientific_scope": (
            "smoke_validates_geoattention_v2_injection_mechanics"
            if input_source == "synthetic_memory_smoke"
            else "observes_provided_hidden_states"
        ),
        "injection_formula": "logits = QK/sqrt(d) - alpha * D_stable",
        "construction": "H -> W_t memory -> D_causal -> normalized D_stable -> attention",
        "alpha": alpha,
        "checks": checks,
        "strict_gate_status": strict_gate_status,
        "strict_gate_checks": strict_gate_checks,
        "metrics": metrics,
        "alpha_zero": alpha_zero,
        "controls": {
            "families": GEOATTENTION_V2_MODES,
            "baseline_policy": "checked by alpha_zero weight-copy neutrality",
        },
        "next_required_step": _next_required_step(status, strict_gate_status),
    }


def attention_alignment_metrics(
    attention_sequence: list[torch.Tensor],
    target_sequence: list[torch.Tensor],
    valid_edge_masks: list[torch.Tensor],
) -> dict[str, float]:
    """Score geometry-only attention against next real stable relational target."""

    if len(attention_sequence) != len(target_sequence):
        raise ValueError("attention and target sequences must have the same length")
    if len(attention_sequence) < 2:
        raise ValueError("attention_sequence must contain at least two items")

    mse_scores = []
    cosines = []
    entropies = []
    for index in range(len(attention_sequence) - 1):
        mask = valid_edge_masks[index + 1]
        predicted = _row_probability(attention_sequence[index], mask)
        target = _row_probability(target_sequence[index + 1], mask)
        mse = _masked_mse(predicted, target, mask)
        variance = _masked_variance(target, mask)
        normalized_error = mse / max(variance, 1e-8)
        mse_scores.append(1.0 / (1.0 + normalized_error))
        cosines.append(_masked_cosine(predicted, target, mask))
        entropies.append(_masked_entropy(predicted, mask))

    prediction_score = _mean(mse_scores)
    cosine = _mean(cosines)
    entropy = _mean(entropies)
    score = 0.65 * prediction_score + 0.25 * max(_finite_or_zero(cosine), 0.0)
    score += 0.10 * (1.0 / (1.0 + max(_finite_or_zero(entropy), 0.0)))
    return {
        "attention_alignment_score": score,
        "target_prediction_score": prediction_score,
        "target_cosine": cosine,
        "valid_attention_entropy": entropy,
    }


def _next_required_step(status: str, strict_gate_status: str) -> str:
    if status != "pass":
        return "fix_geoattention_v2"
    if strict_gate_status == "pass":
        return "auxiliary_physics_loss"
    return "run_geoattention_v2_training_controls"


def alpha_zero_neutrality_check(
    hidden_states: torch.Tensor,
    *,
    graph_config: dict[str, Any],
    distance_config: dict[str, Any],
    memory_config: MemoryConfig,
    max_causal_step: int,
) -> dict[str, float]:
    """Check that GeoAttention v2 with alpha=0 is numerically baseline-neutral."""

    torch.manual_seed(17)
    baseline_config = TransformerBaselineConfig(
        vocab_size=16,
        context_length=hidden_states.size(1),
        n_layers=1,
        n_heads=2,
        hidden_dim=hidden_states.size(-1),
        ffn_dim=hidden_states.size(-1) * 2,
        dropout=0.0,
    )
    baseline = CausalSelfAttention(baseline_config)
    geo = _attention(
        mode="real_stable_causal_d",
        hidden_dim=hidden_states.size(-1),
        alpha=0.0,
        graph_config=graph_config,
        distance_config=distance_config,
        memory_config=memory_config,
        max_causal_step=max_causal_step,
    )
    geo.qkv_proj.load_state_dict(baseline.qkv_proj.state_dict())
    geo.out_proj.load_state_dict(baseline.out_proj.state_dict())
    baseline_output = baseline(hidden_states)
    geo_output = geo(hidden_states)
    delta = (baseline_output - geo_output).abs()
    return {
        "max_abs_delta": _to_float(delta.max()),
        "mean_abs_delta": _to_float(delta.mean()),
    }


def future_attention_forbidden(attention_weights: torch.Tensor) -> bool:
    sequence_length = attention_weights.size(-1)
    positions = torch.arange(sequence_length, device=attention_weights.device)
    future = positions.view(1, sequence_length) > positions.view(sequence_length, 1)
    future = future.view(1, 1, sequence_length, sequence_length)
    future_values = attention_weights[future.expand_as(attention_weights)]
    return bool(torch.allclose(future_values, torch.zeros_like(future_values)))


def _geometry_attention_sequence(
    hidden_layers: list[torch.Tensor],
    attention_mask: torch.Tensor,
    *,
    mode: str,
    graph_config: dict[str, Any],
    distance_config: dict[str, Any],
    memory_config: MemoryConfig,
    alpha: float,
    max_causal_step: int,
) -> dict[str, Any]:
    attention = _attention(
        mode=mode,
        hidden_dim=hidden_layers[0].size(-1),
        alpha=alpha,
        graph_config=graph_config,
        distance_config=distance_config,
        memory_config=memory_config,
        max_causal_step=max_causal_step,
    )
    geometry_memory = None
    attention_weights = []
    diagnostics = []
    for hidden_states in hidden_layers:
        distance_result = attention.compute_distance(
            hidden_states,
            attention_mask=attention_mask,
            geometry_memory=geometry_memory,
            return_memory=True,
        )
        distance = attention._broadcast_distance(  # noqa: SLF001
            distance_result["distance"],
            torch.zeros(hidden_states.size(0), 2, hidden_states.size(1), hidden_states.size(1)),
        )
        geometry_memory = distance_result["geometry_memory"]
        weights = _geometry_only_attention(attention, distance, attention_mask)
        attention_weights.append(weights)
        diagnostics.append(
            {
                "diagnostics": distance_result["metadata"],
                "distance": distance,
            }
        )
    return {"attention_weights": attention_weights, "diagnostics": diagnostics}


def _geometry_only_attention(
    attention: GeoAttention,
    distance: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    qk_logits = torch.zeros_like(distance)
    logits = attention.apply_geometry_bias(qk_logits, distance, attention.alpha())
    logits = attention.apply_attention_mask(logits, attention_mask)
    return F.softmax(logits, dim=-1)


def _real_memory_targets(
    hidden_layers: list[torch.Tensor],
    attention_mask: torch.Tensor,
    *,
    graph_config: dict[str, Any],
    memory_config: MemoryConfig,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    graph_builder = RelationalGraph(graph_config)
    updates = []
    valid_masks = []
    for hidden_states in hidden_layers:
        graph = graph_builder(hidden_states, attention_mask=attention_mask)
        valid_edge_mask = make_valid_edge_mask_like(graph, attention_mask=attention_mask)
        update = stable_memory_update(
            hidden_states,
            graph,
            valid_edge_mask,
            attention_mask=attention_mask,
            graph_config=graph_config,
            memory_config=memory_config,
        )
        updates.append(update["stable_update"])
        valid_masks.append(valid_edge_mask)
    return relational_memory_sequence(updates, memory_config=memory_config), valid_masks


def _attention(
    *,
    mode: str,
    hidden_dim: int,
    alpha: float,
    graph_config: dict[str, Any],
    distance_config: dict[str, Any],
    memory_config: MemoryConfig,
    max_causal_step: int,
) -> GeoAttention:
    return GeoAttention(
        GeoAttentionConfig(
            n_heads=2,
            hidden_dim=hidden_dim,
            dropout=0.0,
            distance_mode=mode,
            alpha_initial_value=alpha,
            gradient_mode="detached_d",
            memory_decay=memory_config.decay,
            memory_eta=memory_config.eta,
            memory_gate_floor=memory_config.gate_floor,
            memory_min_context_edges=memory_config.min_context_edges,
            max_causal_step=max_causal_step,
        ),
        relational_graph_config=graph_config,
        distance_config=distance_config,
    )


def _row_probability(values: torch.Tensor, valid_edge_mask: torch.Tensor) -> torch.Tensor:
    valid_edge_mask = valid_edge_mask.to(dtype=torch.bool, device=values.device)
    if valid_edge_mask.size(1) == 1 and values.size(1) != 1:
        valid_edge_mask = valid_edge_mask.expand_as(values)
    weights = torch.where(valid_edge_mask, values.clamp_min(0.0), torch.zeros_like(values))
    row_sum = weights.sum(dim=-1, keepdim=True)
    return weights / row_sum.clamp_min(1e-12)


def _masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    b = _expand_like(b, a)
    mask = _prepare_mask(mask, a) & torch.isfinite(a) & torch.isfinite(b)
    if not bool(mask.any().item()):
        return math.nan
    return _to_float(((a - b) ** 2)[mask].mean())


def _masked_variance(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask = _prepare_mask(mask, values) & torch.isfinite(values)
    if not bool(mask.any().item()):
        return math.nan
    return _to_float(values[mask].var(unbiased=False))


def _masked_cosine(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    b = _expand_like(b, a)
    mask = _prepare_mask(mask, a) & torch.isfinite(a) & torch.isfinite(b)
    if int(mask.sum().item()) < 2:
        return math.nan
    a_values = a[mask]
    b_values = b[mask]
    denominator = torch.linalg.vector_norm(a_values) * torch.linalg.vector_norm(b_values)
    if denominator <= 0:
        return math.nan
    return _to_float((a_values * b_values).sum() / denominator)


def _masked_entropy(probability: torch.Tensor, mask: torch.Tensor) -> float:
    mask = _prepare_mask(mask, probability) & torch.isfinite(probability)
    if not bool(mask.any().item()):
        return math.nan
    selected = probability[mask].clamp_min(1e-12)
    return _to_float(-(selected * torch.log(selected)).mean())


def _prepare_mask(valid_edge_mask: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
    valid_edge_mask = valid_edge_mask.to(dtype=torch.bool, device=graph.device)
    if valid_edge_mask.shape == graph.shape:
        return valid_edge_mask
    if valid_edge_mask.size(1) == 1:
        return valid_edge_mask.expand_as(graph)
    raise ValueError("valid_edge_mask must match graph shape or be head-shared")


def _expand_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if tensor.shape == reference.shape:
        return tensor
    if tensor.dim() == reference.dim() and tensor.size(1) == 1:
        return tensor.expand_as(reference)
    raise ValueError("tensor must match reference shape or be head-shared")


def _normalize_graph_config(graph_config: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(graph_config or {"kernel": "sigmoid_cosine", "normalize_hidden": True})
    if normalized.get("diagonal_policy", "keep_for_distance") == "keep_for_distance":
        normalized["diagonal_policy"] = "keep"
    return normalized


def _normalize_distance_config(distance_config: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "epsilon": 1e-6,
        "normalization": "offdiag_zscore_clamp",
        "clip_value": 5.0,
        "diagonal_policy": "zero",
        "causal_runtime_distance": True,
        **(distance_config or {}),
    }


def _attention_mask_or_ones(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if attention_mask is not None:
        return attention_mask
    return torch.ones(hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device)


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


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
