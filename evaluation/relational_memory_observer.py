"""Phi-gated relational memory observer for the strengthened ERGT program."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from evaluation.information_potential_phi import (
    PhiWeights,
    information_potential_components,
)
from evaluation.reconstruction_gate import reconstruction_deficits
from layers.relational_graph import (
    RelationalGraph,
    make_random_graph_like,
    make_shuffled_graph,
    make_valid_edge_mask_like,
)

MemoryStatus = Literal["pass", "fail"]


@dataclass(frozen=True)
class MemoryConfig:
    decay: float = 0.7
    eta: float = 0.3
    gate_floor: float = 0.05
    min_context_edges: int = 2
    anti_collapse_threshold: float = 0.1

    def validate(self) -> None:
        if not 0 <= self.decay <= 1:
            raise ValueError("decay must be in [0, 1]")
        if not 0 <= self.eta <= 1:
            raise ValueError("eta must be in [0, 1]")
        if not 0 <= self.gate_floor <= 1:
            raise ValueError("gate_floor must be in [0, 1]")
        if self.min_context_edges <= 0:
            raise ValueError("min_context_edges must be positive")
        if not 0 <= self.anti_collapse_threshold <= 1:
            raise ValueError("anti_collapse_threshold must be in [0, 1]")


def build_relational_memory_observer_report(
    *,
    hidden_layers: list[torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    graph_config: dict[str, Any] | None = None,
    seed: int = 2027,
    memory_config: MemoryConfig | None = None,
    score_margin: float = 1e-4,
) -> dict[str, Any]:
    """Build the Phase 6 Phi-gated Relational Memory Observer report."""

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
    instantaneous_sequence = [update["stable_update"] for update in family_updates["real"]]
    no_memory_sequence = [
        _valid_only(graph, family_graphs["valid_edge_masks"][layer_index])
        for layer_index, graph in enumerate(family_graphs["real"])
    ]
    generic_smoothing_sequence = relational_memory_sequence(
        no_memory_sequence,
        memory_config=memory_config,
    )
    target_sequence = [update["stable_update"] for update in family_updates["real"]]

    observers = {
        "real_memory": memory_sequences["real"],
        "random_memory": memory_sequences["random"],
        "shuffled_memory": memory_sequences["shuffled"],
        "instantaneous_eta_1": instantaneous_sequence,
        "generic_smoothing": generic_smoothing_sequence,
        "no_memory": no_memory_sequence,
    }
    observer_metrics = {
        name: memory_quality_metrics(
            sequence,
            target_sequence,
            family_graphs["valid_edge_masks"],
        )
        for name, sequence in observers.items()
    }
    layer_reports = _layer_reports(
        observers,
        target_sequence,
        family_graphs["valid_edge_masks"],
    )

    real_score = observer_metrics["real_memory"]["memory_quality_score"]
    checks = {
        "model_intervention_none": True,
        "w_level_controls_used": True,
        "future_edges_forbidden": all(
            check["future_edges_forbidden"] for check in leakage_checks
        ),
        "future_inputs_forbidden": all(
            check["future_sources_forbidden"] for check in leakage_checks
        ),
        "real_memory_beats_random_memory": _beats(
            real_score,
            observer_metrics["random_memory"]["memory_quality_score"],
            margin=score_margin,
        ),
        "real_memory_beats_shuffled_memory": _beats(
            real_score,
            observer_metrics["shuffled_memory"]["memory_quality_score"],
            margin=score_margin,
        ),
        "real_memory_beats_instantaneous": _beats(
            real_score,
            observer_metrics["instantaneous_eta_1"]["memory_quality_score"],
            margin=score_margin,
        ),
        "real_memory_beats_generic_smoothing": _beats(
            real_score,
            observer_metrics["generic_smoothing"]["memory_quality_score"],
            margin=score_margin,
        ),
        "real_memory_beats_no_memory": _beats(
            real_score,
            observer_metrics["no_memory"]["memory_quality_score"],
            margin=score_margin,
        ),
        "real_memory_not_collapsed": (
            observer_metrics["real_memory"]["anti_collapse_mean"]
            >= memory_config.anti_collapse_threshold
        ),
    }
    status: MemoryStatus = "pass" if all(checks.values()) else "fail"

    return {
        "phase": "phase6_relational_memory_observer",
        "status": status,
        "seed": seed,
        "input_source": input_source,
        "scientific_scope": (
            "smoke_validates_relational_memory_observer_mechanics"
            if input_source == "synthetic_memory_smoke"
            else "observes_provided_hidden_states"
        ),
        "observer_pipeline": (
            "H_t -> W_t family -> Phi/reconstruction gate -> stable_update -> memory"
        ),
        "model_intervention": "none",
        "memory_formula": {
            "formula": "W_mem_t = decay * W_mem_{t-1} + eta * stable_update_t",
            "stable_update": (
                "W_current * row_gate(Phi_t, reconstruction_t) over causal valid edges"
            ),
            "config": asdict(memory_config),
            "target_for_scoring": (
                "next real stable_update; used only for evaluation, never for memory update"
            ),
        },
        "checks": checks,
        "metrics": observer_metrics,
        "layers": layer_reports,
        "controls": {
            "control_generation_level": "W_level_before_distance_normalization",
            "families": [
                "real_memory",
                "random_memory",
                "shuffled_memory",
                "instantaneous_eta_1",
                "generic_smoothing",
                "no_memory",
            ],
        },
        "next_required_step": (
            "causal_shortest_path_geometry" if status == "pass" else "fix_relational_memory"
        ),
    }


def stable_memory_update(
    hidden_states: torch.Tensor,
    graph: torch.Tensor,
    valid_edge_mask: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    graph_config: dict[str, Any] | None = None,
    memory_config: MemoryConfig | None = None,
) -> dict[str, Any]:
    """Build a Phi/reconstruction-gated causal update from one hidden layer."""

    memory_config = memory_config or MemoryConfig()
    memory_config.validate()
    _validate_hidden_and_graph(hidden_states, graph)
    attention_mask = _attention_mask_or_ones(hidden_states, attention_mask)
    valid_edge_mask = _prepare_mask(valid_edge_mask, graph)

    phi = information_potential_components(
        hidden_states,
        graph,
        valid_edge_mask,
        attention_mask=attention_mask,
        weights=PhiWeights(),
    )
    deficits = reconstruction_deficits(
        hidden_states,
        graph,
        valid_edge_mask,
        attention_mask=attention_mask,
        graph_config=graph_config,
        min_context_edges=memory_config.min_context_edges,
    )
    reconstruction_gate = _deficit_to_gate(
        deficits["relational_deficit"],
        deficits["reconstructible_node_mask"],
        floor=memory_config.gate_floor,
    )
    phi_gate = _normalize_gate(
        phi["score"],
        phi["valid_node_mask"],
        floor=memory_config.gate_floor,
    )
    row_gate = torch.sqrt((phi_gate * reconstruction_gate).clamp_min(0.0))
    edge_gate = row_gate.unsqueeze(-1).expand_as(graph)
    stable_update_tensor = torch.where(
        valid_edge_mask,
        graph * edge_gate,
        torch.zeros_like(graph),
    )
    stable_update_tensor = torch.nan_to_num(stable_update_tensor, nan=0.0)
    return {
        "stable_update": stable_update_tensor,
        "row_gate": row_gate,
        "phi_gate": phi_gate,
        "reconstruction_gate": reconstruction_gate,
        "valid_edge_mask": valid_edge_mask,
        "leakage_checks": deficits["leakage_checks"],
    }


def relational_memory_sequence(
    stable_updates: list[torch.Tensor],
    *,
    memory_config: MemoryConfig | None = None,
) -> list[torch.Tensor]:
    """Apply causal relational memory recurrence to a sequence of updates."""

    memory_config = memory_config or MemoryConfig()
    memory_config.validate()
    if not stable_updates:
        raise ValueError("stable_updates must not be empty")

    memory: list[torch.Tensor] = [stable_updates[0].clamp(0.0, 1.0)]
    previous = memory[0]
    for update in stable_updates[1:]:
        current = memory_config.decay * previous + memory_config.eta * update
        current = current.clamp(0.0, 1.0)
        memory.append(current)
        previous = current
    return memory


def synthetic_memory_hidden_layers(
    *,
    seed: int = 2027,
    batch_size: int = 2,
    sequence_length: int = 6,
    hidden_dim: int = 8,
    layers: int = 5,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Create a noisy stable latent field for memory observer smoke tests."""

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
        latent_drift = 0.005 * layer_index
        step_noise = torch.randn(base.shape, generator=generator) * 0.07
        hidden_layers.append(base + latent_drift + step_noise)

    attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long)
    attention_mask[0, -1] = 0
    return hidden_layers, attention_mask


def memory_quality_metrics(
    memory_sequence: list[torch.Tensor],
    target_sequence: list[torch.Tensor],
    valid_edge_masks: list[torch.Tensor],
) -> dict[str, float]:
    """Score memory against next stable target, persistence, and anti-collapse."""

    if len(memory_sequence) != len(target_sequence):
        raise ValueError("memory and target sequences must have the same length")
    if len(memory_sequence) < 2:
        raise ValueError("memory_sequence must contain at least two items")

    prediction_errors = []
    prediction_scores = []
    prediction_cosines = []
    temporal_cosines = []
    anti_collapse_scores = []
    for index in range(len(memory_sequence) - 1):
        mask = valid_edge_masks[index + 1]
        prediction = memory_sequence[index]
        target = target_sequence[index + 1]
        mse = _masked_mse(prediction, target, mask)
        target_variance = _masked_variance(target, mask)
        normalized_error = mse / max(target_variance, 1e-8)
        prediction_errors.append(normalized_error)
        prediction_scores.append(1.0 / (1.0 + normalized_error))
        prediction_cosines.append(_masked_cosine(prediction, target, mask))

    for index in range(1, len(memory_sequence)):
        temporal_cosines.append(
            _masked_cosine(
                memory_sequence[index - 1],
                memory_sequence[index],
                valid_edge_masks[index],
            )
        )
    for memory, mask in zip(memory_sequence, valid_edge_masks, strict=True):
        anti_collapse_scores.append(_memory_anti_collapse(memory, mask))

    prediction_score = _mean(prediction_scores)
    temporal_stability = _mean(temporal_cosines)
    anti_collapse = _mean(anti_collapse_scores)
    cosine_to_next = _mean(prediction_cosines)
    quality = (
        0.45 * prediction_score
        + 0.25 * max(_finite_or_zero(cosine_to_next), 0.0)
        + 0.2 * max(_finite_or_zero(temporal_stability), 0.0)
        + 0.1 * _finite_or_zero(anti_collapse)
    )
    return {
        "memory_quality_score": quality,
        "next_update_prediction_score": prediction_score,
        "next_update_normalized_mse": _mean(prediction_errors),
        "next_update_cosine": cosine_to_next,
        "temporal_stability_cosine": temporal_stability,
        "anti_collapse_mean": anti_collapse,
    }


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
    observers: dict[str, list[torch.Tensor]],
    target_sequence: list[torch.Tensor],
    valid_edge_masks: list[torch.Tensor],
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for layer_index in range(len(target_sequence) - 1):
        mask = valid_edge_masks[layer_index + 1]
        target = target_sequence[layer_index + 1]
        reports[f"transition_{layer_index}_to_{layer_index + 1}"] = {
            name: {
                "normalized_mse_to_next_update": _masked_mse(sequence[layer_index], target, mask)
                / max(_masked_variance(target, mask), 1e-8),
                "cosine_to_next_update": _masked_cosine(sequence[layer_index], target, mask),
            }
            for name, sequence in observers.items()
        }
    return reports


def _deficit_to_gate(
    deficit: torch.Tensor,
    valid_node_mask: torch.Tensor,
    *,
    floor: float,
) -> torch.Tensor:
    finite = deficit[valid_node_mask & torch.isfinite(deficit)]
    if finite.numel() == 0:
        return torch.full_like(deficit, floor)
    scale = finite.mean().abs().clamp_min(1e-8)
    score = torch.exp(-deficit / scale)
    return _normalize_gate(score, valid_node_mask, floor=floor)


def _normalize_gate(
    values: torch.Tensor,
    valid_node_mask: torch.Tensor,
    *,
    floor: float,
) -> torch.Tensor:
    valid = values[valid_node_mask & torch.isfinite(values)]
    if valid.numel() == 0:
        return torch.full_like(values, floor)
    minimum = valid.min()
    maximum = valid.max()
    if torch.isclose(maximum, minimum):
        normalized = torch.ones_like(values)
    else:
        normalized = (values - minimum) / (maximum - minimum).clamp_min(1e-8)
    gate = floor + (1.0 - floor) * normalized.clamp(0.0, 1.0)
    return torch.where(valid_node_mask, gate, torch.full_like(gate, 0.0))


def _memory_anti_collapse(memory: torch.Tensor, valid_edge_mask: torch.Tensor) -> float:
    valid_edge_mask = _prepare_mask(valid_edge_mask, memory)
    weights = torch.where(valid_edge_mask, memory, torch.zeros_like(memory))
    degree = valid_edge_mask.sum(dim=-1).to(dtype=memory.dtype)
    row_sum = weights.sum(dim=-1, keepdim=True)
    probability = weights / row_sum.clamp_min(1e-12)
    participation = 1.0 / probability.square().sum(dim=-1).clamp_min(1e-8)
    spread = ((participation - 1.0) / (degree - 1.0).clamp_min(1.0)).clamp(0.0, 1.0)
    max_probability = probability.max(dim=-1).values
    no_single_lock = (1.0 - ((max_probability - 0.85) / 0.15).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    row_mean = weights.sum(dim=-1) / degree.clamp_min(1.0)
    variance = torch.where(
        valid_edge_mask,
        (memory - row_mean.unsqueeze(-1)) ** 2,
        torch.zeros_like(memory),
    ).sum(dim=-1) / degree.clamp_min(1.0)
    coefficient = torch.sqrt(variance.clamp_min(0.0)) / row_mean.abs().clamp_min(1e-8)
    non_uniform = (0.25 + 0.75 * (coefficient / 0.03).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    valid_nodes = valid_edge_mask.any(dim=-1)
    anti_collapse = torch.exp(
        torch.log(
            torch.stack(
                [
                    spread.clamp_min(1e-8),
                    no_single_lock.clamp_min(1e-8),
                    non_uniform.clamp_min(1e-8),
                ]
            )
        ).mean(dim=0)
    )
    return _finite_mean(anti_collapse[valid_nodes & torch.isfinite(anti_collapse)])


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


def _validate_hidden_and_graph(hidden_states: torch.Tensor, graph: torch.Tensor) -> None:
    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden_dim]")
    if graph.dim() != 4:
        raise ValueError("graph must have shape [batch, heads, sequence, sequence]")
    if hidden_states.shape[:2] != graph.shape[0:1] + graph.shape[-2:-1]:
        raise ValueError("hidden_states and graph batch/sequence dimensions must match")


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
