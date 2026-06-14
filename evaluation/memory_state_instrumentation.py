"""Memory-state instrumentation for open adaptive ERGT control."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

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

MEMORY_STATE_FIELDS = [
    "memory_decay",
    "memory_eta",
    "gate_floor",
    "memory_stability",
    "memory_turnover",
    "memory_edge_density",
    "memory_persistence",
    "memory_spectral_entropy",
    "memory_effective_rank",
    "memory_rigidity",
    "noise_risk",
]


@dataclass(frozen=True)
class MemoryInstrumentationConfig:
    edge_threshold: float = 0.05
    turnover_scale: float = 0.05
    rigidity_max_probability: float = 0.85
    rigidity_min_effective_rank_ratio: float = 0.2

    def validate(self) -> None:
        if self.edge_threshold < 0:
            raise ValueError("edge_threshold must be non-negative")
        if self.turnover_scale <= 0:
            raise ValueError("turnover_scale must be positive")
        if not 0 < self.rigidity_max_probability <= 1:
            raise ValueError("rigidity_max_probability must be in (0, 1]")
        if not 0 <= self.rigidity_min_effective_rank_ratio <= 1:
            raise ValueError("rigidity_min_effective_rank_ratio must be in [0, 1]")


def build_memory_state_instrumentation_report(
    *,
    hidden_layers: list[torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    graph_config: dict[str, Any] | None = None,
    seed: int = 2027,
    memory_config: MemoryConfig | None = None,
    instrumentation_config: MemoryInstrumentationConfig | None = None,
) -> dict[str, Any]:
    """Build the stage-4 Memory State Instrumentation report."""

    from evaluation.unified_telemetry_schema import build_unified_telemetry_schema_report

    memory_config = memory_config or MemoryConfig()
    instrumentation_config = instrumentation_config or MemoryInstrumentationConfig()
    memory_config.validate()
    instrumentation_config.validate()

    input_source = "provided_hidden_layers"
    if hidden_layers is None:
        hidden_layers, attention_mask = synthetic_memory_hidden_layers(seed=seed)
        input_source = "synthetic_memory_smoke"
    if len(hidden_layers or []) < 2:
        raise ValueError("hidden_layers must contain at least two layers/steps")

    attention_mask = _attention_mask_or_ones(hidden_layers[0], attention_mask)
    graph_builder = RelationalGraph(_normalize_graph_config(graph_config))
    sequences = _build_instrumented_sequences(
        hidden_layers,
        attention_mask,
        graph_builder,
        seed=seed,
        memory_config=memory_config,
    )

    family_metrics = {
        family: memory_sequence_state_metrics(
            sequence,
            valid_edge_masks=sequences["valid_edge_masks"],
            stable_updates=sequences.get(f"{family}_updates"),
            memory_config=memory_config,
            instrumentation_config=instrumentation_config,
        )
        for family, sequence in sequences["memory_sequences"].items()
    }
    transition_metrics = _transition_reports(
        sequences["memory_sequences"],
        sequences["valid_edge_masks"],
        memory_config=memory_config,
        instrumentation_config=instrumentation_config,
    )
    schema_report = build_unified_telemetry_schema_report()
    schema_fields = set(schema_report["fields"])
    checks = {
        "model_intervention_none": True,
        "memory_scope_declared_layer_local_forward": True,
        "required_memory_fields_declared_in_schema": set(MEMORY_STATE_FIELDS).issubset(
            schema_fields
        ),
        "required_memory_fields_emitted": all(
            set(MEMORY_STATE_FIELDS).issubset(metrics)
            for metrics in family_metrics.values()
        ),
        "all_metrics_finite": _all_metrics_finite(family_metrics),
        "real_memory_has_nonzero_turnover": (
            family_metrics["real_memory"]["memory_turnover"] > 0
        ),
        "real_memory_not_fully_rigid": (
            family_metrics["real_memory"]["memory_rigidity"] < 1
        ),
        "controls_instrumented": {
            "real_memory",
            "random_memory",
            "shuffled_memory",
            "instantaneous",
            "no_memory",
        }.issubset(family_metrics),
    }
    status = "pass" if all(checks.values()) else "fail"

    return {
        "stage": "stage4_memory_state_instrumentation",
        "status": status,
        "seed": seed,
        "input_source": input_source,
        "scientific_scope": (
            "instrument_layer_local_forward_memory_state; no cross-batch memory claim"
        ),
        "model_intervention": "none",
        "memory_scope": {
            "scope": "layer_local_within_forward",
            "cross_batch_persistence": False,
            "dataset_memory_bank": False,
            "meaning": (
                "Current ERGT v2 passes geometry_memory between layers in one forward pass. "
                "This stage instruments that state without claiming long-term memory."
            ),
        },
        "memory_config": asdict(memory_config),
        "instrumentation_config": asdict(instrumentation_config),
        "required_fields": list(MEMORY_STATE_FIELDS),
        "checks": checks,
        "metrics": family_metrics,
        "transitions": transition_metrics,
        "next_required_step": (
            "attention_rigidity_and_collapse_monitor"
            if status == "pass"
            else "fix_memory_state_instrumentation"
        ),
    }


def memory_state_metrics(
    memory: torch.Tensor,
    *,
    valid_edge_mask: torch.Tensor | None = None,
    previous_memory: torch.Tensor | None = None,
    stable_update: torch.Tensor | None = None,
    memory_config: MemoryConfig | None = None,
    instrumentation_config: MemoryInstrumentationConfig | None = None,
) -> dict[str, float]:
    """Compute one-step memory metrics for a graph-shaped memory tensor."""

    memory_config = memory_config or MemoryConfig()
    instrumentation_config = instrumentation_config or MemoryInstrumentationConfig()
    memory_config.validate()
    instrumentation_config.validate()
    _validate_memory(memory)
    valid_edge_mask = _valid_edge_mask_or_finite(memory, valid_edge_mask)
    values = _valid_values(memory, valid_edge_mask)
    if values.numel() == 0:
        raise ValueError("memory contains no valid finite edges")

    persistence = (
        _masked_cosine(memory, previous_memory, valid_edge_mask)
        if previous_memory is not None
        else 0.0
    )
    stability = (
        _masked_cosine(memory, stable_update, valid_edge_mask)
        if stable_update is not None
        else persistence
    )
    turnover = (
        _masked_mean_abs(memory - previous_memory, valid_edge_mask)
        if previous_memory is not None
        else 0.0
    )
    spectral = _spectral_metrics(memory, valid_edge_mask)
    rigidity = _memory_rigidity(
        memory,
        valid_edge_mask,
        spectral_entropy=spectral["memory_spectral_entropy"],
        effective_rank=spectral["memory_effective_rank"],
        instrumentation_config=instrumentation_config,
    )
    turnover_risk = min(turnover / instrumentation_config.turnover_scale, 1.0)
    noise_risk = max(turnover_risk, rigidity)

    return {
        "memory_decay": float(memory_config.decay),
        "memory_eta": float(memory_config.eta),
        "gate_floor": float(memory_config.gate_floor),
        "memory_stability": _finite_or_zero(stability),
        "memory_turnover": _finite_or_zero(turnover),
        "memory_edge_density": _edge_density(
            memory,
            valid_edge_mask,
            threshold=instrumentation_config.edge_threshold,
        ),
        "memory_persistence": _finite_or_zero(persistence),
        "memory_spectral_entropy": spectral["memory_spectral_entropy"],
        "memory_effective_rank": spectral["memory_effective_rank"],
        "memory_rigidity": rigidity,
        "noise_risk": noise_risk,
        "memory_mean": _to_float(values.mean()),
        "memory_std": _to_float(values.std(unbiased=False)),
        "memory_max_probability": _mean_row_max_probability(memory, valid_edge_mask),
    }


def memory_sequence_state_metrics(
    memory_sequence: list[torch.Tensor],
    *,
    valid_edge_masks: list[torch.Tensor],
    stable_updates: list[torch.Tensor] | None = None,
    memory_config: MemoryConfig | None = None,
    instrumentation_config: MemoryInstrumentationConfig | None = None,
) -> dict[str, float]:
    """Aggregate memory-state metrics across a layer/step sequence."""

    if len(memory_sequence) != len(valid_edge_masks):
        raise ValueError("memory_sequence and valid_edge_masks must have the same length")
    if stable_updates is not None and len(stable_updates) != len(memory_sequence):
        raise ValueError("stable_updates and memory_sequence must have the same length")

    rows = []
    previous = None
    for index, memory in enumerate(memory_sequence):
        rows.append(
            memory_state_metrics(
                memory,
                valid_edge_mask=valid_edge_masks[index],
                previous_memory=previous,
                stable_update=stable_updates[index] if stable_updates is not None else None,
                memory_config=memory_config,
                instrumentation_config=instrumentation_config,
            )
        )
        previous = memory
    return _average_metric_rows(rows)


def _build_instrumented_sequences(
    hidden_layers: list[torch.Tensor],
    attention_mask: torch.Tensor,
    graph_builder: RelationalGraph,
    *,
    seed: int,
    memory_config: MemoryConfig,
) -> dict[str, Any]:
    real_updates = []
    random_updates = []
    shuffled_updates = []
    no_memory = []
    valid_edge_masks = []
    graph_config = graph_builder.config.__dict__

    for layer_index, hidden_states in enumerate(hidden_layers):
        graph = graph_builder(hidden_states, attention_mask=attention_mask)
        valid_edge_mask = make_valid_edge_mask_like(graph, attention_mask=attention_mask)
        generator = torch.Generator(device=graph.device)
        generator.manual_seed(seed + layer_index)
        random_graph = make_random_graph_like(
            graph,
            generator=generator,
            valid_edge_mask=valid_edge_mask,
        )
        shuffled_graph = make_shuffled_graph(
            graph,
            generator=generator,
            valid_edge_mask=valid_edge_mask,
        )
        real_updates.append(
            stable_memory_update(
                hidden_states,
                graph,
                valid_edge_mask,
                attention_mask=attention_mask,
                graph_config=graph_config,
                memory_config=memory_config,
            )["stable_update"]
        )
        random_updates.append(
            stable_memory_update(
                hidden_states,
                random_graph,
                valid_edge_mask,
                attention_mask=attention_mask,
                graph_config=graph_config,
                memory_config=memory_config,
            )["stable_update"]
        )
        shuffled_updates.append(
            stable_memory_update(
                hidden_states,
                shuffled_graph,
                valid_edge_mask,
                attention_mask=attention_mask,
                graph_config=graph_config,
                memory_config=memory_config,
            )["stable_update"]
        )
        no_memory.append(_valid_only(graph, valid_edge_mask))
        valid_edge_masks.append(valid_edge_mask)

    return {
        "valid_edge_masks": valid_edge_masks,
        "real_memory_updates": real_updates,
        "random_memory_updates": random_updates,
        "shuffled_memory_updates": shuffled_updates,
        "instantaneous_updates": real_updates,
        "no_memory_updates": no_memory,
        "memory_sequences": {
            "real_memory": relational_memory_sequence(
                real_updates,
                memory_config=memory_config,
            ),
            "random_memory": relational_memory_sequence(
                random_updates,
                memory_config=memory_config,
            ),
            "shuffled_memory": relational_memory_sequence(
                shuffled_updates,
                memory_config=memory_config,
            ),
            "instantaneous": real_updates,
            "no_memory": no_memory,
        },
    }


def _transition_reports(
    memory_sequences: dict[str, list[torch.Tensor]],
    valid_edge_masks: list[torch.Tensor],
    *,
    memory_config: MemoryConfig,
    instrumentation_config: MemoryInstrumentationConfig,
) -> dict[str, dict[str, dict[str, float]]]:
    reports: dict[str, dict[str, dict[str, float]]] = {}
    for index in range(1, len(valid_edge_masks)):
        reports[f"transition_{index - 1}_to_{index}"] = {
            family: memory_state_metrics(
                sequence[index],
                valid_edge_mask=valid_edge_masks[index],
                previous_memory=sequence[index - 1],
                memory_config=memory_config,
                instrumentation_config=instrumentation_config,
            )
            for family, sequence in memory_sequences.items()
        }
    return reports


def _spectral_metrics(memory: torch.Tensor, valid_edge_mask: torch.Tensor) -> dict[str, float]:
    valid_edge_mask = _prepare_mask(valid_edge_mask, memory)
    matrices = torch.where(valid_edge_mask, memory, torch.zeros_like(memory))
    ranks = []
    entropies = []
    for matrix in matrices.reshape(-1, matrices.size(-2), matrices.size(-1)):
        singular_values = torch.linalg.svdvals(matrix)
        positive = singular_values[singular_values > 1e-12]
        if positive.numel() == 0:
            ranks.append(0.0)
            entropies.append(0.0)
            continue
        probabilities = positive / positive.sum()
        entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum()
        effective_rank = torch.exp(entropy)
        normalized_entropy = entropy / math.log(max(matrix.size(-1), 2))
        ranks.append(_to_float(effective_rank))
        entropies.append(_to_float(normalized_entropy.clamp(0.0, 1.0)))
    return {
        "memory_spectral_entropy": _mean(entropies),
        "memory_effective_rank": _mean(ranks),
    }


def _memory_rigidity(
    memory: torch.Tensor,
    valid_edge_mask: torch.Tensor,
    *,
    spectral_entropy: float,
    effective_rank: float,
    instrumentation_config: MemoryInstrumentationConfig,
) -> float:
    max_probability = _mean_row_max_probability(memory, valid_edge_mask)
    sequence_length = memory.size(-1)
    rank_ratio = effective_rank / max(float(sequence_length), 1.0)
    concentration_risk = max(
        0.0,
        (max_probability - instrumentation_config.rigidity_max_probability)
        / max(1.0 - instrumentation_config.rigidity_max_probability, 1e-8),
    )
    rank_risk = max(
        0.0,
        (
            instrumentation_config.rigidity_min_effective_rank_ratio
            - _finite_or_zero(rank_ratio)
        )
        / max(instrumentation_config.rigidity_min_effective_rank_ratio, 1e-8),
    )
    entropy_risk = max(0.0, 1.0 - _finite_or_zero(spectral_entropy))
    return min(max(0.5 * concentration_risk + 0.3 * rank_risk + 0.2 * entropy_risk, 0.0), 1.0)


def _edge_density(
    memory: torch.Tensor,
    valid_edge_mask: torch.Tensor,
    *,
    threshold: float,
) -> float:
    values = _valid_values(memory, valid_edge_mask)
    if values.numel() == 0:
        return math.nan
    return _to_float((values > threshold).float().mean())


def _mean_row_max_probability(memory: torch.Tensor, valid_edge_mask: torch.Tensor) -> float:
    valid_edge_mask = _prepare_mask(valid_edge_mask, memory)
    weights = torch.where(valid_edge_mask, memory.clamp_min(0.0), torch.zeros_like(memory))
    row_sum = weights.sum(dim=-1, keepdim=True)
    probabilities = weights / row_sum.clamp_min(1e-12)
    row_valid = valid_edge_mask.any(dim=-1) & torch.isfinite(probabilities).all(dim=-1)
    if not bool(row_valid.any().item()):
        return math.nan
    return _to_float(probabilities.max(dim=-1).values[row_valid].mean())


def _valid_values(memory: torch.Tensor, valid_edge_mask: torch.Tensor) -> torch.Tensor:
    valid_edge_mask = _prepare_mask(valid_edge_mask, memory)
    return memory[valid_edge_mask & torch.isfinite(memory)]


def _valid_edge_mask_or_finite(
    memory: torch.Tensor,
    valid_edge_mask: torch.Tensor | None,
) -> torch.Tensor:
    if valid_edge_mask is not None:
        return _prepare_mask(valid_edge_mask, memory)
    return torch.isfinite(memory)


def _masked_cosine(
    a: torch.Tensor,
    b: torch.Tensor | None,
    mask: torch.Tensor,
) -> float:
    if b is None:
        return math.nan
    mask = _prepare_mask(mask, a) & torch.isfinite(a) & torch.isfinite(b)
    if int(mask.sum().item()) < 2:
        return math.nan
    a_values = a[mask]
    b_values = b[mask]
    denominator = torch.linalg.vector_norm(a_values) * torch.linalg.vector_norm(b_values)
    if denominator <= 0:
        return math.nan
    return _to_float((a_values * b_values).sum() / denominator)


def _masked_mean_abs(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask = _prepare_mask(mask, values) & torch.isfinite(values)
    if not bool(mask.any().item()):
        return math.nan
    return _to_float(values[mask].abs().mean())


def _average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: _mean([row[key] for row in rows if key in row and math.isfinite(row[key])])
        for key in keys
    }


def _all_metrics_finite(family_metrics: dict[str, dict[str, float]]) -> bool:
    for metrics in family_metrics.values():
        for field in MEMORY_STATE_FIELDS:
            value = metrics.get(field)
            if value is None or not math.isfinite(float(value)):
                return False
    return True


def _valid_only(graph: torch.Tensor, valid_edge_mask: torch.Tensor) -> torch.Tensor:
    valid_edge_mask = _prepare_mask(valid_edge_mask, graph)
    return torch.where(valid_edge_mask, graph, torch.zeros_like(graph))


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


def _validate_memory(memory: torch.Tensor) -> None:
    if memory.dim() != 4:
        raise ValueError("memory must have shape [batch, heads, sequence, sequence]")
    if memory.size(-1) != memory.size(-2):
        raise ValueError("memory must be square in the last two dimensions")


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(float(value))]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def _finite_or_zero(value: float) -> float:
    return value if math.isfinite(float(value)) else 0.0


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
