import json

import torch

from evaluation.memory_state_instrumentation import (
    MEMORY_STATE_FIELDS,
    MemoryInstrumentationConfig,
    build_memory_state_instrumentation_report,
    memory_sequence_state_metrics,
    memory_state_metrics,
)
from evaluation.relational_memory_observer import MemoryConfig


def test_memory_state_metrics_emit_required_fields_and_are_finite() -> None:
    memory = torch.tensor(
        [[[[0.0, 0.4, 0.0], [0.2, 0.0, 0.0], [0.3, 0.5, 0.0]]]],
        dtype=torch.float32,
    )
    previous = memory * 0.8
    valid_edge_mask = torch.tensor(
        [[[[False, False, False], [True, False, False], [True, True, False]]]]
    )

    metrics = memory_state_metrics(
        memory,
        valid_edge_mask=valid_edge_mask,
        previous_memory=previous,
        stable_update=memory,
        memory_config=MemoryConfig(decay=0.7, eta=0.3, gate_floor=0.05),
    )

    assert set(MEMORY_STATE_FIELDS).issubset(metrics)
    assert all(torch.isfinite(torch.tensor(metrics[field])) for field in MEMORY_STATE_FIELDS)
    assert metrics["memory_turnover"] > 0
    assert metrics["memory_persistence"] > 0
    assert 0 <= metrics["memory_rigidity"] <= 1


def test_memory_state_metrics_detect_more_rigid_memory() -> None:
    valid_edge_mask = torch.tensor(
        [[[[False, False, False], [True, False, False], [True, True, False]]]]
    )
    rigid = torch.tensor(
        [[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.95, 0.05, 0.0]]]],
        dtype=torch.float32,
    )
    spread = torch.tensor(
        [[[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]]]],
        dtype=torch.float32,
    )

    rigid_metrics = memory_state_metrics(
        rigid,
        valid_edge_mask=valid_edge_mask,
        instrumentation_config=MemoryInstrumentationConfig(rigidity_max_probability=0.7),
    )
    spread_metrics = memory_state_metrics(
        spread,
        valid_edge_mask=valid_edge_mask,
        instrumentation_config=MemoryInstrumentationConfig(rigidity_max_probability=0.7),
    )

    assert rigid_metrics["memory_rigidity"] > spread_metrics["memory_rigidity"]


def test_memory_sequence_state_metrics_aggregate_history() -> None:
    mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    first = torch.tensor([[[[0.0, 0.2], [0.1, 0.0]]]], dtype=torch.float32)
    second = torch.tensor([[[[0.0, 0.4], [0.3, 0.0]]]], dtype=torch.float32)

    metrics = memory_sequence_state_metrics(
        [first, second],
        valid_edge_masks=[mask, mask],
        stable_updates=[first, second],
    )

    assert metrics["memory_turnover"] > 0
    assert metrics["memory_stability"] > 0
    assert metrics["memory_edge_density"] > 0


def test_memory_state_instrumentation_report_passes() -> None:
    report = build_memory_state_instrumentation_report(seed=2027)

    assert report["status"] == "pass"
    assert report["model_intervention"] == "none"
    assert report["memory_scope"]["cross_batch_persistence"] is False
    assert report["checks"]["required_memory_fields_declared_in_schema"]
    assert report["checks"]["required_memory_fields_emitted"]
    assert report["checks"]["all_metrics_finite"]
    assert report["checks"]["controls_instrumented"]
    assert report["next_required_step"] == "attention_rigidity_and_collapse_monitor"
    json.dumps(report)


def test_memory_state_instrumentation_report_accepts_explicit_hidden_layers() -> None:
    hidden_layers = [
        torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(seed))
        for seed in [1, 2]
    ]
    attention_mask = torch.ones(1, 4, dtype=torch.long)

    report = build_memory_state_instrumentation_report(
        hidden_layers=hidden_layers,
        attention_mask=attention_mask,
        seed=3,
    )

    assert report["input_source"] == "provided_hidden_layers"
    assert sorted(report["transitions"]) == ["transition_0_to_1"]
