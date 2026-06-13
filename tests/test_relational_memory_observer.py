import json

import torch

from evaluation.relational_field_observer import synthetic_structured_hidden_layers
from evaluation.relational_memory_observer import (
    MemoryConfig,
    build_relational_memory_observer_report,
    memory_quality_metrics,
    relational_memory_sequence,
    stable_memory_update,
)
from layers.relational_graph import RelationalGraph, make_valid_edge_mask_like


def test_stable_memory_update_masks_future_edges() -> None:
    hidden_layers, attention_mask = synthetic_structured_hidden_layers(seed=1)
    graph = RelationalGraph({"kernel": "sigmoid_cosine", "normalize_hidden": True})(
        hidden_layers[0],
        attention_mask=attention_mask,
    )
    valid_edge_mask = make_valid_edge_mask_like(graph, attention_mask)

    update = stable_memory_update(
        hidden_layers[0],
        graph,
        valid_edge_mask,
        attention_mask=attention_mask,
        graph_config={"kernel": "sigmoid_cosine", "normalize_hidden": True},
    )

    stable_update = update["stable_update"]
    assert torch.isfinite(stable_update).all()
    assert torch.equal(
        stable_update[~valid_edge_mask],
        torch.zeros_like(stable_update[~valid_edge_mask]),
    )
    assert update["leakage_checks"]["future_sources_forbidden"]
    assert update["leakage_checks"]["future_edges_forbidden"]


def test_relational_memory_sequence_blends_history_and_current_update() -> None:
    config = MemoryConfig(decay=0.5, eta=0.5)
    first = torch.ones(1, 1, 2, 2)
    second = torch.zeros(1, 1, 2, 2)

    memory = relational_memory_sequence([first, second], memory_config=config)

    assert torch.allclose(memory[0], first)
    assert torch.allclose(memory[1], torch.full_like(first, 0.5))


def test_memory_quality_prefers_sequence_that_predicts_next_target() -> None:
    mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    target = [
        torch.tensor([[[[0.0, 0.2], [0.1, 0.0]]]]),
        torch.tensor([[[[0.0, 0.8], [0.7, 0.0]]]]),
    ]
    good = [target[1].clone(), target[1].clone()]
    bad = [torch.zeros_like(target[1]), torch.zeros_like(target[1])]

    good_metrics = memory_quality_metrics(good, target, [mask, mask])
    bad_metrics = memory_quality_metrics(bad, target, [mask, mask])

    assert good_metrics["memory_quality_score"] > bad_metrics["memory_quality_score"]


def test_relational_memory_observer_report_passes_on_structured_smoke_field() -> None:
    report = build_relational_memory_observer_report(seed=2027)

    assert report["status"] == "pass"
    assert report["input_source"] == "synthetic_memory_smoke"
    assert report["model_intervention"] == "none"
    assert report["checks"]["real_memory_beats_random_memory"]
    assert report["checks"]["real_memory_beats_shuffled_memory"]
    assert report["checks"]["real_memory_beats_instantaneous"]
    assert report["checks"]["real_memory_beats_generic_smoothing"]
    assert report["checks"]["real_memory_beats_no_memory"]
    assert report["checks"]["real_memory_not_collapsed"]
    assert report["next_required_step"] == "causal_shortest_path_geometry"
    json.dumps(report)


def test_relational_memory_observer_report_accepts_explicit_hidden_layers() -> None:
    hidden_layers, attention_mask = synthetic_structured_hidden_layers(seed=3, layers=2)

    report = build_relational_memory_observer_report(
        hidden_layers=hidden_layers,
        attention_mask=attention_mask,
        seed=3,
    )

    assert report["input_source"] == "provided_hidden_layers"
    assert sorted(report["layers"].keys()) == ["transition_0_to_1"]
