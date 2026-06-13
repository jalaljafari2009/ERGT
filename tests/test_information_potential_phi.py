import json

import torch

from evaluation.information_potential_phi import (
    build_information_potential_report,
    information_potential_components,
    node_neighborhood_overlap,
)
from evaluation.relational_field_observer import synthetic_structured_hidden_layers
from layers.relational_graph import RelationalGraph, make_valid_edge_mask_like


def test_information_potential_components_are_finite_and_causal() -> None:
    hidden_layers, attention_mask = synthetic_structured_hidden_layers(seed=1)
    graph = RelationalGraph({"kernel": "sigmoid_cosine", "normalize_hidden": True})(
        hidden_layers[0],
        attention_mask=attention_mask,
    )
    valid_edge_mask = make_valid_edge_mask_like(graph, attention_mask)

    components = information_potential_components(
        hidden_layers[0],
        graph,
        valid_edge_mask,
        attention_mask=attention_mask,
    )

    valid = components["valid_node_mask"]
    assert torch.isfinite(components["score"][valid]).all()
    assert (components["score"][valid] >= 0).all()
    assert (components["score"][valid] <= 1).all()
    assert torch.allclose(
        components["causal_validity"][valid],
        torch.ones_like(components["causal_validity"][valid]),
    )


def test_information_potential_penalizes_single_token_lock_in() -> None:
    hidden_layers, attention_mask = synthetic_structured_hidden_layers(seed=2)
    graph = RelationalGraph({"kernel": "sigmoid_cosine", "normalize_hidden": True})(
        hidden_layers[0],
        attention_mask=attention_mask,
    )
    valid_edge_mask = make_valid_edge_mask_like(graph, attention_mask)
    locked_graph = graph.clone()
    locked_graph[valid_edge_mask] = 0.0
    for batch_idx in range(locked_graph.size(0)):
        for head_idx in range(locked_graph.size(1)):
            for row_idx in range(locked_graph.size(-2)):
                valid_columns = torch.where(valid_edge_mask[batch_idx, head_idx, row_idx])[0]
                if valid_columns.numel():
                    locked_graph[batch_idx, head_idx, row_idx, valid_columns[0]] = 1.0

    real_components = information_potential_components(
        hidden_layers[0],
        graph,
        valid_edge_mask,
        attention_mask=attention_mask,
    )
    locked_components = information_potential_components(
        hidden_layers[0],
        locked_graph,
        valid_edge_mask,
        attention_mask=attention_mask,
    )

    assert _mean(locked_components["anti_single_token_lock"]) < _mean(
        real_components["anti_single_token_lock"]
    )
    assert _mean(locked_components["anti_collapse"]) < _mean(
        real_components["anti_collapse"]
    )


def test_node_neighborhood_overlap_ignores_future_and_empty_rows() -> None:
    distance = torch.tensor(
        [[[[0.0, float("inf"), float("inf")], [0.2, 0.0, float("inf")], [0.3, 0.1, 0.0]]]]
    )
    shifted = torch.tensor(
        [[[[0.0, float("inf"), float("inf")], [0.2, 0.0, float("inf")], [0.1, 0.3, 0.0]]]]
    )

    overlap = node_neighborhood_overlap(distance, shifted, k=1)

    assert torch.isnan(overlap[0, 0, 0])
    assert overlap[0, 0, 1] == 1.0
    assert overlap[0, 0, 2] == 0.0


def test_information_potential_report_passes_on_structured_smoke_field() -> None:
    report = build_information_potential_report(seed=2027)

    assert report["status"] == "pass"
    assert report["input_source"] == "synthetic_structured_smoke"
    assert report["model_intervention"] == "none"
    assert report["checks"]["real_phi_separates_from_random"]
    assert report["checks"]["real_phi_separates_from_shuffled"]
    assert report["checks"]["phi_not_low_entropy_only"]
    assert report["checks"]["high_phi_not_from_collapse"]
    assert report["next_required_step"] == "reconstruction_gate"
    json.dumps(report)


def test_information_potential_report_accepts_explicit_hidden_layers() -> None:
    hidden_layers, attention_mask = synthetic_structured_hidden_layers(seed=3, layers=2)

    report = build_information_potential_report(
        hidden_layers=hidden_layers,
        attention_mask=attention_mask,
        seed=3,
    )

    assert report["input_source"] == "provided_hidden_layers"
    assert sorted(report["layers"].keys()) == ["layer_0", "layer_1"]


def _mean(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    return float(finite.mean().item())
