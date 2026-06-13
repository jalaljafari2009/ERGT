import json

import torch

from evaluation.reconstruction_gate import (
    build_reconstruction_gate_report,
    causal_prefix_hidden_prediction,
    high_phi_reconstruction_alignment,
    leakage_checks,
    reconstruction_deficits,
)
from evaluation.relational_field_observer import synthetic_structured_hidden_layers
from layers.relational_graph import RelationalGraph, make_valid_edge_mask_like


def test_causal_prefix_hidden_prediction_uses_only_previous_valid_state() -> None:
    hidden = torch.arange(1 * 4 * 2, dtype=torch.float32).view(1, 4, 2)
    attention_mask = torch.tensor([[1, 1, 1, 0]])

    prediction, source_available, source_index = causal_prefix_hidden_prediction(
        hidden,
        attention_mask=attention_mask,
    )

    assert not source_available[0, 0]
    assert source_index[0, 1] == 0
    assert source_index[0, 2] == 1
    assert source_index[0, 3] == -1
    assert torch.equal(prediction[0, 1], hidden[0, 0])
    assert torch.equal(prediction[0, 2], hidden[0, 1])


def test_reconstruction_deficits_are_causal_and_finite_on_structured_field() -> None:
    hidden_layers, attention_mask = synthetic_structured_hidden_layers(seed=1)
    graph = RelationalGraph({"kernel": "sigmoid_cosine", "normalize_hidden": True})(
        hidden_layers[0],
        attention_mask=attention_mask,
    )
    valid_edge_mask = make_valid_edge_mask_like(graph, attention_mask)

    deficits = reconstruction_deficits(
        hidden_layers[0],
        graph,
        valid_edge_mask,
        attention_mask=attention_mask,
        graph_config={"kernel": "sigmoid_cosine", "normalize_hidden": True},
    )

    valid = deficits["reconstructible_node_mask"]
    assert int(valid.sum().item()) > 0
    assert torch.isfinite(deficits["hidden_deficit"][valid]).all()
    assert torch.isfinite(deficits["relational_deficit"][valid]).all()
    assert deficits["leakage_checks"]["future_sources_forbidden"]
    assert deficits["leakage_checks"]["future_edges_forbidden"]


def test_leakage_checks_reject_future_source_indices() -> None:
    source_index = torch.tensor([[1, 0, 1]])
    graph = torch.zeros(1, 1, 3, 3)
    valid_edge_mask = make_valid_edge_mask_like(graph)

    checks = leakage_checks(source_index, valid_edge_mask)

    assert checks["future_sources_forbidden"] is False
    assert checks["future_edges_forbidden"] is True


def test_high_phi_reconstruction_alignment_detects_better_high_phi_regions() -> None:
    phi = torch.tensor([[[0.1, 0.2, 0.8, 0.9]]])
    deficit = torch.tensor([[[0.8, 0.7, 0.2, 0.1]]])
    mask = torch.ones_like(phi, dtype=torch.bool)

    alignment = high_phi_reconstruction_alignment(phi, deficit, mask, fraction=0.5)

    assert alignment["delta_low_minus_high_total_deficit"] > 0


def test_reconstruction_gate_report_passes_on_structured_smoke_field() -> None:
    report = build_reconstruction_gate_report(seed=2027)

    assert report["status"] == "pass"
    assert report["input_source"] == "synthetic_structured_smoke"
    assert report["model_intervention"] == "none"
    assert report["checks"]["real_relational_deficit_lt_random"]
    assert report["checks"]["real_relational_deficit_lt_shuffled"]
    assert report["checks"]["real_total_deficit_lt_random"]
    assert report["checks"]["real_total_deficit_lt_shuffled"]
    assert report["checks"]["high_phi_regions_reconstruct_better"]
    assert report["next_required_step"] == "relational_memory_observer"
    json.dumps(report)


def test_reconstruction_gate_report_accepts_explicit_hidden_layers() -> None:
    hidden_layers, attention_mask = synthetic_structured_hidden_layers(seed=3, layers=2)

    report = build_reconstruction_gate_report(
        hidden_layers=hidden_layers,
        attention_mask=attention_mask,
        seed=3,
    )

    assert report["input_source"] == "provided_hidden_layers"
    assert sorted(report["layers"].keys()) == ["layer_0", "layer_1"]
