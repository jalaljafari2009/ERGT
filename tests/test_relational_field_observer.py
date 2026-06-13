import json

import torch

from evaluation.relational_field_observer import (
    build_relational_field_observer_report,
    field_metrics,
    synthetic_structured_hidden_layers,
)
from geometry.emergent_distance import EmergentDistance
from layers.relational_graph import RelationalGraph, make_valid_edge_mask_like


def test_field_metrics_include_required_observer_metrics() -> None:
    hidden_layers, attention_mask = synthetic_structured_hidden_layers(seed=1)
    graph = RelationalGraph({"kernel": "sigmoid_cosine", "normalize_hidden": True})(
        hidden_layers[0],
        attention_mask=attention_mask,
    )
    distance = EmergentDistance(
        {
            "normalization": "offdiag_zscore_clamp",
            "clip_value": 5.0,
            "diagonal_policy": "zero",
            "causal_runtime_distance": True,
        }
    )(graph, attention_mask=attention_mask)
    valid_edge_mask = make_valid_edge_mask_like(graph, attention_mask)

    metrics = field_metrics(hidden_layers[0], graph, distance, valid_edge_mask)

    assert metrics["relational_entropy_mean"] >= 0
    assert metrics["spectral_entropy_mean"] >= 0
    assert metrics["effective_rank_mean"] >= 1
    assert -1 <= metrics["coherence_mean"] <= 1
    assert metrics["valid_weight_variance"] > 0
    json.dumps(metrics)


def test_relational_field_observer_report_passes_on_structured_smoke_field() -> None:
    report = build_relational_field_observer_report(seed=2027)

    assert report["status"] == "pass"
    assert report["input_source"] == "synthetic_structured_smoke"
    assert report["scientific_scope"] == "smoke_validates_observer_mechanics"
    assert report["observer_pipeline"] == "H -> W -> D"
    assert report["model_intervention"] == "none"
    assert report["checks"]["all_layers_separate_from_random"]
    assert report["checks"]["all_layers_separate_from_shuffled"]
    assert report["checks"]["nearby_layers_stable"]
    assert report["next_required_step"] == "resonant_response_observer"
    json.dumps(report)


def test_relational_field_observer_accepts_explicit_hidden_layers() -> None:
    hidden_layers, attention_mask = synthetic_structured_hidden_layers(seed=2, layers=2)
    hidden_layers = [layer + torch.randn_like(layer) * 0.001 for layer in hidden_layers]

    report = build_relational_field_observer_report(
        hidden_layers=hidden_layers,
        attention_mask=attention_mask,
        seed=2,
    )

    assert sorted(report["layers"].keys()) == ["layer_0", "layer_1"]
    assert report["input_source"] == "provided_hidden_layers"
    assert report["stability"]["pairs"] == 1
