import json

import torch

from evaluation.causal_shortest_path_geometry import (
    build_causal_shortest_path_geometry_report,
    causal_shortest_path_distance,
    distance_diagonal_is_zero,
    distance_to_affinity,
    future_distances_are_infinite,
    geometry_quality_metrics,
    pairwise_edge_distance,
)
from evaluation.relational_memory_observer import synthetic_memory_hidden_layers


def causal_mask(sequence_length: int) -> torch.Tensor:
    positions = torch.arange(sequence_length)
    valid = positions.view(sequence_length, 1) > positions.view(1, sequence_length)
    return valid.view(1, 1, sequence_length, sequence_length)


def test_causal_shortest_path_uses_only_valid_past_bridges() -> None:
    graph = torch.full((1, 1, 4, 4), 0.05)
    mask = causal_mask(4)
    graph = torch.where(mask, graph, torch.zeros_like(graph))
    graph[0, 0, 3, 1] = 0.10
    graph[0, 0, 3, 2] = 0.90
    graph[0, 0, 2, 1] = 0.90

    pairwise = pairwise_edge_distance(graph, mask)
    causal = causal_shortest_path_distance(graph, mask)

    assert causal[0, 0, 3, 1] < pairwise[0, 0, 3, 1]
    assert torch.isinf(causal[0, 0, 1, 3])
    assert torch.isfinite(causal[0, 0, 3, 1])
    assert distance_diagonal_is_zero(causal)
    assert future_distances_are_infinite(causal)


def test_distance_to_affinity_maps_unreachable_pairs_to_zero() -> None:
    distance = torch.tensor([[[[0.0, torch.inf], [0.7, 0.0]]]])

    affinity = distance_to_affinity(distance)

    assert affinity[0, 0, 0, 0] == 1.0
    assert affinity[0, 0, 0, 1] == 0.0
    assert torch.isclose(affinity[0, 0, 1, 0], torch.exp(torch.tensor(-0.7)))


def test_geometry_quality_rewards_shortest_path_context_signal() -> None:
    mask = causal_mask(3)
    pairwise = torch.full((1, 1, 3, 3), torch.inf)
    pairwise = pairwise.masked_fill(torch.eye(3, dtype=torch.bool).view(1, 1, 3, 3), 0.0)
    pairwise[0, 0, 2, 1] = 1.0
    pairwise[0, 0, 1, 0] = 0.1
    pairwise[0, 0, 2, 0] = 1.0

    causal = pairwise.clone()
    causal[0, 0, 2, 0] = 0.2
    target_next = distance_to_affinity(causal)
    targets = [torch.zeros_like(target_next), target_next]

    causal_metrics = geometry_quality_metrics(
        [causal, causal],
        targets,
        [mask, mask],
        pairwise_reference=[pairwise, pairwise],
    )
    pairwise_metrics = geometry_quality_metrics([pairwise, pairwise], targets, [mask, mask])

    assert causal_metrics["path_improvement_fraction"] > 0
    assert causal_metrics["geometry_quality_score"] > pairwise_metrics["geometry_quality_score"]


def test_causal_shortest_path_geometry_report_passes_on_synthetic_memory_field() -> None:
    report = build_causal_shortest_path_geometry_report(seed=2027)

    assert report["status"] == "pass"
    assert report["input_source"] == "synthetic_memory_smoke"
    assert report["model_intervention"] == "none"
    assert report["checks"]["future_edges_forbidden"]
    assert report["checks"]["future_inputs_forbidden"]
    assert report["checks"]["real_causal_path_separates_from_random"]
    assert report["checks"]["real_causal_path_separates_from_shuffled"]
    assert report["checks"]["causal_path_beats_pairwise"]
    assert report["checks"]["causal_path_beats_no_memory"]
    assert report["checks"]["causal_path_adds_contextual_signal"]
    assert report["next_required_step"] == "geoattention_v2"
    json.dumps(report)


def test_causal_shortest_path_geometry_report_accepts_explicit_hidden_layers() -> None:
    hidden_layers, attention_mask = synthetic_memory_hidden_layers(seed=3, layers=2)

    report = build_causal_shortest_path_geometry_report(
        hidden_layers=hidden_layers,
        attention_mask=attention_mask,
        seed=3,
    )

    assert report["input_source"] == "provided_hidden_layers"
    assert sorted(report["layers"].keys()) == ["transition_0_to_1"]
