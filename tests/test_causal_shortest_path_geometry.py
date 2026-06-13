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


def unit_step_causal_mask(sequence_length: int) -> torch.Tensor:
    positions = torch.arange(sequence_length)
    current = positions.view(sequence_length, 1)
    context = positions.view(1, sequence_length)
    valid = (current - context) == 1
    return valid.view(1, 1, sequence_length, sequence_length)


def floyd_reference_shortest_path(graph: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    distance = pairwise_edge_distance(graph, mask)
    sequence_length = distance.size(-1)
    diagonal = torch.eye(sequence_length, dtype=torch.bool, device=distance.device).view(
        1,
        1,
        sequence_length,
        sequence_length,
    )
    distance = distance.masked_fill(diagonal, 0.0)
    for bridge in range(sequence_length):
        via_bridge = (
            distance[..., :, bridge].unsqueeze(-1)
            + distance[..., bridge, :].unsqueeze(-2)
        )
        distance = torch.minimum(distance, via_bridge)
    future = torch.ones(sequence_length, sequence_length, dtype=torch.bool).triu(diagonal=1)
    future = future.view(1, 1, sequence_length, sequence_length)
    return distance.masked_fill(future, torch.inf).masked_fill(diagonal, 0.0)


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


def test_unit_step_causal_shortest_path_matches_floyd_reference() -> None:
    torch.manual_seed(3)
    graph = torch.rand(2, 1, 6, 6).clamp_min(0.05)
    mask = unit_step_causal_mask(6).expand_as(graph).clone()
    mask[0, 0, 3, 2] = False
    graph = torch.where(mask, graph, torch.zeros_like(graph))

    fast = causal_shortest_path_distance(graph, mask)
    reference = floyd_reference_shortest_path(graph, mask)

    assert torch.equal(torch.isfinite(fast), torch.isfinite(reference))
    finite = torch.isfinite(fast) & torch.isfinite(reference)
    assert torch.allclose(fast[finite], reference[finite], atol=1e-6)
    assert future_distances_are_infinite(fast)
    assert distance_diagonal_is_zero(fast)


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
