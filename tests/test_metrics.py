import json

import pytest
import torch

from evaluation.graph_metrics import (
    degree_distribution,
    diagonal_dominance,
    graph_metrics,
    graph_metrics_with_controls,
    layer_to_layer_similarity,
    sparsity,
)


def sample_graph() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [
                    [1.0, 0.2, 0.7],
                    [0.2, 1.0, 0.4],
                    [0.7, 0.4, 1.0],
                ]
            ]
        ]
    )


def test_graph_metrics_are_json_compatible() -> None:
    metrics = graph_metrics(sample_graph(), sparsity_thresholds=[0.5, 0.75])

    encoded = json.dumps(metrics, sort_keys=True)

    assert "entropy" in metrics
    assert "0.5" in metrics["sparsity"]
    assert encoded


def test_graph_metrics_with_controls_are_json_compatible() -> None:
    metrics = graph_metrics_with_controls(sample_graph(), sparsity_thresholds=[0.5], seed=1)

    encoded = json.dumps(metrics, sort_keys=True)

    assert "real_W" in metrics
    assert "random_W" in metrics["controls"]
    assert "shuffled_W" in metrics["controls"]
    assert encoded


def test_sparsity_degree_and_diagonal_dominance() -> None:
    graph = sample_graph()

    assert sparsity(graph, 0.5) == pytest.approx(4 / 6)
    assert degree_distribution(graph, 0.5)["max"] == 2.0
    assert diagonal_dominance(graph) > 1.0


def test_layer_to_layer_similarity() -> None:
    graph = sample_graph()
    similarity = layer_to_layer_similarity(graph, graph.clone())

    assert similarity["cosine_mean"] == 1.0
    assert similarity["frobenius_mean"] == 0.0
