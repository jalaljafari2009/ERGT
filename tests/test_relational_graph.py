import torch

from layers.relational_graph import RelationalGraph, make_random_graph_like, make_shuffled_graph


def test_relational_graph_shape_and_range() -> None:
    torch.manual_seed(1)
    graph_builder = RelationalGraph()
    hidden_states = torch.randn(2, 4, 8)

    graph = graph_builder(hidden_states)

    assert graph.shape == (2, 1, 4, 4)
    assert torch.isfinite(graph).all()
    assert graph.min() >= 0
    assert graph.max() <= 1


def test_relational_graph_attention_mask_zeroes_invalid_pairs() -> None:
    torch.manual_seed(1)
    graph_builder = RelationalGraph()
    hidden_states = torch.randn(2, 4, 8)
    attention_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ]
    )

    graph = graph_builder(hidden_states, attention_mask=attention_mask)

    assert torch.all(graph[0, :, 2:, :] == 0)
    assert torch.all(graph[0, :, :, 2:] == 0)
    assert torch.all(graph[1, :, 3:, :] == 0)
    assert torch.all(graph[1, :, :, 3:] == 0)


def test_relational_graph_zero_diagonal_policy() -> None:
    torch.manual_seed(1)
    graph_builder = RelationalGraph({"diagonal_policy": "zero"})
    hidden_states = torch.randn(2, 4, 8)

    graph = graph_builder(hidden_states)
    diagonal = torch.diagonal(graph, dim1=-2, dim2=-1)

    assert torch.all(diagonal == 0)


def test_relational_graph_mask_diagonal_policy() -> None:
    torch.manual_seed(1)
    graph_builder = RelationalGraph({"diagonal_policy": "mask"})
    hidden_states = torch.randn(2, 4, 8)

    graph = graph_builder(hidden_states)
    diagonal = torch.diagonal(graph, dim1=-2, dim2=-1)

    assert torch.isnan(diagonal).all()


def test_random_and_shuffled_graph_controls_keep_shape() -> None:
    torch.manual_seed(1)
    graph = RelationalGraph()(torch.randn(2, 4, 8))
    generator = torch.Generator()
    generator.manual_seed(1)

    random_graph = make_random_graph_like(graph, generator=generator)
    shuffled_graph = make_shuffled_graph(graph, generator=generator)

    assert random_graph.shape == graph.shape
    assert shuffled_graph.shape == graph.shape
    assert torch.isfinite(random_graph).all()
    assert torch.isfinite(shuffled_graph).all()
