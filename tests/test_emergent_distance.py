import torch

from geometry.emergent_distance import (
    EmergentDistance,
    make_random_distance_like,
    make_shuffled_distance,
)


def sample_graph() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [
                    [1.0, 0.8, 0.2],
                    [0.8, 1.0, 0.4],
                    [0.2, 0.4, 1.0],
                ]
            ]
        ]
    )


def test_emergent_distance_shape_and_finiteness() -> None:
    distance_builder = EmergentDistance({"normalization": "none", "diagonal_policy": "zero"})

    distance = distance_builder(sample_graph())

    assert distance.shape == (1, 1, 3, 3)
    assert torch.isfinite(distance).all()


def test_distance_decreases_as_relation_strength_increases() -> None:
    distance_builder = EmergentDistance({"normalization": "none", "diagonal_policy": "zero"})

    distance = distance_builder(sample_graph())

    assert distance[0, 0, 0, 1] < distance[0, 0, 0, 2]


def test_zero_diagonal_policy() -> None:
    distance_builder = EmergentDistance({"normalization": "none", "diagonal_policy": "zero"})

    distance = distance_builder(sample_graph())
    diagonal = torch.diagonal(distance, dim1=-2, dim2=-1)

    assert torch.all(diagonal == 0)


def test_mask_diagonal_policy() -> None:
    distance_builder = EmergentDistance({"normalization": "none", "diagonal_policy": "mask"})

    distance = distance_builder(sample_graph())
    diagonal = torch.diagonal(distance, dim1=-2, dim2=-1)

    assert torch.isnan(diagonal).all()


def test_offdiag_zscore_clamp_normalization_is_stable() -> None:
    distance_builder = EmergentDistance(
        {
            "normalization": "offdiag_zscore_clamp",
            "clip_value": 5.0,
            "diagonal_policy": "zero",
        }
    )

    distance = distance_builder(sample_graph())
    offdiag = distance[~torch.eye(3, dtype=torch.bool).view(1, 1, 3, 3)]

    assert torch.isfinite(distance).all()
    assert offdiag.mean().abs() < 1e-5
    assert distance.min() >= -5.0
    assert distance.max() <= 5.0


def test_offdiag_zscore_clamp_backward_is_stable_for_degenerate_graph() -> None:
    graph = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)
    distance_builder = EmergentDistance(
        {
            "normalization": "offdiag_zscore_clamp",
            "clip_value": 5.0,
            "diagonal_policy": "zero",
        }
    )

    distance = distance_builder(graph)
    loss = distance.square().sum()
    loss.backward()

    assert torch.isfinite(distance).all()
    assert graph.grad is not None
    assert torch.isfinite(graph.grad).all()


def test_padding_mask_sets_invalid_pairs_to_inf_before_normalization() -> None:
    distance_builder = EmergentDistance({"normalization": "none", "diagonal_policy": "zero"})
    attention_mask = torch.tensor([[1, 1, 0]])

    distance = distance_builder(sample_graph(), attention_mask=attention_mask)

    assert torch.isinf(distance[0, 0, 0, 2])
    assert torch.isinf(distance[0, 0, 2, 0])


def test_causal_runtime_distance_masks_future_positions() -> None:
    distance_builder = EmergentDistance(
        {
            "normalization": "none",
            "diagonal_policy": "zero",
            "causal_runtime_distance": True,
        }
    )

    distance = distance_builder(sample_graph())

    assert torch.isinf(distance[0, 0, 0, 1])
    assert torch.isinf(distance[0, 0, 0, 2])
    assert torch.isfinite(distance[0, 0, 2, 0])


def test_causal_runtime_distance_preserves_future_inf_after_clamp() -> None:
    distance_builder = EmergentDistance(
        {
            "normalization": "offdiag_zscore_clamp",
            "clip_value": 5.0,
            "diagonal_policy": "zero",
            "causal_runtime_distance": True,
        }
    )

    distance = distance_builder(sample_graph())

    assert torch.isinf(distance[0, 0, 0, 1])
    assert torch.isinf(distance[0, 0, 0, 2])
    assert torch.isfinite(distance[0, 0, 2, 0])


def test_random_and_shuffled_distance_controls_keep_shape_and_finiteness() -> None:
    distance = EmergentDistance({"normalization": "none", "diagonal_policy": "zero"})(
        sample_graph()
    )
    generator = torch.Generator()
    generator.manual_seed(1)

    random_distance = make_random_distance_like(distance, generator=generator)
    shuffled_distance = make_shuffled_distance(distance, generator=generator)

    assert random_distance.shape == distance.shape
    assert shuffled_distance.shape == distance.shape
    assert torch.isfinite(random_distance).all()
    assert torch.isfinite(shuffled_distance).all()
