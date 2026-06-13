import torch

from attention.geo_attention import GeoAttention, GeoAttentionConfig
from evaluation.strict_w_controls import build_strict_w_controls_report
from layers.relational_graph import (
    RelationalGraph,
    make_random_graph_like,
    make_shuffled_graph,
    make_valid_edge_mask_like,
)


def test_strict_graph_controls_preserve_invalid_region_and_change_valid_edges() -> None:
    torch.manual_seed(1)
    hidden_states = torch.randn(1, 5, 8)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    graph = RelationalGraph({"diagonal_policy": "keep"})(hidden_states, attention_mask)
    valid_edge_mask = make_valid_edge_mask_like(graph, attention_mask)
    generator = torch.Generator().manual_seed(1)

    random_graph = make_random_graph_like(
        graph,
        generator=generator,
        valid_edge_mask=valid_edge_mask,
    )
    shuffled_graph = make_shuffled_graph(
        graph,
        generator=generator,
        valid_edge_mask=valid_edge_mask,
    )

    invalid = ~valid_edge_mask
    valid = valid_edge_mask

    assert torch.allclose(graph[invalid], random_graph[invalid])
    assert torch.allclose(graph[invalid], shuffled_graph[invalid])
    assert not torch.allclose(graph[valid], random_graph[valid])
    assert not torch.allclose(graph[valid], shuffled_graph[valid])
    assert torch.allclose(graph[valid].sort().values, shuffled_graph[valid].sort().values)


def test_geo_attention_random_and_shuffled_modes_use_w_level_controls() -> None:
    torch.manual_seed(1)
    hidden_states = torch.randn(1, 5, 16)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])

    for distance_mode in ["random_d", "shuffled_d"]:
        attention = GeoAttention(
            GeoAttentionConfig(
                n_heads=2,
                hidden_dim=16,
                dropout=0.0,
                distance_mode=distance_mode,
            ),
            relational_graph_config={"diagonal_policy": "keep"},
            distance_config={
                "normalization": "offdiag_zscore_clamp",
                "clip_value": 5.0,
                "diagonal_policy": "zero",
                "causal_runtime_distance": True,
            },
        )

        real_graph = attention.relational_graph(hidden_states, attention_mask)
        control_graph = attention.compute_control_graph(hidden_states, attention_mask)
        valid_edge_mask = make_valid_edge_mask_like(real_graph, attention_mask)

        assert torch.allclose(real_graph[~valid_edge_mask], control_graph[~valid_edge_mask])
        assert not torch.allclose(real_graph[valid_edge_mask], control_graph[valid_edge_mask])


def test_strict_w_controls_report_passes() -> None:
    report = build_strict_w_controls_report(seed=2027)

    assert report["status"] == "pass"
    assert report["checks"]["controls_built_at_w_level"]
    assert report["checks"]["same_valid_region_random"]
    assert report["checks"]["same_valid_region_shuffled"]
    assert report["checks"]["distance_finite_regions_match"]
    assert report["next_required_step"] == "relational_field_observer"
