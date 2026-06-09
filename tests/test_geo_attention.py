import torch

from attention.geo_attention import GeoAttention, GeoAttentionConfig
from models.transformer_baseline import CausalSelfAttention, TransformerBaselineConfig


def geo_attention(
    *,
    distance_mode: str = "real_d",
    alpha: float = 0.1,
    gradient_mode: str = "grad_d",
) -> GeoAttention:
    return GeoAttention(
        GeoAttentionConfig(
            n_heads=2,
            hidden_dim=16,
            dropout=0.0,
            distance_mode=distance_mode,
            alpha_initial_value=alpha,
            gradient_mode=gradient_mode,
        ),
        relational_graph_config={
            "kernel": "sigmoid_dot_sqrt_d",
            "graph_heads": 1,
            "normalize_hidden": False,
            "diagonal_policy": "keep",
        },
        distance_config={
            "epsilon": 1e-6,
            "normalization": "offdiag_zscore_clamp",
            "clip_value": 5.0,
            "diagonal_policy": "zero",
            "causal_runtime_distance": True,
        },
    )


def test_geo_attention_forward_shapes_and_diagnostics() -> None:
    torch.manual_seed(1)
    attention = geo_attention()
    hidden_states = torch.randn(2, 4, 16)

    result = attention(hidden_states, return_diagnostics=True)

    assert result["output"].shape == (2, 4, 16)
    assert result["attention_weights"].shape == (2, 2, 4, 4)
    assert result["distance"].shape == (2, 2, 4, 4)
    assert result["diagnostics"]["geo_to_qk_ratio"] >= 0


def test_geo_attention_alpha_zero_matches_baseline_attention() -> None:
    torch.manual_seed(1)
    baseline_config = TransformerBaselineConfig(
        vocab_size=16,
        context_length=4,
        n_layers=1,
        n_heads=2,
        hidden_dim=16,
        ffn_dim=32,
        dropout=0.0,
    )
    baseline = CausalSelfAttention(baseline_config)
    geo = geo_attention(alpha=0.0)
    geo.qkv_proj.load_state_dict(baseline.qkv_proj.state_dict())
    geo.out_proj.load_state_dict(baseline.out_proj.state_dict())
    hidden_states = torch.randn(2, 4, 16)

    baseline_output = baseline(hidden_states)
    geo_output = geo(hidden_states)

    assert torch.allclose(geo_output, baseline_output, atol=1e-6)


def test_geo_attention_distance_modes_run_and_are_finite() -> None:
    torch.manual_seed(1)
    hidden_states = torch.randn(2, 4, 16)

    for distance_mode in ["real_d", "random_d", "shuffled_d", "zero_d"]:
        attention = geo_attention(distance_mode=distance_mode, gradient_mode="detached_d")
        result = attention(hidden_states, return_diagnostics=True)

        assert torch.isfinite(result["output"]).all()
        assert torch.isfinite(result["attention_weights"]).all()


def test_geo_attention_detached_distance_has_no_distance_grad_path() -> None:
    torch.manual_seed(1)
    attention = geo_attention(gradient_mode="detached_d")
    hidden_states = torch.randn(2, 4, 16, requires_grad=True)

    result = attention(hidden_states, return_diagnostics=True)
    loss = result["output"].sum()
    loss.backward()

    assert hidden_states.grad is not None
    assert result["distance"].requires_grad is False


def test_geo_attention_zero_distance_mode_has_zero_geo_ratio() -> None:
    torch.manual_seed(1)
    attention = geo_attention(distance_mode="zero_d", alpha=0.1)
    hidden_states = torch.randn(2, 4, 16)

    result = attention(hidden_states, return_diagnostics=True)

    assert result["diagnostics"]["mean_abs_geo"] == 0.0
    assert result["diagnostics"]["geo_to_qk_ratio"] == 0.0
