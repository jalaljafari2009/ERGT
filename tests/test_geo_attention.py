import torch

from attention.geo_attention import GeoAttention, GeoAttentionConfig
from models.transformer_baseline import CausalSelfAttention, TransformerBaselineConfig


def geo_attention(
    *,
    distance_mode: str = "real_d",
    alpha: float = 0.1,
    alpha_warmup_steps: int = 0,
    gradient_mode: str = "grad_d",
) -> GeoAttention:
    return GeoAttention(
        GeoAttentionConfig(
            n_heads=2,
            hidden_dim=16,
            dropout=0.0,
            distance_mode=distance_mode,
            alpha_initial_value=alpha,
            alpha_warmup_steps=alpha_warmup_steps,
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


def geo_attention_v2(
    *,
    distance_mode: str = "real_stable_causal_d",
    alpha: float = 0.1,
) -> GeoAttention:
    return GeoAttention(
        GeoAttentionConfig(
            n_heads=2,
            hidden_dim=16,
            dropout=0.0,
            distance_mode=distance_mode,
            alpha_initial_value=alpha,
            gradient_mode="detached_d",
            max_causal_step=1,
        ),
        relational_graph_config={
            "kernel": "sigmoid_cosine",
            "graph_heads": 1,
            "normalize_hidden": True,
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


def test_geo_attention_v2_alpha_zero_matches_baseline_attention() -> None:
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
    geo = geo_attention_v2(alpha=0.0)
    geo.qkv_proj.load_state_dict(baseline.qkv_proj.state_dict())
    geo.out_proj.load_state_dict(baseline.out_proj.state_dict())
    hidden_states = torch.randn(2, 4, 16)

    baseline_output = baseline(hidden_states)
    geo_output = geo(hidden_states)

    assert torch.allclose(geo_output, baseline_output, atol=1e-6)


def test_geo_attention_v2_alpha_zero_skips_geometry_construction() -> None:
    torch.manual_seed(1)
    attention = geo_attention_v2(alpha=0.0, distance_mode="real_stable_causal_d")
    hidden_states = torch.randn(2, 5, 16)

    result = attention(hidden_states, return_diagnostics=True)

    assert result["geometry_memory"] is None
    assert torch.all(result["distance"] == 0)
    assert result["diagnostics"]["geometry_skipped_by_alpha_zero"] is True
    assert result["diagnostics"]["geo_to_qk_ratio"] == 0.0


def test_geo_attention_distance_modes_run_and_are_finite() -> None:
    torch.manual_seed(1)
    hidden_states = torch.randn(2, 4, 16)

    for distance_mode in ["real_d", "random_d", "shuffled_d", "zero_d"]:
        attention = geo_attention(distance_mode=distance_mode, gradient_mode="detached_d")
        result = attention(hidden_states, return_diagnostics=True)

        assert torch.isfinite(result["output"]).all()
        assert torch.isfinite(result["attention_weights"]).all()


def test_geo_attention_v2_distance_modes_run_and_report_geometry_metadata() -> None:
    torch.manual_seed(1)
    hidden_states = torch.randn(2, 5, 16)
    v2_modes = [
        "real_stable_causal_d",
        "random_stable_causal_d",
        "shuffled_stable_causal_d",
        "instantaneous_real_d",
        "pairwise_real_d",
        "no_memory_real_d",
    ]

    for distance_mode in v2_modes:
        attention = geo_attention_v2(distance_mode=distance_mode)
        result = attention(hidden_states, return_diagnostics=True)

        assert torch.isfinite(result["output"]).all()
        assert torch.isfinite(result["attention_weights"]).all()
        assert result["diagnostics"]["geometry_version"] == "v2"
        assert result["diagnostics"]["distance_mode"] == distance_mode
        assert result["diagnostics"]["max_causal_step"] == 1


def test_geo_attention_v2_memory_blends_previous_layer_geometry() -> None:
    torch.manual_seed(1)
    attention = geo_attention_v2(distance_mode="real_stable_causal_d")
    hidden_states = torch.randn(2, 5, 16)

    first = attention(hidden_states, return_diagnostics=True)
    second = attention(
        hidden_states + 0.01,
        geometry_memory=first["geometry_memory"],
        return_diagnostics=True,
    )

    assert first["geometry_memory"] is not None
    assert second["geometry_memory"] is not None
    assert first["diagnostics"]["geometry_memory_used"] is False
    assert second["diagnostics"]["geometry_memory_used"] is True
    assert not torch.allclose(first["geometry_memory"], second["geometry_memory"])


def test_geo_attention_bias_penalizes_unreachable_past_distance() -> None:
    attention = geo_attention_v2(distance_mode="pairwise_real_d")
    qk_logits = torch.zeros(1, 1, 3, 3)
    distance = torch.tensor(
        [[[[0.0, torch.inf, torch.inf], [0.5, 0.0, torch.inf], [torch.inf, 0.2, 0.0]]]]
    )

    biased = attention.apply_geometry_bias(qk_logits, distance, torch.tensor(1.0))

    assert biased[0, 0, 2, 0] < biased[0, 0, 2, 1]


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


def test_geo_attention_alpha_warmup_scales_effective_alpha() -> None:
    attention = geo_attention(alpha=0.2, alpha_warmup_steps=10)

    assert torch.isclose(attention.alpha(), torch.tensor(0.0))

    attention.set_training_step(5)
    assert torch.isclose(attention.alpha(), torch.tensor(0.1))

    attention.set_training_step(20)
    assert torch.isclose(attention.alpha(), torch.tensor(0.2))

    hidden_states = torch.randn(2, 4, 16)
    result = attention(hidden_states, return_diagnostics=True)

    assert abs(result["diagnostics"]["target_alpha"] - 0.2) < 1e-6
    assert result["diagnostics"]["alpha_warmup_factor"] == 1.0
