import torch

from attention.geo_attention import GeoAttention, GeoAttentionConfig
from models.transformer_baseline import CausalSelfAttention, TransformerBaselineConfig


def test_causal_mask_blocks_future_attention() -> None:
    torch.manual_seed(1)
    config = TransformerBaselineConfig(
        vocab_size=16,
        context_length=5,
        n_layers=1,
        n_heads=1,
        hidden_dim=8,
        ffn_dim=16,
        dropout=0.0,
    )
    attention = CausalSelfAttention(config)
    hidden_states = torch.randn(2, 5, 8)

    _, weights = attention(hidden_states, need_weights=True)
    future_mask = torch.ones(5, 5, dtype=torch.bool).triu(diagonal=1)

    assert torch.all(weights[:, :, future_mask] == 0)


def test_padding_mask_blocks_padded_keys() -> None:
    torch.manual_seed(1)
    config = TransformerBaselineConfig(
        vocab_size=16,
        context_length=5,
        n_layers=1,
        n_heads=1,
        hidden_dim=8,
        ffn_dim=16,
        dropout=0.0,
    )
    attention = CausalSelfAttention(config)
    hidden_states = torch.randn(2, 5, 8)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )

    _, weights = attention(hidden_states, attention_mask=attention_mask, need_weights=True)

    assert torch.all(weights[0, :, :, 3:] == 0)
    assert torch.all(weights[1, :, :, 4:] == 0)


def test_geo_attention_causal_mask_blocks_future_attention() -> None:
    torch.manual_seed(1)
    attention = GeoAttention(
        GeoAttentionConfig(
            n_heads=1,
            hidden_dim=8,
            dropout=0.0,
            distance_mode="real_d",
            alpha_initial_value=0.1,
        ),
        distance_config={
            "normalization": "offdiag_zscore_clamp",
            "diagonal_policy": "zero",
            "causal_runtime_distance": True,
        },
    )
    hidden_states = torch.randn(2, 5, 8)

    _, weights = attention(hidden_states, need_weights=True)
    future_mask = torch.ones(5, 5, dtype=torch.bool).triu(diagonal=1)

    assert torch.all(weights[:, :, future_mask] == 0)
