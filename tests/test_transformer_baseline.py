import torch

from models.transformer_baseline import TransformerBaseline, TransformerBaselineConfig


def tiny_config() -> TransformerBaselineConfig:
    return TransformerBaselineConfig(
        vocab_size=32,
        context_length=8,
        n_layers=2,
        n_heads=2,
        hidden_dim=16,
        ffn_dim=64,
        dropout=0.0,
    )


def test_baseline_forward_shapes_and_loss_are_finite() -> None:
    torch.manual_seed(1)
    model = TransformerBaseline(tiny_config())
    input_ids = torch.randint(0, 32, (2, 8))

    outputs = model(
        input_ids,
        targets=input_ids,
        return_hidden_states=True,
        return_attention_weights=True,
    )

    assert outputs["logits"].shape == (2, 8, 32)
    assert torch.isfinite(outputs["loss"])
    assert len(outputs["hidden_states"]) == 3
    assert len(outputs["attention_weights"]) == 2
    assert outputs["attention_weights"][0].shape == (2, 2, 8, 8)


def test_baseline_rejects_sequence_longer_than_context() -> None:
    model = TransformerBaseline(tiny_config())
    input_ids = torch.randint(0, 32, (2, 9))

    try:
        model(input_ids)
    except ValueError as exc:
        assert "exceeds context_length" in str(exc)
    else:
        raise AssertionError("expected context length ValueError")


def test_baseline_one_training_step_runs() -> None:
    torch.manual_seed(1)
    model = TransformerBaseline(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))

    outputs = model(input_ids, targets=targets)
    loss = outputs["loss"]
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_model_summary_contains_core_fields() -> None:
    model = TransformerBaseline(tiny_config())
    summary = model.model_summary()

    assert summary["type"] == "transformer_baseline"
    assert summary["vocab_size"] == 32
    assert summary["context_length"] == 8
    assert summary["total_params"] > 0
    assert summary["trainable_params"] > 0
