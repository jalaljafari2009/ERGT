import json
import math
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from experiments.adaptive_alpha import (
    AdaptiveAlphaConfig,
    AdaptiveAlphaController,
    AlphaObservation,
    reference_loss_for_step,
    set_model_fixed_alpha,
)
from experiments.data_utils import PreparedDatasetMetadata, save_prepared_blocks
from experiments.progress_logging import format_progress_line
from experiments.train_ergt_adaptive_alpha import main as train_adaptive_main
from models.ergt_v1 import ERGTV1


def tiny_adaptive_config(output_dir: Path) -> dict:
    return {
        "run": {
            "phase": "adaptive_competitive_alpha",
            "condition": "real_memory_d_adaptive",
            "seed": 1,
            "output_dir": str(output_dir),
        },
        "dataset": {
            "name": "smoke",
            "split": "local",
            "tokenizer": "local",
            "context_length": 4,
        },
        "model": {
            "type": "ergt_v1",
            "vocab_size": None,
            "n_layers": 1,
            "n_heads": 1,
            "hidden_dim": 8,
            "ffn_dim": 16,
            "dropout": 0.0,
            "bias": True,
            "positional_encoding": "learned_absolute",
        },
        "attention": {
            "type": "geo_attention",
            "distance_mode": "real_stable_causal_d",
            "head_sharing": "shared_d",
            "alpha": {
                "mode": "fixed",
                "initial_value": 0.0,
                "non_negative": True,
                "warmup_steps": 0,
            },
            "gradient_mode": "detached_d",
            "causal_runtime_distance": True,
            "max_causal_step": 1,
            "memory": {
                "decay": 0.7,
                "eta": 0.3,
                "gate_floor": 0.05,
                "min_context_edges": 2,
            },
        },
        "relational_graph": {
            "kernel": "sigmoid_cosine",
            "graph_heads": 1,
            "normalize_hidden": True,
            "diagonal_policy": "keep_for_distance",
        },
        "distance": {
            "formula": "-log(W + epsilon)",
            "epsilon": 1e-6,
            "normalization": "offdiag_zscore_clamp",
            "clip_value": 5.0,
            "diagonal_policy": "zero",
        },
        "adaptive_alpha": {
            "initial_alpha": 0.0,
            "exploration_points": 1,
            "exploration_step": 0.02,
            "exploration_alpha": 0.02,
            "min_points_for_slope": 2,
            "slope_window_points": 2,
            "inertia": 0.5,
        },
        "training": {
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "batch_size": 2,
            "max_steps": 2,
            "warmup_steps": 0,
            "grad_clip": 1.0,
            "eval_interval": 1,
            "checkpoint_interval": 2,
            "max_eval_batches": 1,
        },
        "logging": {
            "train_log": "train_log.jsonl",
            "progress_log": "progress_log.jsonl",
            "adaptive_alpha_log": "adaptive_alpha_log.jsonl",
            "results": "metrics.json",
            "model_summary": "model_summary.json",
            "log_geometry_diagnostics": True,
        },
    }


def test_adaptive_alpha_requires_windowed_slope_before_competitive_growth() -> None:
    controller = AdaptiveAlphaController(
        AdaptiveAlphaConfig(
            initial_alpha=0.02,
            exploration_points=0,
            min_points_for_slope=4,
            slope_window_points=4,
        )
    )

    first = controller.update(
        AlphaObservation(step=100, validation_loss=5.0, reference_validation_loss=5.1)
    )
    second = controller.update(
        AlphaObservation(step=200, validation_loss=4.8, reference_validation_loss=5.0)
    )

    assert first.decision == "hold_insufficient_slope"
    assert second.decision == "hold_insufficient_slope"
    assert controller.current_alpha == 0.02


def test_adaptive_alpha_grows_from_smoothed_slope_gain() -> None:
    controller = AdaptiveAlphaController(
        AdaptiveAlphaConfig(
            initial_alpha=0.02,
            exploration_points=0,
            min_points_for_slope=4,
            slope_window_points=4,
            positive_margin=0.00001,
            inertia=0.5,
        )
    )
    observations = [
        AlphaObservation(step=100, validation_loss=5.0, reference_validation_loss=5.0),
        AlphaObservation(step=200, validation_loss=4.8, reference_validation_loss=4.85),
        AlphaObservation(step=300, validation_loss=4.6, reference_validation_loss=4.72),
        AlphaObservation(step=400, validation_loss=4.4, reference_validation_loss=4.60),
    ]

    for observation in observations:
        decision = controller.update(observation)

    assert decision.decision == "grow"
    assert decision.slope_gain is not None and decision.slope_gain > 0
    assert controller.current_alpha > 0.02


def test_set_model_fixed_alpha_updates_all_geoattention_layers() -> None:
    config = tiny_adaptive_config(Path("unused"))
    config["model"]["vocab_size"] = 16
    config["model"]["n_layers"] = 2
    model = ERGTV1(config)

    set_model_fixed_alpha(model, 0.037)

    alphas = [block.attn.target_alpha().item() for block in model.blocks]
    assert alphas == pytest.approx([0.037, 0.037])


def test_reference_loss_for_step_uses_latest_previous_point() -> None:
    reference = {100: 5.0, 300: 4.5}

    assert reference_loss_for_step(reference, 300) == 4.5
    assert reference_loss_for_step(reference, 350) == 4.5
    assert reference_loss_for_step(reference, 50) is None


def test_train_ergt_adaptive_alpha_smoke_outputs_controller_metrics() -> None:
    tmp = Path(tempfile.mkdtemp())
    data_dir = tmp / "data"
    output_dir = tmp / "adaptive"
    reference_path = tmp / "baseline_progress.jsonl"
    train_blocks = torch.arange(60).remainder(16).view(12, 5)
    validation_blocks = torch.arange(30).remainder(16).view(6, 5)
    save_prepared_blocks(
        data_dir,
        train_blocks,
        validation_blocks,
        PreparedDatasetMetadata(
            dataset_name="smoke",
            tokenizer="local",
            context_length=4,
            vocab_size=16,
            train_tokens=60,
            validation_tokens=30,
            train_sequences=12,
            validation_sequences=6,
            eos_token_id=None,
        ),
    )
    reference_path.write_text(
        "\n".join(
            [
                json.dumps({"step": 1, "validation_loss": 10.0}),
                json.dumps({"step": 2, "validation_loss": 9.9}),
            ]
        ),
        encoding="utf-8",
    )
    config = tiny_adaptive_config(output_dir)
    config_path = tmp / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    old_argv = sys.argv
    try:
        sys.argv = [
            "train_ergt_adaptive_alpha",
            "--config",
            str(config_path),
            "--data-dir",
            str(data_dir),
            "--reference-progress",
            str(reference_path),
            "--device",
            "cpu",
        ]
        train_adaptive_main()
    finally:
        sys.argv = old_argv

    results = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    progress_rows = [
        json.loads(line)
        for line in (output_dir / "progress_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert math.isfinite(results["final_validation_loss"])
    assert "adaptive_alpha" in results
    assert len(progress_rows) == 2
    assert "adaptive_alpha" in progress_rows[-1]
    assert "alpha_next" in progress_rows[-1]
    assert "alpha_delta" in progress_rows[-1]
    assert "adaptive_slope_gain" in progress_rows[-1]
    assert "adaptive_advantage" in progress_rows[-1]
    assert "geo_qk_risk" in progress_rows[-1]
    assert "entropy_risk" in progress_rows[-1]
    assert "max_probability_risk" in progress_rows[-1]


def test_progress_line_includes_adaptive_telemetry_fields() -> None:
    line = format_progress_line(
        {
            "condition": "real_memory_d_adaptive",
            "step": 100,
            "validation_loss": 5.0,
            "alpha_effective": 0.02,
            "alpha_next": 0.03,
            "alpha_delta": 0.01,
            "alpha_decision": "grow",
            "adaptive_score": 0.0003,
            "adaptive_slope_gain": 0.0002,
            "adaptive_advantage": 0.001,
            "geo_to_qk_ratio": 0.06,
            "geo_qk_risk": 0.0,
            "attention_entropy": 3.2,
            "entropy_risk": 0.1,
            "mean_max_probability": 0.2,
            "max_probability_risk": 0.0,
        }
    )

    assert "decision=grow" in line
    assert "a_next=0.0300" in line
    assert "d_alpha=0.0100" in line
    assert "slope=0.000200" in line
    assert "adv=0.001000" in line
    assert "gRisk=0.000" in line
    assert "ent=3.200" in line
    assert "maxp=0.200" in line
