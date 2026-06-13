import json
import math
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from attention.geo_attention import GeoAttention, GeoAttentionConfig
from evaluation.attention_metrics import aggregate_attention_diagnostics
from experiments.data_utils import PreparedDatasetMetadata, save_prepared_blocks
from experiments.train_ergt_v1 import main as train_ergt_main
from models.ergt_v1 import ERGTV1


def tiny_ergt_config(output_dir: Path) -> dict:
    return {
        "run": {
            "phase": "phase3_geo_attention",
            "condition": "real_d",
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
            "distance_mode": "real_d",
            "head_sharing": "shared_d",
            "alpha": {"mode": "fixed", "initial_value": 0.1, "non_negative": True},
            "gradient_mode": "grad_d",
            "causal_runtime_distance": True,
        },
        "relational_graph": {
            "kernel": "sigmoid_dot_sqrt_d",
            "graph_heads": 1,
            "normalize_hidden": False,
            "diagonal_policy": "keep_for_distance",
        },
        "distance": {
            "formula": "-log(W + epsilon)",
            "epsilon": 1e-6,
            "normalization": "offdiag_zscore_clamp",
            "clip_value": 5.0,
            "diagonal_policy": "zero",
        },
        "training": {
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "batch_size": 2,
            "max_steps": 1,
            "warmup_steps": 0,
            "grad_clip": 1.0,
            "eval_interval": 1,
            "checkpoint_interval": 1,
        },
        "logging": {
            "train_log": "train_log.jsonl",
            "results": "metrics.json",
            "model_summary": "model_summary.json",
            "log_geometry_diagnostics": True,
        },
    }


def test_random_distance_mode_changes_between_forward_calls() -> None:
    torch.manual_seed(1)
    attention = GeoAttention(
        GeoAttentionConfig(
            n_heads=1,
            hidden_dim=8,
            dropout=0.0,
            distance_mode="random_d",
            alpha_initial_value=0.1,
        ),
        distance_config={"normalization": "offdiag_zscore_clamp", "diagonal_policy": "zero"},
    )
    hidden_states = torch.randn(1, 4, 8)

    first = attention.compute_distance(hidden_states)
    second = attention.compute_distance(hidden_states)

    assert not torch.allclose(first, second)


def test_geometry_diagnostics_include_attention_metrics() -> None:
    torch.manual_seed(1)
    config = tiny_ergt_config(Path("unused"))
    config["model"]["vocab_size"] = 16
    model = ERGTV1(config)
    input_ids = torch.randint(0, 16, (2, 4))

    outputs = model(input_ids, targets=input_ids, return_geometry_diagnostics=True)
    diagnostics = aggregate_attention_diagnostics(outputs["geometry_diagnostics"])

    assert "attention_entropy" in diagnostics["layers"]["layer_0"]
    assert "attention_entropy" in diagnostics["summary"]


def test_ergt_v1_geoattention_v2_carries_memory_between_layers() -> None:
    torch.manual_seed(1)
    config = tiny_ergt_config(Path("unused"))
    config["model"]["vocab_size"] = 16
    config["model"]["n_layers"] = 2
    config["attention"]["distance_mode"] = "real_stable_causal_d"
    config["attention"]["gradient_mode"] = "detached_d"
    config["attention"]["max_causal_step"] = 1
    config["attention"]["memory"] = {
        "decay": 0.7,
        "eta": 0.3,
        "gate_floor": 0.05,
        "min_context_edges": 2,
    }
    config["relational_graph"]["kernel"] = "sigmoid_cosine"
    config["relational_graph"]["normalize_hidden"] = True
    model = ERGTV1(config)
    input_ids = torch.randint(0, 16, (2, 4))

    outputs = model(input_ids, targets=input_ids, return_geometry_diagnostics=True)
    diagnostics = outputs["geometry_diagnostics"]

    assert len(diagnostics) == 2
    assert diagnostics[0]["diagnostics"]["geometry_version"] == "v2"
    assert diagnostics[0]["diagnostics"]["geometry_memory_used"] is False
    assert diagnostics[1]["diagnostics"]["geometry_memory_used"] is True
    assert torch.isfinite(outputs["loss"])


@pytest.mark.parametrize(
    ("condition", "distance_mode", "alpha"),
    [
        ("real_d", "real_d", 0.1),
        ("alpha_zero", "real_d", 0.0),
        ("random_d", "random_d", 0.1),
        ("shuffled_d", "shuffled_d", 0.1),
    ],
)
def test_train_ergt_v1_smoke_outputs_finite_metrics_and_exact_step(
    condition: str,
    distance_mode: str,
    alpha: float,
) -> None:
    tmp = Path(tempfile.mkdtemp())
    data_dir = tmp / "data"
    output_dir = tmp / condition
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

    config = tiny_ergt_config(output_dir)
    config["run"]["condition"] = condition
    config["attention"]["distance_mode"] = distance_mode
    config["attention"]["alpha"]["initial_value"] = alpha
    if distance_mode in {"random_d", "shuffled_d"}:
        config["attention"]["gradient_mode"] = "detached_d"
    config_path = tmp / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    old_argv = sys.argv
    try:
        sys.argv = [
            "train_ergt_v1",
            "--config",
            str(config_path),
            "--data-dir",
            str(data_dir),
            "--device",
            "cpu",
        ]
        train_ergt_main()
    finally:
        sys.argv = old_argv

    results = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    progress_rows = [
        json.loads(line)
        for line in (output_dir / "progress_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checkpoint = torch.load(output_dir / "checkpoints" / "last.pt", map_location="cpu")

    assert math.isfinite(results["final_validation_loss"])
    assert math.isfinite(results["perplexity"])
    assert len(progress_rows) == 1
    assert math.isfinite(progress_rows[0]["validation_loss"])
    assert math.isfinite(progress_rows[0]["best_validation_loss"])
    assert "tokens_per_second" in progress_rows[0]
    assert "alpha_effective" in progress_rows[0]
    assert checkpoint["step"] == 1
