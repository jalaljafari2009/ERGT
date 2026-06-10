import json
from pathlib import Path

import pytest

from experiments.compare_phase3_stable_base import (
    build_stable_base_report,
    load_alpha_zero_run,
    load_stable_run,
    parse_run_spec,
)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def metrics_payload(
    condition: str,
    final_loss: float,
    *,
    distance_mode: str | None = None,
    alpha: float | None = None,
    warmup_steps: int = 200,
) -> dict:
    payload = {
        "condition": condition,
        "seed": 2027,
        "final_training_loss": final_loss - 0.2,
        "best_validation_loss": final_loss,
        "final_validation_loss": final_loss,
        "perplexity": 300.0,
        "average_tokens_per_second": 5000.0,
        "device": "cuda",
    }
    if distance_mode is not None and alpha is not None:
        payload["attention"] = {
            "distance_mode": distance_mode,
            "gradient_mode": "detached_d",
            "alpha": {
                "initial_value": alpha,
                "warmup_steps": warmup_steps,
                "mode": "fixed",
                "non_negative": True,
            },
        }
        payload["geometry"] = {
            "summary": {
                "alpha": alpha,
                "target_alpha": alpha,
                "alpha_warmup_factor": 1.0,
                "geo_to_qk_ratio": 0.1,
                "distance_mean": 2.0,
                "distance_std": 1.0,
            }
        }
    return payload


def config_payload(distance_mode: str, alpha: float, *, kernel: str = "sigmoid_cosine") -> dict:
    return {
        "attention": {
            "distance_mode": distance_mode,
            "gradient_mode": "detached_d",
            "alpha": {
                "initial_value": alpha,
                "warmup_steps": 0 if alpha == 0.0 else 200,
                "mode": "fixed",
                "non_negative": True,
            },
        },
        "relational_graph": {
            "kernel": kernel,
            "graph_heads": 1,
            "normalize_hidden": True,
        },
        "distance": {
            "normalization": "offdiag_zscore_clamp",
            "clip_value": 5.0,
        },
    }


def write_run(
    tmp_path: Path,
    label: str,
    condition: str,
    distance_mode: str,
    alpha: float,
    final_loss: float,
    *,
    kernel: str = "sigmoid_cosine",
) -> Path:
    run_dir = tmp_path / label
    metrics_path = run_dir / "metrics.json"
    save_json(
        metrics_path,
        metrics_payload(condition, final_loss, distance_mode=distance_mode, alpha=alpha),
    )
    save_json(run_dir / "config.json", config_payload(distance_mode, alpha, kernel=kernel))
    return metrics_path


def test_stable_base_report_detects_candidate(tmp_path: Path) -> None:
    baseline = metrics_payload("baseline", 5.8)
    baseline_path = tmp_path / "baseline.json"
    save_json(baseline_path, baseline)
    alpha_zero_path = write_run(tmp_path, "alpha_zero", "alpha_zero", "real_d", 0.0, 5.81)
    real_path = write_run(tmp_path, "real", "real_d", "real_d", 0.1, 5.6)
    random_path = write_run(tmp_path, "random", "random_d", "random_d", 0.1, 5.7)
    shuffled_path = write_run(tmp_path, "shuffled", "shuffled_d", "shuffled_d", 0.1, 5.9)

    alpha_zero = load_alpha_zero_run(alpha_zero_path)
    runs = [
        load_stable_run(parse_run_spec(f"real_d:0.1:{real_path.as_posix()}")),
        load_stable_run(parse_run_spec(f"random_d:0.1:{random_path.as_posix()}")),
        load_stable_run(parse_run_spec(f"shuffled_d:0.1:{shuffled_path.as_posix()}")),
    ]
    report = build_stable_base_report(
        baseline=baseline,
        baseline_path=baseline_path,
        alpha_zero=alpha_zero,
        alpha_zero_path=alpha_zero_path,
        runs=runs,
        alpha_zero_tolerance=0.02,
    )

    assert report["checks"]["best_real_beats_baseline"]
    assert report["checks"]["best_real_beats_best_random"]
    assert report["checks"]["best_real_beats_best_shuffled"]
    assert report["summary"]["recommendation"] == "stable_candidate_found_repeat_seeds"


def test_stable_base_loader_rejects_non_cosine_kernel(tmp_path: Path) -> None:
    metrics_path = write_run(
        tmp_path,
        "bad_real",
        "real_d",
        "real_d",
        0.1,
        5.6,
        kernel="sigmoid_dot_sqrt_d",
    )

    with pytest.raises(ValueError, match="kernel=sigmoid_cosine"):
        load_stable_run(parse_run_spec(f"real_d:0.1:{metrics_path.as_posix()}"))
