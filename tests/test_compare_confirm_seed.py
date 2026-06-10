from pathlib import Path

import pytest

from experiments.compare_confirm_seed import (
    build_confirm_seed_report,
    load_required_run,
)
from experiments.data_utils import save_json


def run_payload(condition: str, seed: int, final_loss: float, alpha: float | None = None) -> dict:
    payload = {
        "condition": condition,
        "seed": seed,
        "final_validation_loss": final_loss,
        "best_validation_loss": final_loss,
        "perplexity": 10.0,
        "average_tokens_per_second": 1000.0,
    }
    if alpha is not None:
        payload["attention"] = {"alpha": {"initial_value": alpha}}
    return payload


def confirm_runs(
    *,
    baseline_loss: float,
    real_loss: float,
    matched_random_loss: float,
    best_random_loss: float,
) -> dict[str, dict]:
    return {
        "baseline": run_payload("baseline", 2027, baseline_loss),
        "real_d_alpha_0_2": run_payload("real_d", 2027, real_loss, 0.2),
        "random_d_alpha_0_2": run_payload("random_d", 2027, matched_random_loss, 0.2),
        "random_d_alpha_0_1": run_payload("random_d", 2027, best_random_loss, 0.1),
    }


def confirm_paths(tmp_path) -> dict[str, Path]:
    return {
        "baseline": tmp_path / "baseline.json",
        "real_d_alpha_0_2": tmp_path / "real_d.json",
        "random_d_alpha_0_2": tmp_path / "random_02.json",
        "random_d_alpha_0_1": tmp_path / "random_01.json",
    }


def test_confirm_seed_report_supports_gate1(tmp_path) -> None:
    runs = confirm_runs(
        baseline_loss=5.8,
        real_loss=5.6,
        matched_random_loss=5.7,
        best_random_loss=5.65,
    )

    report = build_confirm_seed_report(runs, confirm_paths(tmp_path))

    assert report["ranking"]["by_final_validation_loss"][0] == "real_d_alpha_0_2"
    assert report["checks"]["all_seeds_equal"]
    assert report["checks"]["real_d_beats_baseline"]
    assert report["checks"]["real_d_beats_random_d_alpha_0_2"]
    assert report["checks"]["real_d_beats_best_random_control"]
    assert report["summary"]["recommendation"] == "confirm_seed_supports_gate1"


def test_confirm_seed_report_detects_conditional_support(tmp_path) -> None:
    runs = confirm_runs(
        baseline_loss=5.8,
        real_loss=5.64,
        matched_random_loss=5.7,
        best_random_loss=5.63,
    )

    report = build_confirm_seed_report(runs, confirm_paths(tmp_path))

    assert report["checks"]["real_d_beats_baseline"]
    assert report["checks"]["real_d_beats_random_d_alpha_0_2"]
    assert not report["checks"]["real_d_beats_best_random_control"]
    assert report["summary"]["recommendation"] == "confirm_seed_conditional_support"


def test_load_required_run_validates_condition_alpha_and_paths(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.json"
    save_json(metrics_path, run_payload("real_d", 2027, 5.9, 0.2))

    run = load_required_run(metrics_path, "real_d", 0.2)

    assert run["condition"] == "real_d"

    with pytest.raises(ValueError, match="expected alpha"):
        load_required_run(metrics_path, "real_d", 0.1)

    with pytest.raises(ValueError, match="expected condition"):
        load_required_run(metrics_path, "random_d", 0.2)
