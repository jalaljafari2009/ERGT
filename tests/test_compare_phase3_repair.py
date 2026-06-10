from pathlib import Path

import pytest

from experiments.compare_phase3_repair import (
    build_repair_report,
    load_required_run,
    parse_candidate,
)
from experiments.data_utils import save_json


def run_payload(
    condition: str,
    final_loss: float,
    *,
    alpha: float | None = None,
    gradient_mode: str | None = None,
    distance_mode: str | None = None,
) -> dict:
    payload = {
        "condition": condition,
        "seed": 2027,
        "final_validation_loss": final_loss,
        "best_validation_loss": final_loss,
        "perplexity": 10.0,
        "average_tokens_per_second": 1000.0,
    }
    if alpha is not None or gradient_mode is not None or distance_mode is not None:
        payload["attention"] = {
            "alpha": {"initial_value": alpha},
            "gradient_mode": gradient_mode,
            "distance_mode": distance_mode,
        }
    return payload


def references() -> dict[str, dict]:
    return {
        "baseline": run_payload("baseline", 5.8),
        "failed_real_d_alpha_0_2_grad": run_payload(
            "real_d", 5.9, alpha=0.2, gradient_mode="grad_d", distance_mode="real_d"
        ),
        "random_d_alpha_0_2": run_payload(
            "random_d", 5.7, alpha=0.2, gradient_mode="detached_d", distance_mode="random_d"
        ),
        "random_d_alpha_0_1": run_payload(
            "random_d", 5.75, alpha=0.1, gradient_mode="detached_d", distance_mode="random_d"
        ),
    }


def paths(tmp_path, labels) -> dict[str, Path]:
    return {label: tmp_path / f"{label}.json" for label in labels}


def test_repair_report_supports_detached_repeat(tmp_path) -> None:
    refs = references()
    candidates = {
        "real_d_alpha_0_2_detached": run_payload(
            "real_d", 5.6, alpha=0.2, gradient_mode="detached_d", distance_mode="real_d"
        )
    }

    report = build_repair_report(
        refs,
        paths(tmp_path, refs),
        candidates,
        paths(tmp_path, candidates),
    )

    assert report["ranking"]["by_final_validation_loss"][0] == "real_d_alpha_0_2_detached"
    assert report["checks"]["best_repair_beats_baseline"]
    assert report["checks"]["best_repair_beats_best_random_control"]
    assert report["summary"]["recommendation"] == "detached_repair_supports_phase3_repeat"


def test_repair_report_detects_failed_real_d_only_improvement(tmp_path) -> None:
    refs = references()
    candidates = {
        "real_d_alpha_0_2_detached": run_payload(
            "real_d", 5.85, alpha=0.2, gradient_mode="detached_d", distance_mode="real_d"
        )
    }

    report = build_repair_report(
        refs,
        paths(tmp_path, refs),
        candidates,
        paths(tmp_path, candidates),
    )

    assert report["checks"]["best_repair_beats_failed_real_d"]
    assert not report["checks"]["best_repair_beats_baseline"]
    assert report["summary"]["recommendation"] == "detached_repair_improves_real_d_only"


def test_parse_candidate_and_load_required_run(tmp_path) -> None:
    spec = parse_candidate("real_d_alpha_0_2_detached:runs/x/metrics.json")

    assert spec.label == "real_d_alpha_0_2_detached"
    assert spec.path == Path("runs/x/metrics.json")

    metrics_path = tmp_path / "metrics.json"
    save_json(metrics_path, run_payload("real_d", 5.9, alpha=0.2))

    run = load_required_run(metrics_path, "real_d")

    assert run["condition"] == "real_d"
    with pytest.raises(ValueError, match="expected condition"):
        load_required_run(metrics_path, "random_d")
