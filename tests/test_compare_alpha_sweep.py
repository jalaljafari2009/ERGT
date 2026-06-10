from pathlib import Path

from experiments.compare_alpha_sweep import (
    SweepRunSpec,
    build_alpha_sweep_report,
    load_sweep_run,
    parse_run_spec,
)
from experiments.data_utils import save_json


def sweep_run(condition: str, alpha: float, final_loss: float) -> dict:
    return {
        "condition": condition,
        "final_validation_loss": final_loss,
        "best_validation_loss": final_loss,
        "perplexity": 10.0,
        "average_tokens_per_second": 1000.0,
        "attention": {"alpha": {"initial_value": alpha}},
    }


def test_parse_run_spec() -> None:
    spec = parse_run_spec("real_d:0.05:runs/x/metrics.json")

    assert spec.family == "real_d"
    assert spec.alpha == 0.05
    assert spec.path == Path("runs/x/metrics.json")
    assert spec.label == "real_d_alpha_0_05"


def test_alpha_sweep_report_detects_best_real_d(tmp_path) -> None:
    baseline = {"condition": "baseline", "final_validation_loss": 5.8}
    runs = [
        {
            "label": "real_d_alpha_0_05",
            "family": "real_d",
            "alpha": 0.05,
            "path": "real_005.json",
            "metrics": sweep_run("real_d", 0.05, 5.6),
            "summary": {
                "condition": "real_d",
                "path": "real_005.json",
                "alpha": 0.05,
                "final_validation_loss": 5.6,
            },
        },
        {
            "label": "random_d_alpha_0_05",
            "family": "random_d",
            "alpha": 0.05,
            "path": "random_005.json",
            "metrics": sweep_run("random_d", 0.05, 5.7),
            "summary": {
                "condition": "random_d",
                "path": "random_005.json",
                "alpha": 0.05,
                "final_validation_loss": 5.7,
            },
        },
    ]

    report = build_alpha_sweep_report(baseline, tmp_path / "baseline.json", runs)

    assert report["ranking"][0] == "real_d_alpha_0_05"
    assert report["checks"]["best_real_d_beats_baseline"]
    assert report["checks"]["best_real_d_beats_best_random_d"]
    assert report["checks"]["real_d_beats_random_d_at_any_matched_alpha"]
    assert report["summary"]["recommendation"] == "alpha_sweep_supports_real_d"


def test_load_sweep_run_validates_condition_and_alpha(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.json"
    save_json(metrics_path, sweep_run("random_d", 0.2, 5.9))

    run = load_sweep_run(SweepRunSpec("random_d", 0.2, metrics_path))

    assert run["family"] == "random_d"
    assert run["summary"]["final_validation_loss"] == 5.9
    assert "\\" not in run["path"]
    assert "\\" not in run["summary"]["path"]
