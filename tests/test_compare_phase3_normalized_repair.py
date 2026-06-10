import pytest

from experiments.compare_phase3_normalized_repair import (
    build_normalized_repair_report,
    load_normalized_run,
    parse_run_spec,
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
    if alpha is not None:
        payload["attention"] = {
            "alpha": {"initial_value": alpha},
            "gradient_mode": gradient_mode,
            "distance_mode": distance_mode,
        }
    return payload


def config_payload(*, normalize_hidden: bool = True) -> dict:
    return {"relational_graph": {"normalize_hidden": normalize_hidden}}


def normalized_run(
    tmp_path,
    label: str,
    condition: str,
    loss: float,
    alpha: float,
    distance_mode: str,
    *,
    normalize_hidden: bool = True,
) -> dict:
    run_dir = tmp_path / label
    run_dir.mkdir()
    metrics_path = run_dir / "metrics.json"
    save_json(
        metrics_path,
        run_payload(
            condition,
            loss,
            alpha=alpha,
            gradient_mode="detached_d",
            distance_mode=distance_mode,
        ),
    )
    save_json(run_dir / "config.json", config_payload(normalize_hidden=normalize_hidden))
    family = condition
    return load_normalized_run(parse_run_spec(f"{family}:{alpha}:{metrics_path.as_posix()}"))


def references() -> tuple[dict, dict, dict]:
    baseline = run_payload("baseline", 5.8)
    prior_detached = run_payload(
        "real_d", 5.75, alpha=0.1, gradient_mode="detached_d", distance_mode="real_d"
    )
    failed_real = run_payload(
        "real_d", 5.9, alpha=0.2, gradient_mode="grad_d", distance_mode="real_d"
    )
    return baseline, prior_detached, failed_real


def test_normalized_repair_report_supports_phase3_repeat(tmp_path) -> None:
    baseline, prior_detached, failed_real = references()
    runs = [
        normalized_run(tmp_path, "real", "real_d", 5.6, 0.1, "real_d"),
        normalized_run(tmp_path, "random", "random_d", 5.7, 0.1, "random_d"),
    ]

    report = build_normalized_repair_report(
        baseline=baseline,
        baseline_path=tmp_path / "baseline.json",
        prior_detached=prior_detached,
        prior_detached_path=tmp_path / "prior.json",
        failed_real_d=failed_real,
        failed_real_d_path=tmp_path / "failed.json",
        runs=runs,
    )

    assert report["checks"]["best_real_norm_beats_baseline"]
    assert report["checks"]["best_real_norm_beats_best_random_norm"]
    assert report["summary"]["recommendation"] == "normalized_repair_supports_phase3_repeat"


def test_normalized_repair_report_detects_not_supported(tmp_path) -> None:
    baseline, prior_detached, failed_real = references()
    runs = [
        normalized_run(tmp_path, "real", "real_d", 5.81, 0.1, "real_d"),
        normalized_run(tmp_path, "random", "random_d", 5.7, 0.1, "random_d"),
    ]

    report = build_normalized_repair_report(
        baseline=baseline,
        baseline_path=tmp_path / "baseline.json",
        prior_detached=prior_detached,
        prior_detached_path=tmp_path / "prior.json",
        failed_real_d=failed_real,
        failed_real_d_path=tmp_path / "failed.json",
        runs=runs,
    )

    assert not report["checks"]["best_real_norm_beats_baseline"]
    assert not report["checks"]["best_real_norm_beats_best_random_norm"]
    assert report["summary"]["recommendation"] == "normalized_repair_not_supported"


def test_load_normalized_run_validates_config(tmp_path) -> None:
    run_dir = tmp_path / "bad_norm"
    run_dir.mkdir()
    metrics_path = run_dir / "metrics.json"
    save_json(
        metrics_path,
        run_payload(
            "real_d",
            5.9,
            alpha=0.1,
            gradient_mode="detached_d",
            distance_mode="real_d",
        ),
    )
    save_json(run_dir / "config.json", config_payload(normalize_hidden=False))

    with pytest.raises(ValueError, match="normalize_hidden=true"):
        load_normalized_run(parse_run_spec(f"real_d:0.1:{metrics_path.as_posix()}"))
