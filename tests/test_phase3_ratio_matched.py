import json
from pathlib import Path

import pytest

from experiments.build_ratio_matched_configs import (
    CalibrationSpec,
    build_ratio_matched_configs,
)
from experiments.compare_phase3_ratio_matched import (
    build_ratio_matched_report,
    load_ratio_matched_run,
    parse_run_spec,
)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def metrics_payload(
    condition: str,
    final_loss: float,
    *,
    alpha: float,
    ratio: float,
) -> dict:
    return {
        "condition": condition,
        "seed": 2027,
        "attention": {
            "distance_mode": condition,
            "gradient_mode": "detached_d",
            "alpha": {
                "mode": "fixed",
                "initial_value": alpha,
                "non_negative": True,
                "warmup_steps": 200,
            },
        },
        "geometry": {
            "summary": {
                "geo_to_qk_ratio": ratio,
                "alpha": alpha,
                "target_alpha": alpha,
            }
        },
        "final_training_loss": final_loss - 0.2,
        "best_validation_loss": final_loss,
        "final_validation_loss": final_loss,
        "perplexity": 300.0,
        "average_tokens_per_second": 5000.0,
    }


def config_payload(
    condition: str,
    alpha: float,
    *,
    output_dir: str = "runs/test",
    ratio_match: dict | None = None,
) -> dict:
    return {
        "run": {
            "phase": "phase3_geo_attention",
            "condition": condition,
            "seed": 2027,
            "output_dir": output_dir,
            **({"ratio_match": ratio_match} if ratio_match else {}),
        },
        "attention": {
            "distance_mode": condition,
            "gradient_mode": "detached_d",
            "alpha": {
                "mode": "fixed",
                "initial_value": alpha,
                "non_negative": True,
                "warmup_steps": 200,
            },
        },
        "relational_graph": {
            "kernel": "sigmoid_cosine",
            "graph_heads": 1,
            "normalize_hidden": True,
        },
        "distance": {
            "normalization": "offdiag_zscore_clamp",
            "clip_value": 5.0,
        },
    }


def write_calibration(
    tmp_path: Path,
    condition: str,
    *,
    alpha: float,
    ratio: float,
    final_loss: float = 5.7,
) -> Path:
    run_dir = tmp_path / f"cal_{condition}"
    metrics_path = run_dir / "metrics.json"
    save_json(metrics_path, metrics_payload(condition, final_loss, alpha=alpha, ratio=ratio))
    save_json(run_dir / "config.json", config_payload(condition, alpha))
    return metrics_path


def write_ratio_run(
    tmp_path: Path,
    condition: str,
    *,
    target_ratio: float,
    observed_ratio: float,
    alpha: float,
    final_loss: float,
) -> Path:
    label = f"{condition}_{target_ratio}"
    run_dir = tmp_path / label
    metrics_path = run_dir / "metrics.json"
    ratio_match = {
        "target_geo_to_qk_ratio": target_ratio,
        "calibration_alpha": alpha,
        "calibration_geo_to_qk_ratio": observed_ratio,
        "generated_alpha": alpha,
    }
    save_json(
        metrics_path,
        metrics_payload(condition, final_loss, alpha=alpha, ratio=observed_ratio),
    )
    save_json(
        run_dir / "config.json",
        config_payload(condition, alpha, output_dir=run_dir.as_posix(), ratio_match=ratio_match),
    )
    return metrics_path


def test_build_ratio_matched_configs_scales_alpha(tmp_path: Path) -> None:
    real_metrics = write_calibration(tmp_path, "real_d", alpha=0.1, ratio=0.25)
    random_metrics = write_calibration(tmp_path, "random_d", alpha=0.05, ratio=0.10)

    manifest = build_ratio_matched_configs(
        target_ratios=[0.2],
        calibrations=[
            CalibrationSpec("real_d", real_metrics),
            CalibrationSpec("random_d", random_metrics),
        ],
        output_dir=tmp_path / "configs",
        run_output_root=tmp_path / "runs",
        min_alpha=1e-4,
        max_alpha=0.5,
    )

    generated = manifest["targets"]["0_2"]["configs"]
    by_family = {item["family"]: item for item in generated}

    assert by_family["real_d"]["generated_alpha"] == pytest.approx(0.08)
    assert by_family["random_d"]["generated_alpha"] == pytest.approx(0.1)

    real_config = json.loads(Path(by_family["real_d"]["config_path"]).read_text())
    assert real_config["run"]["ratio_match"]["target_geo_to_qk_ratio"] == 0.2
    assert real_config["attention"]["alpha"]["initial_value"] == pytest.approx(0.08)


def test_ratio_matched_report_detects_real_win(tmp_path: Path) -> None:
    baseline = {
        "condition": "baseline",
        "seed": 2027,
        "final_validation_loss": 5.8,
        "best_validation_loss": 5.8,
        "perplexity": 330.0,
    }
    baseline_path = tmp_path / "baseline.json"
    save_json(baseline_path, baseline)
    real_path = write_ratio_run(
        tmp_path,
        "real_d",
        target_ratio=0.2,
        observed_ratio=0.205,
        alpha=0.08,
        final_loss=5.6,
    )
    random_path = write_ratio_run(
        tmp_path,
        "random_d",
        target_ratio=0.2,
        observed_ratio=0.19,
        alpha=0.1,
        final_loss=5.7,
    )
    shuffled_path = write_ratio_run(
        tmp_path,
        "shuffled_d",
        target_ratio=0.2,
        observed_ratio=0.21,
        alpha=0.09,
        final_loss=5.75,
    )

    runs = [
        load_ratio_matched_run(
            parse_run_spec(f"real_d:0.2:{real_path.as_posix()}"),
            ratio_tolerance=0.03,
        ),
        load_ratio_matched_run(
            parse_run_spec(f"random_d:0.2:{random_path.as_posix()}"),
            ratio_tolerance=0.03,
        ),
        load_ratio_matched_run(
            parse_run_spec(f"shuffled_d:0.2:{shuffled_path.as_posix()}"),
            ratio_tolerance=0.03,
        ),
    ]
    report = build_ratio_matched_report(
        baseline=baseline,
        baseline_path=baseline_path,
        runs=runs,
        ratio_tolerance=0.03,
    )

    assert report["checks"]["all_ratios_within_tolerance"]
    assert report["checks"]["real_beats_all_controls_at_any_complete_target"]
    assert report["summary"]["recommendation"] == "ratio_matched_supports_real_d_repeat_seeds"


def test_ratio_matched_report_marks_missed_tolerance(tmp_path: Path) -> None:
    metrics_path = write_ratio_run(
        tmp_path,
        "real_d",
        target_ratio=0.2,
        observed_ratio=0.35,
        alpha=0.08,
        final_loss=5.6,
    )

    run = load_ratio_matched_run(
        parse_run_spec(f"real_d:0.2:{metrics_path.as_posix()}"),
        ratio_tolerance=0.03,
    )

    assert run["summary"]["ratio_within_tolerance"] is False
