import json

import pytest

from evaluation.loss_slope_trend_analyzer import (
    TrendAnalyzerConfig,
    build_loss_slope_trend_analyzer_report,
)


def test_loss_slope_trend_analyzer_report_passes_on_synthetic_progress() -> None:
    report = build_loss_slope_trend_analyzer_report()

    assert report["status"] == "pass"
    checks = report["checks"]
    assert checks["baseline_reference_available"]
    assert checks["required_trend_fields_emitted"]
    assert checks["rolling_slope_uses_windowed_points"]
    assert checks["ema_loss_delta_emitted"]
    assert checks["late_window_slope_emitted"]
    assert checks["post_1000_trend_emitted"]
    assert report["summary"]["post_1000"]["trend"] == "improving"
    assert report["next_required_step"] == "parameter_attribution_probe"
    json.dumps(report)


def test_loss_slope_trend_analyzer_computes_baseline_delta_and_slope_gain() -> None:
    baseline = [
        {"step": 100, "validation_loss": 5.0},
        {"step": 200, "validation_loss": 4.9},
        {"step": 300, "validation_loss": 4.8},
    ]
    condition = [
        {"step": 100, "validation_loss": 5.0},
        {"step": 200, "validation_loss": 4.7},
        {"step": 300, "validation_loss": 4.4},
    ]
    report = build_loss_slope_trend_analyzer_report(
        baseline_progress=baseline,
        condition_progress=condition,
        config=TrendAnalyzerConfig(
            min_points_for_slope=3,
            rolling_window_points=3,
            late_window_start=100,
            post_1000_start=100,
        ),
    )

    latest = report["summary"]["latest"]
    assert latest["baseline_relative_validation_delta"] == pytest.approx(0.4)
    assert latest["rolling_slope"] < latest["baseline_rolling_slope"]
    assert latest["loss_slope_gain"] > 0
    assert latest["ema_loss_delta"] > 0


def test_loss_slope_trend_analyzer_requires_windowed_evidence() -> None:
    baseline = [
        {"step": 100, "validation_loss": 5.0},
        {"step": 200, "validation_loss": 4.9},
    ]
    condition = [
        {"step": 100, "validation_loss": 4.95},
        {"step": 200, "validation_loss": 4.8},
    ]
    report = build_loss_slope_trend_analyzer_report(
        baseline_progress=baseline,
        condition_progress=condition,
        config=TrendAnalyzerConfig(
            min_points_for_slope=3,
            rolling_window_points=3,
            late_window_start=100,
            post_1000_start=100,
        ),
    )

    assert report["status"] == "fail"
    assert not report["checks"]["rolling_slope_uses_windowed_points"]
    assert report["summary"]["post_1000"]["trend"] == "insufficient_points"


def test_loss_slope_trend_analyzer_uses_previous_baseline_step() -> None:
    baseline = [
        {"step": 100, "validation_loss": 5.0},
        {"step": 300, "validation_loss": 4.8},
        {"step": 500, "validation_loss": 4.7},
    ]
    condition = [
        {"step": 100, "validation_loss": 4.9},
        {"step": 200, "validation_loss": 4.7},
        {"step": 400, "validation_loss": 4.4},
    ]
    report = build_loss_slope_trend_analyzer_report(
        baseline_progress=baseline,
        condition_progress=condition,
        config=TrendAnalyzerConfig(
            min_points_for_slope=3,
            rolling_window_points=3,
            late_window_start=100,
            post_1000_start=100,
        ),
    )

    rows = report["annotated_progress"]
    assert rows[1]["baseline_validation_loss"] == 5.0
    assert rows[2]["baseline_validation_loss"] == 4.8
