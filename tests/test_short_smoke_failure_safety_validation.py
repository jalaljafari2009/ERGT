import json

from evaluation.short_smoke_failure_safety_validation import (
    REQUIRED_STAGE20_OUTPUTS,
    build_short_smoke_failure_safety_validation_report,
)


def test_short_smoke_failure_safety_report_passes() -> None:
    report = build_short_smoke_failure_safety_validation_report()

    assert report["status"] == "pass"
    assert report["next_required_step"] == "guarded_2000_step_adaptive_run"
    assert set(REQUIRED_STAGE20_OUTPUTS).issubset(report["required_outputs"])
    json.dumps(report)


def test_short_smoke_confirms_live_output_and_schema() -> None:
    report = build_short_smoke_failure_safety_validation_report()

    assert report["checks"]["short_smoke_uses_100_or_200_step_window"]
    assert report["checks"]["live_output_confirmed"]
    assert report["checks"]["schema_validation_passes"]
    assert report["short_smoke_summary"]["live_diagnostic_row_count"] > 0


def test_short_smoke_keeps_real_run_missing_aware_until_controls_exist() -> None:
    report = build_short_smoke_failure_safety_validation_report()

    assert report["checks"][
        "sequential_real_smoke_does_not_peek_at_future_controls"
    ]
    real_rows = [
        row
        for row in report["short_smoke_run"]["progress_log"]
        if row["condition"] == "real_memory_d"
    ]
    assert any(row["pending_control_mask"] is True for row in real_rows)
    assert any(row["real_vs_random_delta"] is None for row in real_rows)


def test_short_smoke_tests_fail_fast_and_auto_shutdown_path() -> None:
    report = build_short_smoke_failure_safety_validation_report()

    assert report["checks"]["fail_fast_path_tested"]
    assert report["checks"]["auto_shutdown_path_exists"]
    assert report["fail_fast_summary"]["trainer_status"] == "failed_fast"
    assert report["fail_fast_summary"]["trainer_fail_fast_reason"] == "future_leak_score"
