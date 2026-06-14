import json

from evaluation.guarded_2000_step_adaptive_run import (
    REQUIRED_STAGE21_CONDITIONS,
    REQUIRED_STAGE21_OUTPUTS,
    build_guarded_2000_step_adaptive_run_report,
)
from experiments.guarded_2000_step_adaptive_run import (
    Guarded2000RunConfig,
    generate_guarded_2000_telemetry_rows,
    summarize_guarded_replay,
)


def test_guarded_2000_step_adaptive_run_report_passes() -> None:
    report = build_guarded_2000_step_adaptive_run_report()

    assert report["status"] == "pass"
    assert report["next_required_step"] == "late_window_and_post_1000_analysis"
    assert set(REQUIRED_STAGE21_OUTPUTS).issubset(report["required_outputs"])
    json.dumps(report)


def test_guarded_2000_run_has_all_required_conditions_and_steps() -> None:
    config = Guarded2000RunConfig()
    rows = generate_guarded_2000_telemetry_rows(config)
    summary = summarize_guarded_replay(rows, config=config)

    assert summary["conditions"] == REQUIRED_STAGE21_CONDITIONS
    assert summary["all_conditions_have_identical_steps"]
    assert summary["all_conditions_reach_2000"]
    assert summary["expected_steps"][0] == 100
    assert summary["expected_steps"][-1] == 2000
    assert len(summary["expected_steps"]) == 20


def test_guarded_2000_replay_exposes_live_rows_for_every_condition() -> None:
    report = build_guarded_2000_step_adaptive_run_report()

    for condition in REQUIRED_STAGE21_CONDITIONS:
        assert report["condition_live_counts"][condition] == 20
        assert report["condition_row_counts"][condition] == 20
    assert report["checks"]["live_output_emitted_for_every_condition"]
    assert report["checks"]["progress_log_emitted_for_every_condition"]


def test_guarded_2000_replay_is_ready_for_late_window_analysis() -> None:
    report = build_guarded_2000_step_adaptive_run_report()

    assert report["checks"]["final_matched_late_window_exists"]
    assert report["checks"]["late_window_analysis_ready"]
    assert report["final_matched_summary"]["late_final_matched_row_count"] > 0
    assert report["final_matched_summary"]["latest_late_matched_window"]["points"] >= 2
    assert report["late_window_analysis"]["stage22_judgment_required"]


def test_guarded_2000_replay_keeps_stage20_as_prerequisite() -> None:
    report = build_guarded_2000_step_adaptive_run_report()

    assert report["checks"]["stage20_smoke_gate_passed"]
    assert report["smoke_gate_summary"]["status"] == "pass"
