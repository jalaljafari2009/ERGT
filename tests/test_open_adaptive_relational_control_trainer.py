import json

from evaluation.open_adaptive_relational_control_trainer import (
    REQUIRED_TRAINER_OUTPUTS,
    build_open_adaptive_relational_control_trainer_report,
)
from experiments.open_adaptive_relational_control_trainer import (
    OpenAdaptiveRelationalControlTrainer,
    OpenAdaptiveTrainerConfig,
    run_open_adaptive_control_trainer,
)
from experiments.progress_logging import format_progress_line


def test_open_adaptive_trainer_report_passes() -> None:
    report = build_open_adaptive_relational_control_trainer_report()

    assert report["status"] == "pass"
    assert report["next_required_step"] == "live_100_step_diagnostic_table"
    assert set(REQUIRED_TRAINER_OUTPUTS).issubset(report["completed_run"])
    assert report["checks"]["real_run_uses_partial_missing_aware_controls"]
    assert report["checks"]["final_matched_replay_available_after_controls"]
    json.dumps(report)


def test_trainer_does_not_peek_at_future_controls_during_real() -> None:
    rows = [
        {"condition": "baseline", "step": 1000, "validation_loss": 5.5},
        {"condition": "real_memory_d", "step": 1000, "validation_loss": 5.3},
        {"condition": "random_memory_d", "step": 1500, "validation_loss": 5.1},
    ]

    result = run_open_adaptive_control_trainer(rows)
    real_row = [
        row
        for row in result["progress_log"]
        if row["condition"] == "real_memory_d" and row["step"] == 1000
    ][0]

    assert real_row["control_separation_mode"] == "partial_live"
    assert real_row["real_vs_baseline_delta"] > 0
    assert real_row["real_vs_random_delta"] is None
    assert real_row["pending_control_mask"] is True


def test_trainer_fail_fast_stops_after_future_leak() -> None:
    trainer = OpenAdaptiveRelationalControlTrainer()
    result = trainer.summary(
        trainer.run(
            [
                {"condition": "baseline", "step": 500, "validation_loss": 5.9},
                {"condition": "real_memory_d", "step": 500, "validation_loss": 5.8},
                {
                    "condition": "real_memory_d",
                    "step": 600,
                    "validation_loss": 5.7,
                    "future_leak_score": 0.2,
                },
                {"condition": "real_memory_d", "step": 700, "validation_loss": 5.6},
            ]
        )
    )

    assert result["trainer_status"] == "failed_fast"
    assert result["trainer_fail_fast_triggered"]
    assert result["trainer_processed_rows"] == 3
    assert result["trainer_fail_fast_reason"] == "future_leak_score"


def test_trainer_produces_controller_and_meta_logs() -> None:
    result = run_open_adaptive_control_trainer(
        [
            {
                "condition": "baseline",
                "step": 100,
                "validation_loss": 6.0,
                "alpha_decision": "observe",
            },
            {
                "condition": "real_memory_d",
                "step": 100,
                "validation_loss": 5.9,
                "alpha_decision": "grow_alpha",
                "loss_slope_gain": 0.08,
                "real_vs_baseline_delta": 0.1,
            },
        ],
        config=OpenAdaptiveTrainerConfig(live_display_interval=100),
    )

    assert result["controller_decision_log"]
    assert result["meta_control_observer_log"]
    assert result["control_separation_log"]
    assert result["live_display_lines"]


def test_trainer_artifact_manifest_excludes_checkpoints() -> None:
    result = run_open_adaptive_control_trainer(
        [{"condition": "baseline", "step": 100, "validation_loss": 6.0}]
    )
    manifest = result["lightweight_artifact_manifest"]

    assert manifest["lightweight_only"]
    assert manifest["checkpoint_artifacts_excluded"]
    assert "checkpoints/" in manifest["excluded_artifact_patterns"]


def test_progress_line_includes_trainer_fields() -> None:
    line = format_progress_line(
        {
            "condition": "real_memory_d",
            "step": 100,
            "validation_loss": 5.9,
            "trainer_status": "running",
            "trainer_fail_fast_triggered": False,
            "controller_decision_count": 2,
            "meta_observer_event_count": 1,
            "control_separation_event_count": 1,
            "live_display_event_count": 1,
        }
    )

    assert "trainer=running" in line
    assert "tFail=0" in line
    assert "ctrlDec=2" in line
