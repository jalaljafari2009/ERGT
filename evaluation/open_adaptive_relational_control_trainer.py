"""Stage-17 Open Adaptive Relational Control Trainer report."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from experiments.open_adaptive_relational_control_trainer import (
    OpenAdaptiveRelationalControlTrainer,
    OpenAdaptiveTrainerConfig,
)

REQUIRED_TRAINER_OUTPUTS = [
    "trainer_status",
    "trainer_fail_fast_triggered",
    "trainer_fail_fast_reason",
    "trainer_processed_rows",
    "trainer_processed_steps",
    "progress_log",
    "live_display_rows",
    "live_display_lines",
    "controller_decision_log",
    "meta_control_observer_log",
    "control_separation_log",
    "safety_log",
    "lightweight_artifact_manifest",
    "trainer_summary",
    "trainer_replay_record",
]


def build_open_adaptive_relational_control_trainer_report(
    *,
    telemetry_rows: list[dict[str, Any]] | None = None,
    hard_stop_rows: list[dict[str, Any]] | None = None,
    config: OpenAdaptiveTrainerConfig | None = None,
) -> dict[str, Any]:
    """Build the stage-17 trainer contract report."""

    from evaluation.unified_telemetry_schema import build_unified_telemetry_schema_report

    config = config or OpenAdaptiveTrainerConfig()
    config.validate()
    trainer = OpenAdaptiveRelationalControlTrainer(config)
    completed = trainer.summary(trainer.run(telemetry_rows or _synthetic_sequential_rows()))
    hard_stop = trainer.summary(trainer.run(hard_stop_rows or _synthetic_hard_stop_rows()))
    schema_fields = set(build_unified_telemetry_schema_report()["fields"])
    required_schema_fields = {
        "trainer_status",
        "trainer_fail_fast_triggered",
        "trainer_fail_fast_reason",
        "trainer_processed_rows",
        "trainer_processed_steps",
        "controller_decision_count",
        "meta_observer_event_count",
        "control_separation_event_count",
        "live_display_event_count",
        "progress_log_ready",
        "controller_decision_log_ready",
        "meta_control_observer_log_ready",
        "control_separation_log_ready",
        "safety_log_ready",
        "lightweight_artifact_bundle_ready",
        "checkpoint_artifacts_excluded",
        "artifact_bundle_name",
    }
    partial_real_rows = [
        row
        for row in completed["progress_log"]
        if row.get("condition") == config.real_condition
    ]
    final_rows = [
        row
        for row in completed["progress_log"]
        if row.get("control_separation_mode") == "final_matched"
    ]
    checks = {
        "required_trainer_outputs_emitted": set(REQUIRED_TRAINER_OUTPUTS).issubset(
            completed
        ),
        "schema_declares_trainer_fields": required_schema_fields.issubset(schema_fields),
        "unbuffered_live_logging_contract_present": bool(completed["live_display_lines"]),
        "per_100_step_telemetry_present": all(
            int(row["step"]) % config.live_display_interval == 0 or int(row["step"]) == 1
            for row in completed["live_display_rows"]
        ),
        "controller_decision_log_exists": bool(completed["controller_decision_log"]),
        "meta_control_observer_log_exists": bool(completed["meta_control_observer_log"]),
        "control_separation_log_exists": bool(completed["control_separation_log"]),
        "real_run_uses_partial_missing_aware_controls": any(
            row.get("condition") == config.real_condition
            and row.get("control_separation_mode") == "partial_live"
            and row.get("pending_control_mask") is True
            and row.get("offline_replay_required") is True
            for row in partial_real_rows
        ),
        "final_matched_replay_available_after_controls": any(
            row.get("meta_control_mode") == "offline_matched_replay"
            and row.get("claim_eligibility") == "eligible_complete_controls"
            for row in final_rows
        ),
        "trainer_does_not_peek_at_future_controls": any(
            row.get("condition") == config.real_condition
            and row.get("step") == 1000
            and row.get("real_vs_random_delta") is None
            for row in partial_real_rows
        ),
        "fail_fast_path_stops_and_records_reason": (
            hard_stop["trainer_status"] == "failed_fast"
            and hard_stop["trainer_fail_fast_triggered"]
            and hard_stop["trainer_processed_rows"]
            < len(hard_stop_rows or _synthetic_hard_stop_rows())
            and bool(hard_stop["safety_log"][-1]["hard_stop_reason"])
        ),
        "lightweight_artifact_manifest_excludes_checkpoints": (
            completed["lightweight_artifact_manifest"]["lightweight_only"]
            and completed["lightweight_artifact_manifest"]["checkpoint_artifacts_excluded"]
        ),
        "trainer_replay_record_present": bool(completed["trainer_replay_record"]),
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "stage": "stage17_open_adaptive_relational_control_trainer",
        "status": status,
        "scientific_scope": (
            "trainer-loop orchestration contract for live telemetry, controller "
            "decisions, missing-aware meta-control observation, fail-fast safety, "
            "and lightweight artifacts"
        ),
        "config": asdict(config),
        "required_outputs": list(REQUIRED_TRAINER_OUTPUTS),
        "checks": checks,
        "completed_run": completed,
        "hard_stop_run": hard_stop,
        "next_required_step": (
            "live_100_step_diagnostic_table"
            if status == "pass"
            else "fix_open_adaptive_relational_control_trainer"
        ),
    }


def _synthetic_sequential_rows() -> list[dict[str, Any]]:
    rows = []
    for condition, losses in [
        ("baseline", [5.90, 5.50, 5.32]),
        ("alpha_zero", [5.89, 5.49, 5.31]),
        ("real_memory_d", [5.86, 5.33, 5.08]),
        ("random_memory_d", [5.87, 5.43, 5.23]),
        ("shuffled_memory_d", [5.88, 5.45, 5.24]),
        ("no_memory_real_d", [5.88, 5.44, 5.25]),
        ("instantaneous_real_d", [5.87, 5.41, 5.22]),
    ]:
        for step, loss in zip([500, 1000, 1500], losses, strict=True):
            rows.append(_row(condition, step, loss))
    return rows


def _synthetic_hard_stop_rows() -> list[dict[str, Any]]:
    return [
        _row("baseline", 500, 5.90),
        _row("real_memory_d", 500, 5.86),
        {
            **_row("real_memory_d", 600, 5.80),
            "future_leak_score": 0.10,
            "hard_stop_reason": "future_leak_score",
        },
        _row("real_memory_d", 700, 5.70),
    ]


def _row(condition: str, step: int, validation_loss: float) -> dict[str, Any]:
    return {
        "condition": condition,
        "step": step,
        "validation_loss": validation_loss,
        "loss_slope_gain": 0.08,
        "real_vs_baseline_delta": 0.05 if condition == "real_memory_d" else None,
        "geo_to_qk_ratio": 0.10,
        "memory_stability": 0.50,
        "memory_persistence": 0.44,
        "noise_risk": 0.12,
        "distance_contrast_retention": 0.50,
        "reach_starvation_score": 0.20,
        "rigidity_risk": 0.08,
        "collapse_risk": 0.04,
        "alpha_decision": "observe",
        "joint_budget_decision": "allocate_budget",
        "tokens_per_second": 1000.0,
    }
