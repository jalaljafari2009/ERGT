"""Stage-21 guarded 2000-step adaptive run contract."""

from __future__ import annotations

from typing import Any

from evaluation.short_smoke_failure_safety_validation import (
    build_short_smoke_failure_safety_validation_report,
)
from evaluation.unified_telemetry_schema import (
    build_unified_telemetry_schema_report,
    validate_telemetry_record,
)
from experiments.guarded_2000_step_adaptive_run import (
    Guarded2000RunConfig,
    build_guarded_run_plan,
    generate_guarded_2000_telemetry_rows,
    guarded_config_asdict,
    summarize_guarded_replay,
)
from experiments.open_adaptive_relational_control_trainer import (
    OpenAdaptiveTrainerConfig,
    run_open_adaptive_control_trainer,
)

REQUIRED_STAGE21_CONDITIONS = [
    "baseline",
    "alpha_zero",
    "real_memory_d",
    "random_memory_d",
    "shuffled_memory_d",
    "no_memory_real_d",
    "instantaneous_real_d",
]

REQUIRED_STAGE21_OUTPUTS = [
    "guarded_run_plan",
    "baseline",
    "alpha_zero",
    "real_adaptive_memory_geometry",
    "random_adaptive_memory_geometry",
    "shuffled_adaptive_memory_geometry",
    "no_memory_real_geometry",
    "instantaneous_real_geometry",
    "comparable_telemetry",
    "late_window_analysis_ready",
]

REVIEW_PROGRESS_FIELDS = [
    "step",
    "condition",
    "validation_loss",
    "trainer_status",
    "trainer_fail_fast_triggered",
    "control_separation_mode",
    "claim_eligibility",
    "control_separation_status",
    "matched_control_window",
    "real_vs_baseline_delta",
    "real_vs_alpha_zero_delta",
    "real_vs_random_delta",
    "real_vs_shuffled_delta",
    "real_vs_no_memory_delta",
    "real_vs_instantaneous_delta",
    "control_separation",
    "scientific_claim_credit",
    "generic_regularization_warning",
    "meta_control_mode",
    "meta_top_signal",
    "alpha_decision",
    "memory_eta_decision",
    "geo_to_qk_ratio",
    "memory_stability",
    "memory_persistence",
    "attention_entropy",
]


def build_guarded_2000_step_adaptive_run_report(
    *,
    config: Guarded2000RunConfig | None = None,
    telemetry_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a compact contract report for the guarded 2000-step run."""

    active = config or Guarded2000RunConfig()
    active.validate()
    rows = telemetry_rows or generate_guarded_2000_telemetry_rows(active)
    plan = build_guarded_run_plan(active)
    replay_summary = summarize_guarded_replay(rows, config=active)
    schema = build_unified_telemetry_schema_report()
    smoke_gate = build_short_smoke_failure_safety_validation_report()
    trainer = run_open_adaptive_control_trainer(
        rows,
        config=OpenAdaptiveTrainerConfig(
            live_display_interval=active.live_display_interval,
            artifact_bundle_name=active.artifact_bundle_name,
        ),
    )
    record_validations = [
        validate_telemetry_record(row, schema=schema)
        for row in trainer["progress_log"]
    ]
    final_matched_rows = [
        row
        for row in trainer["progress_log"]
        if row.get("control_separation_mode") == "final_matched"
    ]
    late_final_rows = [
        row for row in final_matched_rows if int(row["step"]) >= active.late_window_start
    ]
    condition_live_counts = _condition_counts(trainer["live_diagnostic_rows"])
    condition_progress_counts = _condition_counts(trainer["progress_log"])
    latest_final = late_final_rows[-1] if late_final_rows else {}
    checks = {
        "stage20_smoke_gate_passed": smoke_gate["status"] == "pass",
        "required_conditions_present": set(REQUIRED_STAGE21_CONDITIONS).issubset(
            replay_summary["conditions"]
        ),
        "all_conditions_have_identical_steps": replay_summary[
            "all_conditions_have_identical_steps"
        ],
        "all_conditions_reach_2000": replay_summary["all_conditions_reach_2000"],
        "live_output_emitted_for_every_condition": all(
            condition_live_counts.get(condition, 0)
            == len(replay_summary["expected_steps"])
            for condition in REQUIRED_STAGE21_CONDITIONS
        ),
        "progress_log_emitted_for_every_condition": all(
            condition_progress_counts.get(condition, 0)
            == len(replay_summary["expected_steps"])
            for condition in REQUIRED_STAGE21_CONDITIONS
        ),
        "schema_validation_passes": (
            schema["status"] == "pass"
            and all(validation["status"] == "pass" for validation in record_validations)
        ),
        "trainer_completed_without_fail_fast": (
            trainer["trainer_status"] == "completed"
            and not trainer["trainer_fail_fast_triggered"]
        ),
        "controller_and_meta_logs_exist": bool(trainer["controller_decision_log"])
        and bool(trainer["meta_control_observer_log"]),
        "final_matched_late_window_exists": bool(late_final_rows),
        "late_window_has_required_condition_points": _late_window_ready(
            replay_summary,
            active.late_window_start,
        ),
        "late_window_analysis_ready": (
            bool(late_final_rows)
            and latest_final.get("claim_eligibility") == "eligible_complete_controls"
            and latest_final.get("matched_control_window", {}).get("points", 0) >= 2
        ),
        "lightweight_artifact_bundle_ready": (
            trainer["lightweight_artifact_manifest"]["lightweight_only"]
            and trainer["lightweight_artifact_manifest"]["checkpoint_artifacts_excluded"]
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "stage": "stage21_guarded_2000_step_adaptive_run",
        "status": status,
        "scientific_scope": (
            "guarded 2000-step adaptive run contract and replay; verifies that "
            "all required conditions expose comparable 100-step telemetry and "
            "that final matched late-window scoring is available for stage 22. "
            "Synthetic replay is not scientific claim evidence."
        ),
        "config": guarded_config_asdict(active),
        "required_outputs": list(REQUIRED_STAGE21_OUTPUTS),
        "required_conditions": list(REQUIRED_STAGE21_CONDITIONS),
        "checks": checks,
        "guarded_run_plan": plan,
        "condition_row_counts": replay_summary["condition_row_counts"],
        "condition_live_counts": condition_live_counts,
        "smoke_gate_summary": {
            "status": smoke_gate["status"],
            "next_required_step": smoke_gate["next_required_step"],
            "checks": smoke_gate["checks"],
        },
        "trainer_summary": trainer["trainer_summary"],
        "final_matched_summary": {
            "final_matched_row_count": len(final_matched_rows),
            "late_final_matched_row_count": len(late_final_rows),
            "first_late_final_step": late_final_rows[0]["step"] if late_final_rows else None,
            "latest_late_final_step": latest_final.get("step"),
            "latest_late_matched_window": latest_final.get("matched_control_window"),
            "latest_control_separation": latest_final.get("control_separation"),
            "latest_control_separation_status": latest_final.get(
                "control_separation_status"
            ),
        },
        "late_window_analysis": _late_window_analysis(replay_summary),
        "review_progress_tail": [
            _pick(row, REVIEW_PROGRESS_FIELDS) for row in trainer["progress_log"][-8:]
        ],
        "latest_live_diagnostic_rows": trainer["live_diagnostic_rows"][-8:],
        "record_validation_failures": [
            validation
            for validation in record_validations
            if validation["status"] != "pass"
        ],
        "next_required_step": (
            "late_window_and_post_1000_analysis"
            if status == "pass"
            else "fix_guarded_2000_step_adaptive_run"
        ),
    }


def _condition_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        condition = str(row.get("condition"))
        counts[condition] = counts.get(condition, 0) + 1
    return counts


def _late_window_ready(
    replay_summary: dict[str, Any],
    late_window_start: int,
) -> bool:
    late_summary = replay_summary["window_summaries"]["1000_2000"]
    return all(
        condition_summary["points"] >= 2
        and condition_summary["start_step"] >= late_window_start
        for condition_summary in late_summary.values()
    )


def _late_window_analysis(replay_summary: dict[str, Any]) -> dict[str, Any]:
    windows = replay_summary["window_summaries"]
    return {
        "windows": {
            name: {
                condition: {
                    "points": summary["points"],
                    "mean_validation_loss": summary["mean_validation_loss"],
                    "start_step": summary["start_step"],
                    "end_step": summary["end_step"],
                }
                for condition, summary in condition_map.items()
            }
            for name, condition_map in windows.items()
        },
        "stage22_judgment_required": True,
        "decision_note": (
            "Stage 21 only verifies comparable telemetry and late-window readiness; "
            "stage 22 must judge post-1000 trends and control separation."
        ),
    }


def _pick(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {field: row.get(field) for field in fields if field in row}
