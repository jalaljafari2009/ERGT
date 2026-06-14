"""Stage-20 short smoke and failure-safety validation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from evaluation.adaptive_notebook_ergt_03 import build_adaptive_notebook_ergt_03_report
from evaluation.unified_telemetry_schema import (
    build_unified_telemetry_schema_report,
    validate_telemetry_record,
)
from experiments.open_adaptive_relational_control_trainer import (
    OpenAdaptiveTrainerConfig,
    run_open_adaptive_control_trainer,
)

REQUIRED_STAGE20_OUTPUTS = [
    "short_smoke_run",
    "live_output_confirmed",
    "schema_validation",
    "controller_decision_log",
    "meta_control_observer_log",
    "auto_shutdown_path",
    "fail_fast_run",
]

PROGRESS_REVIEW_FIELDS = [
    "step",
    "condition",
    "validation_loss",
    "trainer_status",
    "trainer_fail_fast_triggered",
    "trainer_fail_fast_reason",
    "control_separation_mode",
    "claim_eligibility",
    "pending_control_mask",
    "offline_replay_required",
    "real_vs_baseline_delta",
    "real_vs_random_delta",
    "real_vs_shuffled_delta",
    "meta_control_mode",
    "meta_top_signal",
    "meta_suppressed_signal",
    "meta_observer_decision_summary",
    "alpha_decision",
    "memory_eta_decision",
    "gate_floor_decision",
    "causal_reachability_decision",
    "distance_norm_scale_decision",
    "joint_budget_decision",
]


def build_short_smoke_failure_safety_validation_report(
    *,
    smoke_rows: list[dict[str, Any]] | None = None,
    fail_fast_rows: list[dict[str, Any]] | None = None,
    config: OpenAdaptiveTrainerConfig | None = None,
) -> dict[str, Any]:
    """Run the stage-20 control-loop smoke and return a JSON-ready report."""

    active_config = config or OpenAdaptiveTrainerConfig(live_display_interval=100)
    active_config.validate()
    rows = smoke_rows or _short_smoke_rows()
    hard_stop_rows = fail_fast_rows or _fail_fast_rows()

    short_smoke = run_open_adaptive_control_trainer(rows, config=active_config)
    fail_fast = run_open_adaptive_control_trainer(
        hard_stop_rows,
        config=active_config,
    )
    schema = build_unified_telemetry_schema_report()
    notebook = build_adaptive_notebook_ergt_03_report()
    record_validations = [
        validate_telemetry_record(row, schema=schema)
        for row in short_smoke["progress_log"]
    ]

    smoke_steps = sorted({int(row["step"]) for row in short_smoke["progress_log"]})
    live_steps = [
        int(row["step"]) for row in short_smoke["live_diagnostic_rows"]
    ]
    real_rows = [
        row
        for row in short_smoke["progress_log"]
        if row.get("condition") == active_config.real_condition
    ]
    checks = {
        "short_smoke_uses_100_or_200_step_window": (
            bool(smoke_steps)
            and min(smoke_steps) >= 100
            and max(smoke_steps) <= 200
            and set(smoke_steps).issubset({100, 200})
        ),
        "short_smoke_completed": short_smoke["trainer_status"] == "completed",
        "live_output_confirmed": (
            bool(short_smoke["live_display_rows"])
            and bool(short_smoke["live_display_lines"])
            and bool(short_smoke["live_diagnostic_rows"])
            and bool(short_smoke["live_diagnostic_tables"])
            and bool(short_smoke["live_diagnostic_plot_payloads"])
            and all(step % active_config.live_display_interval == 0 for step in live_steps)
        ),
        "schema_validation_passes": (
            schema["status"] == "pass"
            and all(validation["status"] == "pass" for validation in record_validations)
        ),
        "controller_decision_log_exists": (
            bool(short_smoke["controller_decision_log"])
            and short_smoke["trainer_summary"]["controller_decision_log_ready"]
        ),
        "meta_control_observer_log_exists": (
            bool(short_smoke["meta_control_observer_log"])
            and short_smoke["trainer_summary"]["meta_control_observer_log_ready"]
        ),
        "auto_shutdown_path_exists": (
            notebook["status"] == "pass"
            and notebook["checks"]["auto_shutdown_cell_present"]
        ),
        "fail_fast_path_tested": (
            fail_fast["trainer_status"] == "failed_fast"
            and fail_fast["trainer_fail_fast_triggered"]
            and fail_fast["trainer_fail_fast_reason"] == "future_leak_score"
            and fail_fast["trainer_processed_rows"] < len(hard_stop_rows)
            and fail_fast["live_diagnostic_rows"][-1]["trainer_status"]
            == "failed_fast"
        ),
        "sequential_real_smoke_does_not_peek_at_future_controls": any(
            row.get("control_separation_mode") == "partial_live"
            and row.get("pending_control_mask") is True
            and row.get("real_vs_random_delta") is None
            for row in real_rows
        ),
        "lightweight_artifact_bundle_ready": (
            short_smoke["lightweight_artifact_manifest"]["lightweight_only"]
            and short_smoke["lightweight_artifact_manifest"][
                "checkpoint_artifacts_excluded"
            ]
        ),
        "no_2000_run_allowed_before_pass": max(smoke_steps or [0]) <= 200,
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "stage": "stage20_short_smoke_failure_safety_validation",
        "status": status,
        "scientific_scope": (
            "mechanical smoke gate before a guarded 2000-step adaptive run; "
            "confirms live output, schema compatibility, controller/meta logs, "
            "auto-shutdown availability, fail-fast safety, and sequential no-peek "
            "control behavior"
        ),
        "config": asdict(active_config),
        "required_outputs": list(REQUIRED_STAGE20_OUTPUTS),
        "checks": checks,
        "smoke_steps": smoke_steps,
        "live_display_steps": live_steps,
        "record_validation_summaries": record_validations,
        "short_smoke_summary": _compact_run_summary(short_smoke),
        "fail_fast_summary": _compact_run_summary(fail_fast),
        "schema_status": schema["status"],
        "notebook_status": notebook["status"],
        "short_smoke_run": _review_run_payload(short_smoke),
        "fail_fast_run": _review_run_payload(fail_fast),
        "next_required_step": (
            "guarded_2000_step_adaptive_run"
            if status == "pass"
            else "fix_short_smoke_failure_safety_validation"
        ),
    }


def _short_smoke_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    schedule = [
        ("baseline", [6.05, 5.90], 0.00),
        ("alpha_zero", [6.05, 5.90], 0.00),
        ("real_memory_d", [6.00, 5.78], 0.12),
        ("random_memory_d", [6.03, 5.84], 0.06),
        ("shuffled_memory_d", [6.04, 5.85], 0.05),
    ]
    for condition, losses, improvement in schedule:
        for step, loss in zip([100, 200], losses, strict=True):
            rows.append(_row(condition, step, loss, improvement))
    return rows


def _fail_fast_rows() -> list[dict[str, Any]]:
    return [
        _row("baseline", 100, 6.05, 0.0),
        _row("real_memory_d", 100, 6.00, 0.12),
        {
            **_row("real_memory_d", 150, 5.92, 0.12),
            "future_leak_score": 0.2,
            "hard_stop_reason": "future_leak_score",
        },
        _row("real_memory_d", 200, 5.78, 0.12),
    ]


def _row(
    condition: str,
    step: int,
    validation_loss: float,
    improvement: float,
) -> dict[str, Any]:
    alpha = 0.0 if condition in {"baseline", "alpha_zero"} else 0.02
    return {
        "step": step,
        "condition": condition,
        "train_loss": validation_loss - 0.04,
        "validation_loss": validation_loss,
        "alpha": alpha,
        "alpha_effective": alpha,
        "alpha_next": alpha + (0.005 if condition == "real_memory_d" else 0.0),
        "alpha_delta": 0.005 if condition == "real_memory_d" else 0.0,
        "alpha_decision": "grow_alpha" if condition == "real_memory_d" else "hold",
        "loss_slope_gain": improvement / 3,
        "baseline_centered_improvement": improvement,
        "geo_to_qk_ratio": 0.03 + alpha if alpha else 0.0,
        "attention_entropy": 3.10,
        "mean_max_probability": 0.19,
        "rigidity_risk": 0.04,
        "control_penalty": 0.0,
        "control_rng_isolated": True,
        "trainer_status": "pending",
        "trainer_fail_fast_triggered": False,
        "real_vs_baseline_delta": improvement if condition == "real_memory_d" else None,
        "memory_stability": 0.50 + improvement,
        "memory_turnover": max(0.06, 0.24 - improvement),
        "memory_persistence": 0.45 + improvement,
        "memory_rigidity": 0.08,
        "noise_risk": max(0.04, 0.16 - improvement),
        "attention_behavior_regime": "useful_noncollapsed",
        "attention_behavior_separation": improvement,
        "distance_contrast_retention": 0.60 + improvement,
        "future_leak_score": 0.0,
        "meta_top_signal": "memory_state" if condition == "real_memory_d" else "loss_trend",
        "meta_attention_entropy": 1.85,
        "meta_alpha_weight": 0.20,
        "meta_memory_weight": 0.25,
        "meta_gate_weight": 0.12,
        "meta_reach_weight": 0.17,
        "meta_norm_weight": 0.18,
        "controller_conflict_score": 0.10,
        "meta_control_confidence": 0.74,
        "memory_eta_decision": (
            "increase_eta" if condition == "real_memory_d" else "hold"
        ),
        "memory_decay_decision": "hold",
        "gate_floor_decision": "hold",
        "causal_reachability_decision": "hold",
        "distance_norm_scale_decision": "hold",
        "joint_budget_decision": "balanced",
    }


def _compact_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    summary_keys = [
        "trainer_status",
        "trainer_fail_fast_triggered",
        "trainer_fail_fast_reason",
        "trainer_processed_rows",
        "trainer_processed_steps",
        "live_diagnostic_row_count",
        "live_diagnostic_table_ready",
        "live_diagnostic_plot_ready",
    ]
    return {key: run[key] for key in summary_keys}


def _review_run_payload(run: dict[str, Any]) -> dict[str, Any]:
    latest_plot = (
        run["live_diagnostic_plot_payloads"][-1]
        if run["live_diagnostic_plot_payloads"]
        else {}
    )
    latest_table = (
        run["live_diagnostic_tables"][-1] if run["live_diagnostic_tables"] else ""
    )
    return {
        "trainer_status": run["trainer_status"],
        "trainer_fail_fast_triggered": run["trainer_fail_fast_triggered"],
        "trainer_fail_fast_reason": run["trainer_fail_fast_reason"],
        "trainer_processed_rows": run["trainer_processed_rows"],
        "trainer_processed_steps": run["trainer_processed_steps"],
        "trainer_summary": run["trainer_summary"],
        "progress_log": [_pick(row, PROGRESS_REVIEW_FIELDS) for row in run["progress_log"]],
        "live_display_lines": run["live_display_lines"],
        "live_diagnostic_rows": run["live_diagnostic_rows"],
        "latest_live_diagnostic_table": latest_table,
        "latest_plot_payload_summary": {
            "x_axis": latest_plot.get("x_axis"),
            "latest_step": latest_plot.get("latest_step"),
            "series_groups": sorted(latest_plot.get("series_groups", {}).keys()),
        },
        "controller_decision_log_count": len(run["controller_decision_log"]),
        "meta_control_observer_log_count": len(run["meta_control_observer_log"]),
        "control_separation_log": [
            _pick(row, ["step", "condition", "mode", "status", "stream"])
            for row in run["control_separation_log"]
        ],
        "safety_log": [
            _pick(
                row,
                [
                    "step",
                    "condition",
                    "hard_stop_triggered",
                    "hard_stop_reason",
                    "future_leak_score",
                    "validation_loss",
                ],
            )
            for row in run["safety_log"]
        ],
        "lightweight_artifact_manifest": run["lightweight_artifact_manifest"],
        "trainer_replay_record": run["trainer_replay_record"],
    }


def _pick(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {field: row.get(field) for field in fields if field in row}
