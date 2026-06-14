"""Stage-18 Live 100-Step Diagnostic Table report."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from experiments.live_100_step_diagnostic_table import (
    PLOT_SERIES_GROUPS,
    REQUIRED_LIVE_DIAGNOSTIC_COLUMNS,
    LiveDiagnosticTableBuilder,
    LiveDiagnosticTableConfig,
)
from experiments.open_adaptive_relational_control_trainer import (
    OpenAdaptiveRelationalControlTrainer,
    OpenAdaptiveTrainerConfig,
)

REQUIRED_LIVE_DIAGNOSTIC_OUTPUTS = [
    "live_diagnostic_rows",
    "live_diagnostic_tables",
    "live_diagnostic_plot_payloads",
    "live_diagnostic_row_count",
    "live_diagnostic_table_ready",
    "live_diagnostic_plot_ready",
]


def build_live_100_step_diagnostic_table_report(
    *,
    telemetry_rows: list[dict[str, Any]] | None = None,
    fail_fast_rows: list[dict[str, Any]] | None = None,
    config: LiveDiagnosticTableConfig | None = None,
) -> dict[str, Any]:
    """Build the stage-18 live diagnostic table contract report."""

    from evaluation.unified_telemetry_schema import build_unified_telemetry_schema_report

    config = config or LiveDiagnosticTableConfig(display_interval=100)
    config.validate()
    builder = LiveDiagnosticTableBuilder(config)
    trainer = OpenAdaptiveRelationalControlTrainer(
        OpenAdaptiveTrainerConfig(live_display_interval=config.display_interval)
    )
    completed = trainer.summary(trainer.run(telemetry_rows or _synthetic_live_rows()))
    fail_fast = trainer.summary(trainer.run(fail_fast_rows or _synthetic_fail_fast_rows()))
    sample_row = builder.build_row(_synthetic_full_record())
    sample_snapshot = builder.build_snapshot([sample_row])
    schema_fields = set(build_unified_telemetry_schema_report()["fields"])
    required_schema_fields = {
        "live_diagnostic_row_ready",
        "live_diagnostic_table_ready",
        "live_diagnostic_plot_ready",
        "live_diagnostic_row_count",
        "live_diagnostic_columns",
        "live_diagnostic_table_markdown",
        "live_diagnostic_plot_payload",
    }
    completed_summary = completed["trainer_summary"]
    checks = {
        "required_live_diagnostic_outputs_emitted": set(
            REQUIRED_LIVE_DIAGNOSTIC_OUTPUTS
        ).issubset(completed)
        and {
            "live_diagnostic_row_count",
            "live_diagnostic_table_ready",
            "live_diagnostic_plot_ready",
        }.issubset(completed_summary),
        "schema_declares_live_diagnostic_fields": required_schema_fields.issubset(
            schema_fields
        ),
        "all_required_columns_present": set(REQUIRED_LIVE_DIAGNOSTIC_COLUMNS)
        .issubset(sample_row),
        "table_markdown_has_header_and_rows": (
            "| step | condition |" in sample_snapshot["live_diagnostic_table_markdown"]
            and "real_memory_d" in sample_snapshot["live_diagnostic_table_markdown"]
        ),
        "missing_values_are_explicit_in_table": config.missing_value
        in builder.format_markdown([{"step": 100, "condition": "real_memory_d"}]),
        "plot_payload_has_required_series_groups": set(PLOT_SERIES_GROUPS).issubset(
            sample_snapshot["live_diagnostic_plot_payload"]["series_groups"]
        ),
        "plot_payload_uses_step_axis": (
            sample_snapshot["live_diagnostic_plot_payload"]["x_axis"] == "step"
        ),
        "trainer_emits_live_diagnostic_rows_during_run": bool(
            completed["live_diagnostic_rows"]
        )
        and bool(completed["live_diagnostic_tables"])
        and bool(completed["live_diagnostic_plot_payloads"]),
        "trainer_live_rows_align_with_display_rows": len(
            completed["live_diagnostic_rows"]
        )
        == len(completed["live_display_rows"]),
        "controller_decision_columns_visible": all(
            key in sample_row
            for key in [
                "alpha_decision",
                "gate_floor_decision",
                "causal_reachability_decision",
                "distance_norm_scale_decision",
                "joint_budget_decision",
            ]
        ),
        "meta_control_columns_visible": all(
            key in sample_row
            for key in [
                "meta_top_signal",
                "meta_attention_entropy",
                "meta_alpha_weight",
                "meta_memory_weight",
                "meta_gate_weight",
                "meta_reach_weight",
                "meta_norm_weight",
                "controller_conflict_score",
                "meta_control_confidence",
            ]
        ),
        "fail_fast_row_is_displayed": (
            fail_fast["trainer_status"] == "failed_fast"
            and fail_fast["live_diagnostic_rows"][-1]["trainer_status"] == "failed_fast"
            and fail_fast["live_diagnostic_rows"][-1]["future_leak"] > 0
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "stage": "stage18_live_100_step_diagnostic_table",
        "status": status,
        "scientific_scope": (
            "live display contract for 100-step adaptive ERGT diagnostics; "
            "table and plot payloads are observability artifacts, not claim credit"
        ),
        "config": asdict(config),
        "required_columns": list(REQUIRED_LIVE_DIAGNOSTIC_COLUMNS),
        "required_outputs": list(REQUIRED_LIVE_DIAGNOSTIC_OUTPUTS),
        "plot_series_groups": {
            key: list(value) for key, value in PLOT_SERIES_GROUPS.items()
        },
        "checks": checks,
        "sample_row": sample_row,
        "sample_snapshot": sample_snapshot,
        "completed_run_summary": completed_summary,
        "fail_fast_live_row": fail_fast["live_diagnostic_rows"][-1],
        "next_required_step": (
            "adaptive_notebook_ergt_03"
            if status == "pass"
            else "fix_live_100_step_diagnostic_table"
        ),
    }


def _synthetic_full_record() -> dict[str, Any]:
    return {
        "step": 1000,
        "condition": "real_memory_d",
        "train_loss": 5.20,
        "validation_loss": 5.30,
        "real_vs_baseline_delta": 0.16,
        "loss_slope_gain": 0.04,
        "alpha": 0.035,
        "geo_to_qk_ratio": 0.08,
        "memory_stability": 0.55,
        "memory_turnover": 0.22,
        "memory_persistence": 0.48,
        "memory_rigidity": 0.10,
        "noise_risk": 0.14,
        "attention_behavior_regime": "useful_noncollapsed",
        "attention_behavior_separation": 0.12,
        "distance_contrast_retention": 0.62,
        "future_leak_score": 0.0,
        "meta_top_signal": "memory_state",
        "meta_attention_entropy": 1.92,
        "meta_alpha_weight": 0.18,
        "meta_memory_weight": 0.25,
        "meta_gate_weight": 0.12,
        "meta_reach_weight": 0.17,
        "meta_norm_weight": 0.28,
        "controller_conflict_score": 0.06,
        "meta_control_confidence": 0.62,
        "alpha_decision": "grow_alpha",
        "memory_eta_decision": "increase_eta",
        "memory_decay_decision": "hold_decay",
        "gate_floor_decision": "hold_gate_floor",
        "causal_reachability_decision": "expand_reach",
        "distance_norm_scale_decision": "increase_distance_scale",
        "joint_budget_decision": "allocate_geometry_memory",
        "trainer_status": "running",
        "trainer_fail_fast_triggered": False,
    }


def _synthetic_live_rows() -> list[dict[str, Any]]:
    rows = []
    for condition, offset in [
        ("baseline", 0.00),
        ("alpha_zero", 0.01),
        ("real_memory_d", 0.14),
        ("random_memory_d", 0.07),
        ("shuffled_memory_d", 0.06),
    ]:
        for step, base_loss in [(100, 6.10), (200, 5.80), (300, 5.60)]:
            row = _synthetic_full_record()
            row.update(
                {
                    "condition": condition,
                    "step": step,
                    "validation_loss": base_loss - offset,
                    "real_vs_baseline_delta": offset if condition != "baseline" else None,
                }
            )
            rows.append(row)
    return rows


def _synthetic_fail_fast_rows() -> list[dict[str, Any]]:
    clean = _synthetic_full_record()
    clean.update({"step": 100, "future_leak_score": 0.0})
    leak = _synthetic_full_record()
    leak.update({"step": 200, "future_leak_score": 0.2})
    skipped = _synthetic_full_record()
    skipped.update({"step": 300, "validation_loss": 5.0})
    return [clean, leak, skipped]
