"""Stage-24 decision gate for real geometry vs generic regularization."""

from __future__ import annotations

from typing import Any

from evaluation.late_window_post1000_analysis import (
    build_late_window_post1000_analysis_report,
)
from evaluation.random_shuffled_no_memory_attribution import (
    build_random_shuffled_no_memory_attribution_report,
)
from experiments.decision_gate_real_geometry import (
    REQUIRED_DECISION_GATE_OUTPUTS,
    RealGeometryDecisionGateConfig,
    decide_real_geometry_vs_generic_regularization,
)
from experiments.guarded_2000_step_adaptive_run import (
    Guarded2000RunConfig,
    generate_guarded_2000_telemetry_rows,
    summarize_guarded_replay,
)


def build_decision_gate_real_geometry_report(
    *,
    telemetry_rows: list[dict[str, Any]] | None = None,
    config: RealGeometryDecisionGateConfig | None = None,
) -> dict[str, Any]:
    """Build the stage-24 real-geometry decision gate report."""

    active = config or RealGeometryDecisionGateConfig()
    active.validate()
    rows = telemetry_rows or generate_guarded_2000_telemetry_rows(
        Guarded2000RunConfig()
    )
    guarded_summary = summarize_guarded_replay(rows)
    late_report = build_late_window_post1000_analysis_report(telemetry_rows=rows)
    attribution_report = build_random_shuffled_no_memory_attribution_report(
        telemetry_rows=rows
    )
    gate = decide_real_geometry_vs_generic_regularization(
        rows,
        late_window_report=late_report,
        attribution_report=attribution_report,
        config=active,
    )
    checks = {
        "guarded_run_contract_ready": (
            guarded_summary["all_conditions_present"]
            and guarded_summary["all_conditions_have_identical_steps"]
            and guarded_summary["all_conditions_reach_2000"]
        ),
        "required_outputs_present": set(REQUIRED_DECISION_GATE_OUTPUTS).issubset(gate),
        "stage22_late_window_passed": late_report["status"] == "pass",
        "stage23_attribution_passed": attribution_report["status"] == "pass",
        "all_required_comparisons_positive": gate["checks"][
            "all_required_comparisons_positive"
        ],
        "baseline_only_insufficient_rule_enforced": gate["checks"][
            "baseline_only_is_insufficient"
        ],
        "r1_clear": gate["checks"]["r1_memory_and_causality_clear"],
        "r2_clear": gate["checks"]["r2_distance_scale_clear"],
        "r3_clear": gate["checks"]["r3_attention_behavior_clear"],
        "attention_gate_passed": gate["checks"]["attention_gate_passed"],
        "relation_specific_gate_passed": gate["checks"][
            "relation_specific_gate_passed"
        ],
        "no_failure_labels_for_pass": not gate["failure_labels"],
        "decision_passes": gate["decision"] == "pass_real_geometry_contract",
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "stage": "stage24_decision_gate_real_geometry_vs_generic_regularization",
        "status": status,
        "scientific_scope": (
            "guarded adaptive decision gate; requires real stable causal geometry "
            "to beat baseline, alpha-zero, random, shuffled, no-memory, and "
            "instantaneous controls while clearing memory/causality, distance-scale, "
            "and attention-behavior audits"
        ),
        "input_source": (
            "provided_telemetry_rows"
            if telemetry_rows is not None
            else "guarded_2000_contract_replay"
        ),
        "required_outputs": list(REQUIRED_DECISION_GATE_OUTPUTS),
        "checks": checks,
        "guarded_summary": {
            "condition_row_counts": guarded_summary["condition_row_counts"],
            "all_conditions_have_identical_steps": guarded_summary[
                "all_conditions_have_identical_steps"
            ],
            "all_conditions_reach_2000": guarded_summary[
                "all_conditions_reach_2000"
            ],
        },
        "gate": gate,
        "decision_table": _decision_table(gate),
        "upstream_reports": {
            "late_window_status": late_report["status"],
            "attribution_status": attribution_report["status"],
            "late_window_next_required_step": late_report["next_required_step"],
            "attribution_next_required_step": attribution_report[
                "next_required_step"
            ],
        },
        "next_required_step": (
            "controller_revision_loop"
            if status == "fail"
            else "controller_revision_loop_noop_audit"
        ),
    }


def _decision_table(gate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for comparison, payload in gate["required_comparisons"].items():
        rows.append(
            {
                "gate_item": comparison,
                "real_advantage": payload["real_advantage"],
                "threshold": payload["threshold"],
                "passes": payload["passes"],
            }
        )
    for risk_name, payload in gate["risk_audit"].items():
        rows.append(
            {
                "gate_item": risk_name,
                "real_advantage": None,
                "threshold": None,
                "passes": payload["passes"],
            }
        )
    rows.append(
        {
            "gate_item": "attention_gate",
            "real_advantage": gate["attention_gate"]["minimum_attention_advantage"],
            "threshold": None,
            "passes": gate["attention_gate"]["passes"],
        }
    )
    rows.append(
        {
            "gate_item": "relation_specific_gate",
            "real_advantage": gate["relation_specific_gate"]["estimate"],
            "threshold": None,
            "passes": gate["relation_specific_gate"]["passes"],
        }
    )
    return rows
