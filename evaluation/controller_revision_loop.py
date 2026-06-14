"""Stage-25 controller revision loop report."""

from __future__ import annotations

from typing import Any

from evaluation.decision_gate_real_geometry import (
    build_decision_gate_real_geometry_report,
)
from experiments.controller_revision_loop import (
    REQUIRED_REVISION_LOOP_OUTPUTS,
    REVISION_CATALOG,
    ControllerRevisionLoopConfig,
    build_controller_revision_loop,
)
from experiments.guarded_2000_step_adaptive_run import (
    Guarded2000RunConfig,
    generate_guarded_2000_telemetry_rows,
)

STAGE25_REQUIRED_FAILURE_LABELS = [
    "memory_starved",
    "memory_noisy",
    "memory_rigid",
    "geometry_flattened",
    "alpha_underpowered",
    "alpha_overpowering",
    "causal_reach_too_tight",
    "causal_reach_too_loose",
    "control_regularization_dominance",
    "normalization_erased_contrast",
    "attention_uniformity_drift",
    "attention_control_like",
    "attention_head_lock_in",
    "meta_control_attention_misweighted",
    "controller_conflict_unresolved",
]


def build_controller_revision_loop_report(
    *,
    decision_gate_report: dict[str, Any] | None = None,
    config: ControllerRevisionLoopConfig | None = None,
) -> dict[str, Any]:
    """Build the stage-25 controller revision loop report."""

    active = config or ControllerRevisionLoopConfig()
    active.validate()
    gate_report = decision_gate_report or build_decision_gate_real_geometry_report()
    revision = build_controller_revision_loop(gate_report, config=active)
    synthetic_failure_examples = _synthetic_failure_examples(active)
    checks = {
        "required_outputs_present": set(REQUIRED_REVISION_LOOP_OUTPUTS).issubset(
            revision
        ),
        "all_documented_failure_labels_have_catalog_entries": set(
            STAGE25_REQUIRED_FAILURE_LABELS
        ).issubset(REVISION_CATALOG),
        "active_failure_labels_mapped": not revision["unmapped_failure_labels"],
        "specific_revisions_present": revision["checks"][
            "specific_revisions_present"
        ],
        "validation_gates_present": revision["checks"]["validation_gates_present"],
        "decision_replay_record_present": bool(revision["decision_replay_record"]),
        "noop_audit_allows_stage26_when_gate_passed": (
            gate_report["status"] != "pass"
            or (
                revision["revision_mode"] == "noop_audit"
                and revision["stage26_readiness"]["ready"]
            )
        ),
        "failed_gate_blocks_stage26": (
            gate_report["status"] == "pass"
            or not revision["stage26_readiness"]["ready"]
        ),
        "synthetic_failures_map_to_revisions": all(
            example["status"] == "pass" for example in synthetic_failure_examples
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "stage": "stage25_controller_revision_loop",
        "status": status,
        "scientific_scope": (
            "revision mechanics gate; maps failed decision labels to concrete "
            "controller revisions and blocks stage 26 until failures are resolved"
        ),
        "required_failure_labels": list(STAGE25_REQUIRED_FAILURE_LABELS),
        "required_outputs": list(REQUIRED_REVISION_LOOP_OUTPUTS),
        "checks": checks,
        "source_gate_status": gate_report["status"],
        "revision": revision,
        "synthetic_failure_examples": synthetic_failure_examples,
        "next_required_step": (
            revision["next_required_step"]
            if status == "pass"
            else "fix_controller_revision_loop"
        ),
    }


def _synthetic_failure_examples(
    config: ControllerRevisionLoopConfig,
) -> list[dict[str, Any]]:
    examples = [
        _example_from_rows(
            "random_dominance",
            _rows_with_random_dominance(),
            expected_label="control_regularization_dominance",
            config=config,
        ),
        _example_from_rows(
            "future_leak",
            _rows_with_future_leak(),
            expected_label="future_leak_detected",
            config=config,
        ),
        _example_from_rows(
            "attention_control_like",
            _rows_with_attention_control_like(),
            expected_label="attention_control_like",
            config=config,
        ),
    ]
    return examples


def _example_from_rows(
    name: str,
    rows: list[dict[str, Any]],
    *,
    expected_label: str,
    config: ControllerRevisionLoopConfig,
) -> dict[str, Any]:
    gate_report = build_decision_gate_real_geometry_report(telemetry_rows=rows)
    revision = build_controller_revision_loop(gate_report, config=config)
    labels = revision["failure_labels"]
    status = (
        "pass"
        if expected_label in labels
        and not revision["unmapped_failure_labels"]
        and revision["revision_plan"]
        else "fail"
    )
    return {
        "name": name,
        "status": status,
        "expected_label": expected_label,
        "failure_labels": labels,
        "revision_mode": revision["revision_mode"],
        "revision_plan_count": len(revision["revision_plan"]),
    }


def _rows_with_random_dominance() -> list[dict[str, Any]]:
    rows = generate_guarded_2000_telemetry_rows(Guarded2000RunConfig())
    for row in rows:
        if row["condition"] == "random_memory_d" and row["step"] >= 1000:
            row["validation_loss"] = row["validation_loss"] - 0.30
    return rows


def _rows_with_future_leak() -> list[dict[str, Any]]:
    rows = generate_guarded_2000_telemetry_rows(Guarded2000RunConfig())
    for row in rows:
        if row["condition"] == "real_memory_d" and row["step"] == 1500:
            row["future_leak_score"] = 0.01
    return rows


def _rows_with_attention_control_like() -> list[dict[str, Any]]:
    rows = generate_guarded_2000_telemetry_rows(Guarded2000RunConfig())
    for row in rows:
        if row["condition"] == "real_memory_d" and row["step"] >= 1000:
            row["attention_behavior_score"] = 0.31
    return rows
