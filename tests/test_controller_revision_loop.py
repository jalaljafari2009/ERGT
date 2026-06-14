import json

from evaluation.controller_revision_loop import (
    STAGE25_REQUIRED_FAILURE_LABELS,
    build_controller_revision_loop_report,
)
from evaluation.decision_gate_real_geometry import (
    build_decision_gate_real_geometry_report,
)
from experiments.controller_revision_loop import (
    REQUIRED_REVISION_LOOP_OUTPUTS,
    REVISION_CATALOG,
    build_controller_revision_loop,
)
from experiments.guarded_2000_step_adaptive_run import (
    Guarded2000RunConfig,
    generate_guarded_2000_telemetry_rows,
)


def test_controller_revision_loop_report_passes_with_noop_audit() -> None:
    report = build_controller_revision_loop_report()
    revision = report["revision"]

    assert report["status"] == "pass"
    assert revision["revision_mode"] == "noop_audit"
    assert revision["stage26_readiness"]["ready"]
    assert report["next_required_step"] == "longer_run_or_multi_seed_confirmation"
    json.dumps(report)


def test_controller_revision_loop_declares_required_outputs_and_catalog() -> None:
    report = build_controller_revision_loop_report()

    assert set(REQUIRED_REVISION_LOOP_OUTPUTS).issubset(report["revision"])
    assert set(STAGE25_REQUIRED_FAILURE_LABELS).issubset(REVISION_CATALOG)
    assert report["checks"]["all_documented_failure_labels_have_catalog_entries"]


def test_random_dominance_maps_to_control_revision() -> None:
    rows = generate_guarded_2000_telemetry_rows(Guarded2000RunConfig())
    for row in rows:
        if row["condition"] == "random_memory_d" and row["step"] >= 1000:
            row["validation_loss"] = row["validation_loss"] - 0.30
    gate = build_decision_gate_real_geometry_report(telemetry_rows=rows)

    revision = build_controller_revision_loop(gate)

    assert revision["revision_mode"] == "apply_revisions"
    assert "control_regularization_dominance" in revision["failure_labels"]
    assert not revision["stage26_readiness"]["ready"]
    assert any(
        item["failure_label"] == "control_regularization_dominance"
        for item in revision["revision_plan"]
    )


def test_future_leak_maps_to_hard_stop_revision() -> None:
    rows = generate_guarded_2000_telemetry_rows(Guarded2000RunConfig())
    for row in rows:
        if row["condition"] == "real_memory_d" and row["step"] == 1500:
            row["future_leak_score"] = 0.01
    gate = build_decision_gate_real_geometry_report(telemetry_rows=rows)

    revision = build_controller_revision_loop(gate)

    future_leak = [
        item
        for item in revision["revision_plan"]
        if item["failure_label"] == "future_leak_detected"
    ][0]
    assert future_leak["severity"] == "hard_stop"
    assert revision["rerun_protocol"]["start_from"] == "short_smoke"


def test_attention_control_like_maps_to_attention_revision() -> None:
    rows = generate_guarded_2000_telemetry_rows(Guarded2000RunConfig())
    for row in rows:
        if row["condition"] == "real_memory_d" and row["step"] >= 1000:
            row["attention_behavior_score"] = 0.31
    gate = build_decision_gate_real_geometry_report(telemetry_rows=rows)

    revision = build_controller_revision_loop(gate)

    assert "attention_control_like" in revision["failure_labels"]
    assert any(
        "meta_control_attention_observer" in item["target_components"]
        for item in revision["revision_plan"]
        if item["failure_label"] == "attention_control_like"
    )


def test_unknown_failure_label_is_not_silently_accepted() -> None:
    gate = build_decision_gate_real_geometry_report()
    gate["status"] = "fail"
    gate["gate"]["decision"] = "fail_enter_revision"
    gate["gate"]["failure_labels"] = ["unknown_failure"]

    revision = build_controller_revision_loop(gate)

    assert revision["unmapped_failure_labels"] == ["unknown_failure"]
    assert not revision["checks"]["every_failure_label_mapped"]
