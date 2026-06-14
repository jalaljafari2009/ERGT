import json

from evaluation.decision_gate_real_geometry import (
    build_decision_gate_real_geometry_report,
)
from experiments.decision_gate_real_geometry import REQUIRED_DECISION_GATE_OUTPUTS
from experiments.guarded_2000_step_adaptive_run import (
    Guarded2000RunConfig,
    generate_guarded_2000_telemetry_rows,
)


def test_decision_gate_real_geometry_report_passes() -> None:
    report = build_decision_gate_real_geometry_report()

    assert report["status"] == "pass"
    assert report["gate"]["decision"] == "pass_real_geometry_contract"
    assert report["next_required_step"] == "controller_revision_loop_noop_audit"
    json.dumps(report)


def test_decision_gate_contains_required_outputs_and_comparisons() -> None:
    report = build_decision_gate_real_geometry_report()
    gate = report["gate"]

    assert set(REQUIRED_DECISION_GATE_OUTPUTS).issubset(gate)
    assert set(gate["required_comparisons"]) == {
        "baseline",
        "alpha_zero",
        "random_adaptive",
        "shuffled_adaptive",
        "no_memory_real",
        "instantaneous_real",
    }
    assert all(item["passes"] for item in gate["required_comparisons"].values())


def test_decision_gate_enforces_r1_r2_r3_audits() -> None:
    report = build_decision_gate_real_geometry_report()
    gate = report["gate"]

    assert gate["risk_audit"]["R1"]["passes"]
    assert gate["risk_audit"]["R2"]["passes"]
    assert gate["risk_audit"]["R3"]["passes"]
    assert report["checks"]["r1_clear"]
    assert report["checks"]["r2_clear"]
    assert report["checks"]["r3_clear"]


def test_decision_gate_fails_when_random_dominates_real() -> None:
    rows = generate_guarded_2000_telemetry_rows(Guarded2000RunConfig())
    for row in rows:
        if row["condition"] == "random_memory_d" and row["step"] >= 1000:
            row["validation_loss"] = row["validation_loss"] - 0.30

    report = build_decision_gate_real_geometry_report(telemetry_rows=rows)

    assert report["status"] == "fail"
    assert report["gate"]["decision"] == "fail_enter_revision"
    assert report["next_required_step"] == "controller_revision_loop"
    assert "control_regularization_dominance" in report["gate"]["failure_labels"]


def test_decision_gate_fails_on_future_leak_r1_violation() -> None:
    rows = generate_guarded_2000_telemetry_rows(Guarded2000RunConfig())
    for row in rows:
        if row["condition"] == "real_memory_d" and row["step"] == 1500:
            row["future_leak_score"] = 0.01

    report = build_decision_gate_real_geometry_report(telemetry_rows=rows)

    assert report["status"] == "fail"
    assert not report["checks"]["r1_clear"]
    assert "future_leak_detected" in report["gate"]["failure_labels"]


def test_decision_gate_fails_when_attention_is_control_like() -> None:
    rows = generate_guarded_2000_telemetry_rows(Guarded2000RunConfig())
    for row in rows:
        if row["condition"] == "real_memory_d" and row["step"] >= 1000:
            row["attention_behavior_score"] = 0.31

    report = build_decision_gate_real_geometry_report(telemetry_rows=rows)

    assert report["status"] == "fail"
    assert not report["checks"]["r3_clear"]
    assert "attention_control_like" in report["gate"]["failure_labels"]
