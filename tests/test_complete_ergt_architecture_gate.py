import json

from evaluation.complete_ergt_architecture_gate import (
    ARCHITECTURE_PIPELINE,
    REQUIRED_COMPARISONS,
    build_complete_ergt_architecture_gate_report,
)


def pass_report(phase: str = "phase") -> dict:
    return {"phase": phase, "status": "pass", "next_required_step": "next"}


def upstream_reports(
    *,
    geoattention_strict_pass: bool,
    auxiliary_ready: bool,
) -> dict[str, dict]:
    reports = {
        "measurement_contracts": pass_report("phase0_measurement_contracts"),
        "strict_w_controls": pass_report("phase1_strict_w_controls"),
        "relational_field_observer": pass_report("phase2_relational_field_observer"),
        "resonant_response_observer": pass_report("phase3_resonant_response_observer"),
        "information_potential_phi": pass_report("phase4_information_potential_phi"),
        "reconstruction_gate": pass_report("phase5_reconstruction_gate"),
        "relational_memory_observer": pass_report("phase6_relational_memory_observer"),
        "causal_shortest_path_geometry": pass_report("phase7_causal_shortest_path_geometry"),
        "geoattention_v2": {
            "phase": "phase8_geoattention_v2",
            "status": "pass",
            "strict_gate_status": "pass" if geoattention_strict_pass else "needs_training_run",
            "next_required_step": "auxiliary_physics_loss"
            if geoattention_strict_pass
            else "run_geoattention_v2_training_controls",
        },
        "auxiliary_physics_loss_gate": {
            "phase": "phase9_auxiliary_physics_loss_gate",
            "status": "ready" if auxiliary_ready else "blocked",
            "loss_enabled": auxiliary_ready,
            "next_required_step": "implement_auxiliary_physics_loss"
            if auxiliary_ready
            else "run_geoattention_v2_training_controls",
        },
    }
    return reports


def test_complete_architecture_gate_blocks_when_geoattention_v2_strict_gate_is_missing() -> None:
    report = build_complete_ergt_architecture_gate_report(
        upstream_reports(geoattention_strict_pass=False, auxiliary_ready=False)
    )

    assert report["status"] == "blocked"
    assert report["architecture_enabled"] is False
    assert not report["checks"]["geoattention_v2_strict_training_gate_passed"]
    assert "geoattention_v2_strict_training_gate_not_passed" in report["blocked_by"]
    assert report["next_required_step"] == "run_geoattention_v2_training_controls"
    json.dumps(report)


def test_complete_architecture_gate_requires_auxiliary_loss_after_v2_gate() -> None:
    report = build_complete_ergt_architecture_gate_report(
        upstream_reports(geoattention_strict_pass=True, auxiliary_ready=False)
    )

    assert report["status"] == "blocked"
    assert report["checks"]["geoattention_v2_strict_training_gate_passed"]
    assert not report["checks"]["auxiliary_physics_loss_ready"]
    assert "auxiliary_physics_loss_not_ready_or_not_enabled" in report["blocked_by"]


def test_complete_architecture_gate_ready_when_all_prerequisites_pass() -> None:
    report = build_complete_ergt_architecture_gate_report(
        upstream_reports(geoattention_strict_pass=True, auxiliary_ready=True)
    )

    assert report["status"] == "ready"
    assert report["architecture_enabled"] is True
    assert report["architecture_pipeline"] == ARCHITECTURE_PIPELINE
    assert report["required_comparisons"] == REQUIRED_COMPARISONS
    assert report["missing_reports"] == []
    assert report["blocked_by"] == []
    assert report["next_required_step"] == "implement_complete_ergt_architecture"
