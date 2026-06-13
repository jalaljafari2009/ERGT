import json

import pytest

from evaluation.auxiliary_physics_loss_gate import (
    ALLOWED_REGULARIZERS,
    build_auxiliary_physics_loss_gate_report,
)


def geoattention_report(strict_pass: bool) -> dict:
    checks = {
        "real_stable_beats_random": strict_pass,
        "real_stable_beats_shuffled": strict_pass,
        "real_stable_beats_instantaneous": strict_pass,
        "real_stable_beats_pairwise": strict_pass,
        "real_stable_beats_no_memory": strict_pass,
    }
    return {
        "phase": "phase8_geoattention_v2",
        "status": "pass",
        "strict_gate_status": "pass" if strict_pass else "needs_training_run",
        "strict_gate_checks": checks,
        "next_required_step": "auxiliary_physics_loss"
        if strict_pass
        else "run_geoattention_v2_training_controls",
    }


def test_auxiliary_physics_loss_gate_blocks_until_geoattention_v2_strict_gate_passes() -> None:
    report = build_auxiliary_physics_loss_gate_report(
        geoattention_report(strict_pass=False),
        requested_lambda=0.01,
    )

    assert report["status"] == "blocked"
    assert report["loss_enabled"] is False
    assert report["effective_lambda"] == 0.0
    assert report["checks"]["geoattention_v2_mechanics_passed"]
    assert not report["checks"]["geoattention_v2_strict_gate_passed"]
    assert "real_stable_beats_random" in report["missing_evidence"]
    assert report["next_required_step"] == "run_geoattention_v2_training_controls"
    json.dumps(report)


def test_auxiliary_physics_loss_gate_ready_only_after_all_required_control_wins() -> None:
    report = build_auxiliary_physics_loss_gate_report(
        geoattention_report(strict_pass=True),
        requested_lambda=0.02,
    )

    assert report["status"] == "ready"
    assert report["loss_enabled"] is True
    assert report["effective_lambda"] == 0.02
    assert report["allowed_regularizers"] == ALLOWED_REGULARIZERS
    assert report["missing_evidence"] == []
    assert report["next_required_step"] == "implement_auxiliary_physics_loss"


def test_auxiliary_physics_loss_gate_rejects_negative_lambda() -> None:
    with pytest.raises(ValueError, match="requested_lambda"):
        build_auxiliary_physics_loss_gate_report(
            geoattention_report(strict_pass=True),
            requested_lambda=-0.1,
        )
