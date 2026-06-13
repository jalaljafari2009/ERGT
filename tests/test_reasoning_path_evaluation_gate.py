import json

from evaluation.reasoning_path_evaluation_gate import (
    REASONING_PATH_METRICS,
    REQUIRED_REASONING_CONTROLS,
    build_reasoning_path_evaluation_gate_report,
)


def complete_architecture_report(*, ready: bool) -> dict:
    return {
        "phase": "phase10_complete_ergt_architecture_gate",
        "status": "ready" if ready else "blocked",
        "architecture_enabled": ready,
        "next_required_step": "implement_complete_ergt_architecture"
        if ready
        else "run_geoattention_v2_training_controls",
    }


def test_reasoning_path_gate_blocks_until_complete_architecture_is_ready() -> None:
    report = build_reasoning_path_evaluation_gate_report(
        complete_architecture_report(ready=False)
    )

    assert report["status"] == "blocked"
    assert report["evaluation_enabled"] is False
    assert not report["checks"]["complete_architecture_ready"]
    assert "complete_ergt_architecture_not_ready" in report["blocked_by"]
    assert report["next_required_step"] == "run_geoattention_v2_training_controls"
    json.dumps(report)


def test_reasoning_path_gate_blocks_when_architecture_report_is_missing() -> None:
    report = build_reasoning_path_evaluation_gate_report(None)

    assert report["status"] == "blocked"
    assert report["evaluation_enabled"] is False
    assert "missing_complete_architecture_gate_report" in report["blocked_by"]
    assert report["next_required_step"] == "complete_ergt_architecture_gate"


def test_reasoning_path_gate_ready_after_complete_architecture_is_ready() -> None:
    report = build_reasoning_path_evaluation_gate_report(
        complete_architecture_report(ready=True)
    )

    assert report["status"] == "ready"
    assert report["evaluation_enabled"] is True
    assert report["candidate_metrics"] == REASONING_PATH_METRICS
    assert report["required_controls"] == REQUIRED_REASONING_CONTROLS
    assert report["blocked_by"] == []
    assert report["next_required_step"] == "implement_reasoning_path_evaluation"
