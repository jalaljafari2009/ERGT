import json

from evaluation.intelligence_space_evaluation_gate import (
    INTELLIGENCE_AXES,
    REQUIRED_INTELLIGENCE_CONTROLS,
    build_intelligence_space_evaluation_gate_report,
)


def reasoning_report(*, ready: bool) -> dict:
    return {
        "phase": "phase11_reasoning_path_evaluation_gate",
        "status": "ready" if ready else "blocked",
        "evaluation_enabled": ready,
        "next_required_step": "implement_reasoning_path_evaluation"
        if ready
        else "run_geoattention_v2_training_controls",
    }


def test_intelligence_space_gate_blocks_until_reasoning_path_is_ready() -> None:
    report = build_intelligence_space_evaluation_gate_report(reasoning_report(ready=False))

    assert report["status"] == "blocked"
    assert report["evaluation_enabled"] is False
    assert not report["checks"]["reasoning_path_evaluation_ready"]
    assert "reasoning_path_evaluation_not_ready" in report["blocked_by"]
    assert report["next_required_step"] == "run_geoattention_v2_training_controls"
    json.dumps(report)


def test_intelligence_space_gate_blocks_when_reasoning_report_is_missing() -> None:
    report = build_intelligence_space_evaluation_gate_report(None)

    assert report["status"] == "blocked"
    assert report["evaluation_enabled"] is False
    assert "missing_reasoning_path_gate_report" in report["blocked_by"]
    assert report["next_required_step"] == "reasoning_path_evaluation_gate"


def test_intelligence_space_gate_ready_after_reasoning_path_gate_is_ready() -> None:
    report = build_intelligence_space_evaluation_gate_report(reasoning_report(ready=True))

    assert report["status"] == "ready"
    assert report["evaluation_enabled"] is True
    assert report["axes"] == INTELLIGENCE_AXES
    assert report["required_controls"] == REQUIRED_INTELLIGENCE_CONTROLS
    assert report["blocked_by"] == []
    assert report["next_required_step"] == "implement_intelligence_space_evaluation"
