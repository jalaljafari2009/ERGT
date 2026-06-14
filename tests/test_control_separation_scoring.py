import json

import pytest

from evaluation.control_separation_scoring import (
    REQUIRED_CONTROL_SEPARATION_OUTPUTS,
    build_control_separation_scoring_report,
)
from experiments.control_separation_scoring import (
    ControlSeparationConfig,
    ControlSeparationScorer,
    score_control_separation,
)
from experiments.progress_logging import format_progress_line


def test_partial_live_score_uses_baseline_but_blocks_claim_until_controls_exist() -> None:
    scorer = ControlSeparationScorer()
    score = scorer.score(
        {
            "baseline": [
                {"step": 100, "validation_loss": 6.0},
                {"step": 200, "validation_loss": 5.8},
            ],
            "real_memory_d": [
                {"step": 100, "validation_loss": 5.95},
                {"step": 200, "validation_loss": 5.70},
            ],
        },
        current_step=200,
    )

    assert score.mode == "partial_live"
    assert score.claim_eligibility == "not_eligible_pending_controls"
    assert score.scientific_claim_credit == 0.0
    assert score.real_vs_baseline_delta == pytest.approx(0.10)
    assert score.real_vs_random_delta is None
    assert "random" in score.pending_control_families
    assert "shuffled" in score.pending_control_families
    assert score.generic_regularization_warning == "baseline_only_signal_pending_controls"


def test_final_matched_score_passes_only_when_real_beats_all_controls() -> None:
    report = build_control_separation_scoring_report()
    final = report["final_matched_example"]

    assert report["status"] == "pass"
    assert final["claim_eligibility"] == "eligible_complete_controls"
    assert final["control_separation_status"] == "pass_real_geometry_separated"
    assert final["control_separation"] > 0
    assert final["scientific_claim_credit"] > 0
    assert set(REQUIRED_CONTROL_SEPARATION_OUTPUTS).issubset(final)
    json.dumps(report)


def test_random_dominance_blocks_claim_even_when_baseline_is_beaten() -> None:
    report = build_control_separation_scoring_report()
    failing = report["failing_control_example"]

    assert failing["real_vs_baseline_delta"] > 0
    assert failing["real_vs_random_delta"] < 0
    assert failing["control_separation_status"] == "fail_controls_not_separated"
    assert failing["scientific_claim_credit"] == 0.0
    assert "random" in failing["generic_regularization_warning"]


def test_score_uses_current_step_limit_and_does_not_peek_at_future_controls() -> None:
    summary = score_control_separation(
        {
            "baseline": [
                {"step": 100, "validation_loss": 6.0},
                {"step": 200, "validation_loss": 5.7},
                {"step": 300, "validation_loss": 5.4},
            ],
            "real_memory_d": [
                {"step": 100, "validation_loss": 5.9},
                {"step": 200, "validation_loss": 5.6},
                {"step": 300, "validation_loss": 5.0},
            ],
            "random_memory_d": [
                {"step": 300, "validation_loss": 4.8},
            ],
        },
        current_step=200,
    )

    assert summary["real_vs_baseline_delta"] == pytest.approx(0.10)
    assert summary["real_vs_random_delta"] is None
    assert summary["control_family_status"]["random"] == "pending"
    assert summary["decision_replay_record"]["current_step_limit"] == 200


def test_final_score_uses_only_matched_late_window_steps() -> None:
    progress = build_control_separation_scoring_report()["final_matched_example"]
    per_step = progress["final_matched_score"]["per_step"]

    assert [row["step"] for row in per_step] == [1000, 1500, 2000]


def test_final_score_reports_insufficient_when_controls_exist_without_late_points() -> None:
    scorer = ControlSeparationScorer(
        ControlSeparationConfig(late_window_start=1000, min_matched_points=2)
    )
    progress = {
        condition: [
            {"step": 100, "validation_loss": loss},
            {"step": 500, "validation_loss": loss - 0.1},
        ]
        for condition, loss in {
            "baseline": 6.0,
            "alpha_zero": 5.99,
            "real_memory_d": 5.8,
            "random_memory_d": 5.9,
            "shuffled_memory_d": 5.91,
            "no_memory_real_d": 5.92,
            "instantaneous_real_d": 5.93,
        }.items()
    }

    score = scorer.score(progress)

    assert score.claim_eligibility == "not_eligible_insufficient_matched_steps"
    assert score.scientific_claim_credit == 0.0


def test_progress_line_includes_control_separation_fields() -> None:
    line = format_progress_line(
        {
            "condition": "real_memory_d",
            "step": 1000,
            "validation_loss": 5.4,
            "control_separation_mode": "partial_live",
            "claim_eligibility": "not_eligible_pending_controls",
            "control_separation_status": "partial_signal_only",
            "real_vs_baseline_delta": 0.10,
            "real_vs_random_delta": None,
            "real_vs_shuffled_delta": None,
            "control_separation": 0.10,
            "control_penalty": 0.0,
            "attention_behavior_separation": 0.08,
        }
    )

    assert "sep_mode=partial_live" in line
    assert "claim=not_eligible_pending_controls" in line
    assert "sep_status=partial_signal_only" in line
    assert "rvBase=0.100" in line
    assert "sep=0.100" in line
