"""Stage-15 Control Separation Scoring report."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from experiments.control_separation_scoring import (
    ControlSeparationConfig,
    ControlSeparationScorer,
)

REQUIRED_CONTROL_SEPARATION_OUTPUTS = [
    "control_separation_mode",
    "claim_eligibility",
    "control_separation_status",
    "scientific_claim_credit",
    "available_control_families",
    "pending_control_families",
    "control_family_status",
    "matched_control_steps",
    "matched_control_window",
    "real_vs_baseline_delta",
    "real_vs_alpha_zero_delta",
    "real_vs_random_delta",
    "real_vs_shuffled_delta",
    "real_vs_no_memory_delta",
    "real_vs_instantaneous_delta",
    "control_separation",
    "control_penalty",
    "generic_regularization_warning",
    "attention_behavior_separation",
    "partial_live_score",
    "final_matched_score",
    "decision_replay_record",
]


def build_control_separation_scoring_report(
    *,
    partial_progress: dict[str, list[dict[str, Any]]] | None = None,
    final_progress: dict[str, list[dict[str, Any]]] | None = None,
    failing_control_progress: dict[str, list[dict[str, Any]]] | None = None,
    config: ControlSeparationConfig | None = None,
) -> dict[str, Any]:
    """Build the stage-15 mechanics report for control separation scoring."""

    from evaluation.unified_telemetry_schema import build_unified_telemetry_schema_report

    config = config or ControlSeparationConfig()
    config.validate()
    partial_progress = partial_progress or _synthetic_partial_progress()
    final_progress = final_progress or _synthetic_final_progress()
    failing_control_progress = failing_control_progress or _synthetic_failing_control_progress()

    scorer = ControlSeparationScorer(config)
    partial = scorer.summary(scorer.score(partial_progress, current_step=1000))
    final = scorer.summary(scorer.score(final_progress))
    failing = scorer.summary(scorer.score(failing_control_progress))
    schema_fields = set(build_unified_telemetry_schema_report()["fields"])
    required_schema_fields = {
        "control_separation_mode",
        "claim_eligibility",
        "control_separation_status",
        "scientific_claim_credit",
        "available_control_families",
        "pending_control_families",
        "matched_control_steps",
        "matched_control_window",
        "real_vs_baseline_delta",
        "real_vs_alpha_zero_delta",
        "real_vs_random_delta",
        "real_vs_shuffled_delta",
        "real_vs_no_memory_delta",
        "real_vs_instantaneous_delta",
        "generic_regularization_warning",
        "attention_behavior_separation",
    }
    checks = {
        "required_control_separation_outputs_emitted": set(
            REQUIRED_CONTROL_SEPARATION_OUTPUTS
        ).issubset(partial),
        "schema_declares_control_separation_fields": required_schema_fields.issubset(
            schema_fields
        ),
        "partial_live_score_is_not_claim_eligible": (
            partial["control_separation_mode"] == "partial_live"
            and partial["claim_eligibility"] == "not_eligible_pending_controls"
            and partial["scientific_claim_credit"] == 0.0
            and "random" in partial["pending_control_families"]
            and "shuffled" in partial["pending_control_families"]
        ),
        "partial_live_uses_real_vs_baseline_without_controls": (
            partial["real_vs_baseline_delta"] is not None
            and partial["real_vs_random_delta"] is None
            and partial["real_vs_shuffled_delta"] is None
            and partial["generic_regularization_warning"]
            == "baseline_only_signal_pending_controls"
        ),
        "final_score_requires_complete_matched_controls": (
            final["control_separation_mode"] == "final_matched"
            and final["claim_eligibility"] == "eligible_complete_controls"
            and not final["pending_control_families"]
            and final["matched_control_window"]["points"] >= config.min_matched_points
        ),
        "final_score_passes_only_when_real_beats_all_controls": (
            final["control_separation_status"] == "pass_real_geometry_separated"
            and final["control_separation"] > config.pass_margin
            and final["scientific_claim_credit"] > 0.0
        ),
        "random_or_shuffled_dominance_blocks_claim": (
            failing["control_separation_status"] == "fail_controls_not_separated"
            and failing["scientific_claim_credit"] == 0.0
            and "random" in failing["generic_regularization_warning"]
        ),
        "matched_steps_are_late_window_only_for_claim": all(
            step >= config.late_window_start
            for step in final["final_matched_score"]["matched_steps"]
        ),
        "attention_behavior_separation_is_reported": (
            final["attention_behavior_separation"] is not None
        ),
        "decision_replay_records_present": all(
            bool(record)
            for record in [
                partial["decision_replay_record"],
                final["decision_replay_record"],
                failing["decision_replay_record"],
            ]
        ),
    }
    status = "pass" if all(checks.values()) else "fail"

    return {
        "stage": "stage15_control_separation_scoring",
        "status": status,
        "scientific_scope": (
            "partial live scoring during sequential runs plus final matched "
            "late-window control separation after all controls exist"
        ),
        "config": asdict(config),
        "required_outputs": list(REQUIRED_CONTROL_SEPARATION_OUTPUTS),
        "checks": checks,
        "partial_live_example": partial,
        "final_matched_example": final,
        "failing_control_example": failing,
        "next_required_step": (
            "meta_control_attention_observer"
            if status == "pass"
            else "fix_control_separation_scoring"
        ),
    }


def _synthetic_partial_progress() -> dict[str, list[dict[str, Any]]]:
    return {
        "baseline": [
            {"condition": "baseline", "step": 100, "validation_loss": 6.20},
            {"condition": "baseline", "step": 500, "validation_loss": 5.82},
            {"condition": "baseline", "step": 1000, "validation_loss": 5.50},
            {"condition": "baseline", "step": 1500, "validation_loss": 5.32},
        ],
        "real_memory_d": [
            {
                "condition": "real_memory_d",
                "step": 100,
                "validation_loss": 6.18,
                "attention_control_separation": 0.02,
            },
            {
                "condition": "real_memory_d",
                "step": 500,
                "validation_loss": 5.76,
                "attention_control_separation": 0.06,
            },
            {
                "condition": "real_memory_d",
                "step": 1000,
                "validation_loss": 5.39,
                "attention_control_separation": 0.10,
            },
        ],
    }


def _synthetic_final_progress() -> dict[str, list[dict[str, Any]]]:
    steps = [500, 1000, 1500, 2000]
    return {
        "baseline": _rows("baseline", steps, [5.90, 5.50, 5.32, 5.20], [0.08] * 4),
        "alpha_zero": _rows("alpha_zero", steps, [5.89, 5.49, 5.31, 5.19], [0.08] * 4),
        "real_memory_d": _rows(
            "real_memory_d",
            steps,
            [5.86, 5.33, 5.08, 4.92],
            [0.26, 0.30, 0.34, 0.36],
        ),
        "random_memory_d": _rows(
            "random_memory_d",
            steps,
            [5.87, 5.43, 5.23, 5.10],
            [0.14, 0.15, 0.16, 0.17],
        ),
        "shuffled_memory_d": _rows(
            "shuffled_memory_d",
            steps,
            [5.88, 5.45, 5.24, 5.11],
            [0.13, 0.14, 0.14, 0.15],
        ),
        "no_memory_real_d": _rows(
            "no_memory_real_d",
            steps,
            [5.88, 5.44, 5.25, 5.12],
            [0.12, 0.13, 0.14, 0.15],
        ),
        "instantaneous_real_d": _rows(
            "instantaneous_real_d",
            steps,
            [5.87, 5.41, 5.22, 5.08],
            [0.12, 0.13, 0.14, 0.15],
        ),
    }


def _synthetic_failing_control_progress() -> dict[str, list[dict[str, Any]]]:
    progress = _synthetic_final_progress()
    progress["random_memory_d"] = _rows(
        "random_memory_d",
        [500, 1000, 1500, 2000],
        [5.84, 5.30, 5.04, 4.88],
        [0.18, 0.19, 0.20, 0.21],
    )
    return progress


def _rows(
    condition: str,
    steps: list[int],
    losses: list[float],
    attention_scores: list[float],
) -> list[dict[str, Any]]:
    return [
        {
            "condition": condition,
            "step": step,
            "validation_loss": loss,
            "attention_behavior_score": attention,
        }
        for step, loss, attention in zip(steps, losses, attention_scores, strict=True)
    ]
