"""Stage-13 Normalization and Distance-Scale Controller report."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from experiments.distance_scale_controller import (
    DistanceScaleConfig,
    DistanceScaleController,
    DistanceScaleObservation,
)

REQUIRED_DISTANCE_SCALE_OUTPUTS = [
    "current_distance_norm_scale",
    "proposed_distance_norm_scale",
    "distance_norm_scale_delta",
    "distance_norm_scale_credit",
    "distance_norm_scale_risk_pressure",
    "contrast_evidence",
    "scale_evidence",
    "clipping_evidence",
    "control_distance_evidence",
    "attention_safety_evidence",
    "release_evidence",
    "restraint_evidence",
    "parameter_trajectory",
    "injected_evidence_ledger",
    "controller_state_snapshot",
    "decision_replay_record",
]


def build_distance_scale_controller_report(
    *,
    observations: list[DistanceScaleObservation] | None = None,
    config: DistanceScaleConfig | None = None,
) -> dict[str, Any]:
    """Build the stage-13 mechanics report for distance-scale control."""

    from evaluation.unified_telemetry_schema import build_unified_telemetry_schema_report

    config = config or DistanceScaleConfig()
    config.validate()
    input_source = "provided_observations"
    if observations is None:
        observations = _synthetic_observations()
        input_source = "synthetic_distance_scale_controller_smoke"

    controller = DistanceScaleController(config)
    decisions = [controller.update(observation) for observation in observations]
    decision_rows = [asdict(decision) for decision in decisions]
    latest = decision_rows[-1]
    schema_fields = set(build_unified_telemetry_schema_report()["fields"])
    checks = {
        "required_distance_scale_outputs_emitted": all(
            field in latest for field in REQUIRED_DISTANCE_SCALE_OUTPUTS
        ),
        "schema_declares_distance_scale_fields": {
            "distance_norm_scale",
            "distance_norm_scale_next",
            "distance_norm_scale_delta",
            "distance_norm_scale_decision",
            "distance_norm_scale_credit",
            "distance_norm_scale_risk_pressure",
            "pre_norm_distance_contrast",
            "post_norm_distance_contrast",
            "distance_contrast_retention",
            "clipping_saturation_rate",
            "normalization_erasure_score",
        }.issubset(schema_fields),
        "uses_controller_error_terms": all(
            field in latest["controller_state_snapshot"]
            for field in ["error", "integral_error", "derivative_error"]
        ),
        "can_increase_when_real_contrast_is_erased": any(
            decision.decision == "increase_distance_scale"
            and decision.distance_norm_scale_delta > 0
            for decision in decisions
        ),
        "can_decrease_when_scale_is_noisy_or_clipped": any(
            decision.decision == "decrease_distance_scale"
            and decision.distance_norm_scale_delta < 0
            for decision in decisions
        ),
        "normalization_erasure_becomes_release_pressure": any(
            decision.release_evidence["contrast_erasure_pressure"] > 0
            for decision in decisions
        ),
        "clipping_and_control_become_restraint_pressure": any(
            decision.restraint_evidence["clipping_saturation_pressure"] > 0
            and decision.restraint_evidence["control_dominance_pressure"] > 0
            for decision in decisions
        ),
        "ordinary_distance_risk_does_not_abort": all(
            decision.decision != "hard_stop_hold"
            for decision in decisions
            if not decision.attention_safety_evidence["hard_stop_triggered"]
        ),
        "future_leak_is_validity_hard_stop": (
            DistanceScaleController(config).update(
                DistanceScaleObservation(
                    step=999,
                    pre_norm_distance_contrast=0.60,
                    post_norm_distance_contrast=0.12,
                    distance_std_pre_norm=1.20,
                    distance_std_post_norm=0.40,
                    clipping_saturation_rate=0.02,
                    geo_to_qk_ratio=0.02,
                    attention_entropy_normalized=0.55,
                    collapse_risk=0.05,
                    real_distance_advantage=0.20,
                    future_leak_score=0.10,
                )
            ).decision
            == "hard_stop_hold"
        ),
        "replay_records_present": all(
            decision["decision_replay_record"] for decision in decision_rows
        ),
        "parameter_trajectory_present": all(
            decision["parameter_trajectory"] for decision in decision_rows
        ),
        "injected_evidence_ledger_present": all(
            decision["injected_evidence_ledger"] for decision in decision_rows
        ),
    }
    status = "pass" if all(checks.values()) else "fail"

    return {
        "stage": "stage13_normalization_and_distance_scale_controller",
        "status": status,
        "input_source": input_source,
        "scientific_scope": (
            "distance-scale search controller; contrast erasure, clipping, "
            "control dominance, and attention safety become replayable evidence"
        ),
        "controller_design": "evidence_balance_normalization_distance_scale",
        "config": asdict(config),
        "required_outputs": list(REQUIRED_DISTANCE_SCALE_OUTPUTS),
        "checks": checks,
        "summary": controller.summary(),
        "decisions": decision_rows,
        "next_required_step": (
            "joint_parameter_budget_allocator"
            if status == "pass"
            else "fix_normalization_and_distance_scale_controller"
        ),
    }


def _synthetic_observations() -> list[DistanceScaleObservation]:
    return [
        DistanceScaleObservation(
            step=100,
            pre_norm_distance_contrast=0.60,
            post_norm_distance_contrast=0.12,
            distance_std_pre_norm=1.10,
            distance_std_post_norm=0.35,
            clipping_saturation_rate=0.03,
            geo_to_qk_ratio=0.015,
            attention_entropy_normalized=0.58,
            collapse_risk=0.05,
            real_distance_advantage=0.22,
            random_distance_advantage=0.03,
            shuffled_distance_advantage=0.02,
        ),
        DistanceScaleObservation(
            step=200,
            pre_norm_distance_contrast=0.08,
            post_norm_distance_contrast=0.04,
            distance_std_pre_norm=0.16,
            distance_std_post_norm=2.20,
            clipping_saturation_rate=0.46,
            geo_to_qk_ratio=0.26,
            attention_entropy_normalized=0.95,
            collapse_risk=0.55,
            real_distance_advantage=0.02,
            random_distance_advantage=0.12,
            shuffled_distance_advantage=0.10,
        ),
        DistanceScaleObservation(
            step=300,
            pre_norm_distance_contrast=0.42,
            post_norm_distance_contrast=0.34,
            distance_std_pre_norm=0.90,
            distance_std_post_norm=0.90,
            clipping_saturation_rate=0.05,
            geo_to_qk_ratio=0.08,
            attention_entropy_normalized=0.70,
            collapse_risk=0.08,
            real_distance_advantage=0.04,
            random_distance_advantage=0.03,
            shuffled_distance_advantage=0.02,
        ),
    ]
