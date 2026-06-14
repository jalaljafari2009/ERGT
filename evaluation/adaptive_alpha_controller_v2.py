"""Stage-9 Adaptive Alpha Controller v2 report."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from experiments.adaptive_alpha_v2 import (
    AdaptiveAlphaControllerV2,
    AdaptiveAlphaV2Config,
    AlphaV2Observation,
)

REQUIRED_ALPHA_V2_OUTPUTS = [
    "current_alpha",
    "proposed_alpha",
    "alpha_delta",
    "release_evidence",
    "restraint_evidence",
    "slope_evidence",
    "rigidity_evidence",
    "control_family_evidence",
    "parameter_trajectory",
    "injected_evidence_ledger",
    "controller_state_snapshot",
    "decision_replay_record",
]


def build_adaptive_alpha_controller_v2_report(
    *,
    observations: list[AlphaV2Observation] | None = None,
    config: AdaptiveAlphaV2Config | None = None,
) -> dict[str, Any]:
    """Build the stage-9 mechanics report for alpha controller v2."""

    config = config or AdaptiveAlphaV2Config(initial_alpha=0.02)
    config.validate()
    input_source = "provided_observations"
    if observations is None:
        observations = _synthetic_observations()
        input_source = "synthetic_alpha_v2_smoke"

    controller = AdaptiveAlphaControllerV2(config)
    decisions = [controller.update(observation) for observation in observations]
    decision_rows = [asdict(decision) for decision in decisions]
    latest = decision_rows[-1]
    checks = {
        "required_alpha_outputs_emitted": all(
            field in latest for field in REQUIRED_ALPHA_V2_OUTPUTS
        ),
        "uses_pid_error_terms": all(
            field in latest["controller_state_snapshot"]
            for field in ["error", "integral_error", "derivative_error"]
        ),
        "release_growth_possible": any(
            decision.alpha_delta > 0 and decision.decision in {"grow_pid_release", "probe_up"}
            for decision in decisions
        ),
        "restraint_shrink_possible": any(
            decision.alpha_delta < 0
            and decision.decision in {"shrink_pid_restraint", "probe_down"}
            for decision in decisions
        ),
        "ordinary_risk_does_not_abort": all(
            decision.decision != "hard_stop_hold"
            for decision in decisions
            if not decision.rigidity_evidence["hard_stop_triggered"]
        ),
        "control_family_can_restrain_growth": any(
            decision.control_family_evidence["control_dominates_real"]
            and decision.alpha_delta <= 0
            for decision in decisions
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
        "stage": "stage9_adaptive_alpha_controller_v2",
        "status": status,
        "input_source": input_source,
        "scientific_scope": (
            "alpha search controller; ordinary risk signals become pressure, "
            "while every decision remains replayable"
        ),
        "controller_design": "pid_inspired_evidence_balance",
        "config": asdict(config),
        "required_outputs": list(REQUIRED_ALPHA_V2_OUTPUTS),
        "checks": checks,
        "summary": controller.summary(),
        "decisions": decision_rows,
        "next_required_step": (
            "adaptive_memory_controller" if status == "pass" else "fix_adaptive_alpha_v2"
        ),
    }


def _synthetic_observations() -> list[AlphaV2Observation]:
    return [
        AlphaV2Observation(
            step=100,
            loss_slope_gain=0.12,
            ema_loss_delta=0.08,
            late_window_slope=-0.01,
            post_1000_trend="improving",
            rigidity_risk=0.05,
            collapse_risk=0.02,
            control_penalty=0.0,
            random_loss_slope_gain=0.02,
            shuffled_loss_slope_gain=0.01,
            geo_to_qk_ratio=0.06,
        ),
        AlphaV2Observation(
            step=200,
            loss_slope_gain=0.10,
            ema_loss_delta=0.07,
            late_window_slope=-0.008,
            post_1000_trend="improving",
            rigidity_risk=0.10,
            collapse_risk=0.05,
            control_penalty=0.02,
            random_loss_slope_gain=0.03,
            shuffled_loss_slope_gain=0.02,
            geo_to_qk_ratio=0.09,
        ),
        AlphaV2Observation(
            step=300,
            loss_slope_gain=0.01,
            ema_loss_delta=0.01,
            late_window_slope=0.002,
            post_1000_trend="flat",
            rigidity_risk=0.20,
            collapse_risk=0.10,
            control_penalty=0.10,
            random_loss_slope_gain=0.08,
            shuffled_loss_slope_gain=0.06,
            geo_to_qk_ratio=0.18,
        ),
    ]
