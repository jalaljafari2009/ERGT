"""Stage-12 Causal Reachability Controller report."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from experiments.causal_reachability_controller import (
    CausalReachabilityConfig,
    CausalReachabilityController,
    CausalReachabilityObservation,
)

REQUIRED_CAUSAL_REACHABILITY_OUTPUTS = [
    "current_causal_reachability",
    "proposed_causal_reachability",
    "causal_reachability_delta",
    "causal_reachability_credit",
    "causal_reachability_risk_pressure",
    "future_leak_evidence",
    "memory_readiness_evidence",
    "attention_locality_spread_evidence",
    "control_reach_evidence",
    "expansion_evidence",
    "restraint_evidence",
    "parameter_trajectory",
    "injected_evidence_ledger",
    "controller_state_snapshot",
    "decision_replay_record",
]


def build_causal_reachability_controller_report(
    *,
    observations: list[CausalReachabilityObservation] | None = None,
    config: CausalReachabilityConfig | None = None,
) -> dict[str, Any]:
    """Build the stage-12 mechanics report for causal reach control."""

    from evaluation.unified_telemetry_schema import build_unified_telemetry_schema_report

    config = config or CausalReachabilityConfig()
    config.validate()
    input_source = "provided_observations"
    if observations is None:
        observations = _synthetic_observations()
        input_source = "synthetic_causal_reachability_controller_smoke"

    controller = CausalReachabilityController(config)
    decisions = [controller.update(observation) for observation in observations]
    decision_rows = [asdict(decision) for decision in decisions]
    latest = decision_rows[-1]
    schema_fields = set(build_unified_telemetry_schema_report()["fields"])
    checks = {
        "required_causal_reachability_outputs_emitted": all(
            field in latest for field in REQUIRED_CAUSAL_REACHABILITY_OUTPUTS
        ),
        "schema_declares_causal_reachability_fields": {
            "causal_reachability",
            "causal_reachability_next",
            "causal_reachability_delta",
            "causal_reachability_decision",
            "causal_reachability_credit",
            "causal_reachability_risk_pressure",
            "causal_edge_survival",
            "reach_starvation_score",
            "control_reach_noise_score",
            "attention_locality_score",
            "attention_spread_score",
            "future_leak_score",
        }.issubset(schema_fields),
        "uses_controller_error_terms": all(
            field in latest["controller_state_snapshot"]
            for field in ["error", "integral_error", "derivative_error"]
        ),
        "can_expand_when_reach_is_too_tight": any(
            decision.decision == "expand_reach"
            and decision.causal_reachability_delta > 0
            for decision in decisions
        ),
        "can_restrain_when_reach_is_noisy_or_collapsing": any(
            decision.decision == "restrain_reach"
            and decision.causal_reachability_delta < 0
            for decision in decisions
        ),
        "ordinary_reach_risk_does_not_abort": all(
            decision.decision != "hard_stop_hold"
            for decision in decisions
            if not decision.future_leak_evidence["hard_stop_triggered"]
        ),
        "future_leak_is_validity_hard_stop": (
            CausalReachabilityController(config).update(
                CausalReachabilityObservation(
                    step=999,
                    memory_stability=0.85,
                    causal_edge_survival=0.20,
                    reach_starvation_score=0.80,
                    attention_locality_score=0.85,
                    attention_entropy_normalized=0.55,
                    head_attention_diversity=0.30,
                    layer_attention_diversity=0.25,
                    collapse_risk=0.05,
                    control_reach_noise_score=0.05,
                    control_attention_separation=0.20,
                    real_reach_advantage=0.20,
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
        "stage": "stage12_causal_reachability_controller",
        "status": status,
        "input_source": input_source,
        "scientific_scope": (
            "finite-speed causal reach search controller; ordinary reach risks "
            "become pressure while future leak remains a validity hard stop"
        ),
        "controller_design": "evidence_balance_causal_reachability",
        "config": asdict(config),
        "required_outputs": list(REQUIRED_CAUSAL_REACHABILITY_OUTPUTS),
        "checks": checks,
        "summary": controller.summary(),
        "decisions": decision_rows,
        "next_required_step": (
            "normalization_and_distance_scale_controller"
            if status == "pass"
            else "fix_causal_reachability_controller"
        ),
    }


def _synthetic_observations() -> list[CausalReachabilityObservation]:
    return [
        CausalReachabilityObservation(
            step=100,
            memory_stability=0.82,
            causal_edge_survival=0.18,
            reach_starvation_score=0.78,
            attention_locality_score=0.86,
            attention_entropy_normalized=0.58,
            head_attention_diversity=0.28,
            layer_attention_diversity=0.22,
            collapse_risk=0.06,
            control_reach_noise_score=0.08,
            control_attention_separation=0.24,
            real_reach_advantage=0.22,
            random_reach_advantage=0.03,
            shuffled_reach_advantage=0.02,
        ),
        CausalReachabilityObservation(
            step=200,
            memory_stability=0.46,
            causal_edge_survival=0.72,
            reach_starvation_score=0.08,
            attention_locality_score=0.30,
            attention_entropy_normalized=0.95,
            head_attention_diversity=0.04,
            layer_attention_diversity=0.03,
            collapse_risk=0.58,
            control_reach_noise_score=0.74,
            control_attention_separation=-0.12,
            real_reach_advantage=0.02,
            random_reach_advantage=0.12,
            shuffled_reach_advantage=0.10,
        ),
        CausalReachabilityObservation(
            step=300,
            memory_stability=0.70,
            causal_edge_survival=0.46,
            reach_starvation_score=0.16,
            attention_locality_score=0.48,
            attention_entropy_normalized=0.70,
            head_attention_diversity=0.24,
            layer_attention_diversity=0.20,
            collapse_risk=0.08,
            control_reach_noise_score=0.24,
            control_attention_separation=0.18,
            real_reach_advantage=0.08,
            random_reach_advantage=0.03,
            shuffled_reach_advantage=0.02,
        ),
    ]
