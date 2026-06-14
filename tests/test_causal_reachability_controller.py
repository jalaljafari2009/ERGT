import json

from evaluation.causal_reachability_controller import (
    REQUIRED_CAUSAL_REACHABILITY_OUTPUTS,
    build_causal_reachability_controller_report,
)
from experiments.causal_reachability_controller import (
    CausalReachabilityConfig,
    CausalReachabilityController,
    CausalReachabilityObservation,
)
from experiments.progress_logging import format_progress_line


def test_causal_reachability_controller_expands_when_reach_is_too_tight() -> None:
    controller = CausalReachabilityController()

    decision = controller.update(
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
        )
    )

    assert decision.decision == "expand_reach"
    assert decision.causal_reachability_delta > 0
    assert decision.memory_readiness_evidence["memory_ready"] is True
    assert decision.attention_locality_spread_evidence["attention_too_local"] is True
    assert decision.parameter_trajectory["causal_reachability_next"] == (
        decision.current_causal_reachability
    )
    assert decision.decision_replay_record["decision"] == "expand_reach"


def test_causal_reachability_controller_restrains_when_noisy_or_collapsing() -> None:
    controller = CausalReachabilityController(
        CausalReachabilityConfig(initial_max_causal_step=3)
    )

    decision = controller.update(
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
        )
    )

    assert decision.decision == "restrain_reach"
    assert decision.causal_reachability_delta < 0
    assert decision.control_reach_evidence["control_dominates_real"] is True
    assert decision.attention_locality_spread_evidence["diversity_low"] is True
    assert decision.restraint_evidence["control_reach_noise_pressure"] > 0


def test_causal_reachability_controller_uses_future_leak_as_hard_stop() -> None:
    controller = CausalReachabilityController()

    decision = controller.update(
        CausalReachabilityObservation(
            step=300,
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
            future_leak_score=0.01,
        )
    )

    assert decision.decision == "hard_stop_hold"
    assert decision.causal_reachability_delta == 0
    assert decision.future_leak_evidence["hard_stop_triggered"] is True


def test_causal_reachability_controller_respects_bounds() -> None:
    controller = CausalReachabilityController(
        CausalReachabilityConfig(
            initial_max_causal_step=8,
            max_max_causal_step=8,
        )
    )

    decision = controller.update(
        CausalReachabilityObservation(
            step=400,
            memory_stability=0.90,
            causal_edge_survival=0.10,
            reach_starvation_score=0.90,
            attention_locality_score=0.90,
            attention_entropy_normalized=0.50,
            head_attention_diversity=0.35,
            layer_attention_diversity=0.30,
            collapse_risk=0.05,
            control_reach_noise_score=0.05,
            control_attention_separation=0.30,
            real_reach_advantage=0.30,
            random_reach_advantage=0.01,
            shuffled_reach_advantage=0.01,
        )
    )

    assert decision.current_causal_reachability <= controller.config.max_max_causal_step
    assert decision.causal_reachability_delta == 0


def test_causal_reachability_controller_report_passes() -> None:
    report = build_causal_reachability_controller_report()

    assert report["status"] == "pass"
    assert report["next_required_step"] == "normalization_and_distance_scale_controller"
    assert report["checks"]["required_causal_reachability_outputs_emitted"]
    assert report["checks"]["schema_declares_causal_reachability_fields"]
    assert report["checks"]["can_expand_when_reach_is_too_tight"]
    assert report["checks"]["can_restrain_when_reach_is_noisy_or_collapsing"]
    assert set(REQUIRED_CAUSAL_REACHABILITY_OUTPUTS).issubset(report["decisions"][-1])
    json.dumps(report)


def test_progress_line_includes_causal_reachability_fields() -> None:
    line = format_progress_line(
        {
            "condition": "real_memory_d_adaptive",
            "step": 100,
            "validation_loss": 5.0,
            "causal_reachability": 1,
            "causal_reachability_next": 2,
            "causal_reachability_delta": 1,
            "causal_reachability_decision": "expand_reach",
            "causal_reachability_credit": 0.40,
            "causal_reachability_risk_pressure": 0.10,
            "causal_edge_survival": 0.18,
            "reach_starvation_score": 0.78,
            "control_reach_noise_score": 0.08,
            "attention_locality_score": 0.86,
            "attention_spread_score": 0.14,
            "future_leak_score": 0.00,
        }
    )

    assert "reach_decision=expand_reach" in line
    assert "reach=1" in line
    assert "r_next=2" in line
    assert "d_reach=1" in line
    assert "rCredit=0.400" in line
    assert "cSurv=0.180" in line
    assert "cNoise=0.080" in line
