"""Stage-10 Adaptive Memory Controller report."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from experiments.adaptive_memory_controller import (
    AdaptiveMemoryConfig,
    AdaptiveMemoryController,
    MemoryObservation,
)

REQUIRED_MEMORY_CONTROLLER_OUTPUTS = [
    "current_eta",
    "current_decay",
    "proposed_eta",
    "proposed_decay",
    "eta_delta",
    "decay_delta",
    "memory_stability_trend",
    "memory_turnover_trend",
    "persistence_trend",
    "noise_evidence",
    "rigidity_evidence",
    "release_evidence",
    "restraint_evidence",
    "parameter_trajectory",
    "injected_evidence_ledger",
    "controller_state_snapshot",
    "decision_replay_record",
]


def build_adaptive_memory_controller_report(
    *,
    observations: list[MemoryObservation] | None = None,
    config: AdaptiveMemoryConfig | None = None,
) -> dict[str, Any]:
    """Build the stage-10 mechanics report for memory eta/decay control."""

    config = config or AdaptiveMemoryConfig()
    config.validate()
    input_source = "provided_observations"
    if observations is None:
        observations = _synthetic_observations()
        input_source = "synthetic_memory_controller_smoke"

    controller = AdaptiveMemoryController(config)
    decisions = [controller.update(observation) for observation in observations]
    decision_rows = [asdict(decision) for decision in decisions]
    latest = decision_rows[-1]
    checks = {
        "required_memory_controller_outputs_emitted": all(
            field in latest for field in REQUIRED_MEMORY_CONTROLLER_OUTPUTS
        ),
        "uses_controller_error_terms": all(
            field in latest["controller_state_snapshot"]
            for field in ["error", "integral_error", "derivative_error"]
        ),
        "can_increase_eta_when_memory_is_starved": any(
            decision.decision == "inject_memory" and decision.eta_delta > 0
            for decision in decisions
        ),
        "can_smooth_when_memory_is_noisy": any(
            decision.decision == "smooth_memory"
            and decision.eta_delta < 0
            and decision.decay_delta > 0
            for decision in decisions
        ),
        "can_release_rigid_memory": any(
            decision.decision == "release_rigid_memory"
            and decision.eta_delta < 0
            and decision.decay_delta < 0
            for decision in decisions
        ),
        "ordinary_risk_does_not_abort": all(
            decision.decision != "hard_stop_hold"
            for decision in decisions
            if not decision.rigidity_evidence["hard_stop_triggered"]
        ),
        "future_leak_is_validity_hard_stop": (
            AdaptiveMemoryController(config).update(
                MemoryObservation(
                    step=999,
                    memory_stability=0.8,
                    memory_turnover=0.02,
                    memory_persistence=0.8,
                    memory_edge_density=0.5,
                    memory_rigidity=0.1,
                    noise_risk=0.1,
                    real_memory_advantage=0.2,
                    future_leak_score=0.1,
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
        "stage": "stage10_adaptive_memory_controller",
        "status": status,
        "input_source": input_source,
        "scientific_scope": (
            "memory eta/decay search controller; ordinary memory risk signals "
            "become pressure while future leak remains a validity hard stop"
        ),
        "controller_design": "evidence_balance_memory_eta_decay",
        "config": asdict(config),
        "required_outputs": list(REQUIRED_MEMORY_CONTROLLER_OUTPUTS),
        "checks": checks,
        "summary": controller.summary(),
        "decisions": decision_rows,
        "next_required_step": (
            "gate_floor_and_noise_controller"
            if status == "pass"
            else "fix_adaptive_memory_controller"
        ),
    }


def _synthetic_observations() -> list[MemoryObservation]:
    return [
        MemoryObservation(
            step=100,
            memory_stability=0.42,
            memory_turnover=0.03,
            memory_persistence=0.35,
            memory_edge_density=0.06,
            memory_rigidity=0.12,
            noise_risk=0.10,
            real_memory_advantage=0.18,
            random_memory_advantage=0.02,
            shuffled_memory_advantage=0.01,
        ),
        MemoryObservation(
            step=200,
            memory_stability=0.48,
            memory_turnover=0.22,
            memory_persistence=0.38,
            memory_edge_density=0.24,
            memory_rigidity=0.18,
            noise_risk=0.72,
            real_memory_advantage=0.04,
            random_memory_advantage=0.09,
            shuffled_memory_advantage=0.07,
        ),
        MemoryObservation(
            step=300,
            memory_stability=0.90,
            memory_turnover=0.01,
            memory_persistence=0.93,
            memory_edge_density=0.82,
            memory_rigidity=0.78,
            noise_risk=0.24,
            real_memory_advantage=0.06,
            random_memory_advantage=0.03,
            shuffled_memory_advantage=0.02,
        ),
    ]
