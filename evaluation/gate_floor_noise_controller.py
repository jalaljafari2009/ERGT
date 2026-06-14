"""Stage-11 Gate-Floor and Noise Controller report."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from experiments.gate_floor_noise_controller import (
    GateFloorNoiseConfig,
    GateFloorNoiseController,
    GateFloorObservation,
)

REQUIRED_GATE_FLOOR_OUTPUTS = [
    "current_gate_floor",
    "proposed_gate_floor",
    "gate_floor_delta",
    "gate_floor_credit",
    "gate_floor_risk_pressure",
    "edge_noise_evidence",
    "starvation_evidence",
    "attention_pressure_evidence",
    "control_attention_evidence",
    "release_evidence",
    "restraint_evidence",
    "parameter_trajectory",
    "injected_evidence_ledger",
    "controller_state_snapshot",
    "decision_replay_record",
]


def build_gate_floor_noise_controller_report(
    *,
    observations: list[GateFloorObservation] | None = None,
    config: GateFloorNoiseConfig | None = None,
) -> dict[str, Any]:
    """Build the stage-11 mechanics report for gate-floor/noise control."""

    from evaluation.unified_telemetry_schema import build_unified_telemetry_schema_report

    config = config or GateFloorNoiseConfig()
    config.validate()
    input_source = "provided_observations"
    if observations is None:
        observations = _synthetic_observations()
        input_source = "synthetic_gate_floor_controller_smoke"

    controller = GateFloorNoiseController(config)
    decisions = [controller.update(observation) for observation in observations]
    decision_rows = [asdict(decision) for decision in decisions]
    latest = decision_rows[-1]
    schema_fields = set(build_unified_telemetry_schema_report()["fields"])
    checks = {
        "required_gate_floor_outputs_emitted": all(
            field in latest for field in REQUIRED_GATE_FLOOR_OUTPUTS
        ),
        "schema_declares_gate_floor_fields": {
            "gate_floor",
            "gate_floor_next",
            "gate_floor_delta",
            "gate_floor_decision",
            "gate_floor_credit",
            "gate_floor_risk_pressure",
            "edge_survival",
            "random_edge_noise_score",
            "real_edge_starvation_score",
        }.issubset(schema_fields),
        "uses_controller_error_terms": all(
            field in latest["controller_state_snapshot"]
            for field in ["error", "integral_error", "derivative_error"]
        ),
        "can_raise_gate_floor_when_edges_are_noisy": any(
            decision.decision == "raise_gate_floor" and decision.gate_floor_delta > 0
            for decision in decisions
        ),
        "can_lower_gate_floor_when_real_edges_are_starved": any(
            decision.decision == "lower_gate_floor" and decision.gate_floor_delta < 0
            for decision in decisions
        ),
        "ordinary_noise_does_not_abort": all(
            decision.decision != "hard_stop_hold"
            for decision in decisions
            if not decision.control_attention_evidence["hard_stop_triggered"]
        ),
        "future_leak_is_validity_hard_stop": (
            GateFloorNoiseController(config).update(
                GateFloorObservation(
                    step=999,
                    memory_edge_density=0.3,
                    edge_survival=0.6,
                    random_edge_noise_score=0.1,
                    shuffled_edge_noise_score=0.1,
                    real_edge_starvation_score=0.1,
                    attention_sparsity_0_01=0.3,
                    attention_entropy_normalized=0.8,
                    control_attention_separation=0.2,
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
        "stage": "stage11_gate_floor_and_noise_controller",
        "status": status,
        "input_source": input_source,
        "scientific_scope": (
            "gate-floor search controller; ordinary edge-noise and starvation "
            "signals become pressure while future leak remains a validity hard stop"
        ),
        "controller_design": "evidence_balance_gate_floor_noise_filter",
        "config": asdict(config),
        "required_outputs": list(REQUIRED_GATE_FLOOR_OUTPUTS),
        "checks": checks,
        "summary": controller.summary(),
        "decisions": decision_rows,
        "next_required_step": (
            "causal_reachability_controller"
            if status == "pass"
            else "fix_gate_floor_noise_controller"
        ),
    }


def _synthetic_observations() -> list[GateFloorObservation]:
    return [
        GateFloorObservation(
            step=100,
            memory_edge_density=0.46,
            edge_survival=0.62,
            random_edge_noise_score=0.72,
            shuffled_edge_noise_score=0.63,
            real_edge_starvation_score=0.08,
            attention_sparsity_0_01=0.38,
            attention_entropy_normalized=0.78,
            control_attention_separation=-0.08,
            collapse_risk=0.08,
        ),
        GateFloorObservation(
            step=200,
            memory_edge_density=0.05,
            edge_survival=0.18,
            random_edge_noise_score=0.12,
            shuffled_edge_noise_score=0.10,
            real_edge_starvation_score=0.74,
            attention_sparsity_0_01=0.94,
            attention_entropy_normalized=0.44,
            control_attention_separation=0.22,
            collapse_risk=0.05,
        ),
        GateFloorObservation(
            step=300,
            memory_edge_density=0.24,
            edge_survival=0.48,
            random_edge_noise_score=0.28,
            shuffled_edge_noise_score=0.22,
            real_edge_starvation_score=0.18,
            attention_sparsity_0_01=0.52,
            attention_entropy_normalized=0.70,
            control_attention_separation=0.18,
            collapse_risk=0.08,
        ),
    ]
