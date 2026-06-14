"""Stage-14 Joint Parameter Budget Allocator report."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from experiments.joint_parameter_budget_allocator import (
    JointBudgetConfig,
    JointBudgetObservation,
    JointParameterBudgetAllocator,
    ParameterBudgetRequest,
)

REQUIRED_JOINT_BUDGET_OUTPUTS = [
    "change_budget",
    "allocated_change_budget",
    "geometry_budget",
    "memory_budget",
    "rigidity_budget",
    "noise_budget",
    "qk_competition_state",
    "release_allocation",
    "restraint_allocation",
    "attention_behavior_regime",
    "attention_derived_budget_pressure",
    "budget_conflict_score",
    "budget_allocation",
    "budget_suppression_reasons",
    "parameter_trajectory",
    "injected_evidence_ledger",
    "controller_state_snapshot",
    "decision_replay_record",
]


def build_joint_parameter_budget_allocator_report(
    *,
    observations: list[JointBudgetObservation] | None = None,
    config: JointBudgetConfig | None = None,
) -> dict[str, Any]:
    """Build the stage-14 mechanics report for joint parameter budgeting."""

    from evaluation.unified_telemetry_schema import build_unified_telemetry_schema_report

    config = config or JointBudgetConfig()
    config.validate()
    input_source = "provided_observations"
    if observations is None:
        observations = _synthetic_observations()
        input_source = "synthetic_joint_budget_allocator_smoke"

    allocator = JointParameterBudgetAllocator(config)
    decisions = [allocator.allocate(observation) for observation in observations]
    decision_rows = [asdict(decision) for decision in decisions]
    latest = decision_rows[-1]
    schema_fields = set(build_unified_telemetry_schema_report()["fields"])
    checks = {
        "required_joint_budget_outputs_emitted": all(
            field in latest for field in REQUIRED_JOINT_BUDGET_OUTPUTS
        ),
        "schema_declares_joint_budget_fields": {
            "change_budget",
            "allocated_change_budget",
            "geometry_budget",
            "memory_budget",
            "rigidity_budget",
            "noise_budget",
            "qk_competition_state",
            "attention_behavior_regime",
            "attention_derived_budget_pressure",
            "budget_conflict_score",
            "budget_allocation",
            "budget_suppression_reasons",
        }.issubset(schema_fields),
        "allocation_respects_total_change_budget": all(
            decision.allocated_change_budget <= config.total_change_budget + 1e-12
            for decision in decisions
        ),
        "can_prioritize_geometry_when_geometry_is_supported": (
            decisions[0].geometry_budget > decisions[0].memory_budget
            and decisions[0].decision == "allocate_budget"
        ),
        "can_shift_budget_to_memory_when_edges_are_noisy": (
            decisions[1].memory_budget > decisions[1].geometry_budget
            and decisions[1].decision == "allocate_budget"
        ),
        "suppresses_geometry_growth_when_attention_or_controls_disagree": (
            decisions[2].geometry_budget == 0
            and any(
                "geometry_growth_suppressed_by_attention_or_controls" in reasons
                for reasons in decisions[2].budget_suppression_reasons.values()
            )
        ),
        "future_leak_is_validity_hard_stop": (
            JointParameterBudgetAllocator(config).allocate(_hard_stop_observation()).decision
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
        "stage": "stage14_joint_parameter_budget_allocator",
        "status": status,
        "input_source": input_source,
        "scientific_scope": (
            "joint adaptive budget allocator; coordinates alpha, memory, gate, "
            "reachability, and distance scale before trainer application"
        ),
        "controller_design": "evidence_weighted_joint_budget_allocator",
        "config": asdict(config),
        "required_outputs": list(REQUIRED_JOINT_BUDGET_OUTPUTS),
        "checks": checks,
        "summary": allocator.summary(),
        "decisions": decision_rows,
        "next_required_step": (
            "control_separation_scoring"
            if status == "pass"
            else "fix_joint_parameter_budget_allocator"
        ),
    }


def _synthetic_observations() -> list[JointBudgetObservation]:
    return [
        JointBudgetObservation(
            step=100,
            parameter_requests={
                "alpha": ParameterBudgetRequest(0.10, 0.90, 0.05, 0.10),
                "distance_norm_scale": ParameterBudgetRequest(0.10, 0.65, 0.05, 0.08),
                "causal_reachability": ParameterBudgetRequest(0.20, 0.50, 0.05, 0.04),
                "memory_eta": ParameterBudgetRequest(0.04, 0.15, 0.05, 0.01),
            },
            geo_to_qk_ratio=0.06,
            geometry_takeover_score=0.05,
            rigidity_risk=0.05,
            collapse_risk=0.05,
            noise_risk=0.08,
            control_penalty=0.02,
            attention_entropy_normalized=0.62,
            control_separation=0.24,
            attribution_uncertainty=0.05,
        ),
        JointBudgetObservation(
            step=200,
            parameter_requests={
                "alpha": ParameterBudgetRequest(0.10, 0.55, 0.05, 0.04),
                "memory_decay": ParameterBudgetRequest(0.08, 0.20, 0.05, 0.02),
                "gate_floor": ParameterBudgetRequest(0.06, 0.18, 0.05, 0.01),
                "memory_eta": ParameterBudgetRequest(-0.04, -0.10, 0.60, 0.0),
            },
            geo_to_qk_ratio=0.08,
            geometry_takeover_score=0.12,
            rigidity_risk=0.18,
            collapse_risk=0.10,
            noise_risk=0.82,
            control_penalty=0.18,
            attention_entropy_normalized=0.70,
            control_separation=0.02,
            attribution_uncertainty=0.12,
        ),
        JointBudgetObservation(
            step=300,
            parameter_requests={
                "alpha": ParameterBudgetRequest(0.10, 0.90, 0.05, 0.10),
                "distance_norm_scale": ParameterBudgetRequest(0.10, 0.70, 0.05, 0.08),
                "causal_reachability": ParameterBudgetRequest(0.20, 0.70, 0.05, 0.04),
            },
            geo_to_qk_ratio=0.22,
            geometry_takeover_score=0.72,
            rigidity_risk=0.74,
            collapse_risk=0.76,
            noise_risk=0.20,
            control_penalty=0.55,
            attention_entropy_normalized=0.80,
            control_separation=-0.12,
            attribution_uncertainty=0.25,
        ),
    ]


def _hard_stop_observation() -> JointBudgetObservation:
    return JointBudgetObservation(
        step=999,
        parameter_requests={
            "alpha": ParameterBudgetRequest(0.10, 0.90, 0.05, 0.10),
            "memory_eta": ParameterBudgetRequest(0.04, 0.30, 0.05, 0.02),
        },
        geo_to_qk_ratio=0.06,
        geometry_takeover_score=0.05,
        rigidity_risk=0.05,
        collapse_risk=0.05,
        noise_risk=0.08,
        control_penalty=0.02,
        attention_entropy_normalized=0.62,
        control_separation=0.24,
        future_leak_score=0.10,
    )
