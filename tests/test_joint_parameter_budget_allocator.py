import json

from evaluation.joint_parameter_budget_allocator import (
    REQUIRED_JOINT_BUDGET_OUTPUTS,
    build_joint_parameter_budget_allocator_report,
)
from experiments.joint_parameter_budget_allocator import (
    JointBudgetConfig,
    JointBudgetObservation,
    JointParameterBudgetAllocator,
    ParameterBudgetRequest,
)
from experiments.progress_logging import format_progress_line


def test_joint_budget_allocator_prioritizes_geometry_when_supported() -> None:
    allocator = JointParameterBudgetAllocator()

    decision = allocator.allocate(
        JointBudgetObservation(
            step=100,
            parameter_requests={
                "alpha": ParameterBudgetRequest(0.10, 0.90, 0.05, 0.10),
                "distance_norm_scale": ParameterBudgetRequest(0.10, 0.65, 0.05),
                "memory_eta": ParameterBudgetRequest(0.04, 0.15, 0.05),
            },
            geo_to_qk_ratio=0.06,
            geometry_takeover_score=0.05,
            rigidity_risk=0.05,
            collapse_risk=0.05,
            noise_risk=0.08,
            control_penalty=0.02,
            attention_entropy_normalized=0.62,
            control_separation=0.24,
        )
    )

    assert decision.decision == "allocate_budget"
    assert decision.geometry_budget > decision.memory_budget
    assert decision.budget_allocation["alpha"]["allocated_delta"] > 0
    assert decision.parameter_trajectory["alpha_allocated_delta"] > 0
    assert decision.decision_replay_record["decision"] == "allocate_budget"


def test_joint_budget_allocator_shifts_to_memory_when_noise_is_high() -> None:
    allocator = JointParameterBudgetAllocator()

    decision = allocator.allocate(
        JointBudgetObservation(
            step=200,
            parameter_requests={
                "alpha": ParameterBudgetRequest(0.10, 0.55, 0.05, 0.04),
                "memory_decay": ParameterBudgetRequest(0.08, 0.20, 0.05, 0.02),
                "gate_floor": ParameterBudgetRequest(0.06, 0.18, 0.05, 0.01),
                "memory_eta": ParameterBudgetRequest(-0.04, -0.10, 0.60),
            },
            geo_to_qk_ratio=0.08,
            geometry_takeover_score=0.12,
            rigidity_risk=0.18,
            collapse_risk=0.10,
            noise_risk=0.82,
            control_penalty=0.18,
            attention_entropy_normalized=0.70,
            control_separation=0.02,
        )
    )

    assert decision.decision == "allocate_budget"
    assert decision.memory_budget > decision.geometry_budget
    assert decision.budget_allocation["gate_floor"]["allocated_delta"] > 0
    assert decision.budget_allocation["memory_eta"]["allocated_delta"] < 0


def test_joint_budget_allocator_suppresses_geometry_when_attention_or_controls_disagree() -> None:
    allocator = JointParameterBudgetAllocator()

    decision = allocator.allocate(
        JointBudgetObservation(
            step=300,
            parameter_requests={
                "alpha": ParameterBudgetRequest(0.10, 0.90, 0.05, 0.10),
                "distance_norm_scale": ParameterBudgetRequest(0.10, 0.70, 0.05),
                "causal_reachability": ParameterBudgetRequest(0.20, 0.70, 0.05),
            },
            geo_to_qk_ratio=0.22,
            geometry_takeover_score=0.72,
            rigidity_risk=0.74,
            collapse_risk=0.76,
            noise_risk=0.20,
            control_penalty=0.55,
            attention_entropy_normalized=0.80,
            control_separation=-0.12,
        )
    )

    assert decision.geometry_budget == 0
    assert decision.decision == "freeze_all"
    assert any(
        "geometry_growth_suppressed_by_attention_or_controls" in reasons
        for reasons in decision.budget_suppression_reasons.values()
    )


def test_joint_budget_allocator_freezes_on_future_leak() -> None:
    allocator = JointParameterBudgetAllocator()

    decision = allocator.allocate(
        JointBudgetObservation(
            step=400,
            parameter_requests={
                "alpha": ParameterBudgetRequest(0.10, 0.90, 0.05),
                "memory_eta": ParameterBudgetRequest(0.04, 0.30, 0.05),
            },
            geo_to_qk_ratio=0.06,
            geometry_takeover_score=0.05,
            rigidity_risk=0.05,
            collapse_risk=0.05,
            noise_risk=0.08,
            control_penalty=0.02,
            attention_entropy_normalized=0.62,
            control_separation=0.24,
            future_leak_score=0.01,
        )
    )

    assert decision.decision == "hard_stop_hold"
    assert decision.allocated_change_budget == 0
    assert all(
        allocation["allocated_delta"] == 0
        for allocation in decision.budget_allocation.values()
    )


def test_joint_budget_allocator_respects_total_budget() -> None:
    allocator = JointParameterBudgetAllocator(JointBudgetConfig(total_change_budget=0.25))

    decision = allocator.allocate(
        JointBudgetObservation(
            step=500,
            parameter_requests={
                "alpha": ParameterBudgetRequest(1.0, 2.0, 0.0),
                "distance_norm_scale": ParameterBudgetRequest(1.0, 2.0, 0.0),
                "memory_eta": ParameterBudgetRequest(1.0, 2.0, 0.0),
            },
            geo_to_qk_ratio=0.06,
            geometry_takeover_score=0.05,
            rigidity_risk=0.05,
            collapse_risk=0.05,
            noise_risk=0.08,
            control_penalty=0.02,
            attention_entropy_normalized=0.62,
            control_separation=0.24,
        )
    )

    assert decision.allocated_change_budget <= allocator.config.total_change_budget


def test_joint_parameter_budget_allocator_report_passes() -> None:
    report = build_joint_parameter_budget_allocator_report()

    assert report["status"] == "pass"
    assert report["next_required_step"] == "control_separation_scoring"
    assert report["checks"]["required_joint_budget_outputs_emitted"]
    assert report["checks"]["schema_declares_joint_budget_fields"]
    assert report["checks"]["allocation_respects_total_change_budget"]
    assert report["checks"]["can_prioritize_geometry_when_geometry_is_supported"]
    assert report["checks"]["can_shift_budget_to_memory_when_edges_are_noisy"]
    assert set(REQUIRED_JOINT_BUDGET_OUTPUTS).issubset(report["decisions"][-1])
    json.dumps(report)


def test_progress_line_includes_joint_budget_fields() -> None:
    line = format_progress_line(
        {
            "condition": "real_memory_d_adaptive",
            "step": 100,
            "validation_loss": 5.0,
            "joint_budget_decision": "allocate_budget",
            "change_budget": 1.0,
            "allocated_change_budget": 0.24,
            "geometry_budget": 0.18,
            "memory_budget": 0.06,
            "rigidity_budget": 0.05,
            "noise_budget": 0.08,
            "qk_competition_state": "geo_competitive",
            "attention_behavior_regime": "healthy",
            "attention_derived_budget_pressure": 0.05,
            "budget_conflict_score": 0.22,
        }
    )

    assert "budget_decision=allocate_budget" in line
    assert "qk_state=geo_competitive" in line
    assert "attn_regime=healthy" in line
    assert "budget=1.000" in line
    assert "bUsed=0.240" in line
    assert "bGeom=0.180" in line
    assert "bMem=0.060" in line
    assert "bConflict=0.220" in line
