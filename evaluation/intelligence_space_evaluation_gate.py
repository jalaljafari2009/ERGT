"""Readiness gate for Intelligence Space Evaluation."""

from __future__ import annotations

from typing import Any, Literal

IntelligenceSpaceGateStatus = Literal["blocked", "ready"]

INTELLIGENCE_SPACE_DEFINITION = (
    "stable, compressible, causal, reconstructible relational geometry over hidden states"
)

INTELLIGENCE_AXES = [
    "discovery",
    "compression",
    "stabilization",
    "traversal",
]

AXIS_METRICS = {
    "discovery": [
        "real_relation_separation_from_random",
        "real_relation_separation_from_shuffled",
        "task_relevant_edge_discovery",
    ],
    "compression": [
        "effective_rank_without_collapse",
        "spectral_concentration_without_entropy_collapse",
        "anti_collapse_score",
    ],
    "stabilization": [
        "layer_to_layer_relation_persistence",
        "memory_stability_over_steps",
        "paraphrase_path_stability",
    ],
    "traversal": [
        "reasoning_path_task_alignment",
        "multi_hop_path_consistency",
        "counterfactual_edge_removal_sensitivity",
    ],
}

REQUIRED_INTELLIGENCE_CONTROLS = [
    "transformer_baseline",
    "random_geometry",
    "shuffled_geometry",
    "no_memory_geometry",
    "pairwise_geometry",
    "matched_parameter_baseline",
    "replicated_seeds",
    "replicated_datasets",
]


def build_intelligence_space_evaluation_gate_report(
    reasoning_path_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the gate decision before Intelligence Space Evaluation can run."""

    reasoning_ready = (
        bool(reasoning_path_report)
        and reasoning_path_report.get("status") == "ready"
        and reasoning_path_report.get("evaluation_enabled") is True
    )
    blocked_by = []
    if reasoning_path_report is None:
        blocked_by.append("missing_reasoning_path_gate_report")
    elif not reasoning_ready:
        blocked_by.append("reasoning_path_evaluation_not_ready")

    status: IntelligenceSpaceGateStatus = "ready" if not blocked_by else "blocked"
    evaluation_enabled = status == "ready"

    return {
        "phase": "phase12_intelligence_space_evaluation_gate",
        "status": status,
        "evaluation_enabled": evaluation_enabled,
        "definition": INTELLIGENCE_SPACE_DEFINITION,
        "operational_intelligence_definition": (
            "discovery + compression + stabilization + traversal of relational structures"
        ),
        "axes": INTELLIGENCE_AXES,
        "axis_metrics": AXIS_METRICS,
        "required_controls": REQUIRED_INTELLIGENCE_CONTROLS,
        "source_reasoning_path": {
            "phase": reasoning_path_report.get("phase") if reasoning_path_report else None,
            "status": reasoning_path_report.get("status") if reasoning_path_report else None,
            "evaluation_enabled": reasoning_path_report.get("evaluation_enabled")
            if reasoning_path_report
            else False,
            "next_required_step": reasoning_path_report.get("next_required_step")
            if reasoning_path_report
            else "reasoning_path_evaluation_gate",
        },
        "checks": {
            "reasoning_path_evaluation_ready": reasoning_ready,
            "intelligence_evaluation_not_run_before_reasoning": not evaluation_enabled,
            "all_axes_declared": set(INTELLIGENCE_AXES) == set(AXIS_METRICS),
            "controls_declared": bool(REQUIRED_INTELLIGENCE_CONTROLS),
            "claim_boundary_declared": True,
        },
        "blocked_by": blocked_by,
        "evaluation_contract": {
            "positive_evidence": (
                "real relational structures self-organize, stay reconstructible, "
                "improve reasoning-specific tasks, and survive controls"
            ),
            "replication_rule": "effects must replicate across seeds and datasets",
            "control_rule": (
                "random, shuffled, no-memory, pairwise, and matched-parameter controls "
                "must not explain the same gains"
            ),
            "claim_boundary": (
                "this evaluation can support an ERGT intelligence-space claim; it does "
                "not prove consciousness or general intelligence"
            ),
        },
        "next_required_step": (
            "implement_intelligence_space_evaluation"
            if evaluation_enabled
            else _blocked_next_step(reasoning_path_report)
        ),
    }


def _blocked_next_step(reasoning_path_report: dict[str, Any] | None) -> str:
    if reasoning_path_report is None:
        return "reasoning_path_evaluation_gate"
    return str(
        reasoning_path_report.get(
            "next_required_step",
            "reasoning_path_evaluation_gate",
        )
    )
