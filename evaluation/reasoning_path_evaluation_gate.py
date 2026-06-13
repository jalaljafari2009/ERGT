"""Readiness gate for Reasoning Path Evaluation."""

from __future__ import annotations

from typing import Any, Literal

ReasoningPathGateStatus = Literal["blocked", "ready"]

REASONING_PATH_METRICS = [
    "multi_hop_path_consistency",
    "low_cost_path_answer_alignment",
    "counterfactual_edge_removal_sensitivity",
    "long_range_dependency_score",
    "path_stability_across_paraphrases",
    "compositional_task_performance",
]

REQUIRED_REASONING_CONTROLS = [
    "real_geometry_paths",
    "random_geometry_paths",
    "shuffled_geometry_paths",
    "no_memory_paths",
    "pairwise_paths",
    "edge_removed_counterfactual",
]

REASONING_TASK_FAMILIES = [
    "multi_hop_retrieval",
    "long_range_coreference",
    "compositional_relations",
    "paraphrase_stability",
    "counterfactual_context_edges",
]


def build_reasoning_path_evaluation_gate_report(
    complete_architecture_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the gate decision before reasoning path evaluation can run."""

    architecture_ready = (
        bool(complete_architecture_report)
        and complete_architecture_report.get("status") == "ready"
        and complete_architecture_report.get("architecture_enabled") is True
    )
    blocked_by = []
    if complete_architecture_report is None:
        blocked_by.append("missing_complete_architecture_gate_report")
    elif not architecture_ready:
        blocked_by.append("complete_ergt_architecture_not_ready")

    status: ReasoningPathGateStatus = "ready" if not blocked_by else "blocked"
    evaluation_enabled = status == "ready"

    return {
        "phase": "phase11_reasoning_path_evaluation_gate",
        "status": status,
        "evaluation_enabled": evaluation_enabled,
        "definition": "Reasoning = navigation of stable relational geometry",
        "candidate_metrics": REASONING_PATH_METRICS,
        "required_controls": REQUIRED_REASONING_CONTROLS,
        "task_families": REASONING_TASK_FAMILIES,
        "source_complete_architecture": {
            "phase": complete_architecture_report.get("phase")
            if complete_architecture_report
            else None,
            "status": complete_architecture_report.get("status")
            if complete_architecture_report
            else None,
            "architecture_enabled": complete_architecture_report.get("architecture_enabled")
            if complete_architecture_report
            else False,
            "next_required_step": complete_architecture_report.get("next_required_step")
            if complete_architecture_report
            else "complete_ergt_architecture_gate",
        },
        "checks": {
            "complete_architecture_ready": architecture_ready,
            "reasoning_evaluation_not_run_before_architecture": not evaluation_enabled,
            "controls_declared": bool(REQUIRED_REASONING_CONTROLS),
            "metrics_declared": bool(REASONING_PATH_METRICS),
            "task_families_declared": bool(REASONING_TASK_FAMILIES),
        },
        "blocked_by": blocked_by,
        "evaluation_contract": {
            "path_source": "stable real D_causal paths from the complete architecture",
            "positive_evidence": (
                "real geometry paths are more stable and explanatory than controls"
            ),
            "counterfactual_rule": (
                "removing high-Phi/high-stability edges should harm task behavior"
            ),
            "control_rule": (
                "random, shuffled, no-memory, and pairwise paths must not explain "
                "the same behavior"
            ),
            "claim_boundary": (
                "reasoning-path evidence may support traversal analysis; it does not "
                "prove general intelligence"
            ),
        },
        "next_required_step": (
            "implement_reasoning_path_evaluation"
            if evaluation_enabled
            else _blocked_next_step(complete_architecture_report)
        ),
    }


def _blocked_next_step(complete_architecture_report: dict[str, Any] | None) -> str:
    if complete_architecture_report is None:
        return "complete_ergt_architecture_gate"
    return str(
        complete_architecture_report.get(
            "next_required_step",
            "complete_ergt_architecture_gate",
        )
    )
