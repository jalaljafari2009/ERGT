"""Readiness gate for integrating the Complete ERGT Architecture."""

from __future__ import annotations

from typing import Any, Literal

CompleteArchitectureStatus = Literal["blocked", "ready"]

REQUIRED_REPORTS = [
    "measurement_contracts",
    "strict_w_controls",
    "relational_field_observer",
    "resonant_response_observer",
    "information_potential_phi",
    "reconstruction_gate",
    "relational_memory_observer",
    "causal_shortest_path_geometry",
    "geoattention_v2",
    "auxiliary_physics_loss_gate",
]

ARCHITECTURE_PIPELINE = [
    "HiddenStates",
    "RelationalGraph",
    "Phi",
    "DynamicGraphMemory",
    "EmergentDistance",
    "CausalGeometry",
    "GeoAttention",
    "PhysicsRegularizer",
]

REQUIRED_COMPARISONS = [
    "transformer_baseline",
    "ergt_without_memory",
    "ergt_with_memory",
    "ergt_with_causal_geometry",
    "ergt_with_spectral_regularization",
    "matched_parameter_gpt_style_baseline",
]


def build_complete_ergt_architecture_gate_report(
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return the architecture-integration decision from upstream phase reports."""

    missing_reports = [name for name in REQUIRED_REPORTS if name not in reports]
    phase_statuses = {
        name: {
            "phase": report.get("phase"),
            "status": report.get("status"),
            "next_required_step": report.get("next_required_step"),
        }
        for name, report in reports.items()
        if name in REQUIRED_REPORTS
    }

    observer_reports = [
        "measurement_contracts",
        "strict_w_controls",
        "relational_field_observer",
        "resonant_response_observer",
        "information_potential_phi",
        "reconstruction_gate",
        "relational_memory_observer",
        "causal_shortest_path_geometry",
    ]
    observer_stack_passed = all(
        _status_is_pass(reports.get(name)) for name in observer_reports
    )
    geoattention_report = reports.get("geoattention_v2", {})
    auxiliary_report = reports.get("auxiliary_physics_loss_gate", {})
    geoattention_mechanics_passed = _status_is_pass(geoattention_report)
    geoattention_strict_gate_passed = geoattention_report.get("strict_gate_status") == "pass"
    auxiliary_loss_ready = auxiliary_report.get("status") == "ready"
    auxiliary_loss_enabled = auxiliary_report.get("loss_enabled") is True

    blocked_by = []
    if missing_reports:
        blocked_by.append("missing_required_reports")
    if not observer_stack_passed:
        blocked_by.append("upstream_observer_stack_not_fully_passed")
    if not geoattention_mechanics_passed:
        blocked_by.append("geoattention_v2_mechanics_not_passed")
    if not geoattention_strict_gate_passed:
        blocked_by.append("geoattention_v2_strict_training_gate_not_passed")
    if not auxiliary_loss_ready or not auxiliary_loss_enabled:
        blocked_by.append("auxiliary_physics_loss_not_ready_or_not_enabled")

    ready = not blocked_by
    status: CompleteArchitectureStatus = "ready" if ready else "blocked"

    return {
        "phase": "phase10_complete_ergt_architecture_gate",
        "status": status,
        "architecture_enabled": ready,
        "architecture_pipeline": ARCHITECTURE_PIPELINE,
        "required_comparisons": REQUIRED_COMPARISONS,
        "checks": {
            "required_reports_present": not missing_reports,
            "observer_stack_passed": observer_stack_passed,
            "geoattention_v2_mechanics_passed": geoattention_mechanics_passed,
            "geoattention_v2_strict_training_gate_passed": geoattention_strict_gate_passed,
            "auxiliary_physics_loss_ready": auxiliary_loss_ready,
            "auxiliary_physics_loss_enabled": auxiliary_loss_enabled,
            "architecture_not_integrated_before_gates": not ready,
        },
        "phase_statuses": phase_statuses,
        "missing_reports": missing_reports,
        "blocked_by": blocked_by,
        "integration_contract": {
            "goal": "integrate only validated components into a single architecture",
            "required_pipeline": " -> ".join(ARCHITECTURE_PIPELINE),
            "capacity_control": (
                "complete architecture must beat matched-parameter controls, not just add capacity"
            ),
            "evidence_rule": (
                "role of W_t and D_causal must remain visible against no-memory, "
                "pairwise, random, shuffled, and matched-parameter baselines"
            ),
        },
        "next_required_step": (
            "implement_complete_ergt_architecture"
            if ready
            else _blocked_next_step(geoattention_report, auxiliary_report)
        ),
    }


def _status_is_pass(report: dict[str, Any] | None) -> bool:
    return bool(report and report.get("status") == "pass")


def _blocked_next_step(
    geoattention_report: dict[str, Any],
    auxiliary_report: dict[str, Any],
) -> str:
    if geoattention_report.get("strict_gate_status") != "pass":
        return "run_geoattention_v2_training_controls"
    if auxiliary_report.get("status") != "ready":
        return "complete_auxiliary_physics_loss_gate"
    return "fix_complete_architecture_prerequisites"
