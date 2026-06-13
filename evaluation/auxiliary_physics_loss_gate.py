"""Readiness gate for the Auxiliary Physics-Inspired Loss phase."""

from __future__ import annotations

from typing import Any, Literal

AuxiliaryLossGateStatus = Literal["blocked", "ready"]

REQUIRED_GEOATTENTION_V2_STRICT_CHECKS = [
    "real_stable_beats_random",
    "real_stable_beats_shuffled",
    "real_stable_beats_instantaneous",
    "real_stable_beats_pairwise",
    "real_stable_beats_no_memory",
]

ALLOWED_REGULARIZERS = [
    "spectral_stability",
    "neighborhood_stability",
    "reconstruction_consistency",
    "causal_consistency",
    "anti_collapse",
]


def build_auxiliary_physics_loss_gate_report(
    geoattention_v2_report: dict[str, Any],
    *,
    requested_lambda: float = 0.01,
) -> dict[str, Any]:
    """Return the gate decision before any auxiliary loss can be enabled."""

    if requested_lambda < 0:
        raise ValueError("requested_lambda must be non-negative")

    mechanics_passed = geoattention_v2_report.get("status") == "pass"
    strict_gate_status = geoattention_v2_report.get("strict_gate_status")
    strict_checks = geoattention_v2_report.get("strict_gate_checks", {})
    strict_checks_passed = {
        key: bool(strict_checks.get(key, False))
        for key in REQUIRED_GEOATTENTION_V2_STRICT_CHECKS
    }
    missing_evidence = [
        key
        for key, passed in strict_checks_passed.items()
        if not passed
    ]
    ready = mechanics_passed and strict_gate_status == "pass" and not missing_evidence
    status: AuxiliaryLossGateStatus = "ready" if ready else "blocked"

    return {
        "phase": "phase9_auxiliary_physics_loss_gate",
        "status": status,
        "loss_enabled": ready,
        "requested_lambda": requested_lambda,
        "effective_lambda": requested_lambda if ready else 0.0,
        "loss_formula": "L = L_lm + lambda * physics_regularizer",
        "allowed_regularizers": ALLOWED_REGULARIZERS,
        "forbidden_until_ready": [
            "training_loss_integration",
            "validation_claims_from_regularizer",
            "auxiliary_loss_sweep",
        ],
        "source_geoattention_v2": {
            "phase": geoattention_v2_report.get("phase"),
            "status": geoattention_v2_report.get("status"),
            "strict_gate_status": strict_gate_status,
            "next_required_step": geoattention_v2_report.get("next_required_step"),
        },
        "checks": {
            "geoattention_v2_mechanics_passed": mechanics_passed,
            "geoattention_v2_strict_gate_passed": strict_gate_status == "pass",
            "all_required_control_wins_present": not missing_evidence,
            "loss_not_integrated_before_gate": not ready,
            "lambda_zeroed_while_blocked": (requested_lambda if ready else 0.0) == 0.0
            if not ready
            else True,
        },
        "strict_gate_checks": strict_checks_passed,
        "missing_evidence": missing_evidence,
        "regularizer_contract": {
            "spectral_stability": (
                "penalize abrupt spectral drift; do not minimize entropy unboundedly"
            ),
            "neighborhood_stability": (
                "encourage nearby-layer neighborhood persistence without single-token lock"
            ),
            "reconstruction_consistency": (
                "reward context-reconstructible relations; forbid future-token input"
            ),
            "causal_consistency": "penalize future edges or non-causal support",
            "anti_collapse": (
                "penalize uniform W, over-sparsity, diagonal domination, and rigid collapse"
            ),
        },
        "next_required_step": (
            "implement_auxiliary_physics_loss"
            if ready
            else "run_geoattention_v2_training_controls"
        ),
    }
