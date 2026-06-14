"""Open-control philosophy contract for post Run-02 ERGT adaptation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

HARD_STOPS = {
    "future_leakage": {
        "reason": "Causal violations invalidate the experiment.",
        "required_action": "abort_and_fix_mask_or_reachability",
    },
    "nan_or_inf": {
        "reason": "Non-finite tensors make the trajectory uninterpretable.",
        "required_action": "abort_and_fix_numerics",
    },
    "loss_explosion": {
        "reason": "A runaway objective is not controlled adaptation.",
        "required_action": "abort_or_rewind_to_last_safe_checkpoint",
    },
    "severe_attention_collapse": {
        "reason": "Complete attention collapse prevents meaningful competition.",
        "required_action": "freeze_or_shrink_geometry_and_record_failure",
    },
    "control_unfairness": {
        "reason": "Real/random/shuffled comparisons are invalid if controls differ.",
        "required_action": "abort_and_rebuild_controls",
    },
}

SOFT_PRESSURES = {
    "geo_to_qk_ratio": {
        "role": "pressure",
        "meaning": "Measures geometry strength relative to QK.",
        "rule": "penalize_risk_without_hard_ceiling",
    },
    "attention_entropy_drop": {
        "role": "pressure",
        "meaning": "Detects narrowing attention.",
        "rule": "penalize_when_not_supported_by_loss_and_controls",
    },
    "mean_max_probability": {
        "role": "pressure",
        "meaning": "Detects excessive attention concentration.",
        "rule": "penalize_growth_when_probability_spikes",
    },
    "memory_turnover": {
        "role": "pressure",
        "meaning": "Detects unstable memory edges.",
        "rule": "shift_budget_toward_decay_or_gate_floor_when_noisy",
    },
    "memory_rigidity": {
        "role": "pressure",
        "meaning": "Detects overly persistent or collapsed memory.",
        "rule": "shift_budget_toward_eta_or_lower_decay_when_rigid",
    },
    "control_penalty": {
        "role": "pressure",
        "meaning": "Prevents random/shuffled gains from being counted as real geometry.",
        "rule": "reduce_claim_credit_when_controls_match_or_beat_real",
    },
}

ADAPTIVE_PARAMETERS = {
    "alpha": {
        "meaning": "Strength of geometry injection into attention logits.",
        "growth_evidence": ["positive_loss_slope_gain", "real_beats_controls"],
        "restraint_evidence": ["attention_collapse_pressure", "control_penalty"],
        "hard_ceiling_is_scientific_claim": False,
    },
    "memory_decay": {
        "meaning": "Persistence of previous relational memory.",
        "growth_evidence": ["stable_edges_help_loss", "low_memory_turnover"],
        "restraint_evidence": ["memory_rigidity", "rank_collapse"],
        "hard_ceiling_is_scientific_claim": False,
    },
    "memory_eta": {
        "meaning": "Rate of accepting new relational evidence.",
        "growth_evidence": ["under_adaptation", "stale_memory"],
        "restraint_evidence": ["high_turnover", "noisy_edges"],
        "hard_ceiling_is_scientific_claim": False,
    },
    "gate_floor": {
        "meaning": "Filtering strength for weak/noisy memory edges.",
        "growth_evidence": ["edge_noise", "random_like_edges"],
        "restraint_evidence": ["relation_starvation", "over_sparse_memory"],
        "hard_ceiling_is_scientific_claim": False,
    },
    "distance_norm_scale": {
        "meaning": "Effective scale of normalized distance before geometry bias.",
        "growth_evidence": ["stable_memory_low_geo_effect"],
        "restraint_evidence": ["distance_scale_dominates_qk", "collapse_pressure"],
        "hard_ceiling_is_scientific_claim": False,
    },
    "causal_reachability": {
        "meaning": "Finite-speed graph reachability radius or policy.",
        "growth_evidence": ["local_geometry_underfits", "stable_longer_edges"],
        "restraint_evidence": ["future_leakage_risk", "noisy_long_edges"],
        "hard_ceiling_is_scientific_claim": False,
    },
}

REQUIRED_TELEMETRY = [
    "validation_loss",
    "baseline_centered_improvement",
    "loss_slope_gain",
    "control_separation",
    "alpha",
    "alpha_delta",
    "memory_decay",
    "memory_eta",
    "gate_floor",
    "distance_norm_scale",
    "causal_reachability",
    "geo_to_qk_ratio",
    "attention_entropy",
    "mean_max_probability",
    "memory_stability",
    "memory_turnover",
    "spectral_entropy",
    "effective_rank",
    "rigidity_risk",
    "noise_risk",
    "control_penalty",
    "control_rng_isolated",
    "attribution_summary",
    "parameter_trajectory",
    "injected_evidence_ledger",
    "controller_state_snapshot",
    "decision_replay_record",
]

CONTROLLER_OBLIGATIONS = [
    "Do not treat a soft pressure as a scientific hard stop.",
    "Do not abort a run for ordinary risk flags; convert them into controller pressure.",
    "Use rolling windows or smoothed trends instead of single-point deltas.",
    "Record every parameter change with a reason and evidence fields.",
    "Record the full parameter trajectory for every adaptive degree of freedom.",
    "Record injected evidence, observations, and controller state so decisions can be replayed.",
    "Let a parameter grow when benefit evidence exceeds risk pressure.",
    "Let a parameter shrink or freeze when risk pressure dominates.",
    "Search for better parameter regions instead of treating early gates as final answers.",
    "Separate runtime safety bounds from scientific claims.",
    "Compare real adaptive behavior against random and shuffled controls.",
    "Keep random and shuffled control RNG isolated from training, dropout, and sampler RNG.",
    "Do not count baseline-only improvement as proof of real relational geometry.",
    "Allocate change budget across parameters instead of changing all knobs blindly.",
]

ANTI_PATTERNS = [
    "fixed_small_alpha_as_default_truth",
    "geo_qk_as_absolute_scientific_ceiling",
    "ordinary_risk_flag_aborts_optimization",
    "single_step_loss_delta_controller",
    "changing_alpha_memory_and_normalization_without_attribution",
    "unlogged_parameter_search",
    "controller_decision_without_replay_record",
    "claiming_real_geometry_when_random_or_shuffled_matches_real",
    "opening_more_parameters_when_run02_evidence_is_incomplete",
]


def build_open_control_philosophy_contract() -> dict[str, Any]:
    """Return the canonical open-control contract."""

    report = {
        "contract": "open_control_philosophy",
        "status": "pass",
        "purpose": (
            "Allow geometry and memory parameters to grow or shrink from evidence "
            "without pre-emptively choking the system with hard scientific ceilings."
        ),
        "hard_stops": deepcopy(HARD_STOPS),
        "soft_pressures": deepcopy(SOFT_PRESSURES),
        "adaptive_parameters": deepcopy(ADAPTIVE_PARAMETERS),
        "required_telemetry": list(REQUIRED_TELEMETRY),
        "controller_obligations": list(CONTROLLER_OBLIGATIONS),
        "anti_patterns": list(ANTI_PATTERNS),
        "checks": {},
        "next_required_step": "unified_telemetry_schema",
    }
    report["checks"] = validate_contract(report)
    if not all(report["checks"].values()):
        report["status"] = "fail"
    return report


def validate_contract(contract: dict[str, Any]) -> dict[str, bool]:
    """Validate the minimum invariants of the open-control contract."""

    hard_stops = contract.get("hard_stops", {})
    soft_pressures = contract.get("soft_pressures", {})
    adaptive_parameters = contract.get("adaptive_parameters", {})
    required_telemetry = set(contract.get("required_telemetry", []))
    obligations = contract.get("controller_obligations", [])

    return {
        "hard_stops_are_limited_to_safety_and_validity": set(hard_stops)
        == set(HARD_STOPS),
        "soft_pressures_are_not_hard_stops": not bool(
            set(hard_stops).intersection(soft_pressures)
        ),
        "adaptive_parameters_declared": set(adaptive_parameters)
        == set(ADAPTIVE_PARAMETERS),
        "every_adaptive_parameter_has_growth_and_restraint": all(
            parameter.get("growth_evidence")
            and parameter.get("restraint_evidence")
            and parameter.get("hard_ceiling_is_scientific_claim") is False
            for parameter in adaptive_parameters.values()
        ),
        "telemetry_covers_loss_attention_memory_controls": {
            "validation_loss",
            "loss_slope_gain",
            "geo_to_qk_ratio",
            "attention_entropy",
            "memory_stability",
            "memory_turnover",
            "control_penalty",
            "attribution_summary",
        }.issubset(required_telemetry),
        "telemetry_records_parameter_search_and_injected_evidence": {
            "parameter_trajectory",
            "injected_evidence_ledger",
            "controller_state_snapshot",
            "decision_replay_record",
        }.issubset(required_telemetry),
        "controller_obligations_prevent_soft_flag_abort": any(
            "ordinary risk flags" in obligation for obligation in obligations
        ),
        "controller_obligations_require_full_parameter_trajectory": any(
            "full parameter trajectory" in obligation for obligation in obligations
        ),
        "controller_obligations_require_decision_replay": any(
            "decisions can be replayed" in obligation for obligation in obligations
        ),
        "controller_obligations_require_search_not_static_gate": any(
            "Search for better parameter regions" in obligation
            for obligation in obligations
        ),
        "controller_obligations_require_trend_not_single_delta": any(
            "single-point" in obligation for obligation in obligations
        ),
        "controller_obligations_require_real_controls": any(
            "random and shuffled" in obligation for obligation in obligations
        ),
        "controller_obligations_require_rng_isolation": any(
            "RNG isolated" in obligation for obligation in obligations
        ),
    }
