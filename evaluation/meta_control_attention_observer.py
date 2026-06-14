"""Stage-16 Meta-Control Attention Observer report."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from experiments.meta_control_attention_observer import (
    MetaControlAttentionConfig,
    MetaControlAttentionObserver,
)

REQUIRED_META_CONTROL_OUTPUTS = [
    "meta_control_mode",
    "meta_observer_only",
    "meta_attention_weights",
    "meta_signal_status",
    "meta_available_signal_count",
    "meta_masked_signal_count",
    "evidence_availability_score",
    "pending_control_mask",
    "offline_replay_required",
    "meta_top_signal",
    "meta_suppressed_signal",
    "meta_attention_entropy",
    "meta_attention_entropy_normalized",
    "controller_agreement_score",
    "controller_conflict_score",
    "meta_control_confidence",
    "meta_parameter_allocation",
    "meta_alpha_weight",
    "meta_memory_weight",
    "meta_gate_weight",
    "meta_reach_weight",
    "meta_norm_weight",
    "meta_observer_decision_summary",
    "meta_replay_record",
]


def build_meta_control_attention_observer_report(
    *,
    partial_row: dict[str, Any] | None = None,
    final_row: dict[str, Any] | None = None,
    conflict_row: dict[str, Any] | None = None,
    config: MetaControlAttentionConfig | None = None,
) -> dict[str, Any]:
    """Build the stage-16 mechanics report for meta-control attention."""

    from evaluation.unified_telemetry_schema import build_unified_telemetry_schema_report

    config = config or MetaControlAttentionConfig()
    config.validate()
    partial_row = partial_row or _synthetic_partial_row()
    final_row = final_row or _synthetic_final_row()
    conflict_row = conflict_row or _synthetic_conflict_row()
    observer = MetaControlAttentionObserver(config)
    partial = observer.summary(observer.observe(partial_row))
    final = observer.summary(observer.observe(final_row))
    conflict = observer.summary(observer.observe(conflict_row))
    replay = observer.replay([partial_row, final_row, conflict_row])
    schema_fields = set(build_unified_telemetry_schema_report()["fields"])
    required_schema_fields = {
        "meta_control_mode",
        "meta_observer_only",
        "meta_attention_weights",
        "meta_signal_status",
        "evidence_availability_score",
        "pending_control_mask",
        "offline_replay_required",
        "meta_top_signal",
        "meta_suppressed_signal",
        "meta_attention_entropy",
        "meta_attention_entropy_normalized",
        "controller_agreement_score",
        "controller_conflict_score",
        "meta_control_confidence",
        "meta_alpha_weight",
        "meta_memory_weight",
        "meta_gate_weight",
        "meta_reach_weight",
        "meta_norm_weight",
    }
    checks = {
        "required_meta_control_outputs_emitted": set(REQUIRED_META_CONTROL_OUTPUTS)
        .issubset(partial),
        "schema_declares_meta_control_fields": required_schema_fields.issubset(
            schema_fields
        ),
        "partial_run_masks_pending_control_signals": (
            partial["meta_control_mode"] == "online_partial"
            and partial["pending_control_mask"]
            and partial["offline_replay_required"]
            and partial["meta_signal_status"]["control_separation"]
            == "masked_pending_control"
            and partial["meta_attention_weights"]["control_separation"] == 0.0
        ),
        "partial_run_still_attends_to_available_signals": (
            partial["meta_available_signal_count"] > 0
            and partial["meta_attention_weights"]["baseline_delta"] > 0
            and partial["meta_top_signal"] != "none"
        ),
        "final_replay_uses_control_separation_when_available": (
            final["meta_control_mode"] == "offline_matched_replay"
            and not final["pending_control_mask"]
            and final["meta_signal_status"]["control_separation"] == "available"
            and final["meta_attention_weights"]["control_separation"] > 0
        ),
        "observer_only_never_emits_actuator_delta": (
            partial["meta_observer_only"]
            and final["meta_observer_only"]
            and conflict["meta_observer_only"]
        ),
        "parameter_allocation_sums_to_one": all(
            abs(sum(item["meta_parameter_allocation"].values()) - 1.0) < 1e-9
            for item in [partial, final, conflict]
        ),
        "attention_weights_sum_to_one_for_available_signals": all(
            abs(sum(item["meta_attention_weights"].values()) - 1.0) < 1e-9
            for item in [partial, final, conflict]
        ),
        "conflict_reduces_confidence": (
            conflict["controller_conflict_score"] > final["controller_conflict_score"]
            and conflict["meta_control_confidence"] < final["meta_control_confidence"]
        ),
        "replay_records_present": all(item["meta_replay_record"] for item in replay),
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "stage": "stage16_meta_control_attention_observer",
        "status": status,
        "scientific_scope": (
            "observer-only missing-aware attention over controller evidence; "
            "pending controls are masked until matched offline replay"
        ),
        "config": asdict(config),
        "required_outputs": list(REQUIRED_META_CONTROL_OUTPUTS),
        "checks": checks,
        "partial_live_example": partial,
        "offline_matched_example": final,
        "conflict_example": conflict,
        "replay_examples": replay,
        "next_required_step": (
            "open_adaptive_relational_control_trainer"
            if status == "pass"
            else "fix_meta_control_attention_observer"
        ),
    }


def _synthetic_partial_row() -> dict[str, Any]:
    return {
        "step": 1000,
        "condition": "real_memory_d",
        "claim_eligibility": "not_eligible_pending_controls",
        "pending_control_families": ["random", "shuffled", "no_memory", "instantaneous"],
        "real_vs_baseline_delta": 0.11,
        "loss_slope_gain": 0.08,
        "geo_to_qk_ratio": 0.05,
        "rigidity_risk": 0.08,
        "collapse_risk": 0.04,
        "memory_stability": 0.42,
        "memory_persistence": 0.35,
        "noise_risk": 0.20,
        "reach_starvation_score": 0.24,
        "distance_contrast_retention": 0.36,
        "budget_conflict_score": 0.08,
    }


def _synthetic_final_row() -> dict[str, Any]:
    row = _synthetic_partial_row()
    row.update(
        {
            "step": 2000,
            "claim_eligibility": "eligible_complete_controls",
            "pending_control_families": [],
            "control_separation": 0.14,
            "real_vs_random_delta": 0.12,
            "real_vs_shuffled_delta": 0.13,
            "geo_to_qk_ratio": 0.12,
            "memory_stability": 0.58,
            "distance_contrast_retention": 0.52,
            "budget_conflict_score": 0.04,
        }
    )
    return row


def _synthetic_conflict_row() -> dict[str, Any]:
    row = _synthetic_final_row()
    row.update(
        {
            "control_separation": -0.05,
            "real_vs_random_delta": -0.04,
            "budget_conflict_score": 0.70,
            "rigidity_risk": 0.74,
            "collapse_risk": 0.62,
        }
    )
    return row
