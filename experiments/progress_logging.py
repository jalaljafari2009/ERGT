"""Lightweight progress logging for long ERGT experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_for_json(record), sort_keys=True) + "\n")


def gpu_memory_snapshot(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"gpu_memory_gb": None, "gpu_peak_memory_gb": None}
    return {
        "gpu_memory_gb": torch.cuda.memory_allocated(device) / 1024**3,
        "gpu_peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
    }


def geometry_progress_fields(geometry: dict[str, Any] | None) -> dict[str, Any]:
    if not geometry:
        return {}
    summary = geometry.get("summary", geometry)
    if not isinstance(summary, dict):
        return {}

    fields = {
        "alpha_effective": summary.get("alpha"),
        "alpha_warmup_factor": summary.get("alpha_warmup_factor"),
        "target_alpha": summary.get("target_alpha"),
        "qk_mean_abs": summary.get("mean_abs_qk"),
        "geo_mean_abs": summary.get("mean_abs_geo"),
        "geo_to_qk_ratio": summary.get("geo_to_qk_ratio"),
        "distance_norm_scale": summary.get("distance_norm_scale"),
        "distance_norm_scale_next": summary.get("distance_norm_scale_next"),
        "distance_norm_scale_delta": summary.get("distance_norm_scale_delta"),
        "distance_norm_scale_decision": summary.get("distance_norm_scale_decision"),
        "distance_norm_scale_credit": summary.get("distance_norm_scale_credit"),
        "distance_norm_scale_risk_pressure": summary.get(
            "distance_norm_scale_risk_pressure"
        ),
        "pre_norm_distance_contrast": summary.get("pre_norm_distance_contrast"),
        "post_norm_distance_contrast": summary.get("post_norm_distance_contrast"),
        "distance_contrast_retention": summary.get("distance_contrast_retention"),
        "distance_std_pre_norm": summary.get("distance_std_pre_norm"),
        "distance_std_post_norm": summary.get("distance_std_post_norm"),
        "clipping_saturation_rate": summary.get("clipping_saturation_rate"),
        "normalization_erasure_score": summary.get("normalization_erasure_score"),
        "distance_scale_release_pressure": summary.get(
            "distance_scale_release_pressure"
        ),
        "distance_scale_restraint_pressure": summary.get(
            "distance_scale_restraint_pressure"
        ),
        "real_distance_advantage": summary.get("real_distance_advantage"),
        "random_distance_advantage": summary.get("random_distance_advantage"),
        "shuffled_distance_advantage": summary.get("shuffled_distance_advantage"),
        "distance_mean": summary.get("distance_mean"),
        "distance_std": summary.get("distance_std"),
        "control_separation_mode": summary.get("control_separation_mode"),
        "claim_eligibility": summary.get("claim_eligibility"),
        "control_separation_status": summary.get("control_separation_status"),
        "scientific_claim_credit": summary.get("scientific_claim_credit"),
        "real_vs_baseline_delta": summary.get("real_vs_baseline_delta"),
        "real_vs_alpha_zero_delta": summary.get("real_vs_alpha_zero_delta"),
        "real_vs_random_delta": summary.get("real_vs_random_delta"),
        "real_vs_shuffled_delta": summary.get("real_vs_shuffled_delta"),
        "real_vs_no_memory_delta": summary.get("real_vs_no_memory_delta"),
        "real_vs_instantaneous_delta": summary.get("real_vs_instantaneous_delta"),
        "control_separation": summary.get("control_separation"),
        "control_penalty": summary.get("control_penalty"),
        "generic_regularization_warning": summary.get("generic_regularization_warning"),
        "attention_behavior_separation": summary.get("attention_behavior_separation"),
        "meta_control_mode": summary.get("meta_control_mode"),
        "meta_observer_only": summary.get("meta_observer_only"),
        "meta_available_signal_count": summary.get("meta_available_signal_count"),
        "meta_masked_signal_count": summary.get("meta_masked_signal_count"),
        "evidence_availability_score": summary.get("evidence_availability_score"),
        "pending_control_mask": summary.get("pending_control_mask"),
        "offline_replay_required": summary.get("offline_replay_required"),
        "meta_top_signal": summary.get("meta_top_signal"),
        "meta_suppressed_signal": summary.get("meta_suppressed_signal"),
        "meta_attention_entropy": summary.get("meta_attention_entropy"),
        "meta_attention_entropy_normalized": summary.get(
            "meta_attention_entropy_normalized"
        ),
        "controller_agreement_score": summary.get("controller_agreement_score"),
        "controller_conflict_score": summary.get("controller_conflict_score"),
        "meta_control_confidence": summary.get("meta_control_confidence"),
        "meta_alpha_weight": summary.get("meta_alpha_weight"),
        "meta_memory_weight": summary.get("meta_memory_weight"),
        "meta_gate_weight": summary.get("meta_gate_weight"),
        "meta_reach_weight": summary.get("meta_reach_weight"),
        "meta_norm_weight": summary.get("meta_norm_weight"),
        "meta_observer_decision_summary": summary.get(
            "meta_observer_decision_summary"
        ),
        "trainer_status": summary.get("trainer_status"),
        "trainer_event": summary.get("trainer_event"),
        "trainer_fail_fast_triggered": summary.get("trainer_fail_fast_triggered"),
        "trainer_fail_fast_reason": summary.get("trainer_fail_fast_reason"),
        "trainer_processed_rows": summary.get("trainer_processed_rows"),
        "controller_decision_count": summary.get("controller_decision_count"),
        "meta_observer_event_count": summary.get("meta_observer_event_count"),
        "control_separation_event_count": summary.get(
            "control_separation_event_count"
        ),
        "safety_event_count": summary.get("safety_event_count"),
        "live_display_event_count": summary.get("live_display_event_count"),
        "progress_log_ready": summary.get("progress_log_ready"),
        "controller_decision_log_ready": summary.get("controller_decision_log_ready"),
        "meta_control_observer_log_ready": summary.get(
            "meta_control_observer_log_ready"
        ),
        "control_separation_log_ready": summary.get("control_separation_log_ready"),
        "safety_log_ready": summary.get("safety_log_ready"),
        "lightweight_artifact_bundle_ready": summary.get(
            "lightweight_artifact_bundle_ready"
        ),
        "checkpoint_artifacts_excluded": summary.get("checkpoint_artifacts_excluded"),
        "artifact_bundle_name": summary.get("artifact_bundle_name"),
        "live_diagnostic_row_ready": summary.get("live_diagnostic_row_ready"),
        "live_diagnostic_table_ready": summary.get("live_diagnostic_table_ready"),
        "live_diagnostic_plot_ready": summary.get("live_diagnostic_plot_ready"),
        "live_diagnostic_row_count": summary.get("live_diagnostic_row_count"),
        "change_budget": summary.get("change_budget"),
        "allocated_change_budget": summary.get("allocated_change_budget"),
        "geometry_budget": summary.get("geometry_budget"),
        "memory_budget": summary.get("memory_budget"),
        "rigidity_budget": summary.get("rigidity_budget"),
        "noise_budget": summary.get("noise_budget"),
        "qk_competition_state": summary.get("qk_competition_state"),
        "attention_behavior_regime": summary.get("attention_behavior_regime"),
        "attention_derived_budget_pressure": summary.get(
            "attention_derived_budget_pressure"
        ),
        "budget_conflict_score": summary.get("budget_conflict_score"),
        "joint_budget_decision": summary.get("joint_budget_decision"),
        "causal_reachability": summary.get("causal_reachability"),
        "causal_reachability_next": summary.get("causal_reachability_next"),
        "causal_reachability_delta": summary.get("causal_reachability_delta"),
        "causal_reachability_decision": summary.get("causal_reachability_decision"),
        "causal_reachability_credit": summary.get("causal_reachability_credit"),
        "causal_reachability_risk_pressure": summary.get(
            "causal_reachability_risk_pressure"
        ),
        "causal_edge_survival": summary.get("causal_edge_survival"),
        "reach_starvation_score": summary.get("reach_starvation_score"),
        "reach_expansion_pressure": summary.get("reach_expansion_pressure"),
        "reach_restraint_pressure": summary.get("reach_restraint_pressure"),
        "control_reach_noise_score": summary.get("control_reach_noise_score"),
        "attention_locality_score": summary.get("attention_locality_score"),
        "attention_spread_score": summary.get("attention_spread_score"),
        "future_leak_score": summary.get("future_leak_score"),
        "attention_entropy": summary.get("attention_entropy"),
        "attention_entropy_normalized": summary.get("attention_entropy_normalized"),
        "attention_entropy_drop": summary.get("attention_entropy_drop"),
        "mean_max_probability": summary.get("mean_max_probability"),
        "valid_mean_max_probability": summary.get("valid_mean_max_probability"),
        "valid_attention_sparsity_0_01": summary.get("valid_attention_sparsity_0_01"),
        "valid_attention_sparsity_0_001": summary.get("valid_attention_sparsity_0_001"),
        "valid_sparsity_risk": summary.get("valid_sparsity_risk"),
        "head_attention_diversity": summary.get("head_attention_diversity"),
        "head_collapse_risk": summary.get("head_collapse_risk"),
        "layer_attention_diversity": summary.get("layer_attention_diversity"),
        "layer_collapse_risk": summary.get("layer_collapse_risk"),
        "geometry_takeover_score": summary.get("geometry_takeover_score"),
        "geo_qk_risk": summary.get("geo_qk_risk"),
        "entropy_risk": summary.get("entropy_risk"),
        "max_probability_risk": summary.get("max_probability_risk"),
        "rigidity_risk": summary.get("rigidity_risk"),
        "collapse_risk": summary.get("collapse_risk"),
        "severe_attention_collapse_detected": summary.get(
            "severe_attention_collapse_detected"
        ),
        "memory_decay": summary.get("memory_decay"),
        "memory_eta": summary.get("memory_eta"),
        "gate_floor": summary.get("gate_floor"),
        "gate_floor_next": summary.get("gate_floor_next"),
        "gate_floor_delta": summary.get("gate_floor_delta"),
        "gate_floor_decision": summary.get("gate_floor_decision"),
        "gate_floor_credit": summary.get("gate_floor_credit"),
        "gate_floor_risk_pressure": summary.get("gate_floor_risk_pressure"),
        "edge_survival": summary.get("edge_survival"),
        "random_edge_noise_score": summary.get("random_edge_noise_score"),
        "shuffled_edge_noise_score": summary.get("shuffled_edge_noise_score"),
        "real_edge_starvation_score": summary.get("real_edge_starvation_score"),
        "gate_noise_pressure": summary.get("gate_noise_pressure"),
        "gate_starvation_pressure": summary.get("gate_starvation_pressure"),
        "control_attention_separation": summary.get("control_attention_separation"),
        "memory_stability": summary.get("memory_stability"),
        "memory_turnover": summary.get("memory_turnover"),
        "memory_edge_density": summary.get("memory_edge_density"),
        "memory_persistence": summary.get("memory_persistence"),
        "memory_spectral_entropy": summary.get("memory_spectral_entropy"),
        "memory_effective_rank": summary.get("memory_effective_rank"),
        "memory_rigidity": summary.get("memory_rigidity"),
        "noise_risk": summary.get("noise_risk"),
    }
    control_rng_isolated = as_float_or_none(summary.get("control_rng_isolated"))
    if control_rng_isolated is not None:
        fields["control_rng_isolated"] = control_rng_isolated >= 0.5
    sparsity = summary.get("attention_sparsity")
    if isinstance(sparsity, dict):
        fields["attention_sparsity_0_01"] = sparsity.get("0.01")
        fields["attention_sparsity_0_001"] = sparsity.get("0.001")
    return {key: value for key, value in fields.items() if value is not None}


def format_progress_line(record: dict[str, Any]) -> str:
    condition = str(record.get("condition", "run"))
    parts = [
        f"[{condition}]",
        f"step={record.get('step')}",
    ]
    decision = record.get("alpha_decision")
    if decision:
        parts.append(f"decision={decision}")
    gate_decision = record.get("gate_floor_decision")
    if gate_decision:
        parts.append(f"gate_decision={gate_decision}")
    reach_decision = record.get("causal_reachability_decision")
    if reach_decision:
        parts.append(f"reach_decision={reach_decision}")
    distance_decision = record.get("distance_norm_scale_decision")
    if distance_decision:
        parts.append(f"norm_decision={distance_decision}")
    budget_decision = record.get("joint_budget_decision")
    if budget_decision:
        parts.append(f"budget_decision={budget_decision}")
    sep_mode = record.get("control_separation_mode")
    if sep_mode:
        parts.append(f"sep_mode={sep_mode}")
    claim = record.get("claim_eligibility")
    if claim:
        parts.append(f"claim={claim}")
    sep_status = record.get("control_separation_status")
    if sep_status:
        parts.append(f"sep_status={sep_status}")
    sep_warning = record.get("generic_regularization_warning")
    if sep_warning:
        parts.append(f"sep_warn={sep_warning}")
    meta_mode = record.get("meta_control_mode")
    if meta_mode:
        parts.append(f"meta_mode={meta_mode}")
    meta_top = record.get("meta_top_signal")
    if meta_top:
        parts.append(f"meta_top={meta_top}")
    meta_suppressed = record.get("meta_suppressed_signal")
    if meta_suppressed:
        parts.append(f"meta_supp={meta_suppressed}")
    meta_decision = record.get("meta_observer_decision_summary")
    if meta_decision:
        parts.append(f"meta_decision={meta_decision}")
    trainer_status = record.get("trainer_status")
    if trainer_status:
        parts.append(f"trainer={trainer_status}")
    trainer_event = record.get("trainer_event")
    if trainer_event:
        parts.append(f"t_event={trainer_event}")
    trainer_fail_reason = record.get("trainer_fail_fast_reason")
    if trainer_fail_reason:
        parts.append(f"t_fail_reason={trainer_fail_reason}")
    artifact_bundle = record.get("artifact_bundle_name")
    if artifact_bundle:
        parts.append(f"bundle={artifact_bundle}")
    qk_state = record.get("qk_competition_state")
    if qk_state:
        parts.append(f"qk_state={qk_state}")
    attention_regime = record.get("attention_behavior_regime")
    if attention_regime:
        parts.append(f"attn_regime={attention_regime}")
    for key, label, precision in [
        ("train_loss", "train", 4),
        ("validation_loss", "val", 4),
        ("best_validation_loss", "best", 4),
        ("perplexity", "ppl", 1),
        ("alpha_effective", "alpha", 4),
        ("alpha_next", "a_next", 4),
        ("alpha_delta", "d_alpha", 4),
        ("adaptive_score", "score", 6),
        ("adaptive_slope_gain", "slope", 6),
        ("adaptive_advantage", "adv", 6),
        ("control_rng_isolated", "rngIso", 0),
        ("scientific_claim_credit", "claimCredit", 3),
        ("real_vs_baseline_delta", "rvBase", 3),
        ("real_vs_alpha_zero_delta", "rvAZ", 3),
        ("real_vs_random_delta", "rvRand", 3),
        ("real_vs_shuffled_delta", "rvShuf", 3),
        ("real_vs_no_memory_delta", "rvNoMem", 3),
        ("real_vs_instantaneous_delta", "rvInst", 3),
        ("control_separation", "sep", 3),
        ("control_penalty", "sepPen", 3),
        ("attention_behavior_separation", "attnSep", 3),
        ("meta_available_signal_count", "metaAvail", 0),
        ("meta_masked_signal_count", "metaMasked", 0),
        ("evidence_availability_score", "metaEvidence", 3),
        ("pending_control_mask", "metaPend", 0),
        ("offline_replay_required", "metaReplay", 0),
        ("meta_attention_entropy", "metaEnt", 3),
        ("meta_attention_entropy_normalized", "metaNEnt", 3),
        ("controller_agreement_score", "metaAgree", 3),
        ("controller_conflict_score", "metaConflict", 3),
        ("meta_control_confidence", "metaConf", 3),
        ("meta_alpha_weight", "metaAlpha", 3),
        ("meta_memory_weight", "metaMem", 3),
        ("meta_gate_weight", "metaGate", 3),
        ("meta_reach_weight", "metaReach", 3),
        ("meta_norm_weight", "metaNorm", 3),
        ("trainer_fail_fast_triggered", "tFail", 0),
        ("trainer_processed_rows", "tRows", 0),
        ("controller_decision_count", "ctrlDec", 0),
        ("meta_observer_event_count", "metaObs", 0),
        ("control_separation_event_count", "sepObs", 0),
        ("safety_event_count", "safety", 0),
        ("live_display_event_count", "live", 0),
        ("progress_log_ready", "pLog", 0),
        ("controller_decision_log_ready", "cLog", 0),
        ("meta_control_observer_log_ready", "mLog", 0),
        ("control_separation_log_ready", "sepLog", 0),
        ("safety_log_ready", "sLog", 0),
        ("lightweight_artifact_bundle_ready", "bundleReady", 0),
        ("checkpoint_artifacts_excluded", "ckptExcl", 0),
        ("live_diagnostic_row_ready", "diagRow", 0),
        ("live_diagnostic_table_ready", "diagTbl", 0),
        ("live_diagnostic_plot_ready", "diagPlot", 0),
        ("live_diagnostic_row_count", "diagRows", 0),
        ("geo_to_qk_ratio", "geo/qk", 3),
        ("change_budget", "budget", 3),
        ("allocated_change_budget", "bUsed", 3),
        ("geometry_budget", "bGeom", 3),
        ("memory_budget", "bMem", 3),
        ("rigidity_budget", "bRigid", 3),
        ("noise_budget", "bNoise", 3),
        ("attention_derived_budget_pressure", "bAttn", 3),
        ("budget_conflict_score", "bConflict", 3),
        ("distance_norm_scale", "norm", 3),
        ("distance_norm_scale_next", "n_next", 3),
        ("distance_norm_scale_delta", "d_norm", 3),
        ("distance_norm_scale_credit", "nCredit", 3),
        ("distance_norm_scale_risk_pressure", "nPress", 3),
        ("pre_norm_distance_contrast", "preC", 3),
        ("post_norm_distance_contrast", "postC", 3),
        ("distance_contrast_retention", "ret", 3),
        ("distance_std_pre_norm", "preStd", 3),
        ("distance_std_post_norm", "postStd", 3),
        ("clipping_saturation_rate", "clipSat", 3),
        ("normalization_erasure_score", "erase", 3),
        ("distance_scale_release_pressure", "nRel", 3),
        ("distance_scale_restraint_pressure", "nRest", 3),
        ("real_distance_advantage", "realD", 3),
        ("random_distance_advantage", "randD", 3),
        ("shuffled_distance_advantage", "shufD", 3),
        ("geometry_takeover_score", "gTake", 3),
        ("geo_qk_risk", "gRisk", 3),
        ("attention_entropy", "ent", 3),
        ("attention_entropy_normalized", "nEnt", 3),
        ("entropy_risk", "eRisk", 3),
        ("mean_max_probability", "maxp", 3),
        ("max_probability_risk", "pRisk", 3),
        ("head_attention_diversity", "hDiv", 3),
        ("rigidity_risk", "rigid", 3),
        ("collapse_risk", "collapse", 3),
        ("gate_floor", "gate", 3),
        ("gate_floor_next", "g_next", 3),
        ("gate_floor_delta", "d_gate", 3),
        ("gate_floor_credit", "gCredit", 3),
        ("gate_floor_risk_pressure", "gPress", 3),
        ("edge_survival", "eSurv", 3),
        ("random_edge_noise_score", "rNoise", 3),
        ("real_edge_starvation_score", "rStarv", 3),
        ("causal_reachability", "reach", 0),
        ("causal_reachability_next", "r_next", 0),
        ("causal_reachability_delta", "d_reach", 0),
        ("causal_reachability_credit", "rCredit", 3),
        ("causal_reachability_risk_pressure", "rPress", 3),
        ("causal_edge_survival", "cSurv", 3),
        ("reach_starvation_score", "cStarv", 3),
        ("control_reach_noise_score", "cNoise", 3),
        ("attention_locality_score", "loc", 3),
        ("attention_spread_score", "spread", 3),
        ("future_leak_score", "fLeak", 3),
        ("memory_stability", "mStab", 3),
        ("memory_turnover", "mTurn", 3),
        ("memory_persistence", "mPers", 3),
        ("memory_rigidity", "mRigid", 3),
        ("noise_risk", "nRisk", 3),
        ("grad_norm", "grad", 3),
        ("tokens_per_second", "tok/s", 0),
        ("gpu_memory_gb", "gpu", 2),
        ("elapsed_minutes", "min", 1),
    ]:
        value = as_float_or_none(record.get(key))
        if value is not None:
            parts.append(f"{label}={value:.{precision}f}")
    status = record.get("status")
    if status:
        parts.append(f"status={status}")
    return " ".join(parts)


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return sanitize_for_json(float(value.detach().cpu().item()))
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
