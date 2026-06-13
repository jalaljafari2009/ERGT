"""Unified telemetry schema for ERGT open adaptive control."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from evaluation.open_control_philosophy_contract import (
    ADAPTIVE_PARAMETERS,
    HARD_STOPS,
    REQUIRED_TELEMETRY,
    SOFT_PRESSURES,
)

SCHEMA_VERSION = "unified_telemetry_v1"

FIELD_CATEGORIES = {
    "identity": [
        "telemetry_schema_version",
        "run_id",
        "phase",
        "condition",
        "control_family",
        "step",
        "eval_index",
        "seed",
    ],
    "data_model": [
        "dataset_name",
        "tokenizer",
        "context_length",
        "batch_size",
        "n_layers",
        "n_heads",
        "hidden_dim",
    ],
    "loss_trend": [
        "train_loss",
        "validation_loss",
        "best_validation_loss",
        "baseline_validation_loss",
        "baseline_centered_improvement",
        "late_window",
        "loss_slope",
        "baseline_loss_slope",
        "loss_slope_gain",
        "adaptive_advantage",
        "trend_window_points",
    ],
    "control_separation": [
        "real_vs_random_delta",
        "real_vs_shuffled_delta",
        "real_vs_no_memory_delta",
        "real_vs_instantaneous_delta",
        "control_separation",
        "control_penalty",
        "control_family_fairness_status",
    ],
    "attention_rigidity": [
        "qk_mean_abs",
        "geo_mean_abs",
        "geo_to_qk_ratio",
        "attention_entropy",
        "attention_entropy_drop",
        "mean_max_probability",
        "attention_sparsity_0_01",
        "geo_qk_risk",
        "entropy_risk",
        "max_probability_risk",
        "rigidity_risk",
        "collapse_risk",
    ],
    "memory_state": [
        "memory_decay",
        "memory_eta",
        "gate_floor",
        "memory_stability",
        "memory_turnover",
        "memory_edge_density",
        "memory_persistence",
        "memory_spectral_entropy",
        "memory_effective_rank",
        "memory_rigidity",
        "noise_risk",
    ],
    "geometry_distance": [
        "distance_norm_scale",
        "causal_reachability",
        "max_causal_step",
        "distance_mean",
        "distance_std",
        "spectral_entropy",
        "effective_rank",
        "future_leakage_detected",
    ],
    "adaptive_parameters": [
        "alpha",
        "alpha_previous",
        "alpha_next",
        "alpha_delta",
        "alpha_decision",
        "alpha_credit",
        "alpha_risk_pressure",
        "memory_decay_previous",
        "memory_decay_next",
        "memory_decay_delta",
        "memory_decay_decision",
        "memory_decay_credit",
        "memory_decay_risk_pressure",
        "memory_eta_previous",
        "memory_eta_next",
        "memory_eta_delta",
        "memory_eta_decision",
        "memory_eta_credit",
        "memory_eta_risk_pressure",
        "gate_floor_previous",
        "gate_floor_next",
        "gate_floor_delta",
        "gate_floor_decision",
        "gate_floor_credit",
        "gate_floor_risk_pressure",
        "distance_norm_scale_previous",
        "distance_norm_scale_next",
        "distance_norm_scale_delta",
        "distance_norm_scale_decision",
        "distance_norm_scale_credit",
        "distance_norm_scale_risk_pressure",
        "causal_reachability_previous",
        "causal_reachability_next",
        "causal_reachability_delta",
        "causal_reachability_decision",
        "causal_reachability_credit",
        "causal_reachability_risk_pressure",
    ],
    "attribution": [
        "adaptive_score",
        "attribution_summary",
        "change_budget",
        "decision_summary",
        "alpha_points_used",
    ],
    "safety": [
        "hard_stop_triggered",
        "hard_stop_reason",
        "nan_or_inf_detected",
        "loss_explosion_detected",
        "severe_attention_collapse_detected",
        "control_unfairness_detected",
    ],
    "runtime": [
        "learning_rate",
        "grad_norm",
        "tokens_processed",
        "tokens_per_second",
        "gpu_memory_gb",
        "gpu_peak_memory_gb",
        "elapsed_seconds",
        "elapsed_minutes",
    ],
}

FIELD_TYPES = {
    "string": [
        "telemetry_schema_version",
        "run_id",
        "phase",
        "condition",
        "control_family",
        "dataset_name",
        "tokenizer",
        "control_family_fairness_status",
        "alpha_decision",
        "memory_decay_decision",
        "memory_eta_decision",
        "gate_floor_decision",
        "distance_norm_scale_decision",
        "causal_reachability_decision",
        "attribution_summary",
        "decision_summary",
        "hard_stop_reason",
    ],
    "integer": [
        "step",
        "eval_index",
        "seed",
        "context_length",
        "batch_size",
        "n_layers",
        "n_heads",
        "hidden_dim",
        "trend_window_points",
        "max_causal_step",
        "alpha_points_used",
        "tokens_processed",
    ],
    "boolean": [
        "late_window",
        "future_leakage_detected",
        "hard_stop_triggered",
        "nan_or_inf_detected",
        "loss_explosion_detected",
        "severe_attention_collapse_detected",
        "control_unfairness_detected",
    ],
}

FIELD_ALIASES = {
    "alpha": ["alpha_effective"],
    "loss_slope_gain": ["adaptive_slope_gain"],
    "baseline_centered_improvement": ["base_centered"],
    "mean_max_probability": ["attention_mean_max_probability"],
    "alpha_credit": ["adaptive_score"],
}

MINIMUM_LIVE_FIELDS = [
    "step",
    "condition",
    "validation_loss",
    "alpha",
    "alpha_next",
    "alpha_delta",
    "alpha_decision",
    "loss_slope_gain",
    "baseline_centered_improvement",
    "geo_to_qk_ratio",
    "attention_entropy",
    "mean_max_probability",
    "rigidity_risk",
    "control_penalty",
]

RUN02_COMPATIBILITY_FIELDS = [
    "step",
    "condition",
    "validation_loss",
    "alpha_effective",
    "alpha_next",
    "alpha_delta",
    "alpha_decision",
    "adaptive_score",
    "adaptive_slope_gain",
    "adaptive_advantage",
    "geo_to_qk_ratio",
    "attention_entropy",
    "mean_max_probability",
    "geo_qk_risk",
    "entropy_risk",
    "max_probability_risk",
]


def build_unified_telemetry_schema_report() -> dict[str, Any]:
    """Return the canonical telemetry schema report."""

    fields = _build_field_catalog()
    report = {
        "schema": SCHEMA_VERSION,
        "status": "pass",
        "purpose": (
            "Define one live and machine-readable telemetry language for loss, "
            "attention, memory, geometry, controls, attribution, safety, and runtime."
        ),
        "categories": deepcopy(FIELD_CATEGORIES),
        "fields": fields,
        "aliases": deepcopy(FIELD_ALIASES),
        "minimum_live_fields": list(MINIMUM_LIVE_FIELDS),
        "run02_compatibility_fields": list(RUN02_COMPATIBILITY_FIELDS),
        "open_control_required_telemetry": list(REQUIRED_TELEMETRY),
        "adaptive_parameters": sorted(ADAPTIVE_PARAMETERS),
        "hard_stops": sorted(HARD_STOPS),
        "soft_pressures": sorted(SOFT_PRESSURES),
        "checks": {},
        "next_required_step": "memory_state_instrumentation",
    }
    report["checks"] = validate_unified_telemetry_schema(report)
    if not all(report["checks"].values()):
        report["status"] = "fail"
    return report


def validate_unified_telemetry_schema(schema: dict[str, Any]) -> dict[str, bool]:
    """Validate schema invariants required before ERGT-03 telemetry work."""

    categories = schema.get("categories", {})
    fields = schema.get("fields", {})
    aliases = schema.get("aliases", {})
    all_category_fields = [
        field for field_list in categories.values() for field in field_list
    ]
    unique_category_fields = set(all_category_fields)
    open_required = set(schema.get("open_control_required_telemetry", []))

    return {
        "no_duplicate_category_fields": len(all_category_fields)
        == len(unique_category_fields),
        "every_category_field_has_catalog_entry": unique_category_fields
        .issubset(fields),
        "every_catalog_field_has_category": set(fields).issubset(unique_category_fields),
        "open_control_required_fields_covered": open_required.issubset(
            unique_category_fields
        ),
        "minimum_live_fields_covered": set(schema.get("minimum_live_fields", []))
        .issubset(unique_category_fields),
        "run02_aliases_preserve_current_notebook_fields": set(
            schema.get("run02_compatibility_fields", [])
        ).issubset(unique_category_fields | _alias_values(aliases)),
        "adaptive_parameter_change_fields_declared": _adaptive_change_fields_declared(
            unique_category_fields
        ),
        "hard_stop_fields_declared": {
            "hard_stop_triggered",
            "hard_stop_reason",
            "nan_or_inf_detected",
            "future_leakage_detected",
            "loss_explosion_detected",
        }.issubset(unique_category_fields),
        "field_metadata_complete": all(
            field.get("category")
            and field.get("type")
            and field.get("cadence")
            and field.get("meaning")
            for field in fields.values()
        ),
    }


def validate_telemetry_record(
    record: dict[str, Any],
    *,
    schema: dict[str, Any] | None = None,
    required_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Validate one telemetry record using canonical names plus aliases."""

    active_schema = schema or build_unified_telemetry_schema_report()
    required = required_fields or active_schema["minimum_live_fields"]
    aliases = active_schema.get("aliases", {})
    missing = [
        field
        for field in required
        if field not in record
        and not any(alias in record for alias in aliases.get(field, []))
    ]
    unknown = [
        field
        for field in record
        if field not in active_schema["fields"]
        and field not in _alias_values(aliases)
    ]
    return {
        "status": "pass" if not missing else "fail",
        "missing_required_fields": missing,
        "unknown_fields": unknown,
    }


def _build_field_catalog() -> dict[str, dict[str, str]]:
    type_lookup = _type_lookup()
    fields: dict[str, dict[str, str]] = {}
    for category, field_names in FIELD_CATEGORIES.items():
        for field_name in field_names:
            fields[field_name] = {
                "category": category,
                "type": type_lookup.get(field_name, "number"),
                "cadence": _cadence_for_category(category),
                "meaning": _meaning_for_field(field_name),
            }
    return fields


def _type_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for field_type, names in FIELD_TYPES.items():
        for name in names:
            lookup[name] = field_type
    return lookup


def _cadence_for_category(category: str) -> str:
    if category in {"identity", "data_model"}:
        return "run_start_and_eval"
    if category == "runtime":
        return "every_eval_point"
    if category == "safety":
        return "every_eval_point_and_failure"
    return "every_eval_point"


def _meaning_for_field(field_name: str) -> str:
    return field_name.replace("_", " ")


def _alias_values(aliases: dict[str, list[str]]) -> set[str]:
    return {alias for values in aliases.values() for alias in values}


def _adaptive_change_fields_declared(fields: set[str]) -> bool:
    for parameter in ADAPTIVE_PARAMETERS:
        required = {
            f"{parameter}_next",
            f"{parameter}_delta",
            f"{parameter}_decision",
            f"{parameter}_credit",
            f"{parameter}_risk_pressure",
        }
        if not required.issubset(fields):
            return False
    return True
