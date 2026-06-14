"""Decision gate for real geometry vs generic regularization."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RealGeometryDecisionGateConfig:
    real_condition: str = "real_memory_d"
    decision_window: str = "1000_2000"
    late_window_start: int = 1000
    late_window_end: int = 2000
    min_control_advantage: float = 0.0
    min_relation_specific_advantage: float = 0.0
    max_future_leak_score: float = 0.0
    min_distance_contrast_retention: float = 0.55
    min_geo_to_qk_ratio: float = 0.01
    max_geo_to_qk_ratio: float = 0.16
    min_attention_advantage: float = 0.0

    def validate(self) -> None:
        if not self.real_condition:
            raise ValueError("real_condition must be non-empty")
        if self.late_window_start < 0:
            raise ValueError("late_window_start must be non-negative")
        if self.late_window_end < self.late_window_start:
            raise ValueError("late_window_end must be after late_window_start")
        if self.min_control_advantage < 0:
            raise ValueError("min_control_advantage must be non-negative")
        if self.min_relation_specific_advantage < 0:
            raise ValueError("min_relation_specific_advantage must be non-negative")
        if self.max_future_leak_score < 0:
            raise ValueError("max_future_leak_score must be non-negative")
        if self.min_geo_to_qk_ratio < 0 or self.max_geo_to_qk_ratio < 0:
            raise ValueError("geo/qk thresholds must be non-negative")
        if self.max_geo_to_qk_ratio < self.min_geo_to_qk_ratio:
            raise ValueError("max_geo_to_qk_ratio must be >= min_geo_to_qk_ratio")


REQUIRED_DECISION_GATE_OUTPUTS = [
    "required_comparisons",
    "baseline_only_insufficient_check",
    "risk_audit",
    "attention_gate",
    "relation_specific_gate",
    "failure_labels",
    "decision",
]


def decide_real_geometry_vs_generic_regularization(
    rows: list[dict[str, Any]],
    *,
    late_window_report: dict[str, Any],
    attribution_report: dict[str, Any],
    config: RealGeometryDecisionGateConfig | None = None,
) -> dict[str, Any]:
    """Decide whether real stable causal geometry clears the adaptive gate."""

    active = config or RealGeometryDecisionGateConfig()
    active.validate()
    late_analysis = late_window_report["analysis"]
    attribution = attribution_report["comparison"]
    comparisons = _required_comparisons(late_analysis, attribution, config=active)
    risk_audit = _risk_audit(rows, late_analysis, attribution, config=active)
    attention_gate = _attention_gate(attribution, config=active)
    relation_gate = _relation_specific_gate(attribution, config=active)
    baseline_only = _baseline_only_insufficient_check(comparisons)
    checks = {
        "late_window_report_passed": late_window_report["status"] == "pass",
        "attribution_report_passed": attribution_report["status"] == "pass",
        "all_required_comparisons_positive": all(
            item["passes"] for item in comparisons.values()
        ),
        "baseline_only_is_insufficient": baseline_only["passes"],
        "r1_memory_and_causality_clear": risk_audit["R1"]["passes"],
        "r2_distance_scale_clear": risk_audit["R2"]["passes"],
        "r3_attention_behavior_clear": risk_audit["R3"]["passes"],
        "attention_gate_passed": attention_gate["passes"],
        "relation_specific_gate_passed": relation_gate["passes"],
    }
    failure_labels = _failure_labels(
        checks=checks,
        comparisons=comparisons,
        risk_audit=risk_audit,
        attribution=attribution,
    )
    decision = "pass_real_geometry_contract" if all(checks.values()) else "fail_enter_revision"
    return {
        "config": asdict(active),
        "decision_window": active.decision_window,
        "required_comparisons": comparisons,
        "baseline_only_insufficient_check": baseline_only,
        "risk_audit": risk_audit,
        "attention_gate": attention_gate,
        "relation_specific_gate": relation_gate,
        "checks": checks,
        "failure_labels": failure_labels,
        "decision": decision,
        "anti_overclaim": (
            "Passing this gate supports only the current guarded adaptive "
            "mechanics contract. It does not prove the final physics or "
            "intelligence claim without real notebook telemetry, longer runs, "
            "and multi-seed confirmation."
        ),
    }


def _required_comparisons(
    late_analysis: dict[str, Any],
    attribution: dict[str, Any],
    *,
    config: RealGeometryDecisionGateConfig,
) -> dict[str, dict[str, Any]]:
    deltas = late_analysis["real_vs_control_window_deltas"][config.decision_window][
        "deltas"
    ]
    profiles = attribution["control_profiles"]
    values = {
        "baseline": deltas["baseline"]["mean_delta"],
        "alpha_zero": deltas["alpha_zero"]["mean_delta"],
        "random_adaptive": profiles["random"]["real_minus_control_loss_gain"],
        "shuffled_adaptive": profiles["shuffled"]["real_minus_control_loss_gain"],
        "no_memory_real": profiles["no_memory"]["real_minus_control_loss_gain"],
        "instantaneous_real": profiles["instantaneous"][
            "real_minus_control_loss_gain"
        ],
    }
    return {
        name: {
            "real_advantage": value,
            "threshold": config.min_control_advantage,
            "passes": value is not None and value > config.min_control_advantage,
        }
        for name, value in values.items()
    }


def _baseline_only_insufficient_check(
    comparisons: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline_passes = comparisons["baseline"]["passes"]
    non_baseline = {
        key: value["passes"]
        for key, value in comparisons.items()
        if key != "baseline"
    }
    return {
        "baseline_passes": baseline_passes,
        "non_baseline_comparison_passes": non_baseline,
        "passes": baseline_passes and all(non_baseline.values()),
        "rule": "baseline improvement alone is insufficient for a real geometry claim",
    }


def _risk_audit(
    rows: list[dict[str, Any]],
    late_analysis: dict[str, Any],
    attribution: dict[str, Any],
    *,
    config: RealGeometryDecisionGateConfig,
) -> dict[str, Any]:
    real_rows = [
        row
        for row in rows
        if row.get("condition") == config.real_condition
        and config.late_window_start <= int(row["step"]) <= config.late_window_end
    ]
    real_window = late_analysis["condition_window_summaries"][config.decision_window][
        config.real_condition
    ]
    future_leak = _max_number(real_rows, "future_leak_score")
    distance_contrast = _mean_number(real_rows, "distance_contrast_retention")
    geo_mean = real_window["mean_geo_to_qk_ratio"]
    geo_max = real_window["max_geo_to_qk_ratio"]
    memory_gain = attribution["no_memory_comparison"]["memory_specific_gain"]
    stable_gain = attribution["instantaneous_comparison"]["stable_memory_gain"]
    relation_estimate = attribution["relation_specific_advantage_estimate"]["estimate"]
    attention = attribution["attention_behavior_comparison"]
    return {
        "R1": {
            "name": "memory_and_causal_validity",
            "max_future_leak_score": future_leak,
            "memory_specific_gain": memory_gain,
            "stable_memory_gain": stable_gain,
            "passes": (
                future_leak is not None
                and future_leak <= config.max_future_leak_score
                and memory_gain is not None
                and memory_gain > config.min_control_advantage
                and stable_gain is not None
                and stable_gain > config.min_control_advantage
            ),
        },
        "R2": {
            "name": "distance_contrast_and_scale",
            "mean_distance_contrast_retention": distance_contrast,
            "mean_geo_to_qk_ratio": geo_mean,
            "max_geo_to_qk_ratio": geo_max,
            "relation_specific_advantage": relation_estimate,
            "passes": (
                distance_contrast is not None
                and distance_contrast >= config.min_distance_contrast_retention
                and geo_mean is not None
                and config.min_geo_to_qk_ratio <= geo_mean <= config.max_geo_to_qk_ratio
                and geo_max is not None
                and geo_max <= config.max_geo_to_qk_ratio
                and relation_estimate is not None
                and relation_estimate > config.min_relation_specific_advantage
            ),
        },
        "R3": {
            "name": "attention_behavior",
            "minimum_attention_advantage": attention["minimum_attention_advantage"],
            "collapse_warning": attention["collapse_warning"],
            "uniformity_drift_warning": attention["uniformity_drift_warning"],
            "control_like_attention_warning": attention["control_like_attention_warning"],
            "passes": (
                attention["attention_separated_from_controls"]
                and not attention["collapse_warning"]
                and not attention["uniformity_drift_warning"]
                and not attention["control_like_attention_warning"]
            ),
        },
    }


def _attention_gate(
    attribution: dict[str, Any],
    *,
    config: RealGeometryDecisionGateConfig,
) -> dict[str, Any]:
    attention = attribution["attention_behavior_comparison"]
    advantage = attention["minimum_attention_advantage"]
    return {
        "minimum_attention_advantage": advantage,
        "stage22_attention_safe": attention["stage22_attention_safe"],
        "attention_separated_from_controls": attention[
            "attention_separated_from_controls"
        ],
        "passes": (
            advantage is not None
            and advantage > config.min_attention_advantage
            and attention["stage22_attention_safe"]
            and attention["attention_separated_from_controls"]
        ),
    }


def _relation_specific_gate(
    attribution: dict[str, Any],
    *,
    config: RealGeometryDecisionGateConfig,
) -> dict[str, Any]:
    relation = attribution["relation_specific_advantage_estimate"]
    estimate = relation["estimate"]
    return {
        "estimate": estimate,
        "best_control_family": relation["best_control_family"],
        "best_control_condition": relation["best_control_condition"],
        "best_control_gain_share_of_real_gain": relation[
            "best_control_gain_share_of_real_gain"
        ],
        "passes": estimate is not None
        and estimate > config.min_relation_specific_advantage,
    }


def _failure_labels(
    *,
    checks: dict[str, bool],
    comparisons: dict[str, dict[str, Any]],
    risk_audit: dict[str, Any],
    attribution: dict[str, Any],
) -> list[str]:
    labels = []
    if not checks["late_window_report_passed"]:
        labels.append("late_window_not_ready")
    if not checks["attribution_report_passed"]:
        labels.extend(attribution.get("revision_triggers", []))
    if comparisons["random_adaptive"]["real_advantage"] is not None and not comparisons[
        "random_adaptive"
    ]["passes"]:
        labels.append("control_regularization_dominance")
    if comparisons["shuffled_adaptive"]["real_advantage"] is not None and not comparisons[
        "shuffled_adaptive"
    ]["passes"]:
        labels.append("control_regularization_dominance")
    if not comparisons["no_memory_real"]["passes"]:
        labels.append("memory_starved")
    if not comparisons["instantaneous_real"]["passes"]:
        labels.append("memory_not_stabilizing")
    if not checks["baseline_only_is_insufficient"]:
        labels.append("baseline_only_evidence_insufficient")
    if not risk_audit["R1"]["passes"]:
        if (risk_audit["R1"]["max_future_leak_score"] or 0.0) > 0.0:
            labels.append("future_leak_detected")
        else:
            labels.append("memory_or_causality_unresolved")
    if not risk_audit["R2"]["passes"]:
        labels.append("normalization_erased_contrast")
    if not risk_audit["R3"]["passes"]:
        if risk_audit["R3"]["uniformity_drift_warning"]:
            labels.append("attention_uniformity_drift")
        elif risk_audit["R3"]["control_like_attention_warning"]:
            labels.append("attention_control_like")
        else:
            labels.append("attention_head_lock_in")
    return sorted(set(labels))


def _mean_number(rows: list[dict[str, Any]], field: str) -> float | None:
    values = _numbers(rows, field)
    return statistics.fmean(values) if values else None


def _max_number(rows: list[dict[str, Any]], field: str) -> float | None:
    values = _numbers(rows, field)
    return max(values) if values else None


def _numbers(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            values.append(numeric)
    return values
