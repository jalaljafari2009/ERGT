"""Late-window and post-1000 analysis for guarded adaptive ERGT runs."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from experiments.guarded_2000_step_adaptive_run import (
    GUARDED_WINDOWS,
    REQUIRED_GUARDED_CONDITIONS,
)


@dataclass(frozen=True)
class LateWindowAnalysisConfig:
    real_condition: str = "real_memory_d"
    baseline_condition: str = "baseline"
    control_conditions: tuple[str, ...] = (
        "alpha_zero",
        "random_memory_d",
        "shuffled_memory_d",
        "no_memory_real_d",
        "instantaneous_real_d",
    )
    windows: dict[str, tuple[int, int]] = field(default_factory=lambda: GUARDED_WINDOWS)
    decision_window: str = "1000_2000"
    post_1000_start: int = 1000
    min_window_points: int = 2
    max_rigidity_risk: float = 0.45
    max_mean_max_probability: float = 0.55
    min_attention_entropy: float = 1.80
    min_attention_advantage: float = 0.0

    def validate(self) -> None:
        if self.real_condition not in REQUIRED_GUARDED_CONDITIONS:
            raise ValueError("real_condition must be a guarded condition")
        if self.baseline_condition not in REQUIRED_GUARDED_CONDITIONS:
            raise ValueError("baseline_condition must be a guarded condition")
        if self.decision_window not in self.windows:
            raise ValueError("decision_window must be declared in windows")
        if self.post_1000_start < 0:
            raise ValueError("post_1000_start must be non-negative")
        if self.min_window_points < 1:
            raise ValueError("min_window_points must be positive")


def analyze_late_window_post1000(
    rows: list[dict[str, Any]],
    *,
    config: LateWindowAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Analyze post-1000 loss and attention behavior across fixed windows."""

    active = config or LateWindowAnalysisConfig()
    active.validate()
    clean_rows = _valid_rows(rows)
    by_condition = _group_by_condition(clean_rows)
    window_summaries = {
        window_name: _window_condition_summary(
            by_condition,
            start_step=start,
            end_step=end,
            config=active,
        )
        for window_name, (start, end) in active.windows.items()
    }
    real_vs_controls = {
        window_name: _real_vs_control_deltas(
            by_condition,
            start_step=start,
            end_step=end,
            config=active,
        )
        for window_name, (start, end) in active.windows.items()
    }
    attention = {
        window_name: _attention_window_analysis(
            window_summaries[window_name],
            config=active,
        )
        for window_name in active.windows
    }
    decision = _decision_summary(
        window_summaries[active.decision_window],
        real_vs_controls[active.decision_window],
        attention[active.decision_window],
        config=active,
    )
    return {
        "config": _config_payload(active),
        "windows": _windows_payload(active.windows),
        "condition_window_summaries": window_summaries,
        "real_vs_control_window_deltas": real_vs_controls,
        "attention_window_analysis": attention,
        "post_1000_decision_summary": decision,
        "endpoint_supporting_summary": _endpoint_summary(by_condition, config=active),
        "analysis_rule": (
            "Use the 1000-2000 decision window first; endpoint loss is supporting "
            "context only and cannot override late-window loss or attention behavior."
        ),
    }


def _window_condition_summary(
    by_condition: dict[str, list[dict[str, Any]]],
    *,
    start_step: int,
    end_step: int,
    config: LateWindowAnalysisConfig,
) -> dict[str, dict[str, Any]]:
    summary = {}
    baseline_by_step = _rows_by_step(by_condition.get(config.baseline_condition, []))
    for condition in REQUIRED_GUARDED_CONDITIONS:
        rows = _rows_in_window(by_condition.get(condition, []), start_step, end_step)
        losses = _numbers(rows, "validation_loss")
        baseline_deltas = [
            float(baseline_by_step[int(row["step"])]["validation_loss"])
            - float(row["validation_loss"])
            for row in rows
            if int(row["step"]) in baseline_by_step
        ]
        summary[condition] = {
            "points": len(rows),
            "start_step": rows[0]["step"] if rows else None,
            "end_step": rows[-1]["step"] if rows else None,
            "mean_validation_loss": _mean(losses),
            "validation_slope": _slope(
                [int(row["step"]) for row in rows],
                [float(row["validation_loss"]) for row in rows],
            ),
            "mean_baseline_centered_improvement": _mean(baseline_deltas),
            "mean_geo_to_qk_ratio": _mean(_numbers(rows, "geo_to_qk_ratio")),
            "max_geo_to_qk_ratio": _max(_numbers(rows, "geo_to_qk_ratio")),
            "mean_memory_stability": _mean(_numbers(rows, "memory_stability")),
            "mean_memory_persistence": _mean(_numbers(rows, "memory_persistence")),
            "mean_attention_entropy": _mean(_numbers(rows, "attention_entropy")),
            "attention_entropy_slope": _slope(
                [int(row["step"]) for row in rows],
                [
                    float(row["attention_entropy"])
                    for row in rows
                    if row.get("attention_entropy") is not None
                ],
            )
            if all(row.get("attention_entropy") is not None for row in rows)
            else None,
            "mean_max_probability": _mean(_numbers(rows, "mean_max_probability")),
            "max_mean_max_probability": _max(_numbers(rows, "mean_max_probability")),
            "mean_rigidity_risk": _mean(_numbers(rows, "rigidity_risk")),
            "max_rigidity_risk": _max(_numbers(rows, "rigidity_risk")),
            "mean_attention_behavior_score": _mean(
                _numbers(rows, "attention_behavior_score")
            ),
            "mean_attention_behavior_separation": _mean(
                _numbers(rows, "attention_behavior_separation")
            ),
            "collapse_event_count": sum(
                1
                for row in rows
                if bool(row.get("severe_attention_collapse_detected", False))
            ),
        }
    return summary


def _real_vs_control_deltas(
    by_condition: dict[str, list[dict[str, Any]]],
    *,
    start_step: int,
    end_step: int,
    config: LateWindowAnalysisConfig,
) -> dict[str, Any]:
    real_rows = _rows_by_step(
        _rows_in_window(
            by_condition.get(config.real_condition, []),
            start_step,
            end_step,
        )
    )
    deltas: dict[str, Any] = {}
    matched_steps = set(real_rows)
    for condition in (config.baseline_condition, *config.control_conditions):
        control_rows = _rows_by_step(
            _rows_in_window(by_condition.get(condition, []), start_step, end_step)
        )
        matched_steps &= set(control_rows)
        condition_deltas = [
            float(control_rows[step]["validation_loss"])
            - float(real_rows[step]["validation_loss"])
            for step in sorted(set(real_rows) & set(control_rows))
        ]
        deltas[condition] = {
            "points": len(condition_deltas),
            "mean_delta": _mean(condition_deltas),
            "min_delta": _min(condition_deltas),
            "positive_delta_points": sum(1 for value in condition_deltas if value > 0),
        }
    complete_matched_steps = sorted(matched_steps)
    mean_deltas = [
        item["mean_delta"] for item in deltas.values() if item["mean_delta"] is not None
    ]
    return {
        "matched_steps": complete_matched_steps,
        "matched_points": len(complete_matched_steps),
        "deltas": deltas,
        "control_separation": _min(mean_deltas),
        "real_beats_all_controls": all(
            item["mean_delta"] is not None and item["mean_delta"] > 0
            for item in deltas.values()
        ),
    }


def _attention_window_analysis(
    window_summary: dict[str, dict[str, Any]],
    *,
    config: LateWindowAnalysisConfig,
) -> dict[str, Any]:
    real = window_summary[config.real_condition]
    control_scores = [
        window_summary[condition]["mean_attention_behavior_score"]
        for condition in (config.baseline_condition, *config.control_conditions)
        if window_summary[condition]["mean_attention_behavior_score"] is not None
    ]
    control_best = max(control_scores) if control_scores else None
    real_score = real["mean_attention_behavior_score"]
    attention_advantage = (
        real_score - control_best
        if real_score is not None and control_best is not None
        else None
    )
    uniformity_drift_warning = (
        real["mean_attention_entropy"] is not None
        and real["mean_attention_entropy"] > 3.10
        and (real["mean_geo_to_qk_ratio"] or 0.0) < 0.04
    )
    collapse_warning = (
        (real["max_mean_max_probability"] or 0.0) > config.max_mean_max_probability
        or (real["max_rigidity_risk"] or 0.0) > config.max_rigidity_risk
        or (real["mean_attention_entropy"] or 0.0) < config.min_attention_entropy
        or real["collapse_event_count"] > 0
    )
    control_like_warning = (
        attention_advantage is not None
        and attention_advantage <= config.min_attention_advantage
    )
    return {
        "real_attention_score": real_score,
        "best_control_attention_score": control_best,
        "real_attention_advantage": attention_advantage,
        "uniformity_drift_warning": uniformity_drift_warning,
        "collapse_warning": collapse_warning,
        "control_like_attention_warning": control_like_warning,
        "attention_safe_for_window_decision": not (
            uniformity_drift_warning or collapse_warning or control_like_warning
        ),
    }


def _decision_summary(
    decision_window: dict[str, dict[str, Any]],
    decision_deltas: dict[str, Any],
    attention: dict[str, Any],
    *,
    config: LateWindowAnalysisConfig,
) -> dict[str, Any]:
    real = decision_window[config.real_condition]
    baseline_delta = decision_deltas["deltas"][config.baseline_condition]["mean_delta"]
    return {
        "decision_window": config.decision_window,
        "uses_post_1000_priority": config.decision_window == "1000_2000",
        "real_late_window_points": real["points"],
        "real_late_slope": real["validation_slope"],
        "real_late_trend": _trend_label(real["validation_slope"]),
        "real_late_beats_baseline": baseline_delta is not None and baseline_delta > 0,
        "real_late_beats_all_controls": decision_deltas["real_beats_all_controls"],
        "control_separation": decision_deltas["control_separation"],
        "attention_safe_for_late_decision": attention[
            "attention_safe_for_window_decision"
        ],
        "endpoint_loss_is_supporting_only": True,
        "late_window_decision_ready": (
            real["points"] >= config.min_window_points
            and decision_deltas["matched_points"] >= config.min_window_points
            and real["validation_slope"] is not None
            and attention["attention_safe_for_window_decision"]
        ),
    }


def _endpoint_summary(
    by_condition: dict[str, list[dict[str, Any]]],
    *,
    config: LateWindowAnalysisConfig,
) -> dict[str, Any]:
    endpoints = {}
    for condition in REQUIRED_GUARDED_CONDITIONS:
        rows = by_condition.get(condition, [])
        if not rows:
            continue
        latest = rows[-1]
        endpoints[condition] = {
            "step": latest["step"],
            "validation_loss": latest["validation_loss"],
            "geo_to_qk_ratio": latest.get("geo_to_qk_ratio"),
            "attention_entropy": latest.get("attention_entropy"),
            "rigidity_risk": latest.get("rigidity_risk"),
        }
    real_endpoint = endpoints.get(config.real_condition, {})
    return {
        "endpoints": endpoints,
        "real_endpoint": real_endpoint,
        "endpoint_is_not_decision_source": True,
    }


def _valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = [
        row
        for row in rows
        if row.get("condition") is not None
        and row.get("step") is not None
        and row.get("validation_loss") is not None
    ]
    for row in clean:
        if not math.isfinite(float(row["validation_loss"])):
            raise ValueError("validation_loss must be finite")
    return sorted(clean, key=lambda row: (str(row["condition"]), int(row["step"])))


def _group_by_condition(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)
    for condition_rows in grouped.values():
        condition_rows.sort(key=lambda row: int(row["step"]))
    return grouped


def _rows_in_window(
    rows: list[dict[str, Any]],
    start_step: int,
    end_step: int,
) -> list[dict[str, Any]]:
    return [
        row for row in rows if start_step <= int(row["step"]) <= end_step
    ]


def _rows_by_step(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["step"]): row for row in rows}


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


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _min(values: list[float]) -> float | None:
    return min(values) if values else None


def _max(values: list[float]) -> float | None:
    return max(values) if values else None


def _slope(steps: list[int], values: list[float]) -> float | None:
    if len(steps) != len(values) or len(steps) < 2:
        return None
    x_mean = statistics.fmean(float(step) for step in steps)
    y_mean = statistics.fmean(values)
    denominator = sum((float(step) - x_mean) ** 2 for step in steps)
    if denominator == 0.0:
        return 0.0
    numerator = sum(
        (float(step) - x_mean) * (value - y_mean)
        for step, value in zip(steps, values, strict=True)
    )
    return numerator / denominator


def _trend_label(slope: float | None) -> str:
    if slope is None:
        return "insufficient_points"
    if slope < 0:
        return "improving"
    if slope > 0:
        return "worsening"
    return "flat"


def _config_payload(config: LateWindowAnalysisConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["control_conditions"] = list(config.control_conditions)
    payload["windows"] = _windows_payload(config.windows)
    return payload


def _windows_payload(windows: dict[str, tuple[int, int]]) -> dict[str, dict[str, int]]:
    return {
        name: {"start_step": start, "end_step": end}
        for name, (start, end) in windows.items()
    }
