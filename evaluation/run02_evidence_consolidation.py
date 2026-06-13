"""Consolidate ERGT Run-02 adaptive-alpha evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

DEFAULT_REQUIRED_CONDITIONS = [
    "baseline",
    "alpha_zero",
    "real_memory_d_adaptive",
    "random_memory_d_adaptive",
    "shuffled_memory_d_adaptive",
]

OPTIONAL_CONTROL_CONDITIONS = [
    "no_memory_real_d_adaptive",
    "instantaneous_real_d_adaptive",
]


def build_run02_evidence_consolidation_report(
    run_root: str | Path,
    *,
    late_window: tuple[int, int] = (1000, 2000),
    required_conditions: list[str] | None = None,
    alpha_zero_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Build a machine-readable consolidation report for Run-02 outputs."""

    root = Path(run_root)
    required = required_conditions or DEFAULT_REQUIRED_CONDITIONS
    conditions = sorted(
        path.name for path in root.iterdir() if path.is_dir() and path.name != "checkpoints"
    ) if root.exists() else []
    baseline_progress = _read_condition_progress(root, "baseline")
    baseline_by_step = {
        int(row["step"]): row
        for row in baseline_progress
        if row.get("step") is not None and row.get("validation_loss") is not None
    }

    summaries = {
        condition: _summarize_condition(root, condition, baseline_by_step, late_window)
        for condition in conditions
    }
    checks = _build_checks(
        summaries,
        required,
        alpha_zero_tolerance=alpha_zero_tolerance,
    )
    comparisons = _build_comparisons(summaries)
    status = _classify_status(checks, summaries)

    return {
        "report": "run02_evidence_consolidation",
        "status": status,
        "run_root": root.as_posix(),
        "late_window": {
            "start_step": late_window[0],
            "end_step": late_window[1],
        },
        "required_conditions": required,
        "optional_control_conditions": OPTIONAL_CONTROL_CONDITIONS,
        "conditions_found": conditions,
        "checks": checks,
        "condition_summaries": summaries,
        "comparisons": comparisons,
        "interpretation": _interpretation(status, checks, comparisons),
        "next_required_step": _next_required_step(status),
        "anti_overclaim": (
            "This report consolidates Run-02 evidence. It does not prove "
            "relational geometry unless real adaptive memory separates from "
            "random, shuffled, no-memory, and instantaneous controls."
        ),
    }


def _summarize_condition(
    run_root: Path,
    condition: str,
    baseline_by_step: dict[int, dict[str, Any]],
    late_window: tuple[int, int],
) -> dict[str, Any]:
    condition_dir = run_root / condition
    progress = _read_jsonl(condition_dir / "progress_log.jsonl")
    alpha_log = _read_jsonl(condition_dir / "adaptive_alpha_log.jsonl")
    metrics = _read_json(condition_dir / "metrics.json")

    eval_rows = [row for row in progress if row.get("validation_loss") is not None]
    late_rows = [
        row
        for row in eval_rows
        if late_window[0] <= int(row.get("step", -1)) <= late_window[1]
    ]
    baseline_centered = [
        float(baseline_by_step[int(row["step"])]["validation_loss"])
        - float(row["validation_loss"])
        for row in eval_rows
        if row.get("step") in baseline_by_step
    ]
    late_baseline_centered = [
        float(baseline_by_step[int(row["step"])]["validation_loss"])
        - float(row["validation_loss"])
        for row in late_rows
        if row.get("step") in baseline_by_step
    ]

    final_row = eval_rows[-1] if eval_rows else {}
    adaptive_rows = alpha_log or [
        row.get("adaptive_alpha", {})
        for row in progress
        if isinstance(row.get("adaptive_alpha"), dict)
    ]
    decision_counts = _decision_counts(adaptive_rows)

    return {
        "present": condition_dir.exists(),
        "progress_log_present": bool(progress),
        "metrics_present": bool(metrics),
        "adaptive_alpha_log_present": bool(alpha_log),
        "eval_points": len(eval_rows),
        "late_window_eval_points": len(late_rows),
        "final_step": _last_numeric(eval_rows, "step"),
        "final_validation_loss": _first_finite(
            final_row.get("validation_loss"),
            metrics.get("final_validation_loss"),
        ),
        "best_validation_loss": _first_finite(
            final_row.get("best_validation_loss"),
            metrics.get("best_validation_loss"),
        ),
        "mean_baseline_centered_improvement": _mean(baseline_centered),
        "late_mean_baseline_centered_improvement": _mean(late_baseline_centered),
        "final_baseline_centered_improvement": (
            late_baseline_centered[-1] if late_baseline_centered else None
        ),
        "max_geo_to_qk_ratio": _max_numeric(eval_rows, "geo_to_qk_ratio"),
        "final_geo_to_qk_ratio": _last_numeric(eval_rows, "geo_to_qk_ratio"),
        "min_attention_entropy": _min_numeric(eval_rows, "attention_entropy"),
        "max_mean_max_probability": _max_numeric(eval_rows, "mean_max_probability"),
        "max_geo_qk_risk": _max_numeric(eval_rows, "geo_qk_risk"),
        "max_entropy_risk": _max_numeric(eval_rows, "entropy_risk"),
        "max_probability_risk": _max_numeric(eval_rows, "max_probability_risk"),
        "alpha_initial": _first_numeric(eval_rows, "alpha_effective"),
        "alpha_final": _first_finite(
            _last_numeric(eval_rows, "alpha_next"),
            _last_numeric(eval_rows, "alpha_effective"),
        ),
        "alpha_max": _max_of_keys(eval_rows, ["alpha_next", "alpha_effective"]),
        "alpha_delta_total": _sum_numeric(eval_rows, "alpha_delta"),
        "adaptive_decision_counts": decision_counts,
        "adaptive_score_final": _last_numeric(eval_rows, "adaptive_score"),
        "adaptive_slope_gain_final": _last_numeric(eval_rows, "adaptive_slope_gain"),
        "adaptive_advantage_final": _last_numeric(eval_rows, "adaptive_advantage"),
    }


def _build_checks(
    summaries: dict[str, dict[str, Any]],
    required_conditions: list[str],
    *,
    alpha_zero_tolerance: float,
) -> dict[str, bool]:
    baseline = summaries.get("baseline", {})
    alpha_zero = summaries.get("alpha_zero", {})
    real = summaries.get("real_memory_d_adaptive", {})
    present = {
        condition: summaries.get(condition, {}).get("progress_log_present") is True
        for condition in required_conditions
    }

    baseline_loss = baseline.get("final_validation_loss")
    alpha_zero_loss = alpha_zero.get("final_validation_loss")
    alpha_zero_matches = (
        _is_finite(baseline_loss)
        and _is_finite(alpha_zero_loss)
        and abs(float(baseline_loss) - float(alpha_zero_loss)) <= alpha_zero_tolerance
    )

    return {
        "run_root_has_outputs": bool(summaries),
        "required_conditions_present": all(present.values()),
        "baseline_progress_present": present.get("baseline", False),
        "alpha_zero_progress_present": present.get("alpha_zero", False),
        "alpha_zero_matches_baseline": bool(alpha_zero_matches),
        "real_adaptive_present": present.get("real_memory_d_adaptive", False),
        "real_has_late_window_points": real.get("late_window_eval_points", 0) > 0,
        "real_has_adaptive_telemetry": (
            real.get("adaptive_alpha_log_present") is True
            or real.get("alpha_final") is not None
        ),
        "control_families_present": all(
            present.get(condition, False)
            for condition in [
                "random_memory_d_adaptive",
                "shuffled_memory_d_adaptive",
            ]
        ),
        "rigidity_telemetry_present": any(
            real.get(key) is not None
            for key in [
                "max_geo_to_qk_ratio",
                "min_attention_entropy",
                "max_mean_max_probability",
            ]
        ),
    }


def _build_comparisons(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    real = summaries.get("real_memory_d_adaptive", {})
    comparisons: dict[str, Any] = {}
    for control in [
        "baseline",
        "alpha_zero",
        "random_memory_d_adaptive",
        "shuffled_memory_d_adaptive",
        "no_memory_real_d_adaptive",
        "instantaneous_real_d_adaptive",
    ]:
        control_summary = summaries.get(control)
        if not control_summary:
            continue
        comparisons[f"real_vs_{control}"] = {
            "final_validation_loss_delta": _delta(
                control_summary.get("final_validation_loss"),
                real.get("final_validation_loss"),
            ),
            "late_mean_baseline_centered_improvement_delta": _delta(
                real.get("late_mean_baseline_centered_improvement"),
                control_summary.get("late_mean_baseline_centered_improvement"),
            ),
            "alpha_final_delta": _delta(
                real.get("alpha_final"),
                control_summary.get("alpha_final"),
            ),
        }
    return comparisons


def _classify_status(
    checks: dict[str, bool],
    summaries: dict[str, dict[str, Any]],
) -> str:
    if not checks["run_root_has_outputs"]:
        return "incomplete_needs_run02_bundle"
    if not checks["required_conditions_present"]:
        return "incomplete_required_conditions_missing"
    if not checks["alpha_zero_matches_baseline"]:
        return "needs_investigation_alpha_zero_control"
    if not checks["real_has_adaptive_telemetry"]:
        return "needs_investigation_missing_adaptive_telemetry"
    if not checks["rigidity_telemetry_present"]:
        return "needs_investigation_missing_rigidity_telemetry"
    if not checks["real_has_late_window_points"]:
        return "incomplete_needs_late_window"
    real = summaries.get("real_memory_d_adaptive", {})
    if real.get("final_validation_loss") is None:
        return "incomplete_real_metrics_missing"
    return "consolidated_ready_for_open_control_contract"


def _interpretation(
    status: str,
    checks: dict[str, bool],
    comparisons: dict[str, Any],
) -> list[str]:
    notes = [f"status={status}"]
    if checks.get("alpha_zero_matches_baseline"):
        notes.append("alpha_zero remains a valid neutral control.")
    if checks.get("real_has_adaptive_telemetry"):
        notes.append("adaptive alpha telemetry is available for trajectory analysis.")
    if checks.get("control_families_present"):
        notes.append("random and shuffled adaptive controls are available.")
    if "real_vs_random_memory_d_adaptive" in comparisons:
        notes.append("real-vs-control comparisons can be inspected without overclaiming.")
    return notes


def _next_required_step(status: str) -> str:
    if status == "consolidated_ready_for_open_control_contract":
        return "proceed_to_open_control_philosophy_contract"
    if status.startswith("incomplete"):
        return "complete_or_attach_run02_outputs"
    return "inspect_run02_controls_before_next_stage"


def _read_condition_progress(run_root: Path, condition: str) -> list[dict[str, Any]]:
    return _read_jsonl(run_root / condition / "progress_log.jsonl")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _decision_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        decision = row.get("decision")
        if decision is None:
            continue
        counts[str(decision)] = counts.get(str(decision), 0) + 1
    return counts


def _first_finite(*values: Any) -> float | None:
    for value in values:
        if _is_finite(value):
            return float(value)
    return None


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(values: list[float]) -> float | None:
    finite_values = [float(value) for value in values if _is_finite(value)]
    if not finite_values:
        return None
    return sum(finite_values) / len(finite_values)


def _delta(left: Any, right: Any) -> float | None:
    if not _is_finite(left) or not _is_finite(right):
        return None
    return float(left) - float(right)


def _first_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    for row in rows:
        if _is_finite(row.get(key)):
            return float(row[key])
    return None


def _last_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    for row in reversed(rows):
        if _is_finite(row.get(key)):
            return float(row[key])
    return None


def _max_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if _is_finite(row.get(key))]
    return max(values) if values else None


def _min_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if _is_finite(row.get(key))]
    return min(values) if values else None


def _sum_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if _is_finite(row.get(key))]
    return sum(values) if values else None


def _max_of_keys(rows: list[dict[str, Any]], keys: list[str]) -> float | None:
    values: list[float] = []
    for row in rows:
        for key in keys:
            if _is_finite(row.get(key)):
                values.append(float(row[key]))
    return max(values) if values else None
