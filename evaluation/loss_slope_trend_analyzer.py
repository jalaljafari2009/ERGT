"""Loss-slope and trend analyzer for open adaptive ERGT control."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TrendAnalyzerConfig:
    min_points_for_slope: int = 3
    rolling_window_points: int = 4
    ema_beta: float = 0.7
    late_window_start: int = 1000
    post_1000_start: int = 1000

    def validate(self) -> None:
        if self.min_points_for_slope < 2:
            raise ValueError("min_points_for_slope must be >= 2")
        if self.rolling_window_points < self.min_points_for_slope:
            raise ValueError("rolling_window_points must be >= min_points_for_slope")
        if not 0.0 <= self.ema_beta < 1.0:
            raise ValueError("ema_beta must be in [0, 1)")
        if self.late_window_start < 0:
            raise ValueError("late_window_start must be non-negative")
        if self.post_1000_start < 0:
            raise ValueError("post_1000_start must be non-negative")


REQUIRED_TREND_FIELDS = [
    "baseline_validation_loss",
    "baseline_relative_validation_delta",
    "ema_validation_loss",
    "ema_baseline_validation_loss",
    "ema_loss_delta",
    "rolling_slope",
    "baseline_rolling_slope",
    "loss_slope_gain",
    "late_window_slope",
    "post_1000_trend",
    "trend_window_points",
]


def build_loss_slope_trend_analyzer_report(
    *,
    condition_progress: list[dict[str, Any]] | None = None,
    baseline_progress: list[dict[str, Any]] | None = None,
    config: TrendAnalyzerConfig | None = None,
) -> dict[str, Any]:
    """Build the stage-7 trend report from progress rows."""

    config = config or TrendAnalyzerConfig()
    config.validate()
    input_source = "provided_progress"
    if condition_progress is None or baseline_progress is None:
        baseline_progress, condition_progress = _synthetic_progress()
        input_source = "synthetic_progress_smoke"

    baseline_by_step = _rows_by_step(baseline_progress)
    condition_rows = _valid_progress_rows(condition_progress)
    annotated_rows = _annotate_rows(
        condition_rows,
        baseline_by_step=baseline_by_step,
        config=config,
    )
    summary = _summarize(annotated_rows, config=config)
    checks = {
        "baseline_reference_available": all(
            row.get("baseline_validation_loss") is not None for row in annotated_rows
        ),
        "required_trend_fields_emitted": all(
            field in row for row in annotated_rows for field in REQUIRED_TREND_FIELDS
        ),
        "all_numeric_trend_fields_finite_or_none": _all_numeric_fields_finite_or_none(
            annotated_rows,
            REQUIRED_TREND_FIELDS,
        ),
        "rolling_slope_uses_windowed_points": any(
            row["trend_window_points"] >= config.min_points_for_slope
            and row["rolling_slope"] is not None
            for row in annotated_rows
        ),
        "ema_loss_delta_emitted": any(row["ema_loss_delta"] is not None for row in annotated_rows),
        "late_window_slope_emitted": summary["late_window"]["slope"] is not None,
        "post_1000_trend_emitted": summary["post_1000"]["trend"] != "insufficient_points",
        "controller_evidence_fields_available": all(
            summary["controller_evidence"].get(field) is not None
            for field in [
                "latest_loss_slope_gain",
                "latest_ema_loss_delta",
                "late_window_slope",
                "post_1000_trend",
            ]
        ),
    }
    status = "pass" if all(checks.values()) else "fail"

    return {
        "stage": "stage7_loss_slope_and_trend_analyzer",
        "status": status,
        "input_source": input_source,
        "scientific_scope": (
            "trend diagnostics only; controller decisions must cite windowed "
            "slope and EMA evidence instead of one noisy validation point"
        ),
        "config": asdict(config),
        "required_fields": list(REQUIRED_TREND_FIELDS),
        "checks": checks,
        "summary": summary,
        "annotated_progress": annotated_rows,
        "next_required_step": (
            "parameter_attribution_probe" if status == "pass" else "fix_loss_trend_analyzer"
        ),
    }


def _annotate_rows(
    rows: list[dict[str, Any]],
    *,
    baseline_by_step: dict[int, dict[str, Any]],
    config: TrendAnalyzerConfig,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    ema_loss = None
    ema_baseline = None
    for index, row in enumerate(rows):
        step = int(row["step"])
        validation_loss = float(row["validation_loss"])
        baseline_row = _baseline_row_for_step(baseline_by_step, step)
        baseline_loss = (
            float(baseline_row["validation_loss"])
            if baseline_row is not None
            else None
        )
        ema_loss = _ema(ema_loss, validation_loss, config.ema_beta)
        if baseline_loss is not None:
            ema_baseline = _ema(ema_baseline, baseline_loss, config.ema_beta)
        rolling_rows = annotated[max(0, index + 1 - config.rolling_window_points) : index]
        rolling_steps = [int(item["step"]) for item in rolling_rows] + [step]
        rolling_losses = [
            float(item["validation_loss"]) for item in rolling_rows
        ] + [validation_loss]
        rolling_baseline_losses = [
            item["baseline_validation_loss"] for item in rolling_rows
        ] + [baseline_loss]
        rolling_slope = _windowed_slope(
            rolling_steps,
            rolling_losses,
            min_points=config.min_points_for_slope,
        )
        baseline_rolling_slope = _windowed_slope(
            [
                point_step
                for point_step, point_value in zip(
                    rolling_steps,
                    rolling_baseline_losses,
                    strict=True,
                )
                if point_value is not None
            ],
            [
                float(point_value)
                for point_value in rolling_baseline_losses
                if point_value is not None
            ],
            min_points=config.min_points_for_slope,
        )
        loss_slope_gain = (
            baseline_rolling_slope - rolling_slope
            if baseline_rolling_slope is not None and rolling_slope is not None
            else None
        )
        baseline_delta = (
            baseline_loss - validation_loss if baseline_loss is not None else None
        )
        ema_loss_delta = (
            ema_baseline - ema_loss if ema_baseline is not None else None
        )
        annotated_row = {
            **row,
            "step": step,
            "validation_loss": validation_loss,
            "baseline_validation_loss": baseline_loss,
            "baseline_relative_validation_delta": baseline_delta,
            "baseline_centered_improvement": baseline_delta,
            "ema_validation_loss": ema_loss,
            "ema_baseline_validation_loss": ema_baseline,
            "ema_loss_delta": ema_loss_delta,
            "rolling_slope": rolling_slope,
            "baseline_rolling_slope": baseline_rolling_slope,
            "loss_slope_gain": loss_slope_gain,
            "adaptive_slope_gain": loss_slope_gain,
            "late_window": step >= config.late_window_start,
            "late_window_slope": None,
            "post_1000_trend": None,
            "trend_window_points": min(
                len(rolling_steps),
                config.rolling_window_points,
            ),
        }
        annotated.append(annotated_row)

    late_slope = _subset_slope(
        annotated,
        min_step=config.late_window_start,
        min_points=config.min_points_for_slope,
    )
    post_1000 = _trend_label(
        _subset_slope(
            annotated,
            min_step=config.post_1000_start,
            min_points=config.min_points_for_slope,
        )
    )
    for row in annotated:
        row["late_window_slope"] = late_slope
        row["post_1000_trend"] = post_1000
    return annotated


def _summarize(
    annotated_rows: list[dict[str, Any]],
    *,
    config: TrendAnalyzerConfig,
) -> dict[str, Any]:
    latest = annotated_rows[-1] if annotated_rows else {}
    late_slope = _subset_slope(
        annotated_rows,
        min_step=config.late_window_start,
        min_points=config.min_points_for_slope,
    )
    post_1000_slope = _subset_slope(
        annotated_rows,
        min_step=config.post_1000_start,
        min_points=config.min_points_for_slope,
    )
    return {
        "points": len(annotated_rows),
        "latest": latest,
        "late_window": {
            "start_step": config.late_window_start,
            "points": sum(
                1 for row in annotated_rows if int(row["step"]) >= config.late_window_start
            ),
            "slope": late_slope,
            "trend": _trend_label(late_slope),
        },
        "post_1000": {
            "start_step": config.post_1000_start,
            "points": sum(
                1 for row in annotated_rows if int(row["step"]) >= config.post_1000_start
            ),
            "slope": post_1000_slope,
            "trend": _trend_label(post_1000_slope),
        },
        "controller_evidence": {
            "latest_loss_slope_gain": latest.get("loss_slope_gain"),
            "latest_ema_loss_delta": latest.get("ema_loss_delta"),
            "late_window_slope": late_slope,
            "post_1000_trend": _trend_label(post_1000_slope),
            "latest_baseline_relative_validation_delta": latest.get(
                "baseline_relative_validation_delta"
            ),
        },
    }


def _valid_progress_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [
        row
        for row in rows
        if row.get("step") is not None and row.get("validation_loss") is not None
    ]
    valid.sort(key=lambda row: int(row["step"]))
    if not valid:
        raise ValueError("progress rows must include step and validation_loss")
    for row in valid:
        value = float(row["validation_loss"])
        if not math.isfinite(value):
            raise ValueError("validation_loss must be finite")
    return valid


def _rows_by_step(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {
        int(row["step"]): row
        for row in _valid_progress_rows(rows)
    }


def _baseline_row_for_step(
    baseline_by_step: dict[int, dict[str, Any]],
    step: int,
) -> dict[str, Any] | None:
    if step in baseline_by_step:
        return baseline_by_step[step]
    previous_steps = [candidate for candidate in baseline_by_step if candidate <= step]
    if not previous_steps:
        return None
    return baseline_by_step[max(previous_steps)]


def _subset_slope(
    rows: list[dict[str, Any]],
    *,
    min_step: int,
    min_points: int,
) -> float | None:
    subset = [row for row in rows if int(row["step"]) >= min_step]
    return _windowed_slope(
        [int(row["step"]) for row in subset],
        [float(row["validation_loss"]) for row in subset],
        min_points=min_points,
    )


def _windowed_slope(
    steps: list[int],
    values: list[float],
    *,
    min_points: int,
) -> float | None:
    if len(steps) < min_points:
        return None
    return _linear_slope(steps, values)


def _linear_slope(steps: list[int], values: list[float]) -> float:
    if len(steps) != len(values):
        raise ValueError("steps and values must have the same length")
    if len(steps) < 2:
        return 0.0
    x_mean = sum(float(step) for step in steps) / len(steps)
    y_mean = sum(float(value) for value in values) / len(values)
    numerator = sum(
        (float(step) - x_mean) * (float(value) - y_mean)
        for step, value in zip(steps, values, strict=True)
    )
    denominator = sum((float(step) - x_mean) ** 2 for step in steps)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _trend_label(slope: float | None) -> str:
    if slope is None:
        return "insufficient_points"
    if slope < 0:
        return "improving"
    if slope > 0:
        return "worsening"
    return "flat"


def _ema(previous: float | None, value: float, beta: float) -> float:
    if previous is None:
        return value
    return beta * previous + (1.0 - beta) * value


def _all_numeric_fields_finite_or_none(
    rows: list[dict[str, Any]],
    fields: list[str],
) -> bool:
    for row in rows:
        for field in fields:
            value = row.get(field)
            if value is None or isinstance(value, str):
                continue
            if not math.isfinite(float(value)):
                return False
    return True


def _synthetic_progress() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    steps = [100, 500, 900, 1100, 1300, 1500]
    baseline_losses = [5.20, 5.05, 4.92, 4.84, 4.78, 4.74]
    condition_losses = [5.18, 4.98, 4.78, 4.62, 4.48, 4.35]
    baseline = [
        {"condition": "baseline", "step": step, "validation_loss": loss}
        for step, loss in zip(steps, baseline_losses, strict=True)
    ]
    condition = [
        {"condition": "real_stable_causal_d", "step": step, "validation_loss": loss}
        for step, loss in zip(steps, condition_losses, strict=True)
    ]
    return baseline, condition
