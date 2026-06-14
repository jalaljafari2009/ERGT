"""Guarded 2000-step adaptive ERGT run contract utilities."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

REQUIRED_GUARDED_CONDITIONS = [
    "baseline",
    "alpha_zero",
    "real_memory_d",
    "random_memory_d",
    "shuffled_memory_d",
    "no_memory_real_d",
    "instantaneous_real_d",
]

CONTROL_CONDITION_FAMILIES = {
    "baseline": "baseline",
    "alpha_zero": "alpha_zero",
    "real_memory_d": "real",
    "random_memory_d": "random",
    "shuffled_memory_d": "shuffled",
    "no_memory_real_d": "no_memory",
    "instantaneous_real_d": "instantaneous",
}

GUARDED_WINDOWS = {
    "0_500": (0, 500),
    "500_1000": (500, 1000),
    "1000_1500": (1000, 1500),
    "1500_2000": (1500, 2000),
    "1000_2000": (1000, 2000),
}


@dataclass(frozen=True)
class Guarded2000RunConfig:
    """Configuration for the guarded adaptive run contract."""

    run_profile: str = "adaptive_2000_guarded"
    max_steps: int = 2000
    eval_interval: int = 100
    late_window_start: int = 1000
    live_display_interval: int = 100
    artifact_bundle_name: str = "ergt_03_adaptive_control_report_bundle.zip"
    conditions: tuple[str, ...] = tuple(REQUIRED_GUARDED_CONDITIONS)
    windows: dict[str, tuple[int, int]] = field(default_factory=lambda: GUARDED_WINDOWS)

    def validate(self) -> None:
        if self.max_steps != 2000:
            raise ValueError("guarded run contract requires max_steps=2000")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.live_display_interval != 100:
            raise ValueError("live_display_interval must stay at 100 for this gate")
        missing = set(REQUIRED_GUARDED_CONDITIONS) - set(self.conditions)
        if missing:
            raise ValueError(f"missing guarded conditions: {sorted(missing)}")
        if self.late_window_start < 0 or self.late_window_start >= self.max_steps:
            raise ValueError("late_window_start must be inside the run")


def build_guarded_run_plan(
    config: Guarded2000RunConfig | None = None,
) -> dict[str, Any]:
    """Return a machine-readable guarded run manifest."""

    active = config or Guarded2000RunConfig()
    active.validate()
    steps = list(range(active.eval_interval, active.max_steps + 1, active.eval_interval))
    condition_plan = []
    for order, condition in enumerate(active.conditions, start=1):
        condition_plan.append(
            {
                "order": order,
                "condition": condition,
                "family": CONTROL_CONDITION_FAMILIES[condition],
                "max_steps": active.max_steps,
                "eval_interval": active.eval_interval,
                "live_display_interval": active.live_display_interval,
                "expected_steps": list(steps),
                "required_telemetry": [
                    "validation_loss",
                    "alpha",
                    "geo_to_qk_ratio",
                    "memory_stability",
                    "memory_persistence",
                    "memory_rigidity",
                    "noise_risk",
                    "attention_entropy",
                    "mean_max_probability",
                    "control_rng_isolated",
                    "trainer_status",
                    "trainer_fail_fast_triggered",
                ],
                "claim_role": (
                    "primary_real_geometry"
                    if condition == "real_memory_d"
                    else "control_or_reference"
                ),
            }
        )
    return {
        "run_profile": active.run_profile,
        "max_steps": active.max_steps,
        "eval_interval": active.eval_interval,
        "live_display_interval": active.live_display_interval,
        "late_window_start": active.late_window_start,
        "conditions": list(active.conditions),
        "condition_plan": condition_plan,
        "windows": {
            name: {"start_step": start, "end_step": end}
            for name, (start, end) in active.windows.items()
        },
        "artifact_bundle_name": active.artifact_bundle_name,
        "sequential_order_rule": (
            "baseline -> alpha_zero -> real -> random -> shuffled -> no_memory -> "
            "instantaneous; live real rows cannot read later control rows"
        ),
    }


def generate_guarded_2000_telemetry_rows(
    config: Guarded2000RunConfig | None = None,
) -> list[dict[str, Any]]:
    """Generate deterministic contract telemetry for guarded-run validation."""

    active = config or Guarded2000RunConfig()
    active.validate()
    rows: list[dict[str, Any]] = []
    for condition in active.conditions:
        for step in range(
            active.eval_interval,
            active.max_steps + 1,
            active.eval_interval,
        ):
            rows.append(_telemetry_row(active, condition, step))
    return rows


def summarize_guarded_replay(
    rows: list[dict[str, Any]],
    *,
    config: Guarded2000RunConfig | None = None,
) -> dict[str, Any]:
    """Summarize condition comparability and late-window readiness."""

    active = config or Guarded2000RunConfig()
    active.validate()
    by_condition = {
        condition: [row for row in rows if row.get("condition") == condition]
        for condition in active.conditions
    }
    condition_steps = {
        condition: [int(row["step"]) for row in condition_rows]
        for condition, condition_rows in by_condition.items()
    }
    expected_steps = list(
        range(active.eval_interval, active.max_steps + 1, active.eval_interval)
    )
    window_summaries = {
        name: _window_summary(by_condition, start, end)
        for name, (start, end) in active.windows.items()
    }
    return {
        "conditions": list(active.conditions),
        "expected_steps": expected_steps,
        "condition_steps": condition_steps,
        "condition_row_counts": {
            condition: len(condition_rows)
            for condition, condition_rows in by_condition.items()
        },
        "all_conditions_present": all(by_condition.values()),
        "all_conditions_have_identical_steps": all(
            steps == expected_steps for steps in condition_steps.values()
        ),
        "all_conditions_reach_2000": all(
            steps and max(steps) == active.max_steps for steps in condition_steps.values()
        ),
        "late_window_start": active.late_window_start,
        "late_window_step_count": len(
            [step for step in expected_steps if step >= active.late_window_start]
        ),
        "window_summaries": window_summaries,
    }


def _telemetry_row(
    config: Guarded2000RunConfig,
    condition: str,
    step: int,
) -> dict[str, Any]:
    family = CONTROL_CONDITION_FAMILIES[condition]
    baseline_loss = _baseline_loss(step)
    improvement = _condition_improvement(condition, step)
    alpha = _alpha_for_condition(condition, step)
    return {
        "step": step,
        "condition": condition,
        "condition_family": family,
        "run_profile": config.run_profile,
        "max_steps": config.max_steps,
        "late_window": step >= config.late_window_start,
        "train_loss": baseline_loss - improvement - 0.04,
        "validation_loss": baseline_loss - improvement,
        "alpha": alpha,
        "alpha_effective": alpha,
        "alpha_next": _next_alpha(condition, alpha, step),
        "alpha_delta": _next_alpha(condition, alpha, step) - alpha,
        "alpha_decision": _alpha_decision(condition, step),
        "loss_slope_gain": _loss_slope_gain(condition, step),
        "baseline_centered_improvement": improvement,
        "geo_to_qk_ratio": _geo_to_qk_ratio(condition, alpha, step),
        "attention_entropy": _attention_entropy(condition, step),
        "mean_max_probability": _mean_max_probability(condition, step),
        "rigidity_risk": _rigidity_risk(condition, step),
        "control_penalty": 0.0,
        "control_rng_isolated": condition not in {"baseline", "real_memory_d"},
        "trainer_status": "pending",
        "trainer_fail_fast_triggered": False,
        "real_vs_baseline_delta": improvement if condition == "real_memory_d" else None,
        "memory_stability": _memory_stability(condition, step),
        "memory_turnover": _memory_turnover(condition, step),
        "memory_persistence": _memory_persistence(condition, step),
        "memory_rigidity": _memory_rigidity(condition, step),
        "noise_risk": _noise_risk(condition, step),
        "attention_behavior_regime": "useful_noncollapsed",
        "attention_behavior_score": _attention_behavior_score(condition, step),
        "attention_behavior_separation": _attention_behavior_separation(condition, step),
        "distance_contrast_retention": _distance_contrast_retention(condition, step),
        "future_leak_score": 0.0,
        "meta_top_signal": "memory_state" if condition == "real_memory_d" else "loss_trend",
        "meta_attention_entropy": 1.80,
        "meta_alpha_weight": 0.20,
        "meta_memory_weight": 0.25,
        "meta_gate_weight": 0.12,
        "meta_reach_weight": 0.17,
        "meta_norm_weight": 0.18,
        "controller_conflict_score": 0.10,
        "meta_control_confidence": 0.74,
        "memory_eta_decision": (
            "increase_eta" if condition == "real_memory_d" and step >= 1000 else "hold"
        ),
        "memory_decay_decision": "hold",
        "gate_floor_decision": "hold",
        "causal_reachability_decision": "hold",
        "distance_norm_scale_decision": "hold",
        "joint_budget_decision": "balanced",
    }


def _baseline_loss(step: int) -> float:
    return 6.25 - 0.00042 * step + 0.000000045 * (step - 1200) ** 2


def _condition_improvement(condition: str, step: int) -> float:
    ramp = min(1.0, step / 2000)
    late = max(0.0, (step - 900) / 1100)
    values = {
        "baseline": 0.0,
        "alpha_zero": 0.0,
        "real_memory_d": 0.04 + 0.10 * ramp + 0.10 * late,
        "random_memory_d": 0.02 + 0.04 * ramp + 0.02 * late,
        "shuffled_memory_d": 0.018 + 0.035 * ramp + 0.015 * late,
        "no_memory_real_d": 0.025 + 0.045 * ramp,
        "instantaneous_real_d": 0.03 + 0.05 * ramp + 0.01 * late,
    }
    return values[condition]


def _alpha_for_condition(condition: str, step: int) -> float:
    if condition in {"baseline", "alpha_zero"}:
        return 0.0
    if condition == "real_memory_d":
        return min(0.12, 0.02 + 0.00005 * step)
    return min(0.08, 0.015 + 0.000025 * step)


def _next_alpha(condition: str, alpha: float, step: int) -> float:
    if condition == "real_memory_d" and step < 2000:
        return min(0.12, alpha + 0.004)
    return alpha


def _alpha_decision(condition: str, step: int) -> str:
    if condition == "real_memory_d" and step >= 1000:
        return "grow_alpha_with_late_slope_support"
    if condition == "real_memory_d":
        return "warmup_alpha"
    return "hold"


def _loss_slope_gain(condition: str, step: int) -> float:
    if condition == "real_memory_d":
        return 0.02 + max(0.0, step - 900) / 20000
    if condition in {"random_memory_d", "shuffled_memory_d"}:
        return 0.005
    return 0.0


def _geo_to_qk_ratio(condition: str, alpha: float, step: int) -> float:
    if condition in {"baseline", "alpha_zero"}:
        return 0.0
    base = 0.03 + alpha * 0.55
    return min(0.16, base + (0.01 if step >= 1000 and condition == "real_memory_d" else 0.0))


def _attention_entropy(condition: str, step: int) -> float:
    base = 3.20 - 0.00012 * step
    if condition == "real_memory_d":
        return max(2.45, base - 0.08)
    return max(2.55, base)


def _mean_max_probability(condition: str, step: int) -> float:
    base = 0.18 + 0.00003 * step
    if condition == "real_memory_d":
        return min(0.32, base + 0.02)
    return min(0.30, base)


def _rigidity_risk(condition: str, step: int) -> float:
    if condition == "real_memory_d":
        return max(0.04, 0.10 - 0.00002 * step)
    return 0.08


def _memory_stability(condition: str, step: int) -> float:
    if condition == "real_memory_d":
        return min(0.78, 0.40 + 0.00018 * step)
    if condition in {"random_memory_d", "shuffled_memory_d"}:
        return min(0.55, 0.34 + 0.00008 * step)
    return min(0.60, 0.36 + 0.00009 * step)


def _memory_turnover(condition: str, step: int) -> float:
    if condition == "real_memory_d":
        return max(0.08, 0.32 - 0.00010 * step)
    return max(0.14, 0.30 - 0.00005 * step)


def _memory_persistence(condition: str, step: int) -> float:
    if condition == "real_memory_d":
        return min(0.76, 0.42 + 0.00015 * step)
    return min(0.58, 0.38 + 0.00008 * step)


def _memory_rigidity(condition: str, step: int) -> float:
    if condition == "real_memory_d":
        return min(0.13, 0.07 + 0.000015 * step)
    return 0.09


def _noise_risk(condition: str, step: int) -> float:
    if condition == "real_memory_d":
        return max(0.05, 0.18 - 0.00006 * step)
    return max(0.08, 0.17 - 0.00003 * step)


def _attention_behavior_score(condition: str, step: int) -> float:
    if condition == "real_memory_d":
        return min(0.62, 0.30 + 0.00012 * step)
    if condition in {"random_memory_d", "shuffled_memory_d"}:
        return min(0.44, 0.24 + 0.00006 * step)
    return min(0.48, 0.25 + 0.00007 * step)


def _attention_behavior_separation(condition: str, step: int) -> float:
    if condition == "real_memory_d":
        return min(0.18, 0.04 + 0.00006 * step)
    return 0.02


def _distance_contrast_retention(condition: str, step: int) -> float:
    if condition == "real_memory_d":
        return min(0.78, 0.48 + 0.00013 * step)
    if condition in {"random_memory_d", "shuffled_memory_d"}:
        return min(0.50, 0.38 + 0.00004 * step)
    return min(0.58, 0.40 + 0.00006 * step)


def _window_summary(
    by_condition: dict[str, list[dict[str, Any]]],
    start: int,
    end: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for condition, rows in by_condition.items():
        window_rows = [
            row for row in rows if start <= int(row["step"]) <= end
        ]
        losses = [float(row["validation_loss"]) for row in window_rows]
        summary[condition] = {
            "points": len(window_rows),
            "mean_validation_loss": statistics.fmean(losses) if losses else None,
            "start_step": window_rows[0]["step"] if window_rows else None,
            "end_step": window_rows[-1]["step"] if window_rows else None,
        }
    return summary


def guarded_config_asdict(config: Guarded2000RunConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["conditions"] = list(config.conditions)
    payload["windows"] = {
        name: {"start_step": start, "end_step": end}
        for name, (start, end) in config.windows.items()
    }
    return payload
