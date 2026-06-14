"""Longer-run and multi-seed confirmation contract utilities."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any

from experiments.guarded_2000_step_adaptive_run import REQUIRED_GUARDED_CONDITIONS

CONFIRMATION_PROFILES = {
    "longer_single_seed_5000": {
        "max_steps": 5000,
        "seeds": (2027,),
        "purpose": "single-seed longer-run persistence check",
    },
    "multi_seed_2000": {
        "max_steps": 2000,
        "seeds": (2027, 2028, 2029),
        "purpose": "matched multi-seed robustness check",
    },
}


@dataclass(frozen=True)
class ConfirmationRunConfig:
    eval_interval: int = 100
    live_display_interval: int = 100
    conditions: tuple[str, ...] = tuple(REQUIRED_GUARDED_CONDITIONS)
    profiles: tuple[str, ...] = tuple(CONFIRMATION_PROFILES)
    real_condition: str = "real_memory_d"
    baseline_condition: str = "baseline"
    primary_multi_seed_profile: str = "multi_seed_2000"
    longer_profile: str = "longer_single_seed_5000"
    min_seed_count: int = 3
    min_control_advantage: float = 0.0
    min_median_relation_advantage: float = 0.0
    max_future_leak_score: float = 0.0
    artifact_bundle_name: str = "ergt_04_confirmation_report_bundle.zip"

    def validate(self) -> None:
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.live_display_interval != 100:
            raise ValueError("live_display_interval must remain 100")
        missing = set(REQUIRED_GUARDED_CONDITIONS) - set(self.conditions)
        if missing:
            raise ValueError(f"missing confirmation conditions: {sorted(missing)}")
        unknown_profiles = set(self.profiles) - set(CONFIRMATION_PROFILES)
        if unknown_profiles:
            raise ValueError(f"unknown confirmation profiles: {sorted(unknown_profiles)}")
        if self.primary_multi_seed_profile not in self.profiles:
            raise ValueError("primary_multi_seed_profile must be enabled")
        if self.longer_profile not in self.profiles:
            raise ValueError("longer_profile must be enabled")
        if self.min_seed_count < 1:
            raise ValueError("min_seed_count must be positive")
        if self.min_control_advantage < 0:
            raise ValueError("min_control_advantage must be non-negative")
        if self.min_median_relation_advantage < 0:
            raise ValueError("min_median_relation_advantage must be non-negative")


REQUIRED_STAGE26_OUTPUTS = [
    "confirmation_run_manifest",
    "prerequisite_gate_summary",
    "profile_seed_summaries",
    "aggregate_confirmation_summary",
    "random_shuffled_dominance_audit",
    "stage26_decision",
    "real_confirmation_boundary",
]


def build_confirmation_run_manifest(
    config: ConfirmationRunConfig | None = None,
) -> dict[str, Any]:
    """Build the executable confirmation-run manifest."""

    active = config or ConfirmationRunConfig()
    active.validate()
    profile_plan = []
    for profile in active.profiles:
        spec = CONFIRMATION_PROFILES[profile]
        max_steps = int(spec["max_steps"])
        window = _decision_window(max_steps)
        seeds = list(spec["seeds"])
        profile_plan.append(
            {
                "profile": profile,
                "purpose": spec["purpose"],
                "max_steps": max_steps,
                "eval_interval": active.eval_interval,
                "live_display_interval": active.live_display_interval,
                "decision_window": {
                    "start_step": window[0],
                    "end_step": window[1],
                },
                "seeds": seeds,
                "conditions": list(active.conditions),
                "run_order": list(active.conditions),
                "required_per_step_fields": [
                    "validation_loss",
                    "geo_to_qk_ratio",
                    "attention_entropy",
                    "mean_max_probability",
                    "memory_stability",
                    "memory_persistence",
                    "distance_contrast_retention",
                    "future_leak_score",
                    "attention_behavior_score",
                ],
            }
        )
    return {
        "artifact_bundle_name": active.artifact_bundle_name,
        "checkpoint_artifacts_excluded": True,
        "lightweight_review_artifacts_only": True,
        "profiles": profile_plan,
        "no_peek_rule": (
            "each seed/profile runs conditions sequentially; live real rows cannot "
            "read later random, shuffled, no-memory, or instantaneous controls"
        ),
        "decision_rule": (
            "every seed must show real stable causal geometry beating baseline, "
            "alpha-zero, random, shuffled, no-memory, and instantaneous controls "
            "inside that profile's decision window"
        ),
    }


def generate_confirmation_replay_rows(
    config: ConfirmationRunConfig | None = None,
) -> list[dict[str, Any]]:
    """Generate deterministic confirmation replay rows for contract validation."""

    active = config or ConfirmationRunConfig()
    active.validate()
    rows: list[dict[str, Any]] = []
    for profile in active.profiles:
        spec = CONFIRMATION_PROFILES[profile]
        max_steps = int(spec["max_steps"])
        for seed in spec["seeds"]:
            for condition in active.conditions:
                for step in range(active.eval_interval, max_steps + 1, active.eval_interval):
                    rows.append(_confirmation_row(active, profile, int(seed), condition, step))
    return rows


def analyze_confirmation_replay(
    rows: list[dict[str, Any]],
    *,
    config: ConfirmationRunConfig | None = None,
) -> dict[str, Any]:
    """Analyze confirmation rows by profile and seed."""

    active = config or ConfirmationRunConfig()
    active.validate()
    grouped = _group_rows(rows)
    profile_seed_summaries = []
    for profile in active.profiles:
        spec = CONFIRMATION_PROFILES[profile]
        for seed in spec["seeds"]:
            profile_seed_summaries.append(
                _profile_seed_summary(
                    grouped,
                    profile=profile,
                    seed=int(seed),
                    max_steps=int(spec["max_steps"]),
                    config=active,
                )
            )
    aggregate = _aggregate_summary(profile_seed_summaries, config=active)
    dominance = _random_shuffled_dominance(profile_seed_summaries)
    return {
        "profile_seed_summaries": profile_seed_summaries,
        "aggregate_confirmation_summary": aggregate,
        "random_shuffled_dominance_audit": dominance,
        "stage26_decision": (
            "confirmation_contract_ready"
            if aggregate["all_profiles_pass"]
            and not dominance["random_or_shuffled_dominance_detected"]
            else "confirmation_contract_blocked"
        ),
        "real_confirmation_boundary": (
            "This replay validates confirmation mechanics only. Real longer-run "
            "or multi-seed telemetry is still required before stronger claims."
        ),
    }


def _confirmation_row(
    config: ConfirmationRunConfig,
    profile: str,
    seed: int,
    condition: str,
    step: int,
) -> dict[str, Any]:
    max_steps = int(CONFIRMATION_PROFILES[profile]["max_steps"])
    baseline_loss = _baseline_loss(step, max_steps, seed)
    improvement = _condition_improvement(condition, step, max_steps, seed)
    alpha = _alpha(condition, step, max_steps)
    return {
        "profile": profile,
        "seed": seed,
        "condition": condition,
        "step": step,
        "max_steps": max_steps,
        "validation_loss": baseline_loss - improvement,
        "train_loss": baseline_loss - improvement - 0.04,
        "alpha": alpha,
        "geo_to_qk_ratio": _geo_to_qk(condition, alpha, step, max_steps),
        "attention_entropy": _attention_entropy(condition, step, max_steps),
        "mean_max_probability": _mean_max_probability(condition, step, max_steps),
        "memory_stability": _memory_stability(condition, step, max_steps),
        "memory_persistence": _memory_persistence(condition, step, max_steps),
        "distance_contrast_retention": _distance_contrast(condition, step, max_steps),
        "attention_behavior_score": _attention_score(condition, step, max_steps),
        "future_leak_score": 0.0,
        "control_rng_isolated": condition not in {"baseline", "real_memory_d"},
        "confirmation_replay": True,
        "decision_window": step >= _decision_window(max_steps)[0],
    }


def _profile_seed_summary(
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]],
    *,
    profile: str,
    seed: int,
    max_steps: int,
    config: ConfirmationRunConfig,
) -> dict[str, Any]:
    start, end = _decision_window(max_steps)
    condition_rows = {
        condition: [
            row
            for row in grouped.get((profile, seed, condition), [])
            if start <= int(row["step"]) <= end
        ]
        for condition in config.conditions
    }
    mean_losses = {
        condition: _mean(_numbers(rows, "validation_loss"))
        for condition, rows in condition_rows.items()
    }
    real_loss = mean_losses[config.real_condition]
    deltas = {
        condition: _subtract(mean_losses[condition], real_loss)
        for condition in config.conditions
        if condition != config.real_condition
    }
    finite_deltas = [value for value in deltas.values() if value is not None]
    random_delta = deltas.get("random_memory_d")
    shuffled_delta = deltas.get("shuffled_memory_d")
    return {
        "profile": profile,
        "seed": seed,
        "decision_window": {"start_step": start, "end_step": end},
        "condition_points": {
            condition: len(rows) for condition, rows in condition_rows.items()
        },
        "mean_validation_loss": mean_losses,
        "real_vs_control_deltas": deltas,
        "relation_specific_advantage": min(finite_deltas) if finite_deltas else None,
        "real_beats_all_controls": all(
            value is not None and value > config.min_control_advantage
            for value in deltas.values()
        ),
        "random_dominates_real": random_delta is not None
        and random_delta <= config.min_control_advantage,
        "shuffled_dominates_real": shuffled_delta is not None
        and shuffled_delta <= config.min_control_advantage,
        "max_future_leak_score": max(
            _numbers(
                [
                    row
                    for rows in condition_rows.values()
                    for row in rows
                ],
                "future_leak_score",
            )
            or [None]
        ),
        "mean_attention_behavior_score": {
            condition: _mean(_numbers(rows, "attention_behavior_score"))
            for condition, rows in condition_rows.items()
        },
        "mean_geo_to_qk_ratio": {
            condition: _mean(_numbers(rows, "geo_to_qk_ratio"))
            for condition, rows in condition_rows.items()
        },
    }


def _aggregate_summary(
    summaries: list[dict[str, Any]],
    *,
    config: ConfirmationRunConfig,
) -> dict[str, Any]:
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        by_profile.setdefault(summary["profile"], []).append(summary)
    profile_results = {}
    for profile, rows in by_profile.items():
        advantages = [
            row["relation_specific_advantage"]
            for row in rows
            if row["relation_specific_advantage"] is not None
        ]
        profile_results[profile] = {
            "seed_count": len(rows),
            "passing_seed_count": sum(1 for row in rows if row["real_beats_all_controls"]),
            "all_seeds_pass": all(row["real_beats_all_controls"] for row in rows),
            "median_relation_specific_advantage": (
                statistics.median(advantages) if advantages else None
            ),
            "minimum_relation_specific_advantage": min(advantages) if advantages else None,
        }
    multi = profile_results.get(config.primary_multi_seed_profile, {})
    longer = profile_results.get(config.longer_profile, {})
    all_profiles_pass = all(
        result["all_seeds_pass"]
        and result["median_relation_specific_advantage"] is not None
        and result["median_relation_specific_advantage"]
        > config.min_median_relation_advantage
        for result in profile_results.values()
    )
    return {
        "profile_results": profile_results,
        "multi_seed_profile_passes": bool(
            multi
            and multi["seed_count"] >= config.min_seed_count
            and multi["all_seeds_pass"]
        ),
        "longer_run_profile_passes": bool(longer and longer["all_seeds_pass"]),
        "all_profiles_pass": all_profiles_pass,
    }


def _random_shuffled_dominance(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    random_cases = [
        {
            "profile": row["profile"],
            "seed": row["seed"],
            "delta": row["real_vs_control_deltas"].get("random_memory_d"),
        }
        for row in summaries
        if row["random_dominates_real"]
    ]
    shuffled_cases = [
        {
            "profile": row["profile"],
            "seed": row["seed"],
            "delta": row["real_vs_control_deltas"].get("shuffled_memory_d"),
        }
        for row in summaries
        if row["shuffled_dominates_real"]
    ]
    return {
        "random_dominance_cases": random_cases,
        "shuffled_dominance_cases": shuffled_cases,
        "random_or_shuffled_dominance_detected": bool(random_cases or shuffled_cases),
    }


def _decision_window(max_steps: int) -> tuple[int, int]:
    return max(1000, max_steps // 2), max_steps


def _baseline_loss(step: int, max_steps: int, seed: int) -> float:
    seed_offset = (seed - 2027) * 0.006
    progress = step / max_steps
    return 6.35 - 0.85 * progress + 0.10 * (1.0 - progress) ** 2 + seed_offset


def _condition_improvement(condition: str, step: int, max_steps: int, seed: int) -> float:
    progress = step / max_steps
    late = max(0.0, (step - _decision_window(max_steps)[0]) / max_steps)
    seed_shift = (seed - 2027) * 0.004
    values = {
        "baseline": 0.0,
        "alpha_zero": 0.0,
        "real_memory_d": 0.06 + 0.12 * progress + 0.12 * late + seed_shift,
        "random_memory_d": 0.025 + 0.05 * progress + 0.025 * late + seed_shift * 0.35,
        "shuffled_memory_d": 0.022 + 0.045 * progress + 0.020 * late + seed_shift * 0.25,
        "no_memory_real_d": 0.030 + 0.055 * progress + seed_shift * 0.30,
        "instantaneous_real_d": 0.035 + 0.060 * progress + 0.015 * late + seed_shift * 0.30,
    }
    return values[condition]


def _alpha(condition: str, step: int, max_steps: int) -> float:
    if condition in {"baseline", "alpha_zero"}:
        return 0.0
    if condition == "real_memory_d":
        return min(0.14, 0.02 + 0.12 * step / max_steps)
    return min(0.08, 0.015 + 0.065 * step / max_steps)


def _geo_to_qk(condition: str, alpha: float, step: int, max_steps: int) -> float:
    if condition in {"baseline", "alpha_zero"}:
        return 0.0
    bonus = 0.0
    if condition == "real_memory_d" and step >= _decision_window(max_steps)[0]:
        bonus = 0.015
    return min(0.16, 0.03 + 0.55 * alpha + bonus)


def _attention_entropy(condition: str, step: int, max_steps: int) -> float:
    base = 3.18 - 0.22 * step / max_steps
    return max(2.50, base - (0.08 if condition == "real_memory_d" else 0.0))


def _mean_max_probability(condition: str, step: int, max_steps: int) -> float:
    base = 0.18 + 0.07 * step / max_steps
    return min(0.34, base + (0.02 if condition == "real_memory_d" else 0.0))


def _memory_stability(condition: str, step: int, max_steps: int) -> float:
    if condition == "real_memory_d":
        return min(0.82, 0.48 + 0.28 * step / max_steps)
    if condition in {"random_memory_d", "shuffled_memory_d"}:
        return min(0.58, 0.40 + 0.12 * step / max_steps)
    return min(0.62, 0.42 + 0.12 * step / max_steps)


def _memory_persistence(condition: str, step: int, max_steps: int) -> float:
    if condition == "real_memory_d":
        return min(0.80, 0.46 + 0.25 * step / max_steps)
    return min(0.60, 0.40 + 0.10 * step / max_steps)


def _distance_contrast(condition: str, step: int, max_steps: int) -> float:
    if condition == "real_memory_d":
        return min(0.82, 0.58 + 0.18 * step / max_steps)
    return min(0.58, 0.42 + 0.08 * step / max_steps)


def _attention_score(condition: str, step: int, max_steps: int) -> float:
    if condition == "real_memory_d":
        return min(0.68, 0.36 + 0.22 * step / max_steps)
    if condition in {"random_memory_d", "shuffled_memory_d"}:
        return min(0.46, 0.28 + 0.09 * step / max_steps)
    return min(0.50, 0.30 + 0.10 * step / max_steps)


def _group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["profile"]), int(row["seed"]), str(row["condition"]))
        grouped.setdefault(key, []).append(row)
    for item in grouped.values():
        item.sort(key=lambda row: int(row["step"]))
    return grouped


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


def _subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def confirmation_config_asdict(config: ConfirmationRunConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["conditions"] = list(config.conditions)
    payload["profiles"] = list(config.profiles)
    return payload
