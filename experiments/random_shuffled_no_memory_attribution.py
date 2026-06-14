"""Attribution comparison for random/shuffled/no-memory guarded controls."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from experiments.late_window_post1000_analysis import (
    LateWindowAnalysisConfig,
    analyze_late_window_post1000,
)


@dataclass(frozen=True)
class AttributionComparisonConfig:
    real_condition: str = "real_memory_d"
    baseline_condition: str = "baseline"
    random_condition: str = "random_memory_d"
    shuffled_condition: str = "shuffled_memory_d"
    no_memory_condition: str = "no_memory_real_d"
    instantaneous_condition: str = "instantaneous_real_d"
    alpha_zero_condition: str = "alpha_zero"
    decision_window: str = "1000_2000"
    min_relation_specific_advantage: float = 0.0
    min_attention_advantage: float = 0.0
    dominance_margin: float = 0.0

    def validate(self) -> None:
        if not self.decision_window:
            raise ValueError("decision_window must be non-empty")
        if self.min_relation_specific_advantage < 0:
            raise ValueError("min_relation_specific_advantage must be non-negative")
        if self.min_attention_advantage < 0:
            raise ValueError("min_attention_advantage must be non-negative")
        if self.dominance_margin < 0:
            raise ValueError("dominance_margin must be non-negative")


REQUIRED_ATTRIBUTION_COMPARISON_OUTPUTS = [
    "random_advantage_analysis",
    "shuffled_distribution_bias_analysis",
    "no_memory_comparison",
    "instantaneous_comparison",
    "relation_specific_advantage_estimate",
    "attention_behavior_comparison",
]


def compare_random_shuffled_no_memory_attribution(
    rows: list[dict[str, Any]],
    *,
    config: AttributionComparisonConfig | None = None,
) -> dict[str, Any]:
    """Compare real geometry against generic and ablated controls."""

    active = config or AttributionComparisonConfig()
    active.validate()
    late_config = LateWindowAnalysisConfig(
        real_condition=active.real_condition,
        baseline_condition=active.baseline_condition,
        control_conditions=(
            active.alpha_zero_condition,
            active.random_condition,
            active.shuffled_condition,
            active.no_memory_condition,
            active.instantaneous_condition,
        ),
        decision_window=active.decision_window,
    )
    late_analysis = analyze_late_window_post1000(rows, config=late_config)
    window = late_analysis["condition_window_summaries"][active.decision_window]
    deltas = late_analysis["real_vs_control_window_deltas"][active.decision_window]
    attention = late_analysis["attention_window_analysis"][active.decision_window]

    baseline_loss = _loss(window, active.baseline_condition)
    real_loss = _loss(window, active.real_condition)
    real_gain = _gain_vs_baseline(baseline_loss, real_loss)
    control_profiles = {
        "random": _control_profile(
            window,
            deltas,
            baseline_loss=baseline_loss,
            real_gain=real_gain,
            condition=active.random_condition,
        ),
        "shuffled": _control_profile(
            window,
            deltas,
            baseline_loss=baseline_loss,
            real_gain=real_gain,
            condition=active.shuffled_condition,
        ),
        "no_memory": _control_profile(
            window,
            deltas,
            baseline_loss=baseline_loss,
            real_gain=real_gain,
            condition=active.no_memory_condition,
        ),
        "instantaneous": _control_profile(
            window,
            deltas,
            baseline_loss=baseline_loss,
            real_gain=real_gain,
            condition=active.instantaneous_condition,
        ),
        "alpha_zero": _control_profile(
            window,
            deltas,
            baseline_loss=baseline_loss,
            real_gain=real_gain,
            condition=active.alpha_zero_condition,
        ),
    }

    random_analysis = _random_advantage_analysis(
        control_profiles["random"],
        active,
    )
    shuffled_analysis = _shuffled_distribution_bias_analysis(
        control_profiles["shuffled"],
        control_profiles["random"],
        active,
    )
    no_memory = _ablated_real_comparison(
        control_profiles["no_memory"],
        label="memory_specific_gain",
        explanation=(
            "positive value means stable real memory adds value beyond real "
            "geometry without memory"
        ),
        config=active,
    )
    instantaneous = _ablated_real_comparison(
        control_profiles["instantaneous"],
        label="stable_memory_gain",
        explanation=(
            "positive value means stable memory beats instantaneous real geometry"
        ),
        config=active,
    )
    relation_specific = _relation_specific_advantage(
        control_profiles,
        real_gain=real_gain,
        config=active,
    )
    attention_comparison = _attention_behavior_comparison(
        window,
        attention,
        config=active,
    )

    revision_triggers = _revision_triggers(
        random_analysis=random_analysis,
        shuffled_analysis=shuffled_analysis,
        no_memory=no_memory,
        instantaneous=instantaneous,
        relation_specific=relation_specific,
        attention_comparison=attention_comparison,
    )

    return {
        "config": asdict(active),
        "decision_window": active.decision_window,
        "late_window_source": late_analysis["post_1000_decision_summary"],
        "baseline_mean_validation_loss": baseline_loss,
        "real_mean_validation_loss": real_loss,
        "real_gain_vs_baseline": real_gain,
        "control_profiles": control_profiles,
        "random_advantage_analysis": random_analysis,
        "shuffled_distribution_bias_analysis": shuffled_analysis,
        "no_memory_comparison": no_memory,
        "instantaneous_comparison": instantaneous,
        "relation_specific_advantage_estimate": relation_specific,
        "attention_behavior_comparison": attention_comparison,
        "generic_regularization_present": any(
            bool(profile["control_beats_baseline"])
            for profile in [
                control_profiles["random"],
                control_profiles["shuffled"],
                control_profiles["no_memory"],
                control_profiles["instantaneous"],
            ]
        ),
        "revision_triggers": revision_triggers,
        "attribution_decision": (
            "enter_revision" if revision_triggers else "relation_specific_signal_supported"
        ),
    }


def _control_profile(
    window: dict[str, dict[str, Any]],
    deltas: dict[str, Any],
    *,
    baseline_loss: float | None,
    real_gain: float | None,
    condition: str,
) -> dict[str, Any]:
    control_loss = _loss(window, condition)
    control_gain = _gain_vs_baseline(baseline_loss, control_loss)
    real_delta = deltas["deltas"][condition]["mean_delta"]
    return {
        "condition": condition,
        "mean_validation_loss": control_loss,
        "gain_vs_baseline": control_gain,
        "real_minus_control_loss_gain": real_delta,
        "control_beats_baseline": control_gain is not None and control_gain > 0,
        "control_gain_share_of_real_gain": _ratio(control_gain, real_gain),
        "control_dominates_real": real_delta is not None and real_delta <= 0,
        "matched_points": deltas["deltas"][condition]["points"],
        "positive_delta_points": deltas["deltas"][condition]["positive_delta_points"],
    }


def _random_advantage_analysis(
    random_profile: dict[str, Any],
    config: AttributionComparisonConfig,
) -> dict[str, Any]:
    dominates = _dominates_real(random_profile, config)
    return {
        "condition": random_profile["condition"],
        "random_has_generic_baseline_advantage": random_profile[
            "control_beats_baseline"
        ],
        "random_gain_share_of_real_gain": random_profile[
            "control_gain_share_of_real_gain"
        ],
        "real_advantage_over_random": random_profile["real_minus_control_loss_gain"],
        "random_dominates_real": dominates,
        "interpretation": (
            "revision_required_random_dominates"
            if dominates
            else "random_regularization_present_but_not_dominant"
            if random_profile["control_beats_baseline"]
            else "random_control_not_helpful"
        ),
    }


def _shuffled_distribution_bias_analysis(
    shuffled_profile: dict[str, Any],
    random_profile: dict[str, Any],
    config: AttributionComparisonConfig,
) -> dict[str, Any]:
    dominates = _dominates_real(shuffled_profile, config)
    shuffled_gain = shuffled_profile["gain_vs_baseline"]
    random_gain = random_profile["gain_vs_baseline"]
    return {
        "condition": shuffled_profile["condition"],
        "shuffled_has_distribution_bias_gain": (
            shuffled_gain is not None and shuffled_gain > 0
        ),
        "shuffled_gain_share_of_real_gain": shuffled_profile[
            "control_gain_share_of_real_gain"
        ],
        "shuffled_minus_random_gain": _subtract(shuffled_gain, random_gain),
        "real_advantage_over_shuffled": shuffled_profile["real_minus_control_loss_gain"],
        "shuffled_dominates_real": dominates,
        "interpretation": (
            "revision_required_shuffled_dominates"
            if dominates
            else "distribution_bias_present_but_not_dominant"
            if shuffled_gain is not None and shuffled_gain > 0
            else "shuffled_distribution_bias_not_helpful"
        ),
    }


def _ablated_real_comparison(
    profile: dict[str, Any],
    *,
    label: str,
    explanation: str,
    config: AttributionComparisonConfig,
) -> dict[str, Any]:
    gain = profile["real_minus_control_loss_gain"]
    dominates = _dominates_real(profile, config)
    return {
        "condition": profile["condition"],
        label: gain,
        "control_gain_share_of_real_gain": profile["control_gain_share_of_real_gain"],
        "control_dominates_real": dominates,
        "supports_real_memory_or_stability": gain is not None
        and gain > config.dominance_margin,
        "explanation": explanation,
    }


def _relation_specific_advantage(
    control_profiles: dict[str, dict[str, Any]],
    *,
    real_gain: float | None,
    config: AttributionComparisonConfig,
) -> dict[str, Any]:
    relevant = {
        key: value
        for key, value in control_profiles.items()
        if key in {"random", "shuffled", "no_memory", "instantaneous"}
    }
    advantages = {
        key: profile["real_minus_control_loss_gain"]
        for key, profile in relevant.items()
    }
    finite_advantages = [
        float(value) for value in advantages.values() if _is_finite(value)
    ]
    estimate = min(finite_advantages) if finite_advantages else None
    best_control_key = max(
        relevant,
        key=lambda key: _finite_or_negative_inf(
            relevant[key]["gain_vs_baseline"]
        ),
    )
    best_control = relevant[best_control_key]
    best_control_gain = best_control["gain_vs_baseline"]
    return {
        "advantage_by_control": advantages,
        "estimate": estimate,
        "real_gain_vs_baseline": real_gain,
        "best_control_family": best_control_key,
        "best_control_condition": best_control["condition"],
        "best_control_gain_vs_baseline": best_control_gain,
        "best_control_gain_share_of_real_gain": _ratio(best_control_gain, real_gain),
        "passes_relation_specific_threshold": estimate is not None
        and estimate > config.min_relation_specific_advantage,
        "interpretation": (
            "real_exceeds_best_generic_and_ablation_control"
            if estimate is not None
            and estimate > config.min_relation_specific_advantage
            else "relation_specific_advantage_not_established"
        ),
    }


def _attention_behavior_comparison(
    window: dict[str, dict[str, Any]],
    attention: dict[str, Any],
    *,
    config: AttributionComparisonConfig,
) -> dict[str, Any]:
    real_score = _score(window, config.real_condition)
    controls = {
        "baseline": _score(window, config.baseline_condition),
        "alpha_zero": _score(window, config.alpha_zero_condition),
        "random": _score(window, config.random_condition),
        "shuffled": _score(window, config.shuffled_condition),
        "no_memory": _score(window, config.no_memory_condition),
        "instantaneous": _score(window, config.instantaneous_condition),
    }
    score_deltas = {
        key: _subtract(real_score, value) for key, value in controls.items()
    }
    finite_deltas = [value for value in score_deltas.values() if _is_finite(value)]
    minimum_delta = min(finite_deltas) if finite_deltas else None
    return {
        "real_attention_behavior_score": real_score,
        "control_attention_behavior_scores": controls,
        "real_minus_control_attention_score": score_deltas,
        "minimum_attention_advantage": minimum_delta,
        "stage22_attention_safe": attention["attention_safe_for_window_decision"],
        "attention_separated_from_controls": minimum_delta is not None
        and minimum_delta > config.min_attention_advantage
        and attention["attention_safe_for_window_decision"],
        "collapse_warning": attention["collapse_warning"],
        "uniformity_drift_warning": attention["uniformity_drift_warning"],
        "control_like_attention_warning": attention["control_like_attention_warning"],
    }


def _revision_triggers(
    *,
    random_analysis: dict[str, Any],
    shuffled_analysis: dict[str, Any],
    no_memory: dict[str, Any],
    instantaneous: dict[str, Any],
    relation_specific: dict[str, Any],
    attention_comparison: dict[str, Any],
) -> list[str]:
    triggers = []
    if random_analysis["random_dominates_real"]:
        triggers.append("random_dominates_real")
    if shuffled_analysis["shuffled_dominates_real"]:
        triggers.append("shuffled_dominates_real")
    if no_memory["control_dominates_real"]:
        triggers.append("no_memory_matches_or_beats_real")
    if instantaneous["control_dominates_real"]:
        triggers.append("instantaneous_matches_or_beats_stable_memory")
    if not relation_specific["passes_relation_specific_threshold"]:
        triggers.append("relation_specific_advantage_not_established")
    if not attention_comparison["attention_separated_from_controls"]:
        triggers.append("attention_behavior_not_separated_from_controls")
    return triggers


def _loss(window: dict[str, dict[str, Any]], condition: str) -> float | None:
    value = window[condition]["mean_validation_loss"]
    return float(value) if _is_finite(value) else None


def _score(window: dict[str, dict[str, Any]], condition: str) -> float | None:
    value = window[condition]["mean_attention_behavior_score"]
    return float(value) if _is_finite(value) else None


def _gain_vs_baseline(
    baseline_loss: float | None,
    condition_loss: float | None,
) -> float | None:
    if baseline_loss is None or condition_loss is None:
        return None
    return baseline_loss - condition_loss


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _dominates_real(
    profile: dict[str, Any],
    config: AttributionComparisonConfig,
) -> bool:
    delta = profile["real_minus_control_loss_gain"]
    return delta is not None and delta <= config.dominance_margin


def _is_finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_or_negative_inf(value: Any) -> float:
    return float(value) if _is_finite(value) else float("-inf")
