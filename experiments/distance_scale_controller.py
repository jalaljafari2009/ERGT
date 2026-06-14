"""Normalization and distance-scale controller for ERGT open control."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

DistanceScaleDecisionName = Literal[
    "increase_distance_scale",
    "decrease_distance_scale",
    "hold_distance_scale",
    "hard_stop_hold",
]


@dataclass(frozen=True)
class DistanceScaleConfig:
    initial_distance_norm_scale: float = 1.0
    min_distance_norm_scale: float = 0.25
    max_distance_norm_scale: float = 4.0
    scale_step: float = 0.10
    min_pre_norm_contrast: float = 0.15
    min_post_norm_contrast: float = 0.20
    target_contrast_retention: float = 0.65
    max_clipping_saturation_rate: float = 0.20
    min_geo_to_qk_ratio: float = 0.03
    max_geo_to_qk_ratio: float = 0.18
    min_post_norm_std: float = 0.25
    max_post_norm_std: float = 1.80
    max_collapse_for_increase: float = 0.35
    uniformity_entropy_threshold: float = 0.92
    release_margin: float = 0.03
    restraint_margin: float = 0.03

    def validate(self) -> None:
        if self.min_distance_norm_scale <= 0:
            raise ValueError("min_distance_norm_scale must be positive")
        if self.max_distance_norm_scale < self.min_distance_norm_scale:
            raise ValueError("max_distance_norm_scale must be >= min_distance_norm_scale")
        if not (
            self.min_distance_norm_scale
            <= self.initial_distance_norm_scale
            <= self.max_distance_norm_scale
        ):
            raise ValueError("initial_distance_norm_scale must be inside bounds")
        if self.scale_step <= 0:
            raise ValueError("scale_step must be positive")
        if self.min_geo_to_qk_ratio <= 0 or self.max_geo_to_qk_ratio <= 0:
            raise ValueError("geo/qk ratio thresholds must be positive")
        if self.min_geo_to_qk_ratio >= self.max_geo_to_qk_ratio:
            raise ValueError("min_geo_to_qk_ratio must be < max_geo_to_qk_ratio")


@dataclass(frozen=True)
class DistanceScaleObservation:
    step: int
    pre_norm_distance_contrast: float
    post_norm_distance_contrast: float
    distance_std_pre_norm: float
    distance_std_post_norm: float
    clipping_saturation_rate: float
    geo_to_qk_ratio: float
    attention_entropy_normalized: float
    collapse_risk: float
    real_distance_advantage: float
    random_distance_advantage: float = 0.0
    shuffled_distance_advantage: float = 0.0
    future_leak_score: float = 0.0
    hard_stop_triggered: bool = False
    hard_stop_reason: str | None = None


@dataclass(frozen=True)
class DistanceScaleDecision:
    step: int
    decision: DistanceScaleDecisionName
    previous_distance_norm_scale: float
    proposed_distance_norm_scale: float
    current_distance_norm_scale: float
    distance_norm_scale_delta: float
    distance_norm_scale_credit: float
    distance_norm_scale_risk_pressure: float
    contrast_evidence: dict[str, float | bool]
    scale_evidence: dict[str, float | bool]
    clipping_evidence: dict[str, float | bool]
    control_distance_evidence: dict[str, float | bool]
    attention_safety_evidence: dict[str, float | bool]
    release_evidence: dict[str, float]
    restraint_evidence: dict[str, float]
    controller_state_snapshot: dict[str, float]
    parameter_trajectory: dict[str, float]
    injected_evidence_ledger: dict[str, dict[str, float | bool]]
    decision_replay_record: dict[str, Any]


class DistanceScaleController:
    """Evidence controller for normalization and distance scale."""

    def __init__(self, config: DistanceScaleConfig | None = None) -> None:
        self.config = config or DistanceScaleConfig()
        self.config.validate()
        self.current_distance_norm_scale = self.config.initial_distance_norm_scale
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.decisions: list[DistanceScaleDecision] = []

    def update(self, observation: DistanceScaleObservation) -> DistanceScaleDecision:
        _validate_observation(observation)
        previous = self.current_distance_norm_scale
        contrast = self._contrast_evidence(observation)
        scale = self._scale_evidence(observation)
        clipping = self._clipping_evidence(observation)
        control = self._control_distance_evidence(observation)
        attention = self._attention_safety_evidence(observation)
        release = self._release_evidence(contrast, scale, clipping, control, attention)
        restraint = self._restraint_evidence(contrast, scale, clipping, control, attention)

        release_score = sum(release.values())
        restraint_score = sum(restraint.values())
        error = release_score - restraint_score
        self.integral_error = _clamp(0.8 * self.integral_error + error, -1.0, 1.0)
        derivative_error = error - self.previous_error
        self.previous_error = error

        scale_delta, decision_name = self._delta_and_decision(
            observation,
            contrast=contrast,
            clipping=clipping,
            control=control,
            attention=attention,
            error=error,
            release_score=release_score,
            restraint_score=restraint_score,
        )
        if observation.hard_stop_triggered or observation.future_leak_score > 0:
            scale_delta = 0.0
            decision_name = "hard_stop_hold"

        proposed = _clamp(
            previous + scale_delta,
            self.config.min_distance_norm_scale,
            self.config.max_distance_norm_scale,
        )
        scale_delta = proposed - previous
        self.current_distance_norm_scale = proposed
        credit = release_score - restraint_score
        risk_pressure = restraint_score
        controller_state = {
            "release_score": release_score,
            "restraint_score": restraint_score,
            "error": error,
            "integral_error": self.integral_error,
            "derivative_error": derivative_error,
        }
        trajectory = {
            "distance_norm_scale_previous": previous,
            "distance_norm_scale_proposed": proposed,
            "distance_norm_scale_next": self.current_distance_norm_scale,
            "distance_norm_scale_delta": scale_delta,
        }
        ledger = {
            "contrast_evidence": contrast,
            "scale_evidence": scale,
            "clipping_evidence": clipping,
            "control_distance_evidence": control,
            "attention_safety_evidence": attention,
            "release_evidence": release,
            "restraint_evidence": restraint,
        }
        replay = {
            "controller": "normalization_distance_scale_controller",
            "decision": decision_name,
            "observation": asdict(observation),
            "config": asdict(self.config),
            "state_before_decision": controller_state,
            "parameter_trajectory": trajectory,
        }
        decision = DistanceScaleDecision(
            step=observation.step,
            decision=decision_name,
            previous_distance_norm_scale=previous,
            proposed_distance_norm_scale=proposed,
            current_distance_norm_scale=self.current_distance_norm_scale,
            distance_norm_scale_delta=scale_delta,
            distance_norm_scale_credit=credit,
            distance_norm_scale_risk_pressure=risk_pressure,
            contrast_evidence=contrast,
            scale_evidence=scale,
            clipping_evidence=clipping,
            control_distance_evidence=control,
            attention_safety_evidence=attention,
            release_evidence=release,
            restraint_evidence=restraint,
            controller_state_snapshot=controller_state,
            parameter_trajectory=trajectory,
            injected_evidence_ledger=ledger,
            decision_replay_record=replay,
        )
        self.decisions.append(decision)
        return decision

    def summary(self) -> dict[str, Any]:
        return {
            "current_distance_norm_scale": self.current_distance_norm_scale,
            "controller_state_snapshot": {
                "integral_error": self.integral_error,
                "previous_error": self.previous_error,
            },
            "parameter_trajectory": [
                decision.parameter_trajectory for decision in self.decisions
            ],
            "decisions": [asdict(decision) for decision in self.decisions],
            "config": asdict(self.config),
        }

    def _contrast_evidence(
        self,
        observation: DistanceScaleObservation,
    ) -> dict[str, float | bool]:
        retention = _safe_ratio(
            observation.post_norm_distance_contrast,
            observation.pre_norm_distance_contrast,
        )
        contrast_available = (
            observation.pre_norm_distance_contrast >= self.config.min_pre_norm_contrast
        )
        retention_deficit = max(
            self.config.target_contrast_retention - retention,
            0.0,
        )
        post_contrast_deficit = max(
            self.config.min_post_norm_contrast
            - observation.post_norm_distance_contrast,
            0.0,
        )
        contrast_erased = contrast_available and (
            retention < self.config.target_contrast_retention
            or observation.post_norm_distance_contrast
            < self.config.min_post_norm_contrast
        )
        erasure_score = retention_deficit + post_contrast_deficit
        return {
            "pre_norm_distance_contrast": observation.pre_norm_distance_contrast,
            "post_norm_distance_contrast": observation.post_norm_distance_contrast,
            "distance_contrast_retention": retention,
            "contrast_available_before_normalization": contrast_available,
            "contrast_erased_by_normalization": contrast_erased,
            "contrast_retention_deficit": retention_deficit,
            "post_norm_contrast_deficit": post_contrast_deficit,
            "normalization_erasure_score": erasure_score,
        }

    def _scale_evidence(
        self,
        observation: DistanceScaleObservation,
    ) -> dict[str, float | bool]:
        geo_underpowered_pressure = max(
            self.config.min_geo_to_qk_ratio - observation.geo_to_qk_ratio,
            0.0,
        ) / self.config.min_geo_to_qk_ratio
        geo_overpowered_pressure = max(
            observation.geo_to_qk_ratio - self.config.max_geo_to_qk_ratio,
            0.0,
        ) / self.config.max_geo_to_qk_ratio
        post_std_low_pressure = max(
            self.config.min_post_norm_std - observation.distance_std_post_norm,
            0.0,
        )
        post_std_high_pressure = max(
            observation.distance_std_post_norm - self.config.max_post_norm_std,
            0.0,
        )
        return {
            "distance_std_pre_norm": observation.distance_std_pre_norm,
            "distance_std_post_norm": observation.distance_std_post_norm,
            "geo_to_qk_ratio": observation.geo_to_qk_ratio,
            "geo_underpowered": geo_underpowered_pressure > 0,
            "geo_overpowered": geo_overpowered_pressure > 0,
            "geo_underpowered_pressure": geo_underpowered_pressure,
            "geo_overpowered_pressure": geo_overpowered_pressure,
            "post_std_low_pressure": post_std_low_pressure,
            "post_std_high_pressure": post_std_high_pressure,
        }

    def _clipping_evidence(
        self,
        observation: DistanceScaleObservation,
    ) -> dict[str, float | bool]:
        clipping_pressure = max(
            observation.clipping_saturation_rate
            - self.config.max_clipping_saturation_rate,
            0.0,
        )
        return {
            "clipping_saturation_rate": observation.clipping_saturation_rate,
            "clipping_high": clipping_pressure > 0,
            "clipping_saturation_pressure": clipping_pressure,
        }

    def _control_distance_evidence(
        self,
        observation: DistanceScaleObservation,
    ) -> dict[str, float | bool]:
        best_control_advantage = max(
            observation.random_distance_advantage,
            observation.shuffled_distance_advantage,
        )
        return {
            "real_distance_advantage": observation.real_distance_advantage,
            "random_distance_advantage": observation.random_distance_advantage,
            "shuffled_distance_advantage": observation.shuffled_distance_advantage,
            "best_control_advantage": best_control_advantage,
            "control_dominates_real": best_control_advantage
            >= observation.real_distance_advantage,
        }

    def _attention_safety_evidence(
        self,
        observation: DistanceScaleObservation,
    ) -> dict[str, float | bool]:
        return {
            "attention_entropy_normalized": observation.attention_entropy_normalized,
            "collapse_risk": observation.collapse_risk,
            "attention_uniform": observation.attention_entropy_normalized
            >= self.config.uniformity_entropy_threshold,
            "collapse_high": observation.collapse_risk
            > self.config.max_collapse_for_increase,
            "future_leak_score": observation.future_leak_score,
            "hard_stop_triggered": observation.hard_stop_triggered
            or observation.future_leak_score > 0,
        }

    def _release_evidence(
        self,
        contrast: dict[str, float | bool],
        scale: dict[str, float | bool],
        clipping: dict[str, float | bool],
        control: dict[str, float | bool],
        attention: dict[str, float | bool],
    ) -> dict[str, float]:
        can_release = (
            contrast["contrast_available_before_normalization"]
            and not clipping["clipping_high"]
            and not control["control_dominates_real"]
            and not attention["attention_uniform"]
            and not attention["collapse_high"]
        )
        release_scale = 1.0 if can_release else 0.0
        contrast_or_scale_need = (
            contrast["contrast_erased_by_normalization"] or scale["geo_underpowered"]
        )
        advantage_scale = release_scale if contrast_or_scale_need else 0.0
        return {
            "contrast_erasure_pressure": float(
                contrast["normalization_erasure_score"]
            )
            * release_scale,
            "geo_underpowered_pressure": float(scale["geo_underpowered_pressure"])
            * release_scale,
            "real_distance_advantage": max(
                float(control["real_distance_advantage"]),
                0.0,
            )
            * advantage_scale,
        }

    def _restraint_evidence(
        self,
        contrast: dict[str, float | bool],
        scale: dict[str, float | bool],
        clipping: dict[str, float | bool],
        control: dict[str, float | bool],
        attention: dict[str, float | bool],
    ) -> dict[str, float]:
        control_dominance = max(
            float(control["best_control_advantage"])
            - float(control["real_distance_advantage"]),
            0.0,
        )
        no_pre_signal = (
            0.35 if not contrast["contrast_available_before_normalization"] else 0.0
        )
        return {
            "no_pre_norm_signal_pressure": no_pre_signal,
            "clipping_saturation_pressure": float(
                clipping["clipping_saturation_pressure"]
            ),
            "geo_overpowered_pressure": float(scale["geo_overpowered_pressure"]),
            "post_norm_std_high_pressure": float(scale["post_std_high_pressure"]),
            "post_norm_std_low_pressure": float(scale["post_std_low_pressure"]),
            "control_dominance_pressure": control_dominance,
            "attention_uniformity_pressure": (
                0.25 if attention["attention_uniform"] else 0.0
            ),
            "collapse_pressure": max(float(attention["collapse_risk"]) - 0.15, 0.0),
            "future_leak_pressure": max(float(attention["future_leak_score"]), 0.0),
        }

    def _delta_and_decision(
        self,
        observation: DistanceScaleObservation,
        *,
        contrast: dict[str, float | bool],
        clipping: dict[str, float | bool],
        control: dict[str, float | bool],
        attention: dict[str, float | bool],
        error: float,
        release_score: float,
        restraint_score: float,
    ) -> tuple[float, DistanceScaleDecisionName]:
        can_increase = (
            contrast["contrast_available_before_normalization"]
            and not clipping["clipping_high"]
            and not control["control_dominates_real"]
            and not attention["attention_uniform"]
            and observation.collapse_risk <= self.config.max_collapse_for_increase
        )
        if error > self.config.release_margin and release_score > restraint_score:
            if can_increase:
                return self.config.scale_step, "increase_distance_scale"
        if error < -self.config.restraint_margin or restraint_score > release_score:
            return -self.config.scale_step, "decrease_distance_scale"
        return 0.0, "hold_distance_scale"


def _validate_observation(observation: DistanceScaleObservation) -> None:
    if observation.step <= 0:
        raise ValueError("observation step must be positive")
    for field, value in asdict(observation).items():
        if field in {"hard_stop_reason", "hard_stop_triggered"}:
            continue
        if value is not None and not isinstance(value, str):
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(f"{field} must be finite when provided")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
