"""Adaptive Alpha Controller v2 for replayable ERGT alpha search."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AdaptiveAlphaV2Config:
    initial_alpha: float = 0.0
    min_alpha: float = 0.0
    max_alpha: float = 0.5
    max_delta_per_decision: float = 0.02
    proportional_gain: float = 0.06
    integral_gain: float = 0.02
    derivative_gain: float = 0.01
    integral_decay: float = 0.85
    integral_limit: float = 1.0
    release_margin: float = 0.02
    shrink_margin: float = -0.02
    slope_weight: float = 1.0
    ema_delta_weight: float = 0.5
    late_window_weight: float = 0.4
    post_1000_weight: float = 0.4
    rigidity_weight: float = 0.35
    collapse_weight: float = 0.5
    control_family_weight: float = 0.6
    geo_qk_pressure_weight: float = 0.2
    target_geo_to_qk_ratio: float = 0.10

    def validate(self) -> None:
        if self.max_alpha < self.min_alpha:
            raise ValueError("max_alpha must be >= min_alpha")
        if self.max_delta_per_decision <= 0:
            raise ValueError("max_delta_per_decision must be positive")
        if not 0.0 <= self.integral_decay < 1.0:
            raise ValueError("integral_decay must be in [0, 1)")
        if self.integral_limit <= 0:
            raise ValueError("integral_limit must be positive")
        if self.target_geo_to_qk_ratio <= 0:
            raise ValueError("target_geo_to_qk_ratio must be positive")


@dataclass(frozen=True)
class AlphaV2Observation:
    step: int
    loss_slope_gain: float | None = None
    ema_loss_delta: float | None = None
    late_window_slope: float | None = None
    post_1000_trend: str | None = None
    rigidity_risk: float | None = None
    collapse_risk: float | None = None
    control_penalty: float | None = None
    random_loss_slope_gain: float | None = None
    shuffled_loss_slope_gain: float | None = None
    geo_to_qk_ratio: float | None = None
    hard_stop_triggered: bool = False
    hard_stop_reason: str | None = None


@dataclass(frozen=True)
class AlphaV2Decision:
    step: int
    previous_alpha: float
    proposed_alpha: float
    next_alpha: float
    current_alpha: float
    alpha_delta: float
    decision: str
    release_evidence: dict[str, float]
    restraint_evidence: dict[str, float]
    slope_evidence: dict[str, float | str | None]
    rigidity_evidence: dict[str, float | bool]
    control_family_evidence: dict[str, float | bool | None]
    controller_state_snapshot: dict[str, float | None]
    parameter_trajectory: dict[str, float]
    injected_evidence_ledger: dict[str, Any]
    decision_replay_record: dict[str, Any]


class AdaptiveAlphaControllerV2:
    """PID-inspired alpha search controller.

    This controller uses proportional, integral, and derivative components over
    evidence balance. Ordinary risks become restraint pressure. Only explicit
    hard-stop observations prevent alpha movement.
    """

    def __init__(self, config: AdaptiveAlphaV2Config | dict[str, Any] | None = None) -> None:
        if isinstance(config, AdaptiveAlphaV2Config):
            self.config = config
        else:
            self.config = AdaptiveAlphaV2Config(**(config or {}))
        self.config.validate()
        self.current_alpha = _clamp(
            self.config.initial_alpha,
            self.config.min_alpha,
            self.config.max_alpha,
        )
        self.integral_error = 0.0
        self.previous_error: float | None = None
        self.decisions: list[AlphaV2Decision] = []

    def update(self, observation: AlphaV2Observation) -> AlphaV2Decision:
        _validate_observation(observation)
        previous_alpha = self.current_alpha
        release = self._release_evidence(observation)
        restraint = self._restraint_evidence(observation)
        slope = self._slope_evidence(observation)
        rigidity = self._rigidity_evidence(observation)
        control = self._control_family_evidence(observation)

        release_score = sum(release.values())
        restraint_score = sum(restraint.values())
        error = release_score - restraint_score
        derivative = 0.0 if self.previous_error is None else error - self.previous_error
        self.integral_error = _clamp(
            self.config.integral_decay * self.integral_error + error,
            -self.config.integral_limit,
            self.config.integral_limit,
        )
        raw_delta = (
            self.config.proportional_gain * error
            + self.config.integral_gain * self.integral_error
            + self.config.derivative_gain * derivative
        )
        alpha_delta = _clamp(
            raw_delta,
            -self.config.max_delta_per_decision,
            self.config.max_delta_per_decision,
        )
        decision = self._decision_label(
            observation=observation,
            error=error,
            alpha_delta=alpha_delta,
        )
        if observation.hard_stop_triggered:
            alpha_delta = 0.0
        proposed_alpha = _clamp(
            previous_alpha + alpha_delta,
            self.config.min_alpha,
            self.config.max_alpha,
        )
        next_alpha = proposed_alpha
        self.current_alpha = next_alpha
        self.previous_error = error

        state = {
            "error": error,
            "release_score": release_score,
            "restraint_score": restraint_score,
            "integral_error": self.integral_error,
            "derivative_error": derivative,
            "raw_delta": raw_delta,
        }
        trajectory = {
            "alpha_previous": previous_alpha,
            "alpha_proposed": proposed_alpha,
            "alpha_next": next_alpha,
            "alpha_delta": next_alpha - previous_alpha,
        }
        evidence_ledger = {
            "release_evidence": release,
            "restraint_evidence": restraint,
            "slope_evidence": slope,
            "rigidity_evidence": rigidity,
            "control_family_evidence": control,
        }
        replay = {
            "controller": "adaptive_alpha_v2_pid_inspired",
            "config": asdict(self.config),
            "observation": asdict(observation),
            "state_before_decision": state,
            "parameter_trajectory": trajectory,
            "decision": decision,
        }
        record = AlphaV2Decision(
            step=observation.step,
            previous_alpha=previous_alpha,
            proposed_alpha=proposed_alpha,
            next_alpha=next_alpha,
            current_alpha=next_alpha,
            alpha_delta=next_alpha - previous_alpha,
            decision=decision,
            release_evidence=release,
            restraint_evidence=restraint,
            slope_evidence=slope,
            rigidity_evidence=rigidity,
            control_family_evidence=control,
            controller_state_snapshot=state,
            parameter_trajectory=trajectory,
            injected_evidence_ledger=evidence_ledger,
            decision_replay_record=replay,
        )
        self.decisions.append(record)
        return record

    def summary(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "current_alpha": self.current_alpha,
            "controller_state_snapshot": {
                "integral_error": self.integral_error,
                "previous_error": self.previous_error,
            },
            "parameter_trajectory": [
                decision.parameter_trajectory for decision in self.decisions
            ],
            "decisions": [asdict(decision) for decision in self.decisions],
        }

    def _release_evidence(self, observation: AlphaV2Observation) -> dict[str, float]:
        values = {
            "loss_slope_gain": self.config.slope_weight
            * max(0.0, _finite_or_zero(observation.loss_slope_gain)),
            "ema_loss_delta": self.config.ema_delta_weight
            * max(0.0, _finite_or_zero(observation.ema_loss_delta)),
        }
        if observation.late_window_slope is not None and observation.late_window_slope < 0:
            values["late_window_improving"] = self.config.late_window_weight * min(
                abs(observation.late_window_slope),
                1.0,
            )
        else:
            values["late_window_improving"] = 0.0
        values["post_1000_improving"] = (
            self.config.post_1000_weight
            if observation.post_1000_trend == "improving"
            else 0.0
        )
        return values

    def _restraint_evidence(self, observation: AlphaV2Observation) -> dict[str, float]:
        geo_ratio = _finite_or_none(observation.geo_to_qk_ratio)
        geo_pressure = 0.0
        if geo_ratio is not None:
            geo_pressure = max(0.0, geo_ratio - self.config.target_geo_to_qk_ratio)
            geo_pressure = geo_pressure / self.config.target_geo_to_qk_ratio
        values = {
            "rigidity_pressure": self.config.rigidity_weight
            * max(0.0, _finite_or_zero(observation.rigidity_risk)),
            "collapse_pressure": self.config.collapse_weight
            * max(0.0, _finite_or_zero(observation.collapse_risk)),
            "control_penalty": self.config.control_family_weight
            * max(0.0, _finite_or_zero(observation.control_penalty)),
            "geo_qk_pressure": self.config.geo_qk_pressure_weight * geo_pressure,
        }
        real_gain = _finite_or_zero(observation.loss_slope_gain)
        control_gain = max(
            _finite_or_zero(observation.random_loss_slope_gain),
            _finite_or_zero(observation.shuffled_loss_slope_gain),
        )
        values["control_family_dominance"] = (
            self.config.control_family_weight * max(0.0, control_gain - real_gain)
        )
        return values

    def _slope_evidence(
        self,
        observation: AlphaV2Observation,
    ) -> dict[str, float | str | None]:
        return {
            "loss_slope_gain": observation.loss_slope_gain,
            "ema_loss_delta": observation.ema_loss_delta,
            "late_window_slope": observation.late_window_slope,
            "post_1000_trend": observation.post_1000_trend,
        }

    def _rigidity_evidence(self, observation: AlphaV2Observation) -> dict[str, float | bool]:
        return {
            "rigidity_risk": _finite_or_zero(observation.rigidity_risk),
            "collapse_risk": _finite_or_zero(observation.collapse_risk),
            "hard_stop_triggered": observation.hard_stop_triggered,
        }

    def _control_family_evidence(
        self,
        observation: AlphaV2Observation,
    ) -> dict[str, float | bool | None]:
        real_gain = _finite_or_none(observation.loss_slope_gain)
        random_gain = _finite_or_none(observation.random_loss_slope_gain)
        shuffled_gain = _finite_or_none(observation.shuffled_loss_slope_gain)
        best_control = max(
            _finite_or_zero(random_gain),
            _finite_or_zero(shuffled_gain),
        )
        return {
            "real_loss_slope_gain": real_gain,
            "random_loss_slope_gain": random_gain,
            "shuffled_loss_slope_gain": shuffled_gain,
            "control_penalty": _finite_or_none(observation.control_penalty),
            "control_dominates_real": (
                real_gain is not None and best_control > real_gain
            ),
        }

    def _decision_label(
        self,
        *,
        observation: AlphaV2Observation,
        error: float,
        alpha_delta: float,
    ) -> str:
        if observation.hard_stop_triggered:
            return "hard_stop_hold"
        if error >= self.config.release_margin and alpha_delta > 0:
            return "grow_pid_release"
        if error <= self.config.shrink_margin and alpha_delta < 0:
            return "shrink_pid_restraint"
        if alpha_delta > 0:
            return "probe_up"
        if alpha_delta < 0:
            return "probe_down"
        return "hold_balance"


def _validate_observation(observation: AlphaV2Observation) -> None:
    if observation.step <= 0:
        raise ValueError("observation step must be positive")
    for field, value in asdict(observation).items():
        if field in {"post_1000_trend", "hard_stop_reason", "hard_stop_triggered"}:
            continue
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"{field} must be finite when provided")


def _finite_or_zero(value: float | None) -> float:
    finite = _finite_or_none(value)
    return 0.0 if finite is None else finite


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
