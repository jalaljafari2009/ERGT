"""Adaptive competitive alpha control for ERGT geometry injection."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from attention.geo_attention import GeoAttention


@dataclass(frozen=True)
class AdaptiveAlphaConfig:
    """Configuration for evidence-driven alpha updates.

    Alpha is allowed to grow when the geometry condition improves the validation
    trajectory relative to a reference. Rigidity risks only penalize growth; they
    are not hard geometry-ratio caps.
    """

    initial_alpha: float = 0.0
    min_alpha: float = 0.0
    max_alpha: float = 0.25
    decision_interval_steps: int = 100
    min_points_for_slope: int = 4
    slope_window_points: int = 5
    exploration_points: int = 4
    exploration_alpha: float = 0.025
    exploration_step: float = 0.04
    growth_step: float = 0.04
    decay_step: float = 0.02
    max_change_per_decision: float = 0.01
    inertia: float = 0.8
    score_scale: float = 0.001
    positive_margin: float = 0.0002
    negative_margin: float = -0.0002
    slope_gain_weight: float = 1.0
    advantage_weight: float = 0.25
    geo_qk_target: float = 0.06
    geo_qk_risk_weight: float = 0.0004
    entropy_drop_weight: float = 0.0002
    max_probability_target: float = 0.35
    max_probability_risk_weight: float = 0.0003
    ema_beta: float = 0.7

    def validate(self) -> None:
        if self.max_alpha < self.min_alpha:
            raise ValueError("max_alpha must be >= min_alpha")
        if not 0.0 <= self.inertia < 1.0:
            raise ValueError("inertia must be in [0, 1)")
        if not 0.0 <= self.ema_beta < 1.0:
            raise ValueError("ema_beta must be in [0, 1)")
        if self.decision_interval_steps <= 0:
            raise ValueError("decision_interval_steps must be positive")
        if self.min_points_for_slope < 2:
            raise ValueError("min_points_for_slope must be >= 2")
        if self.slope_window_points < self.min_points_for_slope:
            raise ValueError("slope_window_points must be >= min_points_for_slope")
        for name in [
            "exploration_step",
            "growth_step",
            "decay_step",
            "max_change_per_decision",
            "score_scale",
        ]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class AlphaObservation:
    step: int
    validation_loss: float
    reference_validation_loss: float | None = None
    geo_to_qk_ratio: float | None = None
    attention_entropy: float | None = None
    mean_max_probability: float | None = None


@dataclass(frozen=True)
class AlphaDecision:
    step: int
    previous_alpha: float
    next_alpha: float
    alpha_delta: float
    decision: str
    score: float | None
    slope_gain: float | None
    advantage: float | None
    geo_qk_risk: float
    entropy_risk: float
    max_probability_risk: float
    points_used: int


class AdaptiveAlphaController:
    """Loss-feedback controller for competitive geometry injection."""

    def __init__(self, config: AdaptiveAlphaConfig | dict[str, Any] | None = None) -> None:
        if isinstance(config, AdaptiveAlphaConfig):
            self.config = config
        else:
            self.config = AdaptiveAlphaConfig(**(config or {}))
        self.config.validate()
        self.current_alpha = _clamp(
            self.config.initial_alpha,
            self.config.min_alpha,
            self.config.max_alpha,
        )
        self.history: list[AlphaObservation] = []
        self.decisions: list[AlphaDecision] = []
        self._ema_score: float | None = None
        self._ema_slope_gain: float | None = None
        self._ema_advantage: float | None = None
        self._initial_entropy: float | None = None

    def update(self, observation: AlphaObservation) -> AlphaDecision:
        self._validate_observation(observation)
        self.history.append(observation)
        if observation.attention_entropy is not None and self._initial_entropy is None:
            self._initial_entropy = observation.attention_entropy

        previous_alpha = self.current_alpha
        score, slope_gain, advantage = self._score()
        geo_qk_risk = self._geo_qk_risk(observation.geo_to_qk_ratio)
        entropy_risk = self._entropy_risk(observation.attention_entropy)
        max_probability_risk = self._max_probability_risk(observation.mean_max_probability)

        risk = (
            self.config.geo_qk_risk_weight * geo_qk_risk
            + self.config.entropy_drop_weight * entropy_risk
            + self.config.max_probability_risk_weight * max_probability_risk
        )
        effective_score = None if score is None else score - risk

        proposed_alpha, decision = self._propose_alpha(previous_alpha, effective_score)
        next_alpha = self._inertial_update(previous_alpha, proposed_alpha)
        next_alpha = _clamp(next_alpha, self.config.min_alpha, self.config.max_alpha)
        alpha_delta = next_alpha - previous_alpha
        self.current_alpha = next_alpha

        record = AlphaDecision(
            step=observation.step,
            previous_alpha=previous_alpha,
            next_alpha=next_alpha,
            alpha_delta=alpha_delta,
            decision=decision,
            score=effective_score,
            slope_gain=slope_gain,
            advantage=advantage,
            geo_qk_risk=geo_qk_risk,
            entropy_risk=entropy_risk,
            max_probability_risk=max_probability_risk,
            points_used=min(len(self.history), self.config.slope_window_points),
        )
        self.decisions.append(record)
        return record

    def summary(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "current_alpha": self.current_alpha,
            "decisions": [asdict(decision) for decision in self.decisions],
        }

    def _score(self) -> tuple[float | None, float | None, float | None]:
        window = self.history[-self.config.slope_window_points :]
        if len(window) < self.config.min_points_for_slope:
            return None, None, None

        own_slope = _linear_slope(
            [item.step for item in window],
            [item.validation_loss for item in window],
        )
        reference_items = [
            item for item in window if item.reference_validation_loss is not None
        ]
        if len(reference_items) >= self.config.min_points_for_slope:
            reference_slope = _linear_slope(
                [item.step for item in reference_items],
                [float(item.reference_validation_loss) for item in reference_items],
            )
            latest = window[-1]
            advantage = (
                float(latest.reference_validation_loss) - latest.validation_loss
                if latest.reference_validation_loss is not None
                else None
            )
        else:
            reference_slope = 0.0
            advantage = None

        slope_gain = reference_slope - own_slope
        self._ema_slope_gain = _ema(self._ema_slope_gain, slope_gain, self.config.ema_beta)
        if advantage is not None:
            self._ema_advantage = _ema(self._ema_advantage, advantage, self.config.ema_beta)

        score = self.config.slope_gain_weight * float(self._ema_slope_gain)
        if self._ema_advantage is not None:
            score += self.config.advantage_weight * float(self._ema_advantage)
        self._ema_score = _ema(self._ema_score, score, self.config.ema_beta)
        return float(self._ema_score), float(self._ema_slope_gain), self._ema_advantage

    def _propose_alpha(self, previous_alpha: float, score: float | None) -> tuple[float, str]:
        if len(self.history) <= self.config.exploration_points:
            if previous_alpha < self.config.exploration_alpha:
                return previous_alpha + self.config.exploration_step, "explore_up"
            return previous_alpha, "explore_hold"

        if score is None:
            return previous_alpha, "hold_insufficient_slope"
        if score > self.config.positive_margin:
            strength = min(abs(score) / self.config.score_scale, 1.0)
            return previous_alpha + self.config.growth_step * strength, "grow"
        if score < self.config.negative_margin:
            strength = min(abs(score) / self.config.score_scale, 1.0)
            return previous_alpha - self.config.decay_step * strength, "shrink"
        return previous_alpha, "hold_margin"

    def _inertial_update(self, previous_alpha: float, proposed_alpha: float) -> float:
        raw_delta = (1.0 - self.config.inertia) * (proposed_alpha - previous_alpha)
        delta = _clamp(
            raw_delta,
            -self.config.max_change_per_decision,
            self.config.max_change_per_decision,
        )
        return previous_alpha + delta

    def _geo_qk_risk(self, geo_to_qk_ratio: float | None) -> float:
        if geo_to_qk_ratio is None or not math.isfinite(geo_to_qk_ratio):
            return 0.0
        if self.config.geo_qk_target <= 0:
            return 0.0
        return max(0.0, geo_to_qk_ratio - self.config.geo_qk_target) / self.config.geo_qk_target

    def _entropy_risk(self, attention_entropy: float | None) -> float:
        if (
            attention_entropy is None
            or self._initial_entropy is None
            or not math.isfinite(attention_entropy)
        ):
            return 0.0
        return max(0.0, self._initial_entropy - attention_entropy)

    def _max_probability_risk(self, mean_max_probability: float | None) -> float:
        if mean_max_probability is None or not math.isfinite(mean_max_probability):
            return 0.0
        return max(0.0, mean_max_probability - self.config.max_probability_target)

    @staticmethod
    def _validate_observation(observation: AlphaObservation) -> None:
        if observation.step <= 0:
            raise ValueError("observation step must be positive")
        if not math.isfinite(observation.validation_loss):
            raise ValueError("validation_loss must be finite")
        if (
            observation.reference_validation_loss is not None
            and not math.isfinite(observation.reference_validation_loss)
        ):
            raise ValueError("reference_validation_loss must be finite when provided")


def set_model_fixed_alpha(model: torch.nn.Module, alpha: float) -> None:
    """Set fixed alpha on all GeoAttention modules in a model."""

    value = float(alpha)
    for module in model.modules():
        if isinstance(module, GeoAttention):
            if not hasattr(module, "fixed_alpha"):
                raise ValueError("adaptive alpha requires GeoAttention alpha mode='fixed'")
            module.fixed_alpha.fill_(value)


def load_reference_progress(path: str | Path | None) -> dict[int, float]:
    if path is None:
        return {}
    progress_path = Path(path)
    if not progress_path.exists():
        raise FileNotFoundError(f"reference progress log not found: {progress_path}")
    reference: dict[int, float] = {}
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = row.get("step")
        value = row.get("validation_loss")
        if step is not None and value is not None:
            reference[int(step)] = float(value)
    return reference


def reference_loss_for_step(reference: dict[int, float], step: int) -> float | None:
    if step in reference:
        return reference[step]
    previous_steps = [candidate for candidate in reference if candidate <= step]
    if not previous_steps:
        return None
    return reference[max(previous_steps)]


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
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _ema(previous: float | None, value: float, beta: float) -> float:
    if previous is None:
        return value
    return beta * previous + (1.0 - beta) * value


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
