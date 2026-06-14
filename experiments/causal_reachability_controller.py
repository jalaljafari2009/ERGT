"""Causal reachability controller for ERGT open control."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

CausalReachDecisionName = Literal[
    "expand_reach",
    "restrain_reach",
    "hold_reach",
    "hard_stop_hold",
]


@dataclass(frozen=True)
class CausalReachabilityConfig:
    initial_max_causal_step: int = 1
    min_max_causal_step: int = 1
    max_max_causal_step: int = 8
    step_delta: int = 1
    target_edge_survival: float = 0.45
    min_memory_stability_for_expand: float = 0.60
    max_control_reach_noise: float = 0.45
    max_collapse_for_expand: float = 0.35
    locality_too_tight_threshold: float = 0.70
    uniformity_entropy_threshold: float = 0.92
    min_diversity_for_expand: float = 0.05
    release_margin: float = 0.03
    restraint_margin: float = 0.03

    def validate(self) -> None:
        if self.min_max_causal_step <= 0:
            raise ValueError("min_max_causal_step must be positive")
        if self.max_max_causal_step < self.min_max_causal_step:
            raise ValueError("max_max_causal_step must be >= min_max_causal_step")
        if not self.min_max_causal_step <= self.initial_max_causal_step <= self.max_max_causal_step:
            raise ValueError("initial_max_causal_step must be inside reach bounds")
        if self.step_delta <= 0:
            raise ValueError("step_delta must be positive")


@dataclass(frozen=True)
class CausalReachabilityObservation:
    step: int
    memory_stability: float
    causal_edge_survival: float
    reach_starvation_score: float
    attention_locality_score: float
    attention_entropy_normalized: float
    head_attention_diversity: float
    layer_attention_diversity: float
    collapse_risk: float
    control_reach_noise_score: float
    control_attention_separation: float
    real_reach_advantage: float
    random_reach_advantage: float = 0.0
    shuffled_reach_advantage: float = 0.0
    future_leak_score: float = 0.0
    hard_stop_triggered: bool = False
    hard_stop_reason: str | None = None


@dataclass(frozen=True)
class CausalReachabilityDecision:
    step: int
    decision: CausalReachDecisionName
    previous_causal_reachability: int
    proposed_causal_reachability: int
    current_causal_reachability: int
    causal_reachability_delta: int
    causal_reachability_credit: float
    causal_reachability_risk_pressure: float
    future_leak_evidence: dict[str, float | bool]
    memory_readiness_evidence: dict[str, float | bool]
    attention_locality_spread_evidence: dict[str, float | bool]
    control_reach_evidence: dict[str, float | bool]
    expansion_evidence: dict[str, float | bool]
    restraint_evidence: dict[str, float | bool]
    controller_state_snapshot: dict[str, float]
    parameter_trajectory: dict[str, int]
    injected_evidence_ledger: dict[str, dict[str, float | bool]]
    decision_replay_record: dict[str, Any]


class CausalReachabilityController:
    """Evidence controller for finite-speed causal reach."""

    def __init__(self, config: CausalReachabilityConfig | None = None) -> None:
        self.config = config or CausalReachabilityConfig()
        self.config.validate()
        self.current_max_causal_step = self.config.initial_max_causal_step
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.decisions: list[CausalReachabilityDecision] = []

    def update(
        self,
        observation: CausalReachabilityObservation,
    ) -> CausalReachabilityDecision:
        _validate_observation(observation)
        previous = self.current_max_causal_step
        future_leak = self._future_leak_evidence(observation)
        memory = self._memory_readiness_evidence(observation)
        attention = self._attention_locality_spread_evidence(observation)
        control = self._control_reach_evidence(observation)
        expansion = self._expansion_evidence(memory, attention, control, observation)
        restraint = self._restraint_evidence(memory, attention, control, future_leak)

        expansion_score = sum(float(value) for value in expansion.values())
        restraint_score = sum(float(value) for value in restraint.values())
        error = expansion_score - restraint_score
        self.integral_error = _clamp(0.8 * self.integral_error + error, -1.0, 1.0)
        derivative_error = error - self.previous_error
        self.previous_error = error

        delta, decision_name = self._delta_and_decision(
            observation,
            memory=memory,
            attention=attention,
            control=control,
            error=error,
            expansion_score=expansion_score,
            restraint_score=restraint_score,
        )
        if observation.hard_stop_triggered or observation.future_leak_score > 0:
            delta = 0
            decision_name = "hard_stop_hold"

        proposed = int(
            _clamp(
                previous + delta,
                self.config.min_max_causal_step,
                self.config.max_max_causal_step,
            )
        )
        delta = proposed - previous
        self.current_max_causal_step = proposed
        credit = expansion_score - restraint_score
        risk_pressure = restraint_score
        controller_state = {
            "expansion_score": expansion_score,
            "restraint_score": restraint_score,
            "error": error,
            "integral_error": self.integral_error,
            "derivative_error": derivative_error,
        }
        trajectory = {
            "causal_reachability_previous": previous,
            "causal_reachability_proposed": proposed,
            "causal_reachability_next": self.current_max_causal_step,
            "causal_reachability_delta": delta,
        }
        ledger = {
            "future_leak_evidence": future_leak,
            "memory_readiness_evidence": memory,
            "attention_locality_spread_evidence": attention,
            "control_reach_evidence": control,
            "expansion_evidence": expansion,
            "restraint_evidence": restraint,
        }
        replay = {
            "controller": "causal_reachability_controller",
            "decision": decision_name,
            "observation": asdict(observation),
            "config": asdict(self.config),
            "state_before_decision": controller_state,
            "parameter_trajectory": trajectory,
        }
        decision = CausalReachabilityDecision(
            step=observation.step,
            decision=decision_name,
            previous_causal_reachability=previous,
            proposed_causal_reachability=proposed,
            current_causal_reachability=self.current_max_causal_step,
            causal_reachability_delta=delta,
            causal_reachability_credit=credit,
            causal_reachability_risk_pressure=risk_pressure,
            future_leak_evidence=future_leak,
            memory_readiness_evidence=memory,
            attention_locality_spread_evidence=attention,
            control_reach_evidence=control,
            expansion_evidence=expansion,
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
            "current_causal_reachability": self.current_max_causal_step,
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

    def _future_leak_evidence(
        self,
        observation: CausalReachabilityObservation,
    ) -> dict[str, float | bool]:
        return {
            "future_leak_score": observation.future_leak_score,
            "hard_stop_triggered": observation.hard_stop_triggered
            or observation.future_leak_score > 0,
        }

    def _memory_readiness_evidence(
        self,
        observation: CausalReachabilityObservation,
    ) -> dict[str, float | bool]:
        stability_surplus = max(
            observation.memory_stability - self.config.min_memory_stability_for_expand,
            0.0,
        )
        edge_survival_deficit = max(
            self.config.target_edge_survival - observation.causal_edge_survival,
            0.0,
        )
        return {
            "memory_stability": observation.memory_stability,
            "memory_ready": observation.memory_stability
            >= self.config.min_memory_stability_for_expand,
            "stability_surplus": stability_surplus,
            "causal_edge_survival": observation.causal_edge_survival,
            "edge_survival_deficit": edge_survival_deficit,
            "reach_starvation_score": observation.reach_starvation_score,
        }

    def _attention_locality_spread_evidence(
        self,
        observation: CausalReachabilityObservation,
    ) -> dict[str, float | bool]:
        attention_too_local = (
            observation.attention_locality_score >= self.config.locality_too_tight_threshold
        )
        attention_uniform = (
            observation.attention_entropy_normalized >= self.config.uniformity_entropy_threshold
            and observation.attention_locality_score < 0.35
        )
        diversity_low = min(
            observation.head_attention_diversity,
            observation.layer_attention_diversity,
        ) < self.config.min_diversity_for_expand
        return {
            "attention_locality_score": observation.attention_locality_score,
            "attention_entropy_normalized": observation.attention_entropy_normalized,
            "head_attention_diversity": observation.head_attention_diversity,
            "layer_attention_diversity": observation.layer_attention_diversity,
            "collapse_risk": observation.collapse_risk,
            "attention_too_local": attention_too_local,
            "attention_uniform": attention_uniform,
            "diversity_low": diversity_low,
        }

    def _control_reach_evidence(
        self,
        observation: CausalReachabilityObservation,
    ) -> dict[str, float | bool]:
        best_control_advantage = max(
            observation.random_reach_advantage,
            observation.shuffled_reach_advantage,
        )
        return {
            "control_reach_noise_score": observation.control_reach_noise_score,
            "control_attention_separation": observation.control_attention_separation,
            "real_reach_advantage": observation.real_reach_advantage,
            "random_reach_advantage": observation.random_reach_advantage,
            "shuffled_reach_advantage": observation.shuffled_reach_advantage,
            "best_control_advantage": best_control_advantage,
            "control_dominates_real": best_control_advantage
            >= observation.real_reach_advantage,
            "control_attention_not_separated": observation.control_attention_separation < 0,
        }

    def _expansion_evidence(
        self,
        memory: dict[str, float | bool],
        attention: dict[str, float | bool],
        control: dict[str, float | bool],
        observation: CausalReachabilityObservation,
    ) -> dict[str, float]:
        if not memory["memory_ready"]:
            readiness_scale = 0.0
        else:
            readiness_scale = 1.0
        attention_blocked = attention["attention_uniform"] or attention["diversity_low"]
        attention_scale = 0.0 if attention_blocked else 1.0
        control_scale = 0.0 if control["control_dominates_real"] else 1.0
        return {
            "stable_memory_reach_credit": float(memory["stability_surplus"]) * 0.8,
            "edge_survival_deficit": float(memory["edge_survival_deficit"]) * readiness_scale,
            "reach_starvation_pressure": observation.reach_starvation_score
            * readiness_scale
            * attention_scale
            * control_scale,
            "attention_locality_pressure": (
                observation.attention_locality_score
                if attention["attention_too_local"]
                else 0.0
            )
            * readiness_scale
            * control_scale,
            "real_reach_advantage": max(observation.real_reach_advantage, 0.0)
            * control_scale,
        }

    def _restraint_evidence(
        self,
        memory: dict[str, float | bool],
        attention: dict[str, float | bool],
        control: dict[str, float | bool],
        future_leak: dict[str, float | bool],
    ) -> dict[str, float]:
        low_memory_pressure = max(
            self.config.min_memory_stability_for_expand - float(memory["memory_stability"]),
            0.0,
        )
        control_dominance = max(
            float(control["best_control_advantage"]) - float(control["real_reach_advantage"]),
            0.0,
        )
        return {
            "future_leak_pressure": float(future_leak["future_leak_score"]),
            "low_memory_stability_pressure": low_memory_pressure,
            "control_reach_noise_pressure": float(control["control_reach_noise_score"]),
            "control_dominance_pressure": control_dominance,
            "control_attention_pressure": max(
                -float(control["control_attention_separation"]),
                0.0,
            ),
            "collapse_pressure": max(float(attention["collapse_risk"]) - 0.15, 0.0),
            "attention_uniformity_pressure": 0.25 if attention["attention_uniform"] else 0.0,
            "diversity_collapse_pressure": 0.25 if attention["diversity_low"] else 0.0,
        }

    def _delta_and_decision(
        self,
        observation: CausalReachabilityObservation,
        *,
        memory: dict[str, float | bool],
        attention: dict[str, float | bool],
        control: dict[str, float | bool],
        error: float,
        expansion_score: float,
        restraint_score: float,
    ) -> tuple[int, CausalReachDecisionName]:
        can_expand = (
            memory["memory_ready"]
            and not attention["attention_uniform"]
            and not attention["diversity_low"]
            and observation.collapse_risk <= self.config.max_collapse_for_expand
            and observation.control_reach_noise_score <= self.config.max_control_reach_noise
            and not control["control_dominates_real"]
        )
        if error > self.config.release_margin and expansion_score > restraint_score and can_expand:
            return self.config.step_delta, "expand_reach"
        if error < -self.config.restraint_margin or restraint_score > expansion_score:
            return -self.config.step_delta, "restrain_reach"
        return 0, "hold_reach"


def _validate_observation(observation: CausalReachabilityObservation) -> None:
    if observation.step <= 0:
        raise ValueError("observation step must be positive")
    for field, value in asdict(observation).items():
        if field in {"hard_stop_reason", "hard_stop_triggered"}:
            continue
        if value is not None and not isinstance(value, str):
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(f"{field} must be finite when provided")


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
