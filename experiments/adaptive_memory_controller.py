"""Adaptive memory eta/decay controller for ERGT open control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

MemoryDecisionName = Literal[
    "inject_memory",
    "smooth_memory",
    "release_rigid_memory",
    "hold_memory",
    "hard_stop_hold",
]


@dataclass(frozen=True)
class AdaptiveMemoryConfig:
    initial_eta: float = 0.30
    initial_decay: float = 0.70
    min_eta: float = 0.02
    max_eta: float = 0.80
    min_decay: float = 0.10
    max_decay: float = 0.95
    eta_step: float = 0.04
    decay_step: float = 0.04
    target_stability: float = 0.65
    target_persistence: float = 0.55
    target_turnover: float = 0.08
    starved_density_threshold: float = 0.10
    high_noise_threshold: float = 0.55
    high_rigidity_threshold: float = 0.55
    release_margin: float = 0.03
    restraint_margin: float = 0.03

    def validate(self) -> None:
        for name in ["initial_eta", "min_eta", "max_eta"]:
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ["initial_decay", "min_decay", "max_decay"]:
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_eta > self.max_eta:
            raise ValueError("min_eta must be <= max_eta")
        if self.min_decay > self.max_decay:
            raise ValueError("min_decay must be <= max_decay")
        if not self.min_eta <= self.initial_eta <= self.max_eta:
            raise ValueError("initial_eta must be inside eta bounds")
        if not self.min_decay <= self.initial_decay <= self.max_decay:
            raise ValueError("initial_decay must be inside decay bounds")
        if self.eta_step <= 0 or self.decay_step <= 0:
            raise ValueError("eta_step and decay_step must be positive")


@dataclass(frozen=True)
class MemoryObservation:
    step: int
    memory_stability: float
    memory_turnover: float
    memory_persistence: float
    memory_edge_density: float
    memory_rigidity: float
    noise_risk: float
    real_memory_advantage: float
    random_memory_advantage: float = 0.0
    shuffled_memory_advantage: float = 0.0
    future_leak_score: float = 0.0
    hard_stop_triggered: bool = False
    hard_stop_reason: str | None = None


@dataclass(frozen=True)
class MemoryDecision:
    step: int
    decision: MemoryDecisionName
    previous_eta: float
    previous_decay: float
    proposed_eta: float
    proposed_decay: float
    current_eta: float
    current_decay: float
    eta_delta: float
    decay_delta: float
    memory_stability_trend: dict[str, float]
    memory_turnover_trend: dict[str, float]
    persistence_trend: dict[str, float]
    noise_evidence: dict[str, float | bool]
    rigidity_evidence: dict[str, float | bool]
    release_evidence: dict[str, float | bool]
    restraint_evidence: dict[str, float | bool]
    controller_state_snapshot: dict[str, float]
    parameter_trajectory: dict[str, float]
    injected_evidence_ledger: dict[str, dict[str, float | bool]]
    decision_replay_record: dict[str, object]


class AdaptiveMemoryController:
    """Evidence controller for memory eta and decay."""

    def __init__(self, config: AdaptiveMemoryConfig | None = None) -> None:
        self.config = config or AdaptiveMemoryConfig()
        self.config.validate()
        self.current_eta = self.config.initial_eta
        self.current_decay = self.config.initial_decay
        self.previous_error = 0.0
        self.integral_error = 0.0
        self.decisions: list[MemoryDecision] = []

    def update(self, observation: MemoryObservation) -> MemoryDecision:
        release = self._release_evidence(observation)
        restraint = self._restraint_evidence(observation)
        stability = self._stability_trend(observation)
        turnover = self._turnover_trend(observation)
        persistence = self._persistence_trend(observation)
        noise = self._noise_evidence(observation)
        rigidity = self._rigidity_evidence(observation)

        release_score = sum(float(value) for value in release.values())
        restraint_score = sum(float(value) for value in restraint.values())
        error = release_score - restraint_score
        self.integral_error = _clamp(0.8 * self.integral_error + error, -1.0, 1.0)
        derivative_error = error - self.previous_error
        self.previous_error = error

        previous_eta = self.current_eta
        previous_decay = self.current_decay
        eta_delta, decay_delta, decision_name = self._deltas(
            observation,
            error=error,
            noise_pressure=float(restraint["noise_pressure"]),
            rigidity_pressure=float(restraint["rigidity_pressure"]),
            starved_pressure=float(release["starved_memory_pressure"]),
        )
        if observation.hard_stop_triggered or observation.future_leak_score > 0:
            eta_delta = 0.0
            decay_delta = 0.0
            decision_name = "hard_stop_hold"

        proposed_eta = _clamp(previous_eta + eta_delta, self.config.min_eta, self.config.max_eta)
        proposed_decay = _clamp(
            previous_decay + decay_delta,
            self.config.min_decay,
            self.config.max_decay,
        )
        eta_delta = proposed_eta - previous_eta
        decay_delta = proposed_decay - previous_decay
        self.current_eta = proposed_eta
        self.current_decay = proposed_decay

        controller_state = {
            "release_score": release_score,
            "restraint_score": restraint_score,
            "error": error,
            "integral_error": self.integral_error,
            "derivative_error": derivative_error,
        }
        trajectory = {
            "eta_previous": previous_eta,
            "eta_proposed": proposed_eta,
            "eta_next": self.current_eta,
            "eta_delta": eta_delta,
            "decay_previous": previous_decay,
            "decay_proposed": proposed_decay,
            "decay_next": self.current_decay,
            "decay_delta": decay_delta,
        }
        ledger = {
            "memory_stability_trend": stability,
            "memory_turnover_trend": turnover,
            "persistence_trend": persistence,
            "noise_evidence": noise,
            "rigidity_evidence": rigidity,
            "release_evidence": release,
            "restraint_evidence": restraint,
        }
        replay = {
            "controller": "adaptive_memory_eta_decay",
            "decision": decision_name,
            "observation": asdict(observation),
            "config": asdict(self.config),
            "state_before_decision": controller_state,
            "parameter_trajectory": trajectory,
        }
        decision = MemoryDecision(
            step=observation.step,
            decision=decision_name,
            previous_eta=previous_eta,
            previous_decay=previous_decay,
            proposed_eta=proposed_eta,
            proposed_decay=proposed_decay,
            current_eta=self.current_eta,
            current_decay=self.current_decay,
            eta_delta=eta_delta,
            decay_delta=decay_delta,
            memory_stability_trend=stability,
            memory_turnover_trend=turnover,
            persistence_trend=persistence,
            noise_evidence=noise,
            rigidity_evidence=rigidity,
            release_evidence=release,
            restraint_evidence=restraint,
            controller_state_snapshot=controller_state,
            parameter_trajectory=trajectory,
            injected_evidence_ledger=ledger,
            decision_replay_record=replay,
        )
        self.decisions.append(decision)
        return decision

    def summary(self) -> dict[str, object]:
        return {
            "current_eta": self.current_eta,
            "current_decay": self.current_decay,
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

    def _release_evidence(
        self,
        observation: MemoryObservation,
    ) -> dict[str, float | bool]:
        starved_by_density = (
            observation.memory_edge_density < self.config.starved_density_threshold
        )
        starved_by_persistence = observation.memory_persistence < self.config.target_persistence
        control_dominates = max(
            observation.random_memory_advantage,
            observation.shuffled_memory_advantage,
        ) >= observation.real_memory_advantage
        return {
            "real_memory_advantage": max(observation.real_memory_advantage, 0.0),
            "stability_deficit_recoverable": max(
                self.config.target_stability - observation.memory_stability,
                0.0,
            )
            * (1.0 - observation.noise_risk),
            "persistence_deficit_recoverable": max(
                self.config.target_persistence - observation.memory_persistence,
                0.0,
            )
            * (1.0 - observation.noise_risk),
            "starved_memory_pressure": float(
                (starved_by_density or starved_by_persistence) and not control_dominates
            )
            * 0.35,
        }

    def _restraint_evidence(
        self,
        observation: MemoryObservation,
    ) -> dict[str, float | bool]:
        control_pressure = max(
            observation.random_memory_advantage,
            observation.shuffled_memory_advantage,
        ) - observation.real_memory_advantage
        turnover_pressure = max(
            observation.memory_turnover - self.config.target_turnover,
            0.0,
        )
        return {
            "noise_pressure": max(observation.noise_risk - 0.15, 0.0),
            "rigidity_pressure": max(observation.memory_rigidity - 0.15, 0.0),
            "turnover_pressure": turnover_pressure,
            "control_family_pressure": max(control_pressure, 0.0),
            "future_leak_pressure": max(observation.future_leak_score, 0.0),
        }

    def _stability_trend(self, observation: MemoryObservation) -> dict[str, float]:
        return {
            "memory_stability": observation.memory_stability,
            "target_stability": self.config.target_stability,
            "stability_gap": observation.memory_stability - self.config.target_stability,
        }

    def _turnover_trend(self, observation: MemoryObservation) -> dict[str, float]:
        return {
            "memory_turnover": observation.memory_turnover,
            "target_turnover": self.config.target_turnover,
            "turnover_gap": observation.memory_turnover - self.config.target_turnover,
        }

    def _persistence_trend(self, observation: MemoryObservation) -> dict[str, float]:
        return {
            "memory_persistence": observation.memory_persistence,
            "target_persistence": self.config.target_persistence,
            "persistence_gap": observation.memory_persistence
            - self.config.target_persistence,
        }

    def _noise_evidence(
        self,
        observation: MemoryObservation,
    ) -> dict[str, float | bool]:
        return {
            "noise_risk": observation.noise_risk,
            "memory_turnover": observation.memory_turnover,
            "random_memory_advantage": observation.random_memory_advantage,
            "shuffled_memory_advantage": observation.shuffled_memory_advantage,
            "noise_high": observation.noise_risk >= self.config.high_noise_threshold,
        }

    def _rigidity_evidence(
        self,
        observation: MemoryObservation,
    ) -> dict[str, float | bool]:
        return {
            "memory_rigidity": observation.memory_rigidity,
            "memory_edge_density": observation.memory_edge_density,
            "future_leak_score": observation.future_leak_score,
            "hard_stop_triggered": observation.hard_stop_triggered
            or observation.future_leak_score > 0,
            "rigidity_high": observation.memory_rigidity
            >= self.config.high_rigidity_threshold,
        }

    def _deltas(
        self,
        observation: MemoryObservation,
        *,
        error: float,
        noise_pressure: float,
        rigidity_pressure: float,
        starved_pressure: float,
    ) -> tuple[float, float, MemoryDecisionName]:
        if (
            rigidity_pressure >= self.config.restraint_margin
            and observation.memory_rigidity >= self.config.high_rigidity_threshold
        ):
            return -0.5 * self.config.eta_step, -self.config.decay_step, "release_rigid_memory"
        if noise_pressure >= self.config.restraint_margin:
            return -self.config.eta_step, self.config.decay_step, "smooth_memory"
        if starved_pressure > 0 and error > self.config.release_margin:
            return self.config.eta_step, 0.5 * self.config.decay_step, "inject_memory"
        if error > self.config.release_margin:
            return 0.5 * self.config.eta_step, 0.0, "inject_memory"
        if error < -self.config.restraint_margin:
            return -0.5 * self.config.eta_step, 0.0, "smooth_memory"
        return 0.0, 0.0, "hold_memory"


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
