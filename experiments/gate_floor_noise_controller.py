"""Gate-floor and noise controller for ERGT open control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

GateFloorDecisionName = Literal[
    "raise_gate_floor",
    "lower_gate_floor",
    "hold_gate_floor",
    "hard_stop_hold",
]


@dataclass(frozen=True)
class GateFloorNoiseConfig:
    initial_gate_floor: float = 0.05
    min_gate_floor: float = 0.0
    max_gate_floor: float = 0.35
    gate_floor_step: float = 0.02
    target_edge_survival: float = 0.45
    target_edge_density: float = 0.20
    high_noise_threshold: float = 0.55
    high_starvation_threshold: float = 0.45
    sparse_attention_threshold: float = 0.90
    low_entropy_threshold: float = 0.35
    release_margin: float = 0.03
    restraint_margin: float = 0.03

    def validate(self) -> None:
        for name in ["initial_gate_floor", "min_gate_floor", "max_gate_floor"]:
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_gate_floor > self.max_gate_floor:
            raise ValueError("min_gate_floor must be <= max_gate_floor")
        if not self.min_gate_floor <= self.initial_gate_floor <= self.max_gate_floor:
            raise ValueError("initial_gate_floor must be inside gate bounds")
        if self.gate_floor_step <= 0:
            raise ValueError("gate_floor_step must be positive")


@dataclass(frozen=True)
class GateFloorObservation:
    step: int
    memory_edge_density: float
    edge_survival: float
    random_edge_noise_score: float
    shuffled_edge_noise_score: float
    real_edge_starvation_score: float
    attention_sparsity_0_01: float
    attention_entropy_normalized: float
    control_attention_separation: float
    collapse_risk: float = 0.0
    future_leak_score: float = 0.0
    hard_stop_triggered: bool = False
    hard_stop_reason: str | None = None


@dataclass(frozen=True)
class GateFloorDecision:
    step: int
    decision: GateFloorDecisionName
    previous_gate_floor: float
    proposed_gate_floor: float
    current_gate_floor: float
    gate_floor_delta: float
    gate_floor_credit: float
    gate_floor_risk_pressure: float
    edge_noise_evidence: dict[str, float | bool]
    starvation_evidence: dict[str, float | bool]
    attention_pressure_evidence: dict[str, float | bool]
    control_attention_evidence: dict[str, float | bool]
    release_evidence: dict[str, float | bool]
    restraint_evidence: dict[str, float | bool]
    controller_state_snapshot: dict[str, float]
    parameter_trajectory: dict[str, float]
    injected_evidence_ledger: dict[str, dict[str, float | bool]]
    decision_replay_record: dict[str, Any]


class GateFloorNoiseController:
    """Evidence controller for gate-floor filtering strength."""

    def __init__(self, config: GateFloorNoiseConfig | None = None) -> None:
        self.config = config or GateFloorNoiseConfig()
        self.config.validate()
        self.current_gate_floor = self.config.initial_gate_floor
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.decisions: list[GateFloorDecision] = []

    def update(self, observation: GateFloorObservation) -> GateFloorDecision:
        _validate_observation(observation)
        previous_gate_floor = self.current_gate_floor
        edge_noise = self._edge_noise_evidence(observation)
        starvation = self._starvation_evidence(observation)
        attention = self._attention_pressure_evidence(observation)
        control_attention = self._control_attention_evidence(observation)
        release = self._release_evidence(starvation, attention)
        restraint = self._restraint_evidence(edge_noise, control_attention, attention)

        release_score = sum(float(value) for value in release.values())
        restraint_score = sum(float(value) for value in restraint.values())
        error = restraint_score - release_score
        self.integral_error = _clamp(0.8 * self.integral_error + error, -1.0, 1.0)
        derivative_error = error - self.previous_error
        self.previous_error = error

        gate_delta, decision_name = self._delta_and_decision(
            observation,
            error=error,
            release_score=release_score,
            restraint_score=restraint_score,
        )
        if observation.hard_stop_triggered or observation.future_leak_score > 0:
            gate_delta = 0.0
            decision_name = "hard_stop_hold"

        proposed_gate_floor = _clamp(
            previous_gate_floor + gate_delta,
            self.config.min_gate_floor,
            self.config.max_gate_floor,
        )
        gate_delta = proposed_gate_floor - previous_gate_floor
        self.current_gate_floor = proposed_gate_floor
        gate_credit = release_score - restraint_score
        risk_pressure = restraint_score
        controller_state = {
            "release_score": release_score,
            "restraint_score": restraint_score,
            "error": error,
            "integral_error": self.integral_error,
            "derivative_error": derivative_error,
        }
        trajectory = {
            "gate_floor_previous": previous_gate_floor,
            "gate_floor_proposed": proposed_gate_floor,
            "gate_floor_next": self.current_gate_floor,
            "gate_floor_delta": gate_delta,
        }
        ledger = {
            "edge_noise_evidence": edge_noise,
            "starvation_evidence": starvation,
            "attention_pressure_evidence": attention,
            "control_attention_evidence": control_attention,
            "release_evidence": release,
            "restraint_evidence": restraint,
        }
        replay = {
            "controller": "gate_floor_noise_controller",
            "decision": decision_name,
            "observation": asdict(observation),
            "config": asdict(self.config),
            "state_before_decision": controller_state,
            "parameter_trajectory": trajectory,
        }
        decision = GateFloorDecision(
            step=observation.step,
            decision=decision_name,
            previous_gate_floor=previous_gate_floor,
            proposed_gate_floor=proposed_gate_floor,
            current_gate_floor=self.current_gate_floor,
            gate_floor_delta=gate_delta,
            gate_floor_credit=gate_credit,
            gate_floor_risk_pressure=risk_pressure,
            edge_noise_evidence=edge_noise,
            starvation_evidence=starvation,
            attention_pressure_evidence=attention,
            control_attention_evidence=control_attention,
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
            "current_gate_floor": self.current_gate_floor,
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

    def _edge_noise_evidence(
        self,
        observation: GateFloorObservation,
    ) -> dict[str, float | bool]:
        best_control_noise = max(
            observation.random_edge_noise_score,
            observation.shuffled_edge_noise_score,
        )
        return {
            "random_edge_noise_score": observation.random_edge_noise_score,
            "shuffled_edge_noise_score": observation.shuffled_edge_noise_score,
            "best_control_noise": best_control_noise,
            "memory_edge_density": observation.memory_edge_density,
            "noise_high": best_control_noise >= self.config.high_noise_threshold,
        }

    def _starvation_evidence(
        self,
        observation: GateFloorObservation,
    ) -> dict[str, float | bool]:
        edge_density_deficit = max(
            self.config.target_edge_density - observation.memory_edge_density,
            0.0,
        )
        survival_deficit = max(
            self.config.target_edge_survival - observation.edge_survival,
            0.0,
        )
        return {
            "real_edge_starvation_score": observation.real_edge_starvation_score,
            "edge_survival": observation.edge_survival,
            "edge_density_deficit": edge_density_deficit,
            "edge_survival_deficit": survival_deficit,
            "starvation_high": observation.real_edge_starvation_score
            >= self.config.high_starvation_threshold,
        }

    def _attention_pressure_evidence(
        self,
        observation: GateFloorObservation,
    ) -> dict[str, float | bool]:
        attention_sparse = observation.attention_sparsity_0_01 >= (
            self.config.sparse_attention_threshold
        )
        low_entropy = observation.attention_entropy_normalized <= (
            self.config.low_entropy_threshold
        )
        return {
            "attention_sparsity_0_01": observation.attention_sparsity_0_01,
            "attention_entropy_normalized": observation.attention_entropy_normalized,
            "collapse_risk": observation.collapse_risk,
            "attention_sparse": attention_sparse,
            "attention_low_entropy": low_entropy,
        }

    def _control_attention_evidence(
        self,
        observation: GateFloorObservation,
    ) -> dict[str, float | bool]:
        return {
            "control_attention_separation": observation.control_attention_separation,
            "control_attention_not_separated": observation.control_attention_separation < 0,
            "future_leak_score": observation.future_leak_score,
            "hard_stop_triggered": observation.hard_stop_triggered
            or observation.future_leak_score > 0,
        }

    def _release_evidence(
        self,
        starvation: dict[str, float | bool],
        attention: dict[str, float | bool],
    ) -> dict[str, float]:
        sparse_attention_release = 0.10 if attention["attention_sparse"] else 0.0
        return {
            "real_edge_starvation_pressure": float(
                starvation["real_edge_starvation_score"]
            ),
            "edge_survival_deficit": float(starvation["edge_survival_deficit"]),
            "edge_density_deficit": float(starvation["edge_density_deficit"]),
            "sparse_attention_release": sparse_attention_release,
        }

    def _restraint_evidence(
        self,
        edge_noise: dict[str, float | bool],
        control_attention: dict[str, float | bool],
        attention: dict[str, float | bool],
    ) -> dict[str, float]:
        control_like_pressure = max(
            -float(control_attention["control_attention_separation"]),
            0.0,
        )
        return {
            "edge_noise_pressure": float(edge_noise["best_control_noise"]),
            "control_attention_pressure": control_like_pressure,
            "collapse_pressure": max(float(attention["collapse_risk"]) - 0.15, 0.0),
            "future_leak_pressure": max(
                float(control_attention["future_leak_score"]),
                0.0,
            ),
        }

    def _delta_and_decision(
        self,
        observation: GateFloorObservation,
        *,
        error: float,
        release_score: float,
        restraint_score: float,
    ) -> tuple[float, GateFloorDecisionName]:
        if error > self.config.restraint_margin and restraint_score > release_score:
            return self.config.gate_floor_step, "raise_gate_floor"
        if error < -self.config.release_margin and release_score > restraint_score:
            return -self.config.gate_floor_step, "lower_gate_floor"
        if observation.real_edge_starvation_score >= self.config.high_starvation_threshold:
            return -0.5 * self.config.gate_floor_step, "lower_gate_floor"
        return 0.0, "hold_gate_floor"


def _validate_observation(observation: GateFloorObservation) -> None:
    if observation.step <= 0:
        raise ValueError("observation step must be positive")
    for field, value in asdict(observation).items():
        if field in {"hard_stop_reason", "hard_stop_triggered"}:
            continue
        if value is not None and not isinstance(value, str):
            float(value)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
