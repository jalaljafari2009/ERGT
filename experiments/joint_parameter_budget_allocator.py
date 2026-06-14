"""Joint parameter budget allocator for ERGT adaptive control."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

PARAMETER_FAMILIES = {
    "alpha": "geometry",
    "distance_norm_scale": "geometry",
    "causal_reachability": "geometry",
    "memory_eta": "memory",
    "memory_decay": "memory",
    "gate_floor": "memory",
}

BudgetDecisionName = Literal[
    "allocate_budget",
    "freeze_all",
    "hard_stop_hold",
]


@dataclass(frozen=True)
class ParameterBudgetRequest:
    requested_delta: float
    credit: float
    risk_pressure: float
    attribution_estimate: float = 0.0
    hard_block: bool = False


@dataclass(frozen=True)
class JointBudgetConfig:
    total_change_budget: float = 1.0
    max_geometry_budget_fraction: float = 0.55
    max_memory_budget_fraction: float = 0.65
    target_geo_to_qk_ratio: float = 0.10
    underpowered_geo_to_qk_ratio: float = 0.03
    overpowered_geo_to_qk_ratio: float = 0.18
    high_attention_pressure: float = 0.65
    high_noise_pressure: float = 0.55
    high_control_penalty: float = 0.35
    high_attribution_uncertainty: float = 0.50

    def validate(self) -> None:
        if self.total_change_budget <= 0:
            raise ValueError("total_change_budget must be positive")
        for field in [
            "max_geometry_budget_fraction",
            "max_memory_budget_fraction",
            "high_attention_pressure",
            "high_noise_pressure",
            "high_control_penalty",
            "high_attribution_uncertainty",
        ]:
            value = getattr(self, field)
            if not 0 <= value <= 1:
                raise ValueError(f"{field} must be in [0, 1]")
        if self.underpowered_geo_to_qk_ratio >= self.overpowered_geo_to_qk_ratio:
            raise ValueError("geo/qk underpowered threshold must be below overpowered")


@dataclass(frozen=True)
class JointBudgetObservation:
    step: int
    parameter_requests: dict[str, ParameterBudgetRequest]
    geo_to_qk_ratio: float
    geometry_takeover_score: float
    rigidity_risk: float
    collapse_risk: float
    noise_risk: float
    control_penalty: float
    attention_entropy_normalized: float
    control_separation: float
    attribution_uncertainty: float = 0.0
    future_leak_score: float = 0.0
    hard_stop_triggered: bool = False
    hard_stop_reason: str | None = None


@dataclass(frozen=True)
class JointBudgetDecision:
    step: int
    decision: BudgetDecisionName
    change_budget: float
    allocated_change_budget: float
    geometry_budget: float
    memory_budget: float
    rigidity_budget: float
    noise_budget: float
    qk_competition_state: str
    attention_behavior_regime: str
    attention_derived_budget_pressure: float
    budget_conflict_score: float
    release_allocation: dict[str, float]
    restraint_allocation: dict[str, float]
    budget_allocation: dict[str, dict[str, float | str | bool]]
    budget_suppression_reasons: dict[str, list[str]]
    controller_state_snapshot: dict[str, float | str]
    parameter_trajectory: dict[str, float]
    injected_evidence_ledger: dict[str, Any]
    decision_replay_record: dict[str, Any]


class JointParameterBudgetAllocator:
    """Coordinate parameter deltas before an adaptive trainer applies them."""

    def __init__(self, config: JointBudgetConfig | None = None) -> None:
        self.config = config or JointBudgetConfig()
        self.config.validate()
        self.decisions: list[JointBudgetDecision] = []

    def allocate(self, observation: JointBudgetObservation) -> JointBudgetDecision:
        _validate_observation(observation)
        qk_state = self._qk_competition_state(observation.geo_to_qk_ratio)
        attention_regime = self._attention_behavior_regime(observation)
        attention_pressure = self._attention_pressure(observation)
        noise_pressure = self._noise_pressure(observation)
        hard_stop = observation.hard_stop_triggered or observation.future_leak_score > 0

        preliminary = {}
        suppression_reasons: dict[str, list[str]] = {}
        for parameter, request in observation.parameter_requests.items():
            allocation = self._preliminary_allocation(
                parameter,
                request,
                observation=observation,
                qk_competition_state=qk_state,
                attention_pressure=attention_pressure,
                noise_pressure=noise_pressure,
                hard_stop=hard_stop,
            )
            preliminary[parameter] = allocation
            suppression_reasons[parameter] = allocation["suppression_reasons"]

        scaled = self._apply_budget_caps(preliminary)
        if hard_stop:
            scaled = {
                parameter: {**allocation, "allocated_delta": 0.0, "multiplier": 0.0}
                for parameter, allocation in scaled.items()
            }

        geometry_budget = _family_budget(scaled, "geometry")
        memory_budget = _family_budget(scaled, "memory")
        allocated_change_budget = sum(
            abs(float(allocation["allocated_delta"]))
            for allocation in scaled.values()
        )
        release_allocation = {
            parameter: max(float(allocation["allocated_delta"]), 0.0)
            for parameter, allocation in scaled.items()
        }
        restraint_allocation = {
            parameter: max(-float(allocation["allocated_delta"]), 0.0)
            for parameter, allocation in scaled.items()
        }
        parameter_trajectory = {
            f"{parameter}_allocated_delta": float(allocation["allocated_delta"])
            for parameter, allocation in scaled.items()
        }
        active_requests = sum(
            1
            for request in observation.parameter_requests.values()
            if abs(request.requested_delta) > 0
        )
        budget_conflict_score = _clamp(
            0.15 * max(active_requests - 1, 0)
            + observation.attribution_uncertainty
            + 0.5 * observation.control_penalty,
            0.0,
            1.0,
        )
        decision_name = self._decision_name(
            hard_stop=hard_stop,
            allocated_change_budget=allocated_change_budget,
            active_requests=active_requests,
        )
        state = {
            "qk_competition_state": qk_state,
            "attention_behavior_regime": attention_regime,
            "attention_pressure": attention_pressure,
            "noise_pressure": noise_pressure,
            "budget_conflict_score": budget_conflict_score,
            "allocated_change_budget": allocated_change_budget,
        }
        ledger = {
            "parameter_requests": {
                parameter: asdict(request)
                for parameter, request in observation.parameter_requests.items()
            },
            "preliminary_allocation": preliminary,
            "final_allocation": scaled,
            "budget_suppression_reasons": suppression_reasons,
        }
        replay = {
            "controller": "joint_parameter_budget_allocator",
            "decision": decision_name,
            "observation": asdict(observation),
            "config": asdict(self.config),
            "state_before_decision": state,
            "parameter_trajectory": parameter_trajectory,
        }
        decision = JointBudgetDecision(
            step=observation.step,
            decision=decision_name,
            change_budget=self.config.total_change_budget,
            allocated_change_budget=allocated_change_budget,
            geometry_budget=geometry_budget,
            memory_budget=memory_budget,
            rigidity_budget=attention_pressure,
            noise_budget=noise_pressure,
            qk_competition_state=qk_state,
            attention_behavior_regime=attention_regime,
            attention_derived_budget_pressure=attention_pressure,
            budget_conflict_score=budget_conflict_score,
            release_allocation=release_allocation,
            restraint_allocation=restraint_allocation,
            budget_allocation=scaled,
            budget_suppression_reasons=suppression_reasons,
            controller_state_snapshot=state,
            parameter_trajectory=parameter_trajectory,
            injected_evidence_ledger=ledger,
            decision_replay_record=replay,
        )
        self.decisions.append(decision)
        return decision

    def summary(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "decisions": [asdict(decision) for decision in self.decisions],
        }

    def _preliminary_allocation(
        self,
        parameter: str,
        request: ParameterBudgetRequest,
        *,
        observation: JointBudgetObservation,
        qk_competition_state: str,
        attention_pressure: float,
        noise_pressure: float,
        hard_stop: bool,
    ) -> dict[str, Any]:
        family = PARAMETER_FAMILIES.get(parameter, "unknown")
        release = max(request.credit, 0.0) + max(request.attribution_estimate, 0.0)
        risk = max(request.risk_pressure, 0.0)
        reasons = []
        if family == "geometry" and request.requested_delta > 0:
            risk += 0.55 * attention_pressure + observation.control_penalty
            if qk_competition_state == "geo_overpowered":
                risk += 0.35
                reasons.append("geo_qk_overpowered")
            if noise_pressure >= self.config.high_noise_pressure:
                risk += 0.25 * noise_pressure
                reasons.append("noise_shifts_budget_to_memory")
            if (
                attention_pressure >= self.config.high_attention_pressure
                or observation.control_penalty >= self.config.high_control_penalty
            ):
                reasons.append("geometry_growth_suppressed_by_attention_or_controls")
        if parameter in {"memory_decay", "gate_floor"} and request.requested_delta > 0:
            release += 0.45 * noise_pressure
        if parameter == "memory_eta" and request.requested_delta > 0:
            risk += noise_pressure
            if noise_pressure >= self.config.high_noise_pressure:
                reasons.append("eta_growth_suppressed_by_noise")
        if parameter == "gate_floor" and request.requested_delta < 0:
            risk += noise_pressure
            if noise_pressure >= self.config.high_noise_pressure:
                reasons.append("lower_gate_suppressed_by_noise")
        if observation.attribution_uncertainty >= self.config.high_attribution_uncertainty:
            risk += 0.25 * observation.attribution_uncertainty
            reasons.append("high_attribution_uncertainty")
        if request.hard_block or hard_stop:
            reasons.append("hard_block")

        if request.requested_delta > 0:
            value = max(release - risk, 0.0)
        else:
            value = max(risk + max(-request.credit, 0.0), 0.0)
        multiplier = _clamp(value, 0.0, 1.0)
        if request.hard_block or hard_stop:
            multiplier = 0.0
        if (
            family == "geometry"
            and request.requested_delta > 0
            and "geometry_growth_suppressed_by_attention_or_controls" in reasons
        ):
            multiplier = 0.0
        allocated_delta = request.requested_delta * multiplier
        return {
            "family": family,
            "requested_delta": request.requested_delta,
            "allocated_delta": allocated_delta,
            "multiplier": multiplier,
            "release_score": release,
            "risk_score": risk,
            "suppression_reasons": reasons,
        }

    def _apply_budget_caps(
        self,
        allocations: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        scaled = {parameter: dict(allocation) for parameter, allocation in allocations.items()}
        self._scale_family(
            scaled,
            family="geometry",
            cap=self.config.total_change_budget * self.config.max_geometry_budget_fraction,
        )
        self._scale_family(
            scaled,
            family="memory",
            cap=self.config.total_change_budget * self.config.max_memory_budget_fraction,
        )
        total = sum(abs(float(item["allocated_delta"])) for item in scaled.values())
        if total > self.config.total_change_budget:
            factor = self.config.total_change_budget / total
            for allocation in scaled.values():
                allocation["allocated_delta"] = float(allocation["allocated_delta"]) * factor
                allocation["multiplier"] = float(allocation["multiplier"]) * factor
                allocation["suppression_reasons"].append("global_budget_scaled")
        return scaled

    def _scale_family(
        self,
        allocations: dict[str, dict[str, Any]],
        *,
        family: str,
        cap: float,
    ) -> None:
        total = _family_budget(allocations, family)
        if total <= cap or total <= 0:
            return
        factor = cap / total
        for allocation in allocations.values():
            if allocation["family"] != family:
                continue
            allocation["allocated_delta"] = float(allocation["allocated_delta"]) * factor
            allocation["multiplier"] = float(allocation["multiplier"]) * factor
            allocation["suppression_reasons"].append(f"{family}_budget_scaled")

    def _qk_competition_state(self, geo_to_qk_ratio: float) -> str:
        if geo_to_qk_ratio < self.config.underpowered_geo_to_qk_ratio:
            return "geo_underpowered"
        if geo_to_qk_ratio > self.config.overpowered_geo_to_qk_ratio:
            return "geo_overpowered"
        return "geo_competitive"

    def _attention_behavior_regime(self, observation: JointBudgetObservation) -> str:
        if observation.collapse_risk >= 0.70:
            return "collapsed"
        if observation.attention_entropy_normalized >= 0.92:
            return "uniform"
        if (
            observation.rigidity_risk >= 0.50
            or observation.geometry_takeover_score >= 0.50
        ):
            return "rigid_or_takeover"
        return "healthy"

    def _attention_pressure(self, observation: JointBudgetObservation) -> float:
        uniformity_pressure = max(observation.attention_entropy_normalized - 0.85, 0.0)
        return _clamp(
            max(
                observation.rigidity_risk,
                observation.collapse_risk,
                observation.geometry_takeover_score,
                uniformity_pressure,
            ),
            0.0,
            1.0,
        )

    def _noise_pressure(self, observation: JointBudgetObservation) -> float:
        return _clamp(
            max(observation.noise_risk, observation.control_penalty),
            0.0,
            1.0,
        )

    def _decision_name(
        self,
        *,
        hard_stop: bool,
        allocated_change_budget: float,
        active_requests: int,
    ) -> BudgetDecisionName:
        if hard_stop:
            return "hard_stop_hold"
        if allocated_change_budget > 0:
            return "allocate_budget"
        if active_requests > 0:
            return "freeze_all"
        return "freeze_all"


def _validate_observation(observation: JointBudgetObservation) -> None:
    if observation.step <= 0:
        raise ValueError("observation step must be positive")
    if not observation.parameter_requests:
        raise ValueError("parameter_requests must not be empty")
    unknown = set(observation.parameter_requests) - set(PARAMETER_FAMILIES)
    if unknown:
        raise ValueError(f"unknown adaptive parameters: {sorted(unknown)}")
    for field, value in asdict(observation).items():
        if field in {"hard_stop_reason", "hard_stop_triggered", "parameter_requests"}:
            continue
        if value is not None and not isinstance(value, str):
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(f"{field} must be finite when provided")
    for parameter, request in observation.parameter_requests.items():
        for field, value in asdict(request).items():
            if field == "hard_block":
                continue
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(f"{parameter}.{field} must be finite")


def _family_budget(allocations: dict[str, dict[str, Any]], family: str) -> float:
    return sum(
        abs(float(allocation["allocated_delta"]))
        for allocation in allocations.values()
        if allocation["family"] == family
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
