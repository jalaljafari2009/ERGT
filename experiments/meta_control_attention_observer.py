"""Missing-aware meta-control attention observer for adaptive ERGT."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

MetaControlMode = Literal["online_partial", "offline_matched_replay"]

PARAMETER_TARGETS = ("alpha", "memory", "gate", "reach", "norm")


@dataclass(frozen=True)
class MetaSignalSpec:
    """Defines one controller signal observed by meta-control attention."""

    name: str
    fields: tuple[str, ...]
    parameter_weights: dict[str, float]
    mode: Literal["positive", "risk", "distance_from_target"] = "positive"
    requires_complete_controls: bool = False


SIGNAL_SPECS: tuple[MetaSignalSpec, ...] = (
    MetaSignalSpec(
        "loss_slope",
        ("loss_slope_gain", "adaptive_slope_gain"),
        {"alpha": 0.35, "memory": 0.20, "gate": 0.05, "reach": 0.15, "norm": 0.25},
    ),
    MetaSignalSpec(
        "baseline_delta",
        ("baseline_centered_improvement", "real_vs_baseline_delta"),
        {"alpha": 0.30, "memory": 0.15, "gate": 0.05, "reach": 0.15, "norm": 0.35},
    ),
    MetaSignalSpec(
        "control_separation",
        ("control_separation",),
        {"alpha": 0.25, "memory": 0.20, "gate": 0.20, "reach": 0.15, "norm": 0.20},
        requires_complete_controls=True,
    ),
    MetaSignalSpec(
        "geo_qk",
        ("geo_to_qk_ratio",),
        {"alpha": 0.45, "memory": 0.05, "gate": 0.05, "reach": 0.10, "norm": 0.35},
        mode="distance_from_target",
    ),
    MetaSignalSpec(
        "rigidity_collapse",
        ("rigidity_risk", "collapse_risk", "geometry_takeover_score"),
        {"alpha": 0.05, "memory": 0.20, "gate": 0.35, "reach": 0.10, "norm": 0.30},
        mode="risk",
    ),
    MetaSignalSpec(
        "memory_state",
        ("memory_stability", "memory_persistence", "memory_turnover"),
        {"alpha": 0.05, "memory": 0.55, "gate": 0.10, "reach": 0.20, "norm": 0.10},
    ),
    MetaSignalSpec(
        "gate_noise",
        ("noise_risk", "gate_noise_pressure", "random_edge_noise_score"),
        {"alpha": 0.05, "memory": 0.20, "gate": 0.55, "reach": 0.05, "norm": 0.15},
        mode="risk",
    ),
    MetaSignalSpec(
        "causal_reachability",
        ("reach_starvation_score", "causal_reachability_credit", "attention_locality_score"),
        {"alpha": 0.10, "memory": 0.20, "gate": 0.05, "reach": 0.55, "norm": 0.10},
    ),
    MetaSignalSpec(
        "distance_contrast",
        ("distance_contrast_retention", "post_norm_distance_contrast"),
        {"alpha": 0.15, "memory": 0.05, "gate": 0.05, "reach": 0.15, "norm": 0.60},
    ),
    MetaSignalSpec(
        "attribution_uncertainty",
        ("attribution_uncertainty", "budget_conflict_score"),
        {"alpha": 0.20, "memory": 0.20, "gate": 0.20, "reach": 0.20, "norm": 0.20},
        mode="risk",
    ),
)


@dataclass(frozen=True)
class MetaControlAttentionConfig:
    """Configuration for the observer-only meta-control attention layer."""

    target_geo_to_qk_ratio: float = 0.10
    softmax_temperature: float = 0.35
    minimum_signal_strength: float = 0.01

    def validate(self) -> None:
        if self.target_geo_to_qk_ratio <= 0:
            raise ValueError("target_geo_to_qk_ratio must be positive")
        if self.softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be positive")
        if self.minimum_signal_strength < 0:
            raise ValueError("minimum_signal_strength must be non-negative")


@dataclass
class MetaControlAttentionObservation:
    """One meta-control attention observation."""

    mode: MetaControlMode
    meta_observer_only: bool
    meta_attention_weights: dict[str, float]
    meta_signal_status: dict[str, str]
    meta_signal_strengths: dict[str, float | None]
    meta_available_signal_count: int
    meta_masked_signal_count: int
    evidence_availability_score: float
    pending_control_mask: bool
    offline_replay_required: bool
    meta_top_signal: str
    meta_suppressed_signal: str
    meta_attention_entropy: float
    meta_attention_entropy_normalized: float
    controller_agreement_score: float
    controller_conflict_score: float
    meta_control_confidence: float
    meta_parameter_allocation: dict[str, float]
    meta_alpha_weight: float
    meta_memory_weight: float
    meta_gate_weight: float
    meta_reach_weight: float
    meta_norm_weight: float
    meta_observer_decision_summary: str
    meta_replay_record: dict[str, Any]


class MetaControlAttentionObserver:
    """Attend over controller telemetry without changing parameters."""

    def __init__(self, config: MetaControlAttentionConfig | None = None) -> None:
        self.config = config or MetaControlAttentionConfig()
        self.config.validate()

    def observe(self, telemetry: dict[str, Any]) -> MetaControlAttentionObservation:
        pending_control_mask = _pending_controls(telemetry)
        mode: MetaControlMode = (
            "online_partial" if pending_control_mask else "offline_matched_replay"
        )
        statuses: dict[str, str] = {}
        strengths: dict[str, float | None] = {}
        raw_scores: dict[str, float] = {}

        for spec in SIGNAL_SPECS:
            if spec.requires_complete_controls and pending_control_mask:
                statuses[spec.name] = "masked_pending_control"
                strengths[spec.name] = None
                raw_scores[spec.name] = 0.0
                continue
            raw_value = _first_numeric(telemetry, spec.fields)
            if raw_value is None:
                statuses[spec.name] = "missing"
                strengths[spec.name] = None
                raw_scores[spec.name] = 0.0
                continue
            strength = self._strength(spec, raw_value)
            statuses[spec.name] = "available"
            strengths[spec.name] = strength
            raw_scores[spec.name] = max(strength, self.config.minimum_signal_strength)

        available_names = [
            name for name, status in statuses.items() if status == "available"
        ]
        weights = _softmax_for_available(
            raw_scores,
            available_names,
            temperature=self.config.softmax_temperature,
        )
        allocation = _parameter_allocation(weights)
        entropy = _entropy(weights)
        normalized_entropy = (
            entropy / math.log(len(available_names)) if len(available_names) > 1 else 0.0
        )
        conflict = _controller_conflict_score(telemetry, pending_control_mask)
        agreement = max(0.0, 1.0 - conflict)
        availability = len(available_names) / len(SIGNAL_SPECS)
        confidence = availability * agreement * (1.0 - 0.35 * normalized_entropy)
        top_signal = _top_signal(weights)
        suppressed_signal = _suppressed_signal(statuses, weights)
        replay_required = pending_control_mask
        summary = (
            "observer_only_pending_control_replay_required"
            if replay_required
            else "observer_only_matched_control_attention_ready"
        )

        return MetaControlAttentionObservation(
            mode=mode,
            meta_observer_only=True,
            meta_attention_weights=weights,
            meta_signal_status=statuses,
            meta_signal_strengths=strengths,
            meta_available_signal_count=len(available_names),
            meta_masked_signal_count=sum(
                1 for status in statuses.values() if status != "available"
            ),
            evidence_availability_score=availability,
            pending_control_mask=pending_control_mask,
            offline_replay_required=replay_required,
            meta_top_signal=top_signal,
            meta_suppressed_signal=suppressed_signal,
            meta_attention_entropy=entropy,
            meta_attention_entropy_normalized=normalized_entropy,
            controller_agreement_score=agreement,
            controller_conflict_score=conflict,
            meta_control_confidence=max(0.0, min(1.0, confidence)),
            meta_parameter_allocation=allocation,
            meta_alpha_weight=allocation["alpha"],
            meta_memory_weight=allocation["memory"],
            meta_gate_weight=allocation["gate"],
            meta_reach_weight=allocation["reach"],
            meta_norm_weight=allocation["norm"],
            meta_observer_decision_summary=summary,
            meta_replay_record={
                "mode": mode,
                "observer_only": True,
                "pending_control_mask": pending_control_mask,
                "offline_replay_required": replay_required,
                "top_signal": top_signal,
                "suppressed_signal": suppressed_signal,
                "claim_eligibility": telemetry.get("claim_eligibility"),
                "rule": (
                    "meta-control attention may read available telemetry every step, "
                    "but pending control-family evidence is masked until matched replay"
                ),
            },
        )

    def summary(self, observation: MetaControlAttentionObservation) -> dict[str, Any]:
        row = asdict(observation)
        row["meta_control_mode"] = observation.mode
        return row

    def replay(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Replay observer attention over saved telemetry rows."""

        return [self.summary(self.observe(row)) for row in rows]

    def _strength(self, spec: MetaSignalSpec, value: float) -> float:
        if spec.mode == "distance_from_target":
            return min(
                1.0,
                abs(value - self.config.target_geo_to_qk_ratio)
                / self.config.target_geo_to_qk_ratio,
            )
        if spec.mode == "risk":
            return _bounded(value)
        return _bounded(value)


def observe_meta_control_attention(
    telemetry: dict[str, Any],
    *,
    config: MetaControlAttentionConfig | None = None,
) -> dict[str, Any]:
    """Convenience wrapper returning a JSON-ready observation."""

    observer = MetaControlAttentionObserver(config)
    return observer.summary(observer.observe(telemetry))


def _pending_controls(telemetry: dict[str, Any]) -> bool:
    if telemetry.get("pending_control_mask") is True:
        return True
    pending = telemetry.get("pending_control_families")
    if isinstance(pending, (list, tuple, set)):
        return len(pending) > 0
    return telemetry.get("claim_eligibility") != "eligible_complete_controls"


def _first_numeric(telemetry: dict[str, Any], fields: tuple[str, ...]) -> float | None:
    for field_name in fields:
        value = telemetry.get(field_name)
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _softmax_for_available(
    raw_scores: dict[str, float],
    available_names: list[str],
    *,
    temperature: float,
) -> dict[str, float]:
    weights = {name: 0.0 for name in raw_scores}
    if not available_names:
        return weights
    logits = [raw_scores[name] / temperature for name in available_names]
    max_logit = max(logits)
    exps = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exps)
    for name, exp_value in zip(available_names, exps, strict=True):
        weights[name] = exp_value / total
    return weights


def _parameter_allocation(weights: dict[str, float]) -> dict[str, float]:
    allocation = {target: 0.0 for target in PARAMETER_TARGETS}
    spec_by_name = {spec.name: spec for spec in SIGNAL_SPECS}
    for signal_name, signal_weight in weights.items():
        spec = spec_by_name[signal_name]
        for target, target_weight in spec.parameter_weights.items():
            allocation[target] += signal_weight * target_weight
    total = sum(allocation.values())
    if total <= 0:
        return allocation
    return {target: value / total for target, value in allocation.items()}


def _entropy(weights: dict[str, float]) -> float:
    return -sum(value * math.log(value) for value in weights.values() if value > 0)


def _top_signal(weights: dict[str, float]) -> str:
    if not weights or max(weights.values()) == 0:
        return "none"
    return max(weights, key=weights.get)


def _suppressed_signal(statuses: dict[str, str], weights: dict[str, float]) -> str:
    for status in ("masked_pending_control", "missing"):
        for name, signal_status in statuses.items():
            if signal_status == status:
                return name
    available = {name: weight for name, weight in weights.items() if weight > 0}
    if not available:
        return "none"
    return min(available, key=available.get)


def _controller_conflict_score(
    telemetry: dict[str, Any],
    pending_control_mask: bool,
) -> float:
    candidates = [
        _first_numeric(telemetry, ("control_penalty",)),
        _first_numeric(telemetry, ("budget_conflict_score",)),
        _first_numeric(telemetry, ("attribution_uncertainty",)),
        _first_numeric(telemetry, ("rigidity_risk",)),
        _first_numeric(telemetry, ("collapse_risk",)),
    ]
    control_separation = _first_numeric(telemetry, ("control_separation",))
    if not pending_control_mask and control_separation is not None and control_separation < 0:
        candidates.append(abs(control_separation))
    return _bounded(max((value for value in candidates if value is not None), default=0.0))
