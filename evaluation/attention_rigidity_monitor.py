"""Attention rigidity and collapse monitor for open adaptive ERGT control."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import torch

from evaluation.attention_metrics import (
    ATTENTION_RIGIDITY_FIELDS,
    AttentionRigidityConfig,
    aggregate_attention_diagnostics,
    attention_metrics,
)


def build_attention_rigidity_monitor_report(
    *,
    attention_weights: torch.Tensor | None = None,
    seed: int = 2027,
    rigidity_config: AttentionRigidityConfig | None = None,
) -> dict[str, Any]:
    """Build the stage-5 Attention Rigidity and Collapse Monitor report."""

    from evaluation.unified_telemetry_schema import build_unified_telemetry_schema_report

    rigidity_config = rigidity_config or AttentionRigidityConfig()
    rigidity_config.validate()
    input_source = "provided_attention_weights"
    if attention_weights is None:
        attention_weights = _causal_uniform_attention(batch=2, heads=2, sequence=6)
        input_source = "synthetic_attention_smoke"
    _validate_attention(attention_weights)

    spread_attention = _causal_uniform_attention(
        attention_weights.size(0),
        1,
        attention_weights.size(-1),
        device=attention_weights.device,
    )
    rigid_attention = _causal_self_attention(
        attention_weights.size(0),
        1,
        attention_weights.size(-1),
        device=attention_weights.device,
    )
    diverse_attention = _diverse_two_head_attention(
        attention_weights.size(0),
        max(attention_weights.size(1), 2),
        attention_weights.size(-1),
        device=attention_weights.device,
    )
    identical_heads = _causal_self_attention(
        attention_weights.size(0),
        max(attention_weights.size(1), 2),
        attention_weights.size(-1),
        device=attention_weights.device,
    )

    metrics = {
        "provided_or_default": attention_metrics(
            attention_weights,
            geo_to_qk_ratio=0.02,
            rigidity_config=rigidity_config,
        ),
        "spread_attention": attention_metrics(
            spread_attention,
            geo_to_qk_ratio=0.02,
            rigidity_config=rigidity_config,
        ),
        "rigid_attention": attention_metrics(
            rigid_attention,
            geo_to_qk_ratio=0.02,
            rigidity_config=rigidity_config,
        ),
        "geo_takeover_attention": attention_metrics(
            spread_attention,
            geo_to_qk_ratio=0.5,
            rigidity_config=rigidity_config,
        ),
        "diverse_heads": attention_metrics(
            diverse_attention,
            geo_to_qk_ratio=0.02,
            rigidity_config=rigidity_config,
        ),
        "identical_heads": attention_metrics(
            identical_heads,
            geo_to_qk_ratio=0.02,
            rigidity_config=rigidity_config,
        ),
    }
    aggregate = aggregate_attention_diagnostics(
        [
            {
                "diagnostics": {"geo_to_qk_ratio": 0.02},
                "attention_weights": spread_attention,
            },
            {
                "diagnostics": {"geo_to_qk_ratio": 0.02},
                "attention_weights": rigid_attention,
            },
        ]
    )
    schema_report = build_unified_telemetry_schema_report()
    schema_fields = set(schema_report["fields"])
    checks = {
        "model_intervention_none": True,
        "required_attention_fields_declared_in_schema": set(
            ATTENTION_RIGIDITY_FIELDS
        ).issubset(schema_fields),
        "required_attention_fields_emitted": all(
            set(ATTENTION_RIGIDITY_FIELDS).issubset(row) for row in metrics.values()
        ),
        "all_metrics_finite": _all_monitor_metrics_finite(metrics),
        "rigid_attention_has_higher_collapse_risk": (
            metrics["rigid_attention"]["collapse_risk"]
            > metrics["spread_attention"]["collapse_risk"]
        ),
        "geo_takeover_has_higher_geo_qk_risk": (
            metrics["geo_takeover_attention"]["geo_qk_risk"]
            > metrics["spread_attention"]["geo_qk_risk"]
        ),
        "identical_heads_have_higher_head_collapse": (
            metrics["identical_heads"]["head_collapse_risk"]
            > metrics["diverse_heads"]["head_collapse_risk"]
        ),
        "aggregate_exposes_layer_diversity": (
            "layer_attention_diversity" in aggregate.get("summary", {})
        ),
    }
    status = "pass" if all(checks.values()) else "fail"

    return {
        "stage": "stage5_attention_rigidity_and_collapse_monitor",
        "status": status,
        "seed": seed,
        "input_source": input_source,
        "model_intervention": "none",
        "scientific_scope": (
            "diagnose attention rigidity, collapse, and geometry takeover; "
            "do not alter attention or loss"
        ),
        "rigidity_config": asdict(rigidity_config),
        "required_fields": list(ATTENTION_RIGIDITY_FIELDS),
        "checks": checks,
        "metrics": metrics,
        "aggregate_smoke": aggregate,
        "next_required_step": (
            "control_family_fairness_audit_v2"
            if status == "pass"
            else "fix_attention_rigidity_monitor"
        ),
    }


def _causal_uniform_attention(
    batch: int,
    heads: int,
    sequence: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    mask = torch.ones(sequence, sequence, dtype=torch.bool, device=device).tril()
    weights = mask.to(dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return weights.view(1, 1, sequence, sequence).expand(batch, heads, sequence, sequence).clone()


def _causal_self_attention(
    batch: int,
    heads: int,
    sequence: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    weights = torch.eye(sequence, dtype=torch.float32, device=device)
    return weights.view(1, 1, sequence, sequence).expand(batch, heads, sequence, sequence).clone()


def _diverse_two_head_attention(
    batch: int,
    heads: int,
    sequence: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    uniform = _causal_uniform_attention(batch, 1, sequence, device=device)
    self_attention = _causal_self_attention(batch, 1, sequence, device=device)
    layers = [uniform, self_attention]
    while len(layers) < heads:
        layers.append(uniform if len(layers) % 2 == 0 else self_attention)
    return torch.cat(layers[:heads], dim=1)


def _all_monitor_metrics_finite(metrics: dict[str, dict[str, Any]]) -> bool:
    for row in metrics.values():
        for field in ATTENTION_RIGIDITY_FIELDS:
            value = row.get(field)
            if isinstance(value, bool):
                continue
            if value is None or not math.isfinite(float(value)):
                return False
    return True


def _validate_attention(attention_weights: torch.Tensor) -> None:
    if attention_weights.dim() != 4:
        raise ValueError("attention_weights must have shape [batch, heads, sequence, sequence]")
    if attention_weights.size(-1) != attention_weights.size(-2):
        raise ValueError("attention_weights must be square in the last two dimensions")
