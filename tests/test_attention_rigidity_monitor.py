import json

import torch

from evaluation.attention_metrics import (
    ATTENTION_RIGIDITY_FIELDS,
    aggregate_attention_diagnostics,
    attention_metrics,
    head_attention_diversity,
)
from evaluation.attention_rigidity_monitor import (
    build_attention_rigidity_monitor_report,
)


def causal_uniform_attention(batch: int = 1, heads: int = 2, sequence: int = 5) -> torch.Tensor:
    mask = torch.ones(sequence, sequence, dtype=torch.bool).tril()
    weights = mask.float()
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.view(1, 1, sequence, sequence).expand(batch, heads, sequence, sequence).clone()


def causal_self_attention(batch: int = 1, heads: int = 2, sequence: int = 5) -> torch.Tensor:
    weights = torch.eye(sequence, dtype=torch.float32)
    return weights.view(1, 1, sequence, sequence).expand(batch, heads, sequence, sequence).clone()


def test_attention_rigidity_metrics_detect_rigid_attention() -> None:
    spread = causal_uniform_attention(heads=1)
    rigid = causal_self_attention(heads=1)

    spread_metrics = attention_metrics(spread, geo_to_qk_ratio=0.02)
    rigid_metrics = attention_metrics(rigid, geo_to_qk_ratio=0.02)

    assert set(ATTENTION_RIGIDITY_FIELDS).issubset(spread_metrics)
    assert rigid_metrics["collapse_risk"] > spread_metrics["collapse_risk"]
    assert rigid_metrics["attention_entropy_normalized"] < spread_metrics[
        "attention_entropy_normalized"
    ]


def test_attention_rigidity_metrics_detect_geometry_takeover() -> None:
    attention = causal_uniform_attention()

    low_geo = attention_metrics(attention, geo_to_qk_ratio=0.02)
    high_geo = attention_metrics(attention, geo_to_qk_ratio=0.5)

    assert high_geo["geo_qk_risk"] > low_geo["geo_qk_risk"]
    assert high_geo["geometry_takeover_score"] > low_geo["geometry_takeover_score"]


def test_head_attention_diversity_detects_identical_heads() -> None:
    identical = causal_self_attention(heads=2)
    diverse = torch.cat(
        [
            causal_uniform_attention(heads=1),
            causal_self_attention(heads=1),
        ],
        dim=1,
    )

    assert head_attention_diversity(diverse) > head_attention_diversity(identical)


def test_aggregate_attention_diagnostics_exposes_collapse_summary() -> None:
    spread = causal_uniform_attention()
    rigid = causal_self_attention()

    diagnostics = aggregate_attention_diagnostics(
        [
            {"diagnostics": {"geo_to_qk_ratio": 0.02}, "attention_weights": spread},
            {"diagnostics": {"geo_to_qk_ratio": 0.02}, "attention_weights": rigid},
        ]
    )

    assert "collapse_risk" in diagnostics["summary"]
    assert "layer_attention_diversity" in diagnostics["summary"]
    assert "head_attention_diversity" in diagnostics["layers"]["layer_0"]


def test_attention_rigidity_monitor_report_passes() -> None:
    report = build_attention_rigidity_monitor_report(seed=2027)

    assert report["status"] == "pass"
    assert report["model_intervention"] == "none"
    assert report["checks"]["required_attention_fields_declared_in_schema"]
    assert report["checks"]["required_attention_fields_emitted"]
    assert report["checks"]["rigid_attention_has_higher_collapse_risk"]
    assert report["checks"]["geo_takeover_has_higher_geo_qk_risk"]
    assert report["next_required_step"] == "control_family_fairness_audit_v2"
    json.dumps(report)
