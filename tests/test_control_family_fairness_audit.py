import json

import torch

from evaluation.control_family_fairness_audit import (
    build_control_family_fairness_audit_report,
)
from evaluation.relational_memory_observer import synthetic_memory_hidden_layers


def test_control_family_fairness_audit_passes_on_synthetic_controls() -> None:
    report = build_control_family_fairness_audit_report(seed=2027)

    assert report["status"] == "pass"
    checks = report["checks"]
    assert checks["same_data_path"]
    assert checks["same_batch_path"]
    assert checks["same_graph_policy"]
    assert checks["same_distance_policy"]
    assert checks["same_memory_policy_for_memory_modes"]
    assert checks["control_rng_isolated_random"]
    assert checks["control_rng_isolated_shuffled"]
    assert checks["control_rng_does_not_touch_global_rng"]
    assert checks["no_cross_family_real_distance_reuse"]
    assert checks["no_cross_family_real_memory_reuse"]
    assert checks["random_and_shuffled_built_before_distance"]
    assert checks["matched_normalization_policy"]
    assert checks["matched_clipping_policy"]
    assert checks["matched_distance_diagonal_policy"]
    assert checks["matched_graph_diagonal_policy"]
    assert checks["geo_to_qk_ratio_finite"]
    assert checks["geo_to_qk_ratio_same_qk_denominator"]
    assert checks["control_isolation_gate_passed"]
    assert report["next_required_step"] == "loss_slope_and_trend_analyzer"
    json.dumps(report)


def test_control_family_fairness_audit_uses_same_batch_fingerprints() -> None:
    hidden_layers, attention_mask = synthetic_memory_hidden_layers(seed=2027)
    report = build_control_family_fairness_audit_report(
        hidden_layers=hidden_layers,
        attention_mask=attention_mask,
        seed=2027,
    )

    masks = {
        summary["attention_mask_fingerprint"]["sum"]
        for summary in report["mode_summaries"].values()
    }
    hidden_sums = {
        tuple(item["sum"] for item in summary["hidden_fingerprints"])
        for summary in report["mode_summaries"].values()
    }

    assert len(masks) == 1
    assert len(hidden_sums) == 1


def test_control_family_fairness_audit_preserves_global_rng_state() -> None:
    torch.manual_seed(1234)
    before = torch.random.get_rng_state()
    report = build_control_family_fairness_audit_report(seed=2027)
    after = torch.random.get_rng_state()

    assert torch.equal(before, after)
    assert report["checks"]["control_rng_does_not_touch_global_rng"]
