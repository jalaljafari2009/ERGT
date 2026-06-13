import json

from attention.geo_attention import GeoAttention, GeoAttentionConfig
from evaluation.control_isolation_audit import build_control_isolation_audit_report
from evaluation.relational_memory_observer import synthetic_memory_hidden_layers


def test_control_isolation_audit_passes_on_synthetic_memory_field() -> None:
    report = build_control_isolation_audit_report(seed=2027)

    assert report["status"] == "pass"
    checks = report["checks"]
    assert checks["same_hidden_inputs_for_all_modes"]
    assert checks["random_built_at_w_level"]
    assert checks["shuffled_built_at_w_level"]
    assert checks["random_uses_self_family_normalization"]
    assert checks["shuffled_uses_self_family_normalization"]
    assert checks["random_uses_self_family_memory"]
    assert checks["shuffled_uses_self_family_memory"]
    assert checks["random_distance_not_real_distance"]
    assert checks["shuffled_distance_not_real_distance"]
    assert checks["random_memory_not_real_memory"]
    assert checks["shuffled_memory_not_real_memory"]
    assert checks["no_real_distance_reuse_random"]
    assert checks["no_real_memory_reuse_shuffled"]
    assert report["next_required_step"] == "run_geoattention_v2_training_controls"
    json.dumps(report)


def test_geoattention_diagnostics_expose_control_isolation_contract() -> None:
    hidden_layers, attention_mask = synthetic_memory_hidden_layers(seed=2027)
    attention = GeoAttention(
        GeoAttentionConfig(
            n_heads=2,
            hidden_dim=hidden_layers[0].size(-1),
            dropout=0.0,
            distance_mode="random_stable_causal_d",
        ),
        relational_graph_config={"kernel": "sigmoid_cosine", "normalize_hidden": True},
        distance_config={"causal_runtime_distance": True},
    )

    result = attention.compute_distance(
        hidden_layers[0],
        attention_mask=attention_mask,
        return_memory=True,
    )
    metadata = result["metadata"]

    assert metadata["geometry_source_family"] == "random"
    assert metadata["control_generation_level"] == "W_before_distance"
    assert metadata["normalization_source"] == "self_family_distance"
    assert metadata["memory_source"] == "self_family_memory_recurrence"
    assert metadata["cross_family_real_distance_reuse"] is False
    assert metadata["cross_family_real_memory_reuse"] is False
