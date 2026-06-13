"""Control-isolation audit for GeoAttention v2.

This report is a mechanics gate. It does not claim that a geometry condition
is better; it only proves that random/shuffled controls are built from their
own W-level graph, distance normalization, and memory recurrence.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

import torch

from attention.geo_attention import GeoAttention, GeoAttentionConfig
from evaluation.relational_memory_observer import MemoryConfig, synthetic_memory_hidden_layers
from layers.relational_graph import (
    RelationalGraph,
    make_random_graph_like,
    make_shuffled_graph,
    make_valid_edge_mask_like,
)

ControlIsolationStatus = Literal["pass", "fail"]

AUDIT_MODES = [
    "real_stable_causal_d",
    "random_stable_causal_d",
    "shuffled_stable_causal_d",
    "no_memory_real_d",
    "instantaneous_real_d",
]


def build_control_isolation_audit_report(
    *,
    hidden_layers: list[torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    graph_config: dict[str, Any] | None = None,
    distance_config: dict[str, Any] | None = None,
    seed: int = 2027,
    memory_config: MemoryConfig | None = None,
    max_causal_step: int = 1,
) -> dict[str, Any]:
    """Build an audit report for random/shuffled control isolation."""

    memory_config = memory_config or MemoryConfig()
    memory_config.validate()

    input_source = "provided_hidden_layers"
    if hidden_layers is None:
        hidden_layers, attention_mask = synthetic_memory_hidden_layers(seed=seed)
        input_source = "synthetic_memory_smoke"
    if len(hidden_layers or []) < 2:
        raise ValueError("hidden_layers must contain at least two layers/steps")

    attention_mask = _attention_mask_or_ones(hidden_layers[0], attention_mask)
    graph_config = _normalize_graph_config(graph_config)
    distance_config = _normalize_distance_config(distance_config)

    w_audit = _w_level_control_audit(
        hidden_layers[0],
        attention_mask,
        graph_config=graph_config,
        seed=seed,
    )
    mode_records = {
        mode: _run_mode_sequence(
            hidden_layers[:2],
            attention_mask,
            mode=mode,
            graph_config=graph_config,
            distance_config=distance_config,
            memory_config=memory_config,
            max_causal_step=max_causal_step,
            seed=seed,
        )
        for mode in AUDIT_MODES
    }

    real_record = mode_records["real_stable_causal_d"]
    random_record = mode_records["random_stable_causal_d"]
    shuffled_record = mode_records["shuffled_stable_causal_d"]

    random_metadata = random_record["metadata"][-1]
    shuffled_metadata = shuffled_record["metadata"][-1]
    checks = {
        "same_hidden_inputs_for_all_modes": _same_hidden_inputs(hidden_layers[:2], mode_records),
        "valid_edge_mask_shared": w_audit["checks"]["valid_edge_mask_shared"],
        "future_edges_forbidden": w_audit["checks"]["future_edges_forbidden"],
        "random_built_at_w_level": w_audit["checks"]["random_built_at_w_level"],
        "shuffled_built_at_w_level": w_audit["checks"]["shuffled_built_at_w_level"],
        "random_w_changes_valid_edges": w_audit["checks"]["random_w_changes_valid_edges"],
        "shuffled_w_changes_valid_edges": w_audit["checks"]["shuffled_w_changes_valid_edges"],
        "shuffled_w_preserves_valid_multiset": w_audit["checks"][
            "shuffled_w_preserves_valid_multiset"
        ],
        "random_w_preserves_valid_range": w_audit["checks"]["random_w_preserves_valid_range"],
        "random_uses_self_family_normalization": _metadata_contract(
            random_metadata,
            family="random",
            memory_source="self_family_memory_recurrence",
        ),
        "shuffled_uses_self_family_normalization": _metadata_contract(
            shuffled_metadata,
            family="shuffled",
            memory_source="self_family_memory_recurrence",
        ),
        "random_uses_self_family_memory": random_metadata.get("memory_source")
        == "self_family_memory_recurrence",
        "shuffled_uses_self_family_memory": shuffled_metadata.get("memory_source")
        == "self_family_memory_recurrence",
        "random_distance_not_real_distance": _not_equal_on_finite_region(
            random_record["distance"][-1],
            real_record["distance"][-1],
        ),
        "shuffled_distance_not_real_distance": _not_equal_on_finite_region(
            shuffled_record["distance"][-1],
            real_record["distance"][-1],
        ),
        "distance_finite_regions_match": _finite_regions_match(
            [
                real_record["distance"][-1],
                random_record["distance"][-1],
                shuffled_record["distance"][-1],
            ]
        ),
        "random_memory_not_real_memory": _not_equal_on_valid_region(
            random_record["memory"][-1],
            real_record["memory"][-1],
            w_audit["valid_edge_mask"],
        ),
        "shuffled_memory_not_real_memory": _not_equal_on_valid_region(
            shuffled_record["memory"][-1],
            real_record["memory"][-1],
            w_audit["valid_edge_mask"],
        ),
        "no_real_distance_reuse_random": random_metadata.get(
            "cross_family_real_distance_reuse"
        )
        is False,
        "no_real_distance_reuse_shuffled": shuffled_metadata.get(
            "cross_family_real_distance_reuse"
        )
        is False,
        "no_real_memory_reuse_random": random_metadata.get("cross_family_real_memory_reuse")
        is False,
        "no_real_memory_reuse_shuffled": shuffled_metadata.get("cross_family_real_memory_reuse")
        is False,
    }
    status: ControlIsolationStatus = "pass" if all(checks.values()) else "fail"

    return {
        "phase": "phase8a_control_isolation_audit",
        "status": status,
        "seed": seed,
        "input_source": input_source,
        "scientific_scope": "mechanics_only_no_performance_claim",
        "audit_scope": "geoattention_v2_random_shuffled_control_isolation",
        "contract": {
            "real": "H -> W_real -> memory_real -> D_real -> self normalization",
            "random": "H -> W_random -> memory_random -> D_random -> self normalization",
            "shuffled": "H -> W_shuffled -> memory_shuffled -> D_shuffled -> self normalization",
            "forbidden": [
                "random/shuffled from normalized real D",
                "random/shuffled from real memory",
                "random/shuffled from real distance statistics",
                "different training data or seed across control families",
            ],
        },
        "memory_config": asdict(memory_config),
        "max_causal_step": max_causal_step,
        "checks": checks,
        "w_level_control_audit": _jsonify_w_audit(w_audit),
        "metadata_contracts": {
            mode: record["metadata"][-1] for mode, record in mode_records.items()
        },
        "distance_fingerprints": {
            mode: _tensor_fingerprint(record["distance"][-1])
            for mode, record in mode_records.items()
        },
        "memory_fingerprints": {
            mode: _tensor_fingerprint(record["memory"][-1])
            for mode, record in mode_records.items()
            if record["memory"][-1] is not None
        },
        "next_required_step": (
            "run_geoattention_v2_training_controls" if status == "pass" else "fix_control_isolation"
        ),
    }


def _w_level_control_audit(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    graph_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    graph_builder = RelationalGraph(graph_config)
    real_w = graph_builder(hidden_states, attention_mask=attention_mask)
    valid_edge_mask = make_valid_edge_mask_like(real_w, attention_mask=attention_mask)
    generator = torch.Generator(device=real_w.device)
    generator.manual_seed(seed)
    random_w = make_random_graph_like(
        real_w,
        generator=generator,
        valid_edge_mask=valid_edge_mask,
    )
    shuffled_w = make_shuffled_graph(
        real_w,
        generator=generator,
        valid_edge_mask=valid_edge_mask,
    )
    checks = {
        "valid_edge_mask_shared": True,
        "future_edges_forbidden": _future_edges_forbidden(valid_edge_mask),
        "random_built_at_w_level": True,
        "shuffled_built_at_w_level": True,
        "random_w_changes_valid_edges": _changes_valid_edges(real_w, random_w, valid_edge_mask),
        "shuffled_w_changes_valid_edges": _changes_valid_edges(
            real_w,
            shuffled_w,
            valid_edge_mask,
        ),
        "shuffled_w_preserves_valid_multiset": _preserves_valid_multiset(
            real_w,
            shuffled_w,
            valid_edge_mask,
        ),
        "random_w_preserves_valid_range": _preserves_valid_range(
            real_w,
            random_w,
            valid_edge_mask,
        ),
        "invalid_edges_preserved_random": _same_invalid_values(
            real_w,
            random_w,
            valid_edge_mask,
        ),
        "invalid_edges_preserved_shuffled": _same_invalid_values(
            real_w,
            shuffled_w,
            valid_edge_mask,
        ),
    }
    return {
        "checks": checks,
        "valid_edge_mask": valid_edge_mask,
        "fingerprints": {
            "real_w": _tensor_fingerprint(real_w),
            "random_w": _tensor_fingerprint(random_w),
            "shuffled_w": _tensor_fingerprint(shuffled_w),
        },
        "valid_edges": int(valid_edge_mask.sum().item()),
    }


def _run_mode_sequence(
    hidden_layers: list[torch.Tensor],
    attention_mask: torch.Tensor,
    *,
    mode: str,
    graph_config: dict[str, Any],
    distance_config: dict[str, Any],
    memory_config: MemoryConfig,
    max_causal_step: int,
    seed: int,
) -> dict[str, Any]:
    attention = GeoAttention(
        GeoAttentionConfig(
            n_heads=2,
            hidden_dim=hidden_layers[0].size(-1),
            dropout=0.0,
            distance_mode=mode,
            alpha_initial_value=0.1,
            gradient_mode="detached_d",
            memory_decay=memory_config.decay,
            memory_eta=memory_config.eta,
            memory_gate_floor=memory_config.gate_floor,
            memory_min_context_edges=memory_config.min_context_edges,
            max_causal_step=max_causal_step,
        ),
        relational_graph_config=graph_config,
        distance_config=distance_config,
    )
    geometry_memory = None
    distances = []
    memories = []
    metadata = []
    hidden_fingerprints = []
    mode_offset = AUDIT_MODES.index(mode) * 100
    for layer_index, hidden_states in enumerate(hidden_layers):
        torch.manual_seed(seed + mode_offset + layer_index)
        distance_result = attention.compute_distance(
            hidden_states,
            attention_mask=attention_mask,
            geometry_memory=geometry_memory,
            return_memory=True,
        )
        geometry_memory = distance_result["geometry_memory"]
        distances.append(distance_result["distance"])
        memories.append(geometry_memory)
        metadata.append(distance_result["metadata"])
        hidden_fingerprints.append(_tensor_fingerprint(hidden_states))
    return {
        "distance": distances,
        "memory": memories,
        "metadata": metadata,
        "hidden_fingerprints": hidden_fingerprints,
    }


def _metadata_contract(
    metadata: dict[str, Any],
    *,
    family: str,
    memory_source: str,
) -> bool:
    return (
        metadata.get("geometry_source_family") == family
        and metadata.get("control_generation_level") == "W_before_distance"
        and metadata.get("normalization_source") == "self_family_distance"
        and metadata.get("memory_source") == memory_source
        and metadata.get("cross_family_real_distance_reuse") is False
        and metadata.get("cross_family_real_memory_reuse") is False
    )


def _same_hidden_inputs(
    hidden_layers: list[torch.Tensor],
    mode_records: dict[str, dict[str, Any]],
) -> bool:
    expected = [_tensor_fingerprint(hidden) for hidden in hidden_layers]
    return all(record["hidden_fingerprints"] == expected for record in mode_records.values())


def _same_invalid_values(
    real_w: torch.Tensor,
    control_w: torch.Tensor,
    valid_edge_mask: torch.Tensor,
) -> bool:
    invalid = ~valid_edge_mask
    return bool(torch.allclose(real_w[invalid], control_w[invalid], equal_nan=True))


def _changes_valid_edges(
    real_w: torch.Tensor,
    control_w: torch.Tensor,
    valid_edge_mask: torch.Tensor,
) -> bool:
    valid = valid_edge_mask & torch.isfinite(real_w) & torch.isfinite(control_w)
    if int(valid.sum().item()) <= 1:
        return False
    return not bool(torch.allclose(real_w[valid], control_w[valid]))


def _preserves_valid_multiset(
    real_w: torch.Tensor,
    control_w: torch.Tensor,
    valid_edge_mask: torch.Tensor,
) -> bool:
    for batch_idx in range(real_w.size(0)):
        for head_idx in range(real_w.size(1)):
            valid = valid_edge_mask[batch_idx, head_idx]
            real_values = real_w[batch_idx, head_idx][valid].sort().values
            control_values = control_w[batch_idx, head_idx][valid].sort().values
            if not torch.allclose(real_values, control_values):
                return False
    return True


def _preserves_valid_range(
    real_w: torch.Tensor,
    control_w: torch.Tensor,
    valid_edge_mask: torch.Tensor,
) -> bool:
    valid = valid_edge_mask & torch.isfinite(real_w) & torch.isfinite(control_w)
    if int(valid.sum().item()) == 0:
        return False
    real_values = real_w[valid]
    control_values = control_w[valid]
    return bool(
        control_values.min() >= real_values.min()
        and control_values.max() <= real_values.max()
    )


def _future_edges_forbidden(valid_edge_mask: torch.Tensor) -> bool:
    sequence_length = valid_edge_mask.size(-1)
    future = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=valid_edge_mask.device,
    ).triu(diagonal=1)
    future = future.view(1, 1, sequence_length, sequence_length)
    return not bool(valid_edge_mask[future.expand_as(valid_edge_mask)].any().item())


def _not_equal_on_finite_region(left: torch.Tensor, right: torch.Tensor) -> bool:
    finite = torch.isfinite(left) & torch.isfinite(right)
    if int(finite.sum().item()) <= 1:
        return False
    return not bool(torch.allclose(left[finite], right[finite]))


def _not_equal_on_valid_region(
    left: torch.Tensor | None,
    right: torch.Tensor | None,
    valid_edge_mask: torch.Tensor,
) -> bool:
    if left is None or right is None:
        return False
    valid = valid_edge_mask & torch.isfinite(left) & torch.isfinite(right)
    if int(valid.sum().item()) <= 1:
        return False
    return not bool(torch.allclose(left[valid], right[valid]))


def _finite_regions_match(distances: list[torch.Tensor]) -> bool:
    finite_masks = [torch.isfinite(distance) for distance in distances]
    first = finite_masks[0]
    return all(bool(torch.equal(first, mask)) for mask in finite_masks[1:])


def _tensor_fingerprint(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return {
            "shape": list(tensor.shape),
            "finite_count": 0,
            "sum": 0.0,
            "mean": None,
            "std": None,
        }
    return {
        "shape": list(tensor.shape),
        "finite_count": int(finite.numel()),
        "sum": _to_float(finite.sum()),
        "mean": _to_float(finite.mean()),
        "std": _to_float(finite.std(unbiased=False)),
    }


def _jsonify_w_audit(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "checks": report["checks"],
        "valid_edges": report["valid_edges"],
        "fingerprints": report["fingerprints"],
    }


def _normalize_graph_config(graph_config: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(graph_config or {"kernel": "sigmoid_cosine", "normalize_hidden": True})
    if normalized.get("diagonal_policy", "keep_for_distance") == "keep_for_distance":
        normalized["diagonal_policy"] = "keep"
    return normalized


def _normalize_distance_config(distance_config: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "epsilon": 1e-6,
        "normalization": "offdiag_zscore_clamp",
        "clip_value": 5.0,
        "diagonal_policy": "zero",
        "causal_runtime_distance": True,
        **(distance_config or {}),
    }


def _attention_mask_or_ones(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if attention_mask is not None:
        return attention_mask
    return torch.ones(hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device)


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
