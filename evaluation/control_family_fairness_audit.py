"""Control-family fairness audit for open adaptive ERGT control."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import torch

from attention.geo_attention import GeoAttention, GeoAttentionConfig
from evaluation.control_isolation_audit import build_control_isolation_audit_report
from evaluation.relational_memory_observer import MemoryConfig, synthetic_memory_hidden_layers

FAIRNESS_MODES = [
    "real_stable_causal_d",
    "random_stable_causal_d",
    "shuffled_stable_causal_d",
    "no_memory_real_d",
    "instantaneous_real_d",
]


def build_control_family_fairness_audit_report(
    *,
    hidden_layers: list[torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    graph_config: dict[str, Any] | None = None,
    distance_config: dict[str, Any] | None = None,
    seed: int = 2027,
    memory_config: MemoryConfig | None = None,
    max_causal_step: int = 1,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """Build the stage-6 fairness audit across all planned control families."""

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
    hidden_layers = hidden_layers[:2]

    with torch.random.fork_rng(devices=[]):
        isolation_report = build_control_isolation_audit_report(
            hidden_layers=hidden_layers,
            attention_mask=attention_mask,
            graph_config=graph_config,
            distance_config=distance_config,
            seed=seed,
            memory_config=memory_config,
            max_causal_step=max_causal_step,
        )
        mode_records = {
            mode: _run_mode_sequence(
                hidden_layers,
                attention_mask,
                mode=mode,
                graph_config=graph_config,
                distance_config=distance_config,
                memory_config=memory_config,
                max_causal_step=max_causal_step,
                seed=seed,
                alpha=alpha,
            )
            for mode in FAIRNESS_MODES
        }
    final_records = {mode: record["steps"][-1] for mode, record in mode_records.items()}

    real = final_records["real_stable_causal_d"]
    random = final_records["random_stable_causal_d"]
    shuffled = final_records["shuffled_stable_causal_d"]
    ratios = {
        mode: final["diagnostics"]["geo_to_qk_ratio"]
        for mode, final in final_records.items()
    }
    checks = {
        "same_data_path": _same_field(mode_records, "hidden_fingerprints"),
        "same_batch_path": _same_field(mode_records, "attention_mask_fingerprint"),
        "same_model_shape": _same_field(mode_records, "model_shape"),
        "same_graph_policy": _same_field(mode_records, "graph_policy"),
        "same_distance_policy": _same_field(mode_records, "distance_policy"),
        "same_memory_policy_for_memory_modes": _same_memory_policy(mode_records),
        "control_rng_isolated_random": random["metadata"].get("control_rng_isolated") is True,
        "control_rng_isolated_shuffled": shuffled["metadata"].get("control_rng_isolated") is True,
        "control_rng_does_not_touch_global_rng": all(
            record["global_rng_unchanged"] for record in final_records.values()
        ),
        "no_cross_family_real_distance_reuse": all(
            final["metadata"].get("cross_family_real_distance_reuse") is False
            for final in final_records.values()
        ),
        "no_cross_family_real_memory_reuse": all(
            final["metadata"].get("cross_family_real_memory_reuse") is False
            for final in final_records.values()
        ),
        "random_and_shuffled_built_before_distance": (
            random["metadata"]["control_isolation_contract"][
                "random_or_shuffled_generated_before_distance"
            ]
            and shuffled["metadata"]["control_isolation_contract"][
                "random_or_shuffled_generated_before_distance"
            ]
        ),
        "finite_distance_regions_match": _finite_regions_match(
            [final["distance"] for final in final_records.values()]
        ),
        "random_distance_not_real_distance": _not_equal_on_finite_region(
            random["distance"],
            real["distance"],
        ),
        "shuffled_distance_not_real_distance": _not_equal_on_finite_region(
            shuffled["distance"],
            real["distance"],
        ),
        "matched_normalization_policy": _matched_policy_value(
            mode_records,
            "distance_policy",
            "normalization",
        ),
        "matched_clipping_policy": _matched_policy_value(
            mode_records,
            "distance_policy",
            "clip_value",
        ),
        "matched_distance_diagonal_policy": _matched_policy_value(
            mode_records,
            "distance_policy",
            "diagonal_policy",
        ),
        "matched_graph_diagonal_policy": _matched_policy_value(
            mode_records,
            "graph_policy",
            "diagonal_policy",
        ),
        "geo_to_qk_ratio_finite": all(math.isfinite(float(value)) for value in ratios.values()),
        "geo_to_qk_ratio_same_qk_denominator": _same_field(mode_records, "qk_fingerprint"),
        "control_isolation_gate_passed": isolation_report["status"] == "pass",
    }
    status = "pass" if all(checks.values()) else "fail"

    return {
        "stage": "stage6_control_family_fairness_audit_v2",
        "status": status,
        "seed": seed,
        "input_source": input_source,
        "scientific_scope": (
            "mechanics gate only; proves comparable control-family construction "
            "before adaptive training or performance claims"
        ),
        "families": list(FAIRNESS_MODES),
        "contract": {
            "data_path": "same hidden tensors and attention mask for every family",
            "batch_path": "same synthetic or provided batch fingerprints for every family",
            "rng": "random/shuffled controls use isolated local generators",
            "distance": "each family builds W before D and normalizes its own distance",
            "memory": "stable memory families keep self-family memory recurrence only",
            "forbidden": [
                "cross-family real distance reuse",
                "cross-family real memory reuse",
                "control generation after distance normalization",
                "family-specific normalization, clipping, or diagonal policies",
            ],
        },
        "graph_policy": graph_config,
        "distance_policy": distance_config,
        "memory_config": asdict(memory_config),
        "max_causal_step": max_causal_step,
        "alpha": alpha,
        "checks": checks,
        "geo_to_qk_ratio": ratios,
        "mode_summaries": {
            mode: _json_mode_summary(record)
            for mode, record in mode_records.items()
        },
        "control_isolation_audit_status": isolation_report["status"],
        "control_isolation_checks": isolation_report["checks"],
        "next_required_step": (
            "loss_slope_and_trend_analyzer"
            if status == "pass"
            else "fix_control_family_fairness"
        ),
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
    alpha: float,
) -> dict[str, Any]:
    attention = GeoAttention(
        GeoAttentionConfig(
            n_heads=2,
            hidden_dim=hidden_layers[0].size(-1),
            dropout=0.0,
            distance_mode=mode,
            alpha_initial_value=alpha,
            gradient_mode="detached_d",
            memory_decay=memory_config.decay,
            memory_eta=memory_config.eta,
            memory_gate_floor=memory_config.gate_floor,
            memory_min_context_edges=memory_config.min_context_edges,
            max_causal_step=max_causal_step,
            control_seed=seed,
            control_seed_offset=FAIRNESS_MODES.index(mode),
        ),
        relational_graph_config=graph_config,
        distance_config=distance_config,
    )
    qk_logits = _shared_qk_logits(hidden_layers[-1], n_heads=attention.n_heads)
    alpha_tensor = torch.tensor(
        alpha,
        dtype=hidden_layers[-1].dtype,
        device=hidden_layers[-1].device,
    )
    geometry_memory = None
    steps = []
    hidden_fingerprints = []
    for step_index, hidden_states in enumerate(hidden_layers):
        attention.set_training_step(step_index)
        before_rng = torch.random.get_rng_state()
        result = attention.compute_distance(
            hidden_states,
            attention_mask=attention_mask,
            geometry_memory=geometry_memory,
            return_memory=True,
        )
        after_rng = torch.random.get_rng_state()
        geometry_memory = result["geometry_memory"]
        distance = attention._broadcast_distance(result["distance"], qk_logits)
        steps.append(
            {
                "distance": distance,
                "memory": geometry_memory,
                "metadata": result["metadata"],
                "diagnostics": attention.diagnostics(
                    qk_logits,
                    distance,
                    alpha_tensor,
                    geometry_metadata=result["metadata"],
                ),
                "global_rng_unchanged": bool(torch.equal(before_rng, after_rng)),
            }
        )
        hidden_fingerprints.append(_tensor_fingerprint(hidden_states))

    return {
        "mode": mode,
        "steps": steps,
        "hidden_fingerprints": hidden_fingerprints,
        "attention_mask_fingerprint": _tensor_fingerprint(attention_mask),
        "model_shape": {
            "n_heads": attention.n_heads,
            "hidden_dim": attention.hidden_dim,
            "head_dim": attention.head_dim,
        },
        "graph_policy": dict(attention.relational_graph.config.__dict__),
        "distance_policy": asdict(attention.emergent_distance.config),
        "memory_policy": asdict(attention.memory_config),
        "qk_fingerprint": _tensor_fingerprint(qk_logits),
    }


def _shared_qk_logits(hidden_states: torch.Tensor, *, n_heads: int) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
    logits = normalized @ normalized.transpose(-2, -1)
    logits = logits.unsqueeze(1).expand(hidden_states.size(0), n_heads, -1, -1).clone()
    return logits / math.sqrt(hidden_states.size(-1))


def _json_mode_summary(record: dict[str, Any]) -> dict[str, Any]:
    final = record["steps"][-1]
    metadata = final["metadata"]
    diagnostics = final["diagnostics"]
    return {
        "hidden_fingerprints": record["hidden_fingerprints"],
        "attention_mask_fingerprint": record["attention_mask_fingerprint"],
        "model_shape": record["model_shape"],
        "graph_policy": record["graph_policy"],
        "distance_policy": record["distance_policy"],
        "memory_policy": record["memory_policy"],
        "qk_fingerprint": record["qk_fingerprint"],
        "distance_fingerprint": _tensor_fingerprint(final["distance"]),
        "memory_fingerprint": _tensor_fingerprint(final["memory"]),
        "metadata": metadata,
        "geo_to_qk_ratio": diagnostics["geo_to_qk_ratio"],
        "distance_mean": diagnostics["distance_mean"],
        "distance_std": diagnostics["distance_std"],
        "global_rng_unchanged": final["global_rng_unchanged"],
    }


def _same_field(records: dict[str, dict[str, Any]], field: str) -> bool:
    values = [record[field] for record in records.values()]
    return all(value == values[0] for value in values[1:])


def _same_memory_policy(records: dict[str, dict[str, Any]]) -> bool:
    policies = [
        record["memory_policy"]
        for mode, record in records.items()
        if mode
        in {
            "real_stable_causal_d",
            "random_stable_causal_d",
            "shuffled_stable_causal_d",
        }
    ]
    return all(policy == policies[0] for policy in policies[1:])


def _matched_policy_value(
    records: dict[str, dict[str, Any]],
    policy_field: str,
    value_field: str,
) -> bool:
    values = [record[policy_field][value_field] for record in records.values()]
    return all(value == values[0] for value in values[1:])


def _finite_regions_match(distances: list[torch.Tensor]) -> bool:
    finite_masks = [torch.isfinite(distance) for distance in distances]
    first = finite_masks[0]
    return all(bool(torch.equal(first, mask)) for mask in finite_masks[1:])


def _not_equal_on_finite_region(left: torch.Tensor, right: torch.Tensor) -> bool:
    finite = torch.isfinite(left) & torch.isfinite(right)
    if int(finite.sum().item()) <= 1:
        return False
    return not bool(torch.allclose(left[finite], right[finite]))


def _tensor_fingerprint(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    finite = tensor[torch.isfinite(tensor)] if torch.is_floating_point(tensor) else tensor
    if finite.numel() == 0:
        return {
            "shape": list(tensor.shape),
            "finite_count": 0,
            "sum": 0.0,
            "mean": None,
            "std": None,
        }
    finite_float = finite.to(dtype=torch.float32)
    return {
        "shape": list(tensor.shape),
        "finite_count": int(finite.numel()),
        "sum": _to_float(finite_float.sum()),
        "mean": _to_float(finite_float.mean()),
        "std": _to_float(finite_float.std(unbiased=False)),
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
