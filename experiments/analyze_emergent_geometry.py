"""Analyze emergent distance and geometry from baseline hidden states."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.distance_metrics import (  # noqa: E402
    distance_metrics_with_controls,
    neighborhood_overlap,
)
from evaluation.graph_metrics import graph_metrics  # noqa: E402
from experiments.data_utils import load_json, load_prepared_datasets, save_json  # noqa: E402
from experiments.train_baseline import build_model_config, default_data_dir  # noqa: E402
from geometry.emergent_distance import EmergentDistance  # noqa: E402
from layers.relational_graph import RelationalGraph  # noqa: E402
from models.transformer_baseline import TransformerBaseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ERGT Phase 2 emergent geometry.")
    parser.add_argument(
        "--config",
        default="configs/analysis/distance_geometry.json",
        help="Path to distance geometry analysis config.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Prepared data directory. Defaults from the source model config.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint override.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Analysis device. Defaults to cuda if available, otherwise cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    source_config = load_json(config["source_model"]["config"])
    checkpoint_path = Path(args.checkpoint or config["source_model"]["checkpoint"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    output_dir = Path(config["run"]["output_dir"])
    artifacts_dir = output_dir / config["logging"].get("artifacts_dir", "artifacts")
    matrices_dir = artifacts_dir / "sample_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir(source_config)
    _, validation_dataset, metadata = load_prepared_or_raise(data_dir)
    data_loader = DataLoader(
        validation_dataset,
        batch_size=int(source_config["training"]["batch_size"]),
        shuffle=False,
        drop_last=False,
    )

    model_config = build_model_config(source_config, metadata)
    model = TransformerBaseline(model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    graph_config = dict(config["relational_graph"])
    graph_config.pop("layers", None)
    graph_config["diagonal_policy"] = _normalize_graph_diagonal_policy(
        graph_config.get("diagonal_policy", "keep")
    )
    graph_builder = RelationalGraph(graph_config).to(device)
    distance_builder = EmergentDistance(config["distance"]).to(device)

    neighborhood_k = [int(value) for value in config["metrics"].get("neighborhood_k", [])]
    max_batches = int(config["dataset"].get("max_batches", 0))
    max_saved_matrices = int(config["metrics"].get("max_saved_matrices", 0))
    save_sample_matrices = bool(config["metrics"].get("save_sample_matrices", False))

    layer_distance_accumulators: dict[str, list[dict[str, Any]]] = {}
    layer_graph_accumulators: dict[str, list[dict[str, Any]]] = {}
    neighborhood_overlap_accumulators: dict[str, list[float]] = {}
    saved_matrices = 0

    with torch.no_grad():
        for batch_index, (input_ids, _) in enumerate(data_loader):
            if max_batches > 0 and batch_index >= max_batches:
                break

            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                return_hidden_states=True,
                return_attention_weights=True,
            )
            hidden_states = outputs["hidden_states"]
            attention_weights = outputs["attention_weights"]
            distances: list[torch.Tensor] = []

            for layer_index, layer_hidden_states in enumerate(hidden_states):
                graph = graph_builder(layer_hidden_states, attention_mask=attention_mask)
                distance = distance_builder(graph, attention_mask=attention_mask)
                distances.append(distance)

                attention_logits_proxy = None
                if layer_index < len(attention_weights):
                    attention_logits_proxy = _broadcast_distance_target(
                        distance,
                        attention_weights[layer_index],
                    )

                distance_report = distance_metrics_with_controls(
                    distance,
                    attention_logits=attention_logits_proxy,
                    neighborhood_k=neighborhood_k,
                    seed=int(config["run"].get("seed", 1337)) + batch_index + layer_index,
                )
                graph_report = graph_metrics(
                    graph,
                    exclude_diagonal=True,
                    sparsity_thresholds=[0.5, 0.75, 0.9, 0.95],
                )

                layer_key = f"layer_{layer_index}"
                layer_distance_accumulators.setdefault(layer_key, []).append(distance_report)
                layer_graph_accumulators.setdefault(layer_key, []).append(graph_report)

                if save_sample_matrices and saved_matrices < max_saved_matrices:
                    torch.save(
                        {"W": graph.detach().cpu(), "D": distance.detach().cpu()},
                        matrices_dir / f"batch_{batch_index}_layer_{layer_index}_geometry.pt",
                    )
                    saved_matrices += 1

            for layer_index in range(len(distances) - 1):
                for k in neighborhood_k:
                    key = f"layer_{layer_index}_to_{layer_index + 1}_k_{k}"
                    overlap = neighborhood_overlap(
                        distances[layer_index], distances[layer_index + 1], k
                    )
                    neighborhood_overlap_accumulators.setdefault(key, []).append(overlap)

    distance_stats = {
        "run_id": config["run"]["output_dir"],
        "phase": config["run"]["phase"],
        "condition": config["run"]["condition"],
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_step": int(checkpoint.get("step", -1)),
        "data_dir": str(data_dir),
        "distance_formula": config["distance"].get("formula", "-log(W + epsilon)"),
        "epsilon": config["distance"].get("epsilon"),
        "normalization": config["distance"].get("normalization"),
        "diagonal_policy": config["distance"].get("diagonal_policy"),
        "batches_analyzed": _count_batches(layer_distance_accumulators),
        "layers": {
            layer_key: average_nested_metrics(metrics)
            for layer_key, metrics in sorted(layer_distance_accumulators.items())
        },
        "neighborhood_overlap": {
            key: sum(values) / len(values)
            for key, values in sorted(neighborhood_overlap_accumulators.items())
        },
        "artifacts": {
            "sample_matrices_saved": saved_matrices,
            "sample_matrices_dir": str(matrices_dir),
        },
    }

    geometry_report = build_geometry_report(distance_stats, layer_graph_accumulators)
    save_json(
        output_dir / config["logging"].get("distance_stats", "distance_stats.json"), distance_stats
    )
    save_json(
        output_dir / config["logging"].get("geometry_report", "geometry_report.json"),
        geometry_report,
    )
    print(
        json.dumps(
            {"distance_stats": distance_stats, "geometry_report": geometry_report},
            indent=2,
            sort_keys=True,
        )
    )


def build_geometry_report(
    distance_stats: dict[str, Any],
    layer_graph_accumulators: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    graph_summary = {
        layer_key: average_nested_metrics(metrics)
        for layer_key, metrics in sorted(layer_graph_accumulators.items())
    }
    risks: list[str] = []
    evidence: list[str] = []

    for layer_key, layer_metrics in distance_stats["layers"].items():
        real_distance = layer_metrics.get("real_D", {})
        real_std = real_distance.get("std", 0.0)
        real_entropy = real_distance.get("entropy", 0.0)
        if real_std and real_std > 0:
            evidence.append(f"{layer_key}: non-zero distance variance")
        else:
            risks.append(f"{layer_key}: distance variance is zero or missing")
        if real_entropy and real_entropy > 0:
            evidence.append(f"{layer_key}: positive distance entropy")

    return {
        "summary": {
            "non_trivial_geometry_detected": bool(evidence) and not risks,
            "main_evidence": evidence,
            "main_risks": risks,
        },
        "graph_summary": graph_summary,
        "neighborhood_overlap": distance_stats.get("neighborhood_overlap", {}),
        "notes": [
            "This report is diagnostic. Phase 2 does not prove reasoning, memory, or intelligence.",
            "Geometry is treated as neighborhood and distance structure induced by W.",
        ],
    }


def load_prepared_or_raise(data_dir: Path):
    if not (data_dir / "metadata.json").exists():
        raise FileNotFoundError(
            f"prepared dataset not found at {data_dir}. "
            "Run scripts/prepare_wikitext2.py first, or pass --data-dir."
        )
    return load_prepared_datasets(data_dir)


def _normalize_graph_diagonal_policy(policy: str) -> str:
    if policy == "keep_for_distance":
        return "keep"
    return policy


def _broadcast_distance_target(distance: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    if distance.shape == attention.shape:
        return attention
    if distance.size(1) == 1:
        return attention
    return attention[:, : distance.size(1)]


def average_nested_metrics(items: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    return _average_values(items)


def _average_values(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, Mapping):
        keys = sorted({key for value in values for key in value.keys()})
        return {
            key: _average_values([value[key] for value in values if key in value]) for key in keys
        }
    if isinstance(first, list):
        return first
    if isinstance(first, str | bool) or first is None:
        return first
    if isinstance(first, int | float):
        numeric_values = [float(value) for value in values]
        return sum(numeric_values) / len(numeric_values)
    return first


def _count_batches(layer_accumulators: dict[str, list[dict[str, Any]]]) -> int:
    if not layer_accumulators:
        return 0
    return max(len(metrics) for metrics in layer_accumulators.values())


if __name__ == "__main__":
    main()
