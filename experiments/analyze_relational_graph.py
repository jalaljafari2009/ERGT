"""Analyze relational graphs extracted from a trained baseline model."""

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

from evaluation.graph_metrics import (  # noqa: E402
    graph_metrics_with_controls,
    layer_to_layer_similarity,
)
from experiments.data_utils import load_json, load_prepared_datasets, save_json  # noqa: E402
from experiments.train_baseline import build_model_config, default_data_dir  # noqa: E402
from layers.relational_graph import RelationalGraph  # noqa: E402
from models.transformer_baseline import TransformerBaseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ERGT Phase 1 relational graphs.")
    parser.add_argument(
        "--config",
        default="configs/analysis/graph_observer.json",
        help="Path to graph observer analysis config.",
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
    graph_config["diagonal_policy"] = _normalize_diagonal_policy(
        graph_config.get("diagonal_policy", "keep")
    )
    graph_builder = RelationalGraph(graph_config).to(device)

    exclude_diagonal = bool(config["metrics"].get("exclude_diagonal", True))
    thresholds = [float(value) for value in config["metrics"].get("sparsity_thresholds", [])]
    max_batches = int(config["dataset"].get("max_batches", 0))
    max_saved_matrices = int(config["metrics"].get("max_saved_matrices", 0))
    save_sample_matrices = bool(config["metrics"].get("save_sample_matrices", False))

    layer_metric_accumulators: dict[str, list[dict[str, Any]]] = {}
    layer_similarity_accumulators: dict[str, list[dict[str, float]]] = {}
    saved_matrices = 0

    with torch.no_grad():
        for batch_index, (input_ids, _) in enumerate(data_loader):
            if max_batches > 0 and batch_index >= max_batches:
                break

            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
            outputs = model(input_ids, attention_mask=attention_mask, return_hidden_states=True)
            hidden_states = outputs["hidden_states"]
            graphs: list[torch.Tensor] = []

            for layer_index, layer_hidden_states in enumerate(hidden_states):
                graph = graph_builder(layer_hidden_states, attention_mask=attention_mask)
                graphs.append(graph)

                metrics = graph_metrics_with_controls(
                    graph,
                    exclude_diagonal=exclude_diagonal,
                    sparsity_thresholds=thresholds,
                    seed=int(config["run"].get("seed", 1337)) + batch_index + layer_index,
                )
                layer_key = f"layer_{layer_index}"
                layer_metric_accumulators.setdefault(layer_key, []).append(metrics)

                if save_sample_matrices and saved_matrices < max_saved_matrices:
                    torch.save(
                        graph.detach().cpu(),
                        matrices_dir / f"batch_{batch_index}_layer_{layer_index}_W.pt",
                    )
                    saved_matrices += 1

            for layer_index in range(len(graphs) - 1):
                similarity = layer_to_layer_similarity(graphs[layer_index], graphs[layer_index + 1])
                key = f"layer_{layer_index}_to_{layer_index + 1}"
                layer_similarity_accumulators.setdefault(key, []).append(similarity)

    report = {
        "run_id": config["run"]["output_dir"],
        "phase": config["run"]["phase"],
        "condition": config["run"]["condition"],
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_step": int(checkpoint.get("step", -1)),
        "data_dir": str(data_dir),
        "graph_kernel": graph_config["kernel"],
        "diagonal_policy": graph_config["diagonal_policy"],
        "exclude_diagonal": exclude_diagonal,
        "batches_analyzed": _count_batches(layer_metric_accumulators),
        "layers": {
            layer_key: average_nested_metrics(metrics)
            for layer_key, metrics in sorted(layer_metric_accumulators.items())
        },
        "layer_to_layer_similarity": {
            key: average_nested_metrics(metrics)
            for key, metrics in sorted(layer_similarity_accumulators.items())
        },
        "artifacts": {
            "sample_matrices_saved": saved_matrices,
            "sample_matrices_dir": str(matrices_dir),
        },
    }

    save_json(output_dir / config["logging"].get("results", "graph_stats.json"), report)
    print(json.dumps(report, indent=2, sort_keys=True))


def load_prepared_or_raise(data_dir: Path):
    if not (data_dir / "metadata.json").exists():
        raise FileNotFoundError(
            f"prepared dataset not found at {data_dir}. "
            "Run scripts/prepare_wikitext2.py first, or pass --data-dir."
        )
    return load_prepared_datasets(data_dir)


def _normalize_diagonal_policy(policy: str) -> str:
    if policy == "report_separately":
        return "keep"
    return policy


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
    if isinstance(first, str | bool):
        return first
    if isinstance(first, int | float):
        numeric_values = [float(value) for value in values]
        return sum(numeric_values) / len(numeric_values)
    return first


def _count_batches(layer_metric_accumulators: dict[str, list[dict[str, Any]]]) -> int:
    if not layer_metric_accumulators:
        return 0
    return max(len(metrics) for metrics in layer_metric_accumulators.values())


if __name__ == "__main__":
    main()
