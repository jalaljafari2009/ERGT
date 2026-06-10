"""Compare a same-seed confirmation run for ERGT real_d alpha=0.2."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_utils import load_json, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ERGT confirm-seed runs.")
    parser.add_argument(
        "--baseline",
        default="runs/phase0_baseline/confirm_seed_wikitext2/baseline_results.json",
        help="Path to confirm-seed baseline_results.json.",
    )
    parser.add_argument(
        "--real-d",
        default="runs/phase3_geo_attention/confirm_seed/real_d_alpha_0_2/metrics.json",
        help="Path to real_d alpha=0.2 metrics.json.",
    )
    parser.add_argument(
        "--random-d-matched",
        default="runs/phase3_geo_attention/confirm_seed/random_d_alpha_0_2/metrics.json",
        help="Path to matched random_d alpha=0.2 metrics.json.",
    )
    parser.add_argument(
        "--random-d-best-control",
        default="runs/phase3_geo_attention/confirm_seed/random_d_alpha_0_1/metrics.json",
        help="Path to random_d alpha=0.1 control metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/phase3_geo_attention/confirm_seed",
        help="Directory where confirm_seed_results.json is saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "baseline": Path(args.baseline),
        "real_d_alpha_0_2": Path(args.real_d),
        "random_d_alpha_0_2": Path(args.random_d_matched),
        "random_d_alpha_0_1": Path(args.random_d_best_control),
    }
    runs = {
        "baseline": load_required_run(paths["baseline"], "baseline"),
        "real_d_alpha_0_2": load_required_run(paths["real_d_alpha_0_2"], "real_d", 0.2),
        "random_d_alpha_0_2": load_required_run(
            paths["random_d_alpha_0_2"], "random_d", 0.2
        ),
        "random_d_alpha_0_1": load_required_run(
            paths["random_d_alpha_0_1"], "random_d", 0.1
        ),
    }
    report = build_confirm_seed_report(runs, paths)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "confirm_seed_results.json", sanitize_for_json(report))
    print(json.dumps(sanitize_for_json(report), indent=2, sort_keys=True))


def load_required_run(
    path: Path,
    expected_condition: str,
    expected_alpha: float | None = None,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing confirm-seed metrics file: {path}")

    run = load_json(path)
    condition = run.get("condition")
    if condition != expected_condition:
        raise ValueError(
            f"expected condition {expected_condition!r} in {path}, found {condition!r}"
        )

    if expected_alpha is not None:
        alpha = run.get("attention", {}).get("alpha", {}).get("initial_value")
        if alpha is None or not math.isclose(
            float(alpha), expected_alpha, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"expected alpha {expected_alpha} in {path}, found {alpha}")

    return run


def build_confirm_seed_report(
    runs: dict[str, dict[str, Any]],
    paths: dict[str, Path],
) -> dict[str, Any]:
    summaries = {
        label: extract_summary(label, runs[label], paths[label])
        for label in (
            "baseline",
            "real_d_alpha_0_2",
            "random_d_alpha_0_2",
            "random_d_alpha_0_1",
        )
    }
    best_random_label = rank_labels(
        {
            "random_d_alpha_0_2": summaries["random_d_alpha_0_2"],
            "random_d_alpha_0_1": summaries["random_d_alpha_0_1"],
        },
        "final_validation_loss",
    )[0]

    deltas = {
        "real_d_vs_baseline": metric_delta(
            summaries["real_d_alpha_0_2"],
            summaries["baseline"],
        ),
        "real_d_vs_random_d_alpha_0_2": metric_delta(
            summaries["real_d_alpha_0_2"],
            summaries["random_d_alpha_0_2"],
        ),
        "real_d_vs_best_random_control": metric_delta(
            summaries["real_d_alpha_0_2"],
            summaries[best_random_label],
        ),
    }
    checks = build_checks(summaries, deltas)
    ranking = rank_labels(summaries, "final_validation_loss")

    return {
        "runs": summaries,
        "ranking": {
            "by_final_validation_loss": ranking,
        },
        "best_random_control": {
            "label": best_random_label,
            "summary": summaries[best_random_label],
        },
        "deltas": deltas,
        "checks": checks,
        "summary": {
            "recommendation": recommendation(checks),
            "note": (
                "This same-seed confirmation tests whether real_d alpha=0.2 remains "
                "stronger than baseline and random distance controls under seed 2027."
            ),
        },
    }


def extract_summary(label: str, run: dict[str, Any], path: Path) -> dict[str, Any]:
    geometry = run.get("geometry", {})
    geometry_summary = geometry.get("summary", {}) if isinstance(geometry, dict) else {}
    return {
        "label": label,
        "condition": run.get("condition"),
        "path": report_path(path),
        "seed": run.get("seed"),
        "alpha": as_float_or_none(run.get("attention", {}).get("alpha", {}).get("initial_value")),
        "final_training_loss": as_float_or_none(run.get("final_training_loss")),
        "best_validation_loss": as_float_or_none(run.get("best_validation_loss")),
        "final_validation_loss": as_float_or_none(run.get("final_validation_loss")),
        "perplexity": as_float_or_none(run.get("perplexity")),
        "average_tokens_per_second": as_float_or_none(run.get("average_tokens_per_second")),
        "peak_memory_bytes": as_float_or_none(run.get("peak_memory_bytes")),
        "total_training_tokens": as_float_or_none(run.get("total_training_tokens")),
        "device": run.get("device"),
        "geometry_summary": sanitize_for_json(geometry_summary),
    }


def build_checks(
    summaries: dict[str, dict[str, Any]],
    deltas: dict[str, dict[str, dict[str, float | None]]],
) -> dict[str, bool]:
    seeds = [summary.get("seed") for summary in summaries.values()]
    return {
        "all_losses_finite": all(
            summary["final_validation_loss"] is not None for summary in summaries.values()
        ),
        "all_seeds_present": all(seed is not None for seed in seeds),
        "all_seeds_equal": len(set(seeds)) == 1 and all(seed is not None for seed in seeds),
        "real_d_beats_baseline": is_improvement(
            deltas["real_d_vs_baseline"]["final_validation_loss"]
        ),
        "real_d_beats_random_d_alpha_0_2": is_improvement(
            deltas["real_d_vs_random_d_alpha_0_2"]["final_validation_loss"]
        ),
        "real_d_beats_best_random_control": is_improvement(
            deltas["real_d_vs_best_random_control"]["final_validation_loss"]
        ),
    }


def recommendation(checks: dict[str, bool]) -> str:
    if (
        checks["all_losses_finite"]
        and checks["all_seeds_equal"]
        and checks["real_d_beats_baseline"]
        and checks["real_d_beats_random_d_alpha_0_2"]
        and checks["real_d_beats_best_random_control"]
    ):
        return "confirm_seed_supports_gate1"
    if (
        checks["all_losses_finite"]
        and checks["all_seeds_equal"]
        and checks["real_d_beats_baseline"]
        and checks["real_d_beats_random_d_alpha_0_2"]
    ):
        return "confirm_seed_conditional_support"
    if checks["all_losses_finite"] and checks["real_d_beats_baseline"]:
        return "baseline_replication_only"
    return "real_d_not_replicated"


def metric_delta(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    metrics = [
        "final_validation_loss",
        "best_validation_loss",
        "perplexity",
        "average_tokens_per_second",
        "peak_memory_bytes",
    ]
    return {
        metric: numeric_delta(candidate.get(metric), reference.get(metric)) for metric in metrics
    }


def numeric_delta(candidate: Any, reference: Any) -> dict[str, float | None]:
    candidate_value = as_float_or_none(candidate)
    reference_value = as_float_or_none(reference)
    if candidate_value is None or reference_value is None:
        return {
            "candidate": candidate_value,
            "reference": reference_value,
            "absolute": None,
            "relative": None,
        }
    absolute = candidate_value - reference_value
    relative = absolute / abs(reference_value) if reference_value != 0 else None
    return {
        "candidate": candidate_value,
        "reference": reference_value,
        "absolute": absolute,
        "relative": relative,
    }


def rank_labels(summaries: dict[str, dict[str, Any]], metric: str) -> list[str]:
    return [
        label
        for label, _ in sorted(
            summaries.items(),
            key=lambda item: (
                item[1].get(metric) is None,
                item[1].get(metric) if item[1].get(metric) is not None else math.inf,
            ),
        )
    ]


def is_improvement(delta: dict[str, float | None]) -> bool:
    absolute = delta.get("absolute")
    return absolute is not None and absolute < 0


def as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def report_path(path: Path) -> str:
    return path.as_posix()


if __name__ == "__main__":
    main()
