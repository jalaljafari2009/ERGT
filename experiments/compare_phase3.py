"""Compare Phase 3 baseline and ERGT-v1 ablation runs."""

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

REQUIRED_CONDITIONS = ("baseline", "alpha_zero", "real_d", "random_d", "shuffled_d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ERGT Phase 3 ablation runs.")
    parser.add_argument(
        "--baseline",
        default="runs/phase0_baseline/proof_wikitext2/baseline_results.json",
        help="Path to baseline_results.json.",
    )
    parser.add_argument(
        "--alpha-zero",
        default="runs/phase3_geo_attention/alpha_zero/metrics.json",
        help="Path to alpha_zero metrics.json.",
    )
    parser.add_argument(
        "--real-d",
        default="runs/phase3_geo_attention/real_d/metrics.json",
        help="Path to real_d metrics.json.",
    )
    parser.add_argument(
        "--random-d",
        default="runs/phase3_geo_attention/random_d/metrics.json",
        help="Path to random_d metrics.json.",
    )
    parser.add_argument(
        "--shuffled-d",
        default="runs/phase3_geo_attention/shuffled_d/metrics.json",
        help="Path to shuffled_d metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/phase3_geo_attention",
        help="Directory where comparison_results.json and ablation_report.json are saved.",
    )
    parser.add_argument(
        "--alpha-zero-tolerance",
        type=float,
        default=0.02,
        help="Allowed relative validation-loss delta for alpha_zero vs baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "baseline": Path(args.baseline),
        "alpha_zero": Path(args.alpha_zero),
        "real_d": Path(args.real_d),
        "random_d": Path(args.random_d),
        "shuffled_d": Path(args.shuffled_d),
    }
    runs = {condition: load_required_run(path, condition) for condition, path in paths.items()}

    comparison = build_comparison_results(runs, paths)
    ablation = build_ablation_report(
        comparison,
        alpha_zero_tolerance=float(args.alpha_zero_tolerance),
    )

    save_json(output_dir / "comparison_results.json", sanitize_for_json(comparison))
    save_json(output_dir / "ablation_report.json", sanitize_for_json(ablation))
    print(
        json.dumps(
            {
                "comparison_results": sanitize_for_json(comparison),
                "ablation_report": sanitize_for_json(ablation),
            },
            indent=2,
            sort_keys=True,
        )
    )


def load_required_run(path: Path, expected_condition: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing {expected_condition} metrics file: {path}")
    run = load_json(path)
    condition = run.get("condition", expected_condition)
    if condition != expected_condition:
        raise ValueError(
            f"expected condition {expected_condition!r} in {path}, found {condition!r}"
        )
    return run


def build_comparison_results(
    runs: dict[str, dict[str, Any]],
    paths: dict[str, Path],
) -> dict[str, Any]:
    conditions = {
        condition: extract_condition_summary(condition, runs[condition], paths[condition])
        for condition in REQUIRED_CONDITIONS
    }
    ranking = rank_conditions(conditions, "final_validation_loss")

    return {
        "required_conditions": list(REQUIRED_CONDITIONS),
        "conditions": conditions,
        "ranking": {
            "by_final_validation_loss": ranking,
        },
        "deltas": {
            "real_d_vs_baseline": metric_delta(conditions["real_d"], conditions["baseline"]),
            "real_d_vs_alpha_zero": metric_delta(
                conditions["real_d"],
                conditions["alpha_zero"],
            ),
            "real_d_vs_random_d": metric_delta(conditions["real_d"], conditions["random_d"]),
            "real_d_vs_shuffled_d": metric_delta(
                conditions["real_d"],
                conditions["shuffled_d"],
            ),
            "alpha_zero_vs_baseline": metric_delta(
                conditions["alpha_zero"],
                conditions["baseline"],
            ),
        },
    }


def extract_condition_summary(
    condition: str,
    run: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    geometry = run.get("geometry", {})
    geometry_summary = geometry.get("summary", {}) if isinstance(geometry, dict) else {}
    return {
        "condition": condition,
        "path": str(path),
        "final_training_loss": as_float_or_none(run.get("final_training_loss")),
        "best_validation_loss": as_float_or_none(run.get("best_validation_loss")),
        "final_validation_loss": as_float_or_none(run.get("final_validation_loss")),
        "perplexity": as_float_or_none(run.get("perplexity")),
        "average_tokens_per_second": as_float_or_none(run.get("average_tokens_per_second")),
        "peak_memory_bytes": as_float_or_none(run.get("peak_memory_bytes")),
        "total_training_tokens": as_float_or_none(run.get("total_training_tokens")),
        "seed": run.get("seed"),
        "device": run.get("device"),
        "attention": run.get("attention", {}),
        "distance": run.get("distance", {}),
        "geometry_summary": sanitize_for_json(geometry_summary),
    }


def build_ablation_report(
    comparison: dict[str, Any],
    *,
    alpha_zero_tolerance: float,
) -> dict[str, Any]:
    conditions = comparison["conditions"]
    deltas = comparison["deltas"]

    alpha_zero_delta = deltas["alpha_zero_vs_baseline"]["final_validation_loss"]
    real_vs_baseline = deltas["real_d_vs_baseline"]["final_validation_loss"]
    real_vs_random = deltas["real_d_vs_random_d"]["final_validation_loss"]
    real_vs_shuffled = deltas["real_d_vs_shuffled_d"]["final_validation_loss"]

    checks = {
        "all_required_conditions_present": all(
            condition in conditions for condition in REQUIRED_CONDITIONS
        ),
        "alpha_zero_matches_baseline": delta_within_relative_tolerance(
            alpha_zero_delta,
            alpha_zero_tolerance,
        ),
        "real_d_beats_baseline": is_improvement(real_vs_baseline),
        "real_d_beats_random_d": is_improvement(real_vs_random),
        "real_d_beats_shuffled_d": is_improvement(real_vs_shuffled),
        "real_d_validation_loss_finite": conditions["real_d"]["final_validation_loss"] is not None,
    }

    if (
        checks["alpha_zero_matches_baseline"]
        and checks["real_d_beats_baseline"]
        and checks["real_d_beats_random_d"]
        and checks["real_d_beats_shuffled_d"]
    ):
        recommendation = "gate_ready_positive"
    elif not checks["all_required_conditions_present"]:
        recommendation = "missing_required_conditions"
    elif not checks["alpha_zero_matches_baseline"]:
        recommendation = "implementation_check_required"
    elif checks["real_d_beats_random_d"] or checks["real_d_beats_shuffled_d"]:
        recommendation = "conditional_repeat_with_more_controls"
    else:
        recommendation = "redesign_or_repeat_phase3"

    return {
        "summary": {
            "recommendation": recommendation,
            "note": (
                "This is not the Gate 1 decision. It is the Phase 3 comparison input "
                "for evaluation/gate_decision.py."
            ),
        },
        "checks": checks,
        "alpha_zero_tolerance": alpha_zero_tolerance,
        "deltas": deltas,
        "interpretation": build_interpretation(checks),
    }


def build_interpretation(checks: dict[str, bool]) -> list[str]:
    notes: list[str] = []
    if checks["alpha_zero_matches_baseline"]:
        notes.append("alpha_zero is close to baseline; GeoAttention neutral path is plausible.")
    else:
        notes.append(
            "alpha_zero differs from baseline; inspect implementation before claiming gains."
        )

    if checks["real_d_beats_baseline"]:
        notes.append("real_d improves final validation loss over baseline.")
    else:
        notes.append("real_d does not improve final validation loss over baseline.")

    if checks["real_d_beats_random_d"] and checks["real_d_beats_shuffled_d"]:
        notes.append("real_d beats both non-relational distance controls.")
    else:
        notes.append("real_d does not beat all required distance controls.")
    return notes


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


def rank_conditions(conditions: dict[str, dict[str, Any]], metric: str) -> list[str]:
    return [
        condition
        for condition, _ in sorted(
            conditions.items(),
            key=lambda item: (
                item[1].get(metric) is None,
                item[1].get(metric) if item[1].get(metric) is not None else math.inf,
            ),
        )
    ]


def is_improvement(delta: dict[str, float | None]) -> bool:
    absolute = delta.get("absolute")
    return absolute is not None and absolute < 0


def delta_within_relative_tolerance(
    delta: dict[str, float | None],
    tolerance: float,
) -> bool:
    relative = delta.get("relative")
    absolute = delta.get("absolute")
    if relative is not None:
        return abs(relative) <= tolerance
    return absolute is not None and abs(absolute) <= tolerance


def as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


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


if __name__ == "__main__":
    main()
