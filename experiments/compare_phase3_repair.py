"""Compare Phase 3 repair candidates after a failed confirm-seed run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_utils import load_json, save_json  # noqa: E402


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Phase 3 repair candidates.")
    parser.add_argument(
        "--baseline",
        default="runs/phase0_baseline/confirm_seed_wikitext2/baseline_results.json",
        help="Path to confirm-seed baseline results.",
    )
    parser.add_argument(
        "--failed-real-d",
        default="runs/phase3_geo_attention/confirm_seed/real_d_alpha_0_2/metrics.json",
        help="Path to failed real_d grad_d alpha=0.2 metrics.",
    )
    parser.add_argument(
        "--random-d-matched",
        default="runs/phase3_geo_attention/confirm_seed/random_d_alpha_0_2/metrics.json",
        help="Path to random_d alpha=0.2 control.",
    )
    parser.add_argument(
        "--random-d-best-control",
        default="runs/phase3_geo_attention/confirm_seed/random_d_alpha_0_1/metrics.json",
        help="Path to random_d alpha=0.1 control.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help=(
            "Repair candidate as label:path, e.g. "
            "real_d_alpha_0_2_detached:runs/.../metrics.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="runs/phase3_geo_attention/phase3_repair",
        help="Directory where phase3_repair_results.json is saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    references = {
        "baseline": load_required_run(Path(args.baseline), "baseline"),
        "failed_real_d_alpha_0_2_grad": load_required_run(Path(args.failed_real_d), "real_d"),
        "random_d_alpha_0_2": load_required_run(Path(args.random_d_matched), "random_d"),
        "random_d_alpha_0_1": load_required_run(Path(args.random_d_best_control), "random_d"),
    }
    reference_paths = {
        "baseline": Path(args.baseline),
        "failed_real_d_alpha_0_2_grad": Path(args.failed_real_d),
        "random_d_alpha_0_2": Path(args.random_d_matched),
        "random_d_alpha_0_1": Path(args.random_d_best_control),
    }
    candidates = {
        spec.label: load_required_run(spec.path, "real_d")
        for spec in [parse_candidate(value) for value in args.candidate]
    }
    candidate_paths = {
        spec.label: spec.path for spec in [parse_candidate(value) for value in args.candidate]
    }
    report = build_repair_report(references, reference_paths, candidates, candidate_paths)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "phase3_repair_results.json", sanitize_for_json(report))
    print(json.dumps(sanitize_for_json(report), indent=2, sort_keys=True))


def parse_candidate(value: str) -> CandidateSpec:
    label, separator, path = value.partition(":")
    if not separator or not label or not path:
        raise ValueError(f"candidate must be label:path, got {value!r}")
    return CandidateSpec(label=label, path=Path(path))


def load_required_run(path: Path, expected_condition: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing metrics file: {path}")
    run = load_json(path)
    condition = run.get("condition")
    if condition != expected_condition:
        raise ValueError(
            f"expected condition {expected_condition!r} in {path}, found {condition!r}"
        )
    return run


def build_repair_report(
    references: dict[str, dict[str, Any]],
    reference_paths: dict[str, Path],
    candidates: dict[str, dict[str, Any]],
    candidate_paths: dict[str, Path],
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one repair candidate is required")

    reference_summaries = {
        label: extract_summary(label, run, reference_paths[label])
        for label, run in references.items()
    }
    candidate_summaries = {
        label: extract_summary(label, run, candidate_paths[label])
        for label, run in candidates.items()
    }
    all_summaries = {**reference_summaries, **candidate_summaries}
    ranking = rank_labels(all_summaries, "final_validation_loss")
    best_candidate_label = rank_labels(candidate_summaries, "final_validation_loss")[0]
    best_random_label = rank_labels(
        {
            "random_d_alpha_0_2": reference_summaries["random_d_alpha_0_2"],
            "random_d_alpha_0_1": reference_summaries["random_d_alpha_0_1"],
        },
        "final_validation_loss",
    )[0]

    deltas = {
        "best_repair_vs_baseline": metric_delta(
            candidate_summaries[best_candidate_label],
            reference_summaries["baseline"],
        ),
        "best_repair_vs_failed_real_d": metric_delta(
            candidate_summaries[best_candidate_label],
            reference_summaries["failed_real_d_alpha_0_2_grad"],
        ),
        "best_repair_vs_random_d_alpha_0_2": metric_delta(
            candidate_summaries[best_candidate_label],
            reference_summaries["random_d_alpha_0_2"],
        ),
        "best_repair_vs_best_random_control": metric_delta(
            candidate_summaries[best_candidate_label],
            reference_summaries[best_random_label],
        ),
    }
    checks = build_checks(reference_summaries, candidate_summaries, deltas)

    return {
        "references": reference_summaries,
        "candidates": candidate_summaries,
        "ranking": {"by_final_validation_loss": ranking},
        "best_candidate": {
            "label": best_candidate_label,
            "summary": candidate_summaries[best_candidate_label],
        },
        "best_random_control": {
            "label": best_random_label,
            "summary": reference_summaries[best_random_label],
        },
        "deltas": deltas,
        "checks": checks,
        "summary": {
            "recommendation": recommendation(checks),
            "note": (
                "Phase 3 repair tests whether detaching real_d from the gradient path "
                "stabilizes the failed seed-2027 confirmation."
            ),
        },
    }


def extract_summary(label: str, run: dict[str, Any], path: Path) -> dict[str, Any]:
    attention = run.get("attention", {})
    geometry = run.get("geometry", {})
    geometry_summary = geometry.get("summary", {}) if isinstance(geometry, dict) else {}
    return {
        "label": label,
        "condition": run.get("condition"),
        "path": path.as_posix(),
        "seed": run.get("seed"),
        "alpha": as_float_or_none(attention.get("alpha", {}).get("initial_value")),
        "gradient_mode": attention.get("gradient_mode"),
        "distance_mode": attention.get("distance_mode"),
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
    references: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    deltas: dict[str, dict[str, dict[str, float | None]]],
) -> dict[str, bool]:
    summaries = [*references.values(), *candidates.values()]
    return {
        "all_losses_finite": all(
            summary["final_validation_loss"] is not None for summary in summaries
        ),
        "all_candidates_detached": all(
            summary.get("gradient_mode") == "detached_d" for summary in candidates.values()
        ),
        "best_repair_beats_baseline": is_improvement(
            deltas["best_repair_vs_baseline"]["final_validation_loss"]
        ),
        "best_repair_beats_failed_real_d": is_improvement(
            deltas["best_repair_vs_failed_real_d"]["final_validation_loss"]
        ),
        "best_repair_beats_random_d_alpha_0_2": is_improvement(
            deltas["best_repair_vs_random_d_alpha_0_2"]["final_validation_loss"]
        ),
        "best_repair_beats_best_random_control": is_improvement(
            deltas["best_repair_vs_best_random_control"]["final_validation_loss"]
        ),
    }


def recommendation(checks: dict[str, bool]) -> str:
    if (
        checks["all_losses_finite"]
        and checks["all_candidates_detached"]
        and checks["best_repair_beats_baseline"]
        and checks["best_repair_beats_best_random_control"]
    ):
        return "detached_repair_supports_phase3_repeat"
    if (
        checks["all_losses_finite"]
        and checks["best_repair_beats_baseline"]
        and checks["best_repair_beats_failed_real_d"]
    ):
        return "detached_repair_partial_support"
    if checks["all_losses_finite"] and checks["best_repair_beats_failed_real_d"]:
        return "detached_repair_improves_real_d_only"
    return "repair_not_supported"


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


if __name__ == "__main__":
    main()
