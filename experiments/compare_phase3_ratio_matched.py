"""Compare Phase 3 runs matched by geo_to_qk_ratio."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_utils import load_json, save_json  # noqa: E402

Family = Literal["real_d", "random_d", "shuffled_d"]


@dataclass(frozen=True)
class RatioMatchedRunSpec:
    family: Family
    target_ratio: float
    path: Path

    @property
    def label(self) -> str:
        return f"{self.family}_target_{format_ratio(self.target_ratio)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ratio-matched Stable Base runs.")
    parser.add_argument(
        "--baseline",
        default="runs/phase0_baseline/phase3_stable_base_seed2027/baseline_results.json",
        help="Path to the matched baseline result.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Ratio-matched run as family:target_ratio:path, e.g. "
            "real_d:0.15:runs/.../metrics.json"
        ),
    )
    parser.add_argument(
        "--ratio-tolerance",
        type=float,
        default=0.03,
        help="Allowed absolute deviation from target geo_to_qk_ratio.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/phase3_geo_attention/phase3_ratio_matched",
        help="Directory where phase3_ratio_matched_results.json is saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline)
    baseline = load_json(baseline_path)
    runs = [
        load_ratio_matched_run(parse_run_spec(value), ratio_tolerance=args.ratio_tolerance)
        for value in args.run
    ]
    report = build_ratio_matched_report(
        baseline=baseline,
        baseline_path=baseline_path,
        runs=runs,
        ratio_tolerance=float(args.ratio_tolerance),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "phase3_ratio_matched_results.json", sanitize_for_json(report))
    print(json.dumps(sanitize_for_json(report), indent=2, sort_keys=True))


def parse_run_spec(value: str) -> RatioMatchedRunSpec:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"run spec must be family:target_ratio:path, got {value!r}")
    family_raw, ratio_raw, path_raw = parts
    if family_raw not in {"real_d", "random_d", "shuffled_d"}:
        raise ValueError(f"unsupported family: {family_raw!r}")
    return RatioMatchedRunSpec(
        family=family_raw,  # type: ignore[arg-type]
        target_ratio=float(ratio_raw),
        path=Path(path_raw),
    )


def load_ratio_matched_run(
    spec: RatioMatchedRunSpec,
    *,
    ratio_tolerance: float,
) -> dict[str, Any]:
    if not spec.path.exists():
        raise FileNotFoundError(f"missing metrics file: {spec.path}")
    metrics = load_json(spec.path)
    if metrics.get("condition") != spec.family:
        raise ValueError(
            f"expected condition {spec.family!r} in {spec.path}, "
            f"found {metrics.get('condition')!r}"
        )
    config_path = spec.path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing run config file: {config_path}")
    config = load_json(config_path)
    validate_ratio_matched_config(
        config,
        expected_family=spec.family,
        target_ratio=spec.target_ratio,
    )

    summary = extract_summary(spec.label, metrics, spec.path, config)
    ratio_error = numeric_delta(summary["geo_to_qk_ratio"], spec.target_ratio)
    summary["family"] = spec.family
    summary["target_geo_to_qk_ratio"] = spec.target_ratio
    summary["ratio_error"] = ratio_error
    summary["ratio_within_tolerance"] = (
        ratio_error["absolute"] is not None and abs(ratio_error["absolute"]) <= ratio_tolerance
    )
    return {
        "label": spec.label,
        "family": spec.family,
        "target_ratio": spec.target_ratio,
        "path": spec.path.as_posix(),
        "metrics": metrics,
        "config": config,
        "summary": summary,
    }


def validate_ratio_matched_config(
    config: dict[str, Any],
    *,
    expected_family: str,
    target_ratio: float,
) -> None:
    attention = config.get("attention", {})
    relational_graph = config.get("relational_graph", {})
    distance = config.get("distance", {})
    ratio_match = config.get("run", {}).get("ratio_match")

    if not isinstance(ratio_match, dict):
        raise ValueError("ratio-matched comparison requires run.ratio_match metadata")
    config_target = as_float_or_none(ratio_match.get("target_geo_to_qk_ratio"))
    if config_target is None or not math.isclose(
        config_target, target_ratio, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(f"expected target ratio {target_ratio}, found {config_target}")
    if attention.get("distance_mode") != expected_family:
        raise ValueError(
            f"expected distance_mode={expected_family!r}, found {attention.get('distance_mode')!r}"
        )
    if attention.get("gradient_mode") != "detached_d":
        raise ValueError("ratio-matched run requires gradient_mode=detached_d")
    if relational_graph.get("kernel") != "sigmoid_cosine":
        raise ValueError("ratio-matched run requires sigmoid_cosine")
    if relational_graph.get("normalize_hidden") is not True:
        raise ValueError("ratio-matched run requires normalize_hidden=true")
    if distance.get("normalization") != "offdiag_zscore_clamp":
        raise ValueError("ratio-matched run requires offdiag_zscore_clamp")


def build_ratio_matched_report(
    *,
    baseline: dict[str, Any],
    baseline_path: Path,
    runs: list[dict[str, Any]],
    ratio_tolerance: float,
) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one ratio-matched run is required")

    baseline_summary = extract_baseline_summary(baseline, baseline_path)
    groups = build_target_groups(runs, baseline_summary)
    checks = build_checks(groups, runs)
    ranked = sorted(
        runs,
        key=lambda run: (
            run["summary"]["final_validation_loss"] is None,
            run["summary"]["final_validation_loss"] or math.inf,
        ),
    )

    return {
        "baseline": baseline_summary,
        "runs": [run["summary"] for run in ranked],
        "groups": groups,
        "ranking": {
            "by_final_validation_loss": [
                run["summary"]["label"]
                for run in ranked
                if run["summary"]["final_validation_loss"] is not None
            ]
        },
        "checks": checks,
        "summary": {
            "recommendation": recommendation(checks),
            "ratio_tolerance": ratio_tolerance,
            "note": (
                "This comparison holds geo_to_qk_ratio approximately constant so "
                "real_d is judged against random/shuffled at equal geometry strength."
            ),
        },
    }


def build_target_groups(
    runs: list[dict[str, Any]],
    baseline_summary: dict[str, Any],
) -> dict[str, Any]:
    target_ratios = sorted({run["target_ratio"] for run in runs})
    groups: dict[str, Any] = {}
    for target_ratio in target_ratios:
        target_key = format_ratio(target_ratio)
        target_runs = [run for run in runs if run["target_ratio"] == target_ratio]
        by_family = {run["family"]: run for run in target_runs}
        real = by_family.get("real_d")
        random = by_family.get("random_d")
        shuffled = by_family.get("shuffled_d")
        groups[target_key] = {
            "target_geo_to_qk_ratio": target_ratio,
            "runs": {family: run["summary"] for family, run in by_family.items()},
            "deltas": {
                "real_vs_baseline": metric_delta(
                    real["summary"] if real else None,
                    baseline_summary,
                ),
                "real_vs_random": metric_delta(
                    real["summary"] if real else None,
                    random["summary"] if random else None,
                ),
                "real_vs_shuffled": metric_delta(
                    real["summary"] if real else None,
                    shuffled["summary"] if shuffled else None,
                ),
            },
        }
    return groups


def build_checks(groups: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, bool]:
    complete_groups = [
        group
        for group in groups.values()
        if {"real_d", "random_d", "shuffled_d"} <= set(group["runs"])
    ]
    return {
        "all_losses_finite": all(
            run["summary"].get("final_validation_loss") is not None for run in runs
        ),
        "all_ratios_within_tolerance": all(
            bool(run["summary"].get("ratio_within_tolerance")) for run in runs
        ),
        "has_complete_ratio_group": bool(complete_groups),
        "real_beats_baseline_at_any_target": any(
            is_improvement(group["deltas"]["real_vs_baseline"]["final_validation_loss"])
            for group in groups.values()
        ),
        "real_beats_random_at_any_complete_target": any(
            is_improvement(group["deltas"]["real_vs_random"]["final_validation_loss"])
            for group in complete_groups
        ),
        "real_beats_shuffled_at_any_complete_target": any(
            is_improvement(group["deltas"]["real_vs_shuffled"]["final_validation_loss"])
            for group in complete_groups
        ),
        "real_beats_all_controls_at_any_complete_target": any(
            is_improvement(group["deltas"]["real_vs_baseline"]["final_validation_loss"])
            and is_improvement(group["deltas"]["real_vs_random"]["final_validation_loss"])
            and is_improvement(group["deltas"]["real_vs_shuffled"]["final_validation_loss"])
            for group in complete_groups
        ),
    }


def recommendation(checks: dict[str, bool]) -> str:
    if (
        checks["all_losses_finite"]
        and checks["all_ratios_within_tolerance"]
        and checks["has_complete_ratio_group"]
        and checks["real_beats_all_controls_at_any_complete_target"]
    ):
        return "ratio_matched_supports_real_d_repeat_seeds"
    if (
        checks["all_losses_finite"]
        and checks["all_ratios_within_tolerance"]
        and checks["real_beats_baseline_at_any_target"]
    ):
        return "ratio_matched_partial_support"
    if checks["all_losses_finite"] and not checks["all_ratios_within_tolerance"]:
        return "ratio_match_needs_recalibration"
    return "ratio_matched_not_supported"


def extract_baseline_summary(run: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "label": "baseline",
        "condition": run.get("condition"),
        "path": path.as_posix(),
        "seed": run.get("seed"),
        "final_validation_loss": as_float_or_none(run.get("final_validation_loss")),
        "best_validation_loss": as_float_or_none(run.get("best_validation_loss")),
        "perplexity": as_float_or_none(run.get("perplexity")),
        "average_tokens_per_second": as_float_or_none(run.get("average_tokens_per_second")),
    }


def extract_summary(
    label: str,
    run: dict[str, Any],
    path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    attention = run.get("attention", {})
    alpha = attention.get("alpha", {})
    ratio_match = config.get("run", {}).get("ratio_match", {})
    geometry = run.get("geometry", {})
    geometry_summary = geometry.get("summary", {}) if isinstance(geometry, dict) else {}
    return {
        "label": label,
        "condition": run.get("condition"),
        "path": path.as_posix(),
        "seed": run.get("seed"),
        "alpha": as_float_or_none(alpha.get("initial_value")),
        "generated_alpha": as_float_or_none(ratio_match.get("generated_alpha")),
        "calibration_alpha": as_float_or_none(ratio_match.get("calibration_alpha")),
        "calibration_geo_to_qk_ratio": as_float_or_none(
            ratio_match.get("calibration_geo_to_qk_ratio")
        ),
        "target_geo_to_qk_ratio": as_float_or_none(
            ratio_match.get("target_geo_to_qk_ratio")
        ),
        "geo_to_qk_ratio": as_float_or_none(geometry_summary.get("geo_to_qk_ratio")),
        "final_training_loss": as_float_or_none(run.get("final_training_loss")),
        "best_validation_loss": as_float_or_none(run.get("best_validation_loss")),
        "final_validation_loss": as_float_or_none(run.get("final_validation_loss")),
        "perplexity": as_float_or_none(run.get("perplexity")),
        "average_tokens_per_second": as_float_or_none(run.get("average_tokens_per_second")),
        "geometry_summary": sanitize_for_json(geometry_summary),
    }


def metric_delta(
    candidate: dict[str, Any] | None,
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = ["final_validation_loss", "best_validation_loss", "perplexity"]
    return {
        metric: numeric_delta(
            candidate.get(metric) if candidate else None,
            reference.get(metric) if reference else None,
        )
        for metric in metrics
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


def format_ratio(value: float) -> str:
    return f"{value:g}".replace(".", "_")


if __name__ == "__main__":
    main()
