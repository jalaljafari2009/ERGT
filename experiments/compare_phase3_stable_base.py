"""Compare Phase 3 Stable Base runs."""

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
class StableRunSpec:
    family: Family
    alpha: float
    path: Path

    @property
    def label(self) -> str:
        return f"{self.family}_alpha_{format_alpha(self.alpha)}_warmup_cosine"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Phase 3 Stable Base runs.")
    parser.add_argument(
        "--baseline",
        default="runs/phase0_baseline/phase3_stable_base_seed2027/baseline_results.json",
        help="Path to the matched Stable Base baseline result.",
    )
    parser.add_argument(
        "--alpha-zero",
        default="runs/phase3_geo_attention/phase3_stable_base/alpha_zero_cosine/metrics.json",
        help="Path to alpha_zero Stable Base result.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Stable run in the form family:alpha:path, e.g. "
            "real_d:0.1:runs/.../metrics.json"
        ),
    )
    parser.add_argument(
        "--alpha-zero-tolerance",
        type=float,
        default=0.02,
        help="Allowed relative final-validation delta for alpha_zero vs baseline.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/phase3_geo_attention/phase3_stable_base",
        help="Directory where phase3_stable_base_results.json is saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_json(args.baseline)
    alpha_zero = load_alpha_zero_run(Path(args.alpha_zero))
    runs = [load_stable_run(parse_run_spec(value)) for value in args.run]
    report = build_stable_base_report(
        baseline=baseline,
        baseline_path=Path(args.baseline),
        alpha_zero=alpha_zero,
        alpha_zero_path=Path(args.alpha_zero),
        runs=runs,
        alpha_zero_tolerance=float(args.alpha_zero_tolerance),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "phase3_stable_base_results.json", sanitize_for_json(report))
    print(json.dumps(sanitize_for_json(report), indent=2, sort_keys=True))


def parse_run_spec(value: str) -> StableRunSpec:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"run spec must be family:alpha:path, got {value!r}")
    family_raw, alpha_raw, path_raw = parts
    if family_raw not in {"real_d", "random_d", "shuffled_d"}:
        raise ValueError(f"unsupported family: {family_raw!r}")
    return StableRunSpec(
        family=family_raw,  # type: ignore[arg-type]
        alpha=float(alpha_raw),
        path=Path(path_raw),
    )


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


def load_alpha_zero_run(path: Path) -> dict[str, Any]:
    payload = load_required_run(path, "alpha_zero")
    config = load_run_config(path)
    attention = payload.get("attention", {})
    alpha = attention.get("alpha", {}).get("initial_value")
    if alpha is None or not math.isclose(float(alpha), 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"expected alpha_zero initial_value=0.0 in {path}, found {alpha}")
    validate_stable_config(config, expected_distance_mode="real_d", allow_zero_alpha=True)
    return {
        "label": "alpha_zero_cosine",
        "path": path.as_posix(),
        "metrics": payload,
        "config": config,
        "summary": extract_summary("alpha_zero_cosine", payload, path, config),
    }


def load_stable_run(spec: StableRunSpec) -> dict[str, Any]:
    payload = load_required_run(spec.path, spec.family)
    config = load_run_config(spec.path)
    attention = payload.get("attention", {})
    alpha = attention.get("alpha", {}).get("initial_value")
    if alpha is None or not math.isclose(float(alpha), spec.alpha, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"expected alpha {spec.alpha} in {spec.path}, found {alpha}")
    validate_stable_config(config, expected_distance_mode=spec.family)
    summary = extract_summary(spec.label, payload, spec.path, config)
    summary["family"] = spec.family
    return {
        "label": spec.label,
        "family": spec.family,
        "alpha": spec.alpha,
        "path": spec.path.as_posix(),
        "metrics": payload,
        "config": config,
        "summary": summary,
    }


def load_run_config(metrics_path: Path) -> dict[str, Any]:
    config_path = metrics_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing run config file: {config_path}")
    return load_json(config_path)


def validate_stable_config(
    config: dict[str, Any],
    *,
    expected_distance_mode: str,
    allow_zero_alpha: bool = False,
) -> None:
    attention = config.get("attention", {})
    relational_graph = config.get("relational_graph", {})
    distance = config.get("distance", {})
    alpha = attention.get("alpha", {})

    if attention.get("distance_mode") != expected_distance_mode:
        raise ValueError(
            f"expected distance_mode={expected_distance_mode!r}, "
            f"found {attention.get('distance_mode')!r}"
        )
    if attention.get("gradient_mode") != "detached_d":
        raise ValueError("Stable Base requires gradient_mode=detached_d")
    if relational_graph.get("kernel") != "sigmoid_cosine":
        raise ValueError("Stable Base requires relational_graph.kernel=sigmoid_cosine")
    if relational_graph.get("normalize_hidden") is not True:
        raise ValueError("Stable Base requires normalize_hidden=true")
    if distance.get("normalization") != "offdiag_zscore_clamp":
        raise ValueError("Stable Base requires offdiag_zscore_clamp distance normalization")
    if as_float_or_none(distance.get("clip_value")) is None:
        raise ValueError("Stable Base requires finite distance.clip_value")
    warmup_steps = int(alpha.get("warmup_steps", 0))
    target_alpha = as_float_or_none(alpha.get("initial_value"))
    if not allow_zero_alpha:
        if warmup_steps <= 0:
            raise ValueError("Stable Base requires alpha.warmup_steps > 0")
        if target_alpha is None or target_alpha <= 0:
            raise ValueError("Stable Base requires positive target alpha")


def build_stable_base_report(
    *,
    baseline: dict[str, Any],
    baseline_path: Path,
    alpha_zero: dict[str, Any],
    alpha_zero_path: Path,
    runs: list[dict[str, Any]],
    alpha_zero_tolerance: float,
) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one Stable Base run is required")

    references = {
        "baseline": extract_summary("baseline", baseline, baseline_path),
        "alpha_zero_cosine": alpha_zero["summary"],
    }
    ranked = sorted(
        runs,
        key=lambda run: (
            run["summary"]["final_validation_loss"] is None,
            run["summary"]["final_validation_loss"] or math.inf,
        ),
    )
    by_family = {
        family: [run for run in ranked if run["family"] == family]
        for family in ("real_d", "random_d", "shuffled_d")
    }
    best_by_family = {
        family: family_runs[0] if family_runs else None
        for family, family_runs in by_family.items()
    }
    best_real = best_by_family["real_d"]
    best_random = best_by_family["random_d"]
    best_shuffled = best_by_family["shuffled_d"]
    matched_alpha = build_matched_alpha_deltas(by_family)
    deltas = {
        "alpha_zero_vs_baseline": metric_delta(
            references["alpha_zero_cosine"],
            references["baseline"],
        ),
        "best_real_vs_baseline": metric_delta(
            best_real["summary"] if best_real else None,
            references["baseline"],
        ),
        "best_real_vs_alpha_zero": metric_delta(
            best_real["summary"] if best_real else None,
            references["alpha_zero_cosine"],
        ),
        "best_real_vs_best_random": metric_delta(
            best_real["summary"] if best_real else None,
            best_random["summary"] if best_random else None,
        ),
        "best_real_vs_best_shuffled": metric_delta(
            best_real["summary"] if best_real else None,
            best_shuffled["summary"] if best_shuffled else None,
        ),
    }
    checks = build_checks(
        runs=runs,
        best_real=best_real,
        best_random=best_random,
        best_shuffled=best_shuffled,
        deltas=deltas,
        matched_alpha=matched_alpha,
        alpha_zero_tolerance=alpha_zero_tolerance,
    )
    summaries = {run["label"]: run["summary"] for run in ranked}
    all_for_ranking = {**references, **summaries}

    return {
        "references": references,
        "runs": [run["summary"] for run in ranked],
        "ranking": {
            "by_final_validation_loss": rank_labels(all_for_ranking, "final_validation_loss")
        },
        "best_by_family": {
            family: run["summary"] if run else None for family, run in best_by_family.items()
        },
        "matched_alpha_deltas": matched_alpha,
        "deltas": deltas,
        "checks": checks,
        "summary": {
            "recommendation": recommendation(checks),
            "alpha_zero_tolerance": alpha_zero_tolerance,
            "note": (
                "Stable Base treats raw ERGT as diagnostic and tests detached, "
                "cosine, clipped, alpha-warmup geometry against random/shuffled controls."
            ),
        },
    }


def extract_summary(
    label: str,
    run: dict[str, Any],
    path: Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attention = run.get("attention", {})
    alpha = attention.get("alpha", {})
    relational_graph = config.get("relational_graph", {}) if config else {}
    distance = config.get("distance", {}) if config else {}
    geometry = run.get("geometry", {})
    geometry_summary = geometry.get("summary", {}) if isinstance(geometry, dict) else {}
    return {
        "label": label,
        "condition": run.get("condition"),
        "path": path.as_posix(),
        "seed": run.get("seed"),
        "alpha": as_float_or_none(alpha.get("initial_value")),
        "alpha_warmup_steps": as_float_or_none(alpha.get("warmup_steps")),
        "gradient_mode": attention.get("gradient_mode"),
        "distance_mode": attention.get("distance_mode"),
        "kernel": relational_graph.get("kernel"),
        "normalize_hidden": relational_graph.get("normalize_hidden"),
        "distance_normalization": distance.get("normalization"),
        "distance_clip_value": as_float_or_none(distance.get("clip_value")),
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


def build_matched_alpha_deltas(
    by_family: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    real_by_alpha = {run["alpha"]: run for run in by_family["real_d"]}
    random_by_alpha = {run["alpha"]: run for run in by_family["random_d"]}
    shared_alphas = sorted(set(real_by_alpha) & set(random_by_alpha))

    deltas: dict[str, dict[str, Any]] = {}
    for alpha in shared_alphas:
        real = real_by_alpha[alpha]["summary"]
        random = random_by_alpha[alpha]["summary"]
        deltas[format_alpha(alpha)] = {
            "real_d_final_validation_loss": real["final_validation_loss"],
            "random_d_final_validation_loss": random["final_validation_loss"],
            "real_d_minus_random_d": numeric_delta(
                real["final_validation_loss"],
                random["final_validation_loss"],
            ),
        }
    return deltas


def build_checks(
    *,
    runs: list[dict[str, Any]],
    best_real: dict[str, Any] | None,
    best_random: dict[str, Any] | None,
    best_shuffled: dict[str, Any] | None,
    deltas: dict[str, dict[str, Any]],
    matched_alpha: dict[str, dict[str, Any]],
    alpha_zero_tolerance: float,
) -> dict[str, bool]:
    summaries = [run["summary"] for run in runs]
    required = [
        best_real["summary"] if best_real else None,
        best_random["summary"] if best_random else None,
        best_shuffled["summary"] if best_shuffled else None,
    ]
    return {
        "all_losses_finite": all(
            summary.get("final_validation_loss") is not None for summary in summaries
        ),
        "has_real_random_and_shuffled": all(summary is not None for summary in required),
        "alpha_zero_matches_baseline": delta_within_relative_tolerance(
            deltas["alpha_zero_vs_baseline"]["final_validation_loss"],
            alpha_zero_tolerance,
        ),
        "best_real_beats_baseline": is_improvement(
            deltas["best_real_vs_baseline"]["final_validation_loss"]
        ),
        "best_real_beats_alpha_zero": is_improvement(
            deltas["best_real_vs_alpha_zero"]["final_validation_loss"]
        ),
        "best_real_beats_best_random": is_improvement(
            deltas["best_real_vs_best_random"]["final_validation_loss"]
        ),
        "best_real_beats_best_shuffled": is_improvement(
            deltas["best_real_vs_best_shuffled"]["final_validation_loss"]
        ),
        "real_beats_random_at_any_matched_alpha": any(
            is_improvement(delta["real_d_minus_random_d"])
            for delta in matched_alpha.values()
        ),
    }


def recommendation(checks: dict[str, bool]) -> str:
    if (
        checks["all_losses_finite"]
        and checks["has_real_random_and_shuffled"]
        and checks["alpha_zero_matches_baseline"]
        and checks["best_real_beats_baseline"]
        and checks["best_real_beats_best_random"]
        and checks["best_real_beats_best_shuffled"]
    ):
        return "stable_candidate_found_repeat_seeds"
    if (
        checks["all_losses_finite"]
        and checks["best_real_beats_baseline"]
        and checks["real_beats_random_at_any_matched_alpha"]
    ):
        return "stable_signal_partial_repeat_or_extend"
    if checks["all_losses_finite"] and checks["best_real_beats_baseline"]:
        return "control_confounded_signal"
    if checks["all_losses_finite"]:
        return "stable_base_needs_redesign"
    return "unstable_or_incomplete"


def metric_delta(
    candidate: dict[str, Any] | None,
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = [
        "final_validation_loss",
        "best_validation_loss",
        "perplexity",
        "average_tokens_per_second",
        "peak_memory_bytes",
    ]
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


def delta_within_relative_tolerance(delta: dict[str, float | None], tolerance: float) -> bool:
    absolute = delta.get("absolute")
    reference = delta.get("reference")
    if absolute is None or reference is None:
        return False
    if reference == 0:
        return abs(absolute) <= tolerance
    return abs(absolute) / abs(reference) <= tolerance


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


def format_alpha(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "_")


if __name__ == "__main__":
    main()
