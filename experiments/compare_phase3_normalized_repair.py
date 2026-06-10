"""Compare normalized-hidden Phase 3 repair runs."""

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

Family = Literal["real_d", "random_d"]


@dataclass(frozen=True)
class NormalizedRunSpec:
    family: Family
    alpha: float
    path: Path

    @property
    def label(self) -> str:
        return f"{self.family}_alpha_{format_alpha(self.alpha)}_detached_norm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare normalized-hidden repair runs.")
    parser.add_argument(
        "--baseline",
        default="runs/phase0_baseline/confirm_seed_wikitext2/baseline_results.json",
        help="Path to seed-2027 baseline result.",
    )
    parser.add_argument(
        "--prior-detached-real",
        default="runs/phase3_geo_attention/phase3_repair/real_d_alpha_0_1_detached/metrics.json",
        help="Path to best previous detached real_d repair.",
    )
    parser.add_argument(
        "--failed-real-d",
        default="runs/phase3_geo_attention/confirm_seed/real_d_alpha_0_2/metrics.json",
        help="Path to failed grad_d real_d alpha=0.2 run.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Normalized repair run as family:alpha:path, e.g. "
            "real_d:0.1:runs/.../metrics.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="runs/phase3_geo_attention/phase3_repair_normalized",
        help="Directory where phase3_normalized_repair_results.json is saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_json(args.baseline)
    prior_detached = load_required_run(Path(args.prior_detached_real), "real_d")
    failed_real_d = load_required_run(Path(args.failed_real_d), "real_d")
    specs = [parse_run_spec(value) for value in args.run]
    runs = [load_normalized_run(spec) for spec in specs]
    report = build_normalized_repair_report(
        baseline=baseline,
        baseline_path=Path(args.baseline),
        prior_detached=prior_detached,
        prior_detached_path=Path(args.prior_detached_real),
        failed_real_d=failed_real_d,
        failed_real_d_path=Path(args.failed_real_d),
        runs=runs,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "phase3_normalized_repair_results.json",
        sanitize_for_json(report),
    )
    print(json.dumps(sanitize_for_json(report), indent=2, sort_keys=True))


def parse_run_spec(value: str) -> NormalizedRunSpec:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"run spec must be family:alpha:path, got {value!r}")
    family_raw, alpha_raw, path_raw = parts
    if family_raw not in {"real_d", "random_d"}:
        raise ValueError(f"unsupported family: {family_raw!r}")
    return NormalizedRunSpec(
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


def load_normalized_run(spec: NormalizedRunSpec) -> dict[str, Any]:
    payload = load_required_run(spec.path, spec.family)
    attention = payload.get("attention", {})
    alpha = attention.get("alpha", {}).get("initial_value")
    if alpha is None or not math.isclose(float(alpha), spec.alpha, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"expected alpha {spec.alpha} in {spec.path}, found {alpha}")
    if attention.get("gradient_mode") != "detached_d":
        raise ValueError(f"expected detached_d in {spec.path}")

    config_path = spec.path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing run config file: {config_path}")
    config = load_json(config_path)
    if config.get("relational_graph", {}).get("normalize_hidden") is not True:
        raise ValueError(f"expected normalize_hidden=true in {config_path}")

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


def build_normalized_repair_report(
    *,
    baseline: dict[str, Any],
    baseline_path: Path,
    prior_detached: dict[str, Any],
    prior_detached_path: Path,
    failed_real_d: dict[str, Any],
    failed_real_d_path: Path,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one normalized repair run is required")

    references = {
        "baseline": extract_summary("baseline", baseline, baseline_path),
        "prior_real_d_alpha_0_1_detached": extract_summary(
            "prior_real_d_alpha_0_1_detached",
            prior_detached,
            prior_detached_path,
        ),
        "failed_real_d_alpha_0_2_grad": extract_summary(
            "failed_real_d_alpha_0_2_grad",
            failed_real_d,
            failed_real_d_path,
        ),
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
        for family in ("real_d", "random_d")
    }
    best_by_family = {
        family: family_runs[0] if family_runs else None
        for family, family_runs in by_family.items()
    }
    best_real = best_by_family["real_d"]
    best_random = best_by_family["random_d"]
    matched_alpha = build_matched_alpha_deltas(by_family)
    deltas = {
        "best_real_norm_vs_baseline": metric_delta(
            best_real["summary"] if best_real else None,
            references["baseline"],
        ),
        "best_real_norm_vs_prior_detached": metric_delta(
            best_real["summary"] if best_real else None,
            references["prior_real_d_alpha_0_1_detached"],
        ),
        "best_real_norm_vs_failed_grad": metric_delta(
            best_real["summary"] if best_real else None,
            references["failed_real_d_alpha_0_2_grad"],
        ),
        "best_real_norm_vs_best_random_norm": metric_delta(
            best_real["summary"] if best_real else None,
            best_random["summary"] if best_random else None,
        ),
    }
    checks = build_checks(best_real, best_random, deltas, matched_alpha)
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
            "note": (
                "This repair tests whether normalize_hidden=true makes real_d more "
                "stable than the detached repair and normalized random controls."
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
    geometry = run.get("geometry", {})
    geometry_summary = geometry.get("summary", {}) if isinstance(geometry, dict) else {}
    relational_graph = config.get("relational_graph", {}) if config else {}
    return {
        "label": label,
        "condition": run.get("condition"),
        "path": path.as_posix(),
        "seed": run.get("seed"),
        "alpha": as_float_or_none(attention.get("alpha", {}).get("initial_value")),
        "gradient_mode": attention.get("gradient_mode"),
        "distance_mode": attention.get("distance_mode"),
        "normalize_hidden": relational_graph.get("normalize_hidden"),
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
    best_real: dict[str, Any] | None,
    best_random: dict[str, Any] | None,
    deltas: dict[str, dict[str, Any]],
    matched_alpha: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    return {
        "all_losses_finite": all(
            summary is not None and summary.get("final_validation_loss") is not None
            for summary in [
                best_real["summary"] if best_real else None,
                best_random["summary"] if best_random else None,
            ]
        ),
        "best_real_norm_beats_baseline": is_improvement(
            deltas["best_real_norm_vs_baseline"]["final_validation_loss"]
        ),
        "best_real_norm_beats_prior_detached": is_improvement(
            deltas["best_real_norm_vs_prior_detached"]["final_validation_loss"]
        ),
        "best_real_norm_beats_failed_grad": is_improvement(
            deltas["best_real_norm_vs_failed_grad"]["final_validation_loss"]
        ),
        "best_real_norm_beats_best_random_norm": is_improvement(
            deltas["best_real_norm_vs_best_random_norm"]["final_validation_loss"]
        ),
        "real_norm_beats_random_norm_at_any_matched_alpha": any(
            is_improvement(delta["real_d_minus_random_d"])
            for delta in matched_alpha.values()
        ),
    }


def recommendation(checks: dict[str, bool]) -> str:
    if (
        checks["all_losses_finite"]
        and checks["best_real_norm_beats_baseline"]
        and checks["best_real_norm_beats_best_random_norm"]
    ):
        return "normalized_repair_supports_phase3_repeat"
    if (
        checks["all_losses_finite"]
        and checks["best_real_norm_beats_prior_detached"]
        and checks["real_norm_beats_random_norm_at_any_matched_alpha"]
    ):
        return "normalized_repair_partial_support"
    if checks["all_losses_finite"] and checks["best_real_norm_beats_prior_detached"]:
        return "normalized_repair_improves_real_only"
    return "normalized_repair_not_supported"


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
