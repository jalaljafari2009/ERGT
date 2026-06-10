"""Compare ERGT alpha sweep runs for real_d and random_d controls."""

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
class SweepRunSpec:
    family: Family
    alpha: float
    path: Path

    @property
    def label(self) -> str:
        return f"{self.family}_alpha_{format_alpha(self.alpha)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ERGT alpha sweep runs.")
    parser.add_argument(
        "--baseline",
        default="runs/phase0_baseline/short_proof_wikitext2/baseline_results.json",
        help="Path to the baseline result used as the loss reference.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Sweep run in the form family:alpha:path, e.g. "
            "real_d:0.1:runs/phase3_geo_attention/short_proof_real_d/metrics.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="runs/phase3_geo_attention/alpha_sweep_short_proof",
        help="Directory where alpha_sweep_results.json is saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_json(args.baseline)
    specs = [parse_run_spec(value) for value in args.run]
    runs = [load_sweep_run(spec) for spec in specs]
    report = build_alpha_sweep_report(baseline, Path(args.baseline), runs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "alpha_sweep_results.json", sanitize_for_json(report))
    print(json.dumps(sanitize_for_json(report), indent=2, sort_keys=True))


def parse_run_spec(value: str) -> SweepRunSpec:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"run spec must be family:alpha:path, got {value!r}")
    family_raw, alpha_raw, path_raw = parts
    if family_raw not in {"real_d", "random_d"}:
        raise ValueError(f"unsupported sweep family: {family_raw!r}")
    return SweepRunSpec(
        family=family_raw,  # type: ignore[arg-type]
        alpha=float(alpha_raw),
        path=Path(path_raw),
    )


def load_sweep_run(spec: SweepRunSpec) -> dict[str, Any]:
    if not spec.path.exists():
        raise FileNotFoundError(f"missing sweep metrics file: {spec.path}")
    payload = load_json(spec.path)
    condition = payload.get("condition")
    if condition != spec.family:
        raise ValueError(f"expected condition {spec.family!r} in {spec.path}, found {condition!r}")

    attention = payload.get("attention", {})
    alpha = attention.get("alpha", {}).get("initial_value")
    if alpha is not None and not math.isclose(float(alpha), spec.alpha, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"expected alpha {spec.alpha} in {spec.path}, found {alpha}")

    summary = extract_summary(payload, spec.path)
    summary["label"] = spec.label
    summary["family"] = spec.family
    summary["alpha"] = spec.alpha

    return {
        "label": spec.label,
        "family": spec.family,
        "alpha": spec.alpha,
        "path": str(spec.path),
        "metrics": payload,
        "summary": summary,
    }


def build_alpha_sweep_report(
    baseline: dict[str, Any],
    baseline_path: Path,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one sweep run is required")

    baseline_summary = extract_summary(baseline, baseline_path)
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
    matched_alpha = build_matched_alpha_deltas(by_family)
    checks = build_checks(baseline_summary, best_by_family, matched_alpha)

    return {
        "baseline": baseline_summary,
        "runs": [run["summary"] for run in ranked],
        "ranking": [run["label"] for run in ranked],
        "best_by_family": {
            family: run["summary"] if run else None for family, run in best_by_family.items()
        },
        "matched_alpha_deltas": matched_alpha,
        "checks": checks,
        "summary": {
            "recommendation": recommendation(checks),
            "note": (
                "This sweep is diagnostic. It tests alpha sensitivity after short proof "
                "random_d beat real_d at alpha=0.1."
            ),
        },
    }


def extract_summary(run: dict[str, Any], path: Path) -> dict[str, Any]:
    attention = run.get("attention", {})
    alpha = attention.get("alpha", {}).get("initial_value")
    geometry = run.get("geometry", {})
    geometry_summary = geometry.get("summary", {}) if isinstance(geometry, dict) else {}
    return {
        "label": None,
        "condition": run.get("condition"),
        "path": str(path),
        "alpha": as_float_or_none(alpha),
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
    baseline: dict[str, Any],
    best_by_family: dict[str, dict[str, Any] | None],
    matched_alpha: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    best_real = best_by_family["real_d"]
    best_random = best_by_family["random_d"]
    return {
        "best_real_d_beats_baseline": is_improvement(
            numeric_delta(
                best_real["summary"]["final_validation_loss"] if best_real else None,
                baseline["final_validation_loss"],
            )
        ),
        "best_real_d_beats_best_random_d": is_improvement(
            numeric_delta(
                best_real["summary"]["final_validation_loss"] if best_real else None,
                best_random["summary"]["final_validation_loss"] if best_random else None,
            )
        ),
        "real_d_beats_random_d_at_any_matched_alpha": any(
            is_improvement(delta["real_d_minus_random_d"])
            for delta in matched_alpha.values()
        ),
        "all_losses_finite": all(
            item is not None
            for item in [
                baseline["final_validation_loss"],
                best_real["summary"]["final_validation_loss"] if best_real else None,
                best_random["summary"]["final_validation_loss"] if best_random else None,
            ]
        ),
    }


def recommendation(checks: dict[str, bool]) -> str:
    if (
        checks["all_losses_finite"]
        and checks["best_real_d_beats_baseline"]
        and checks["best_real_d_beats_best_random_d"]
    ):
        return "alpha_sweep_supports_real_d"
    if checks["best_real_d_beats_baseline"] and checks[
        "real_d_beats_random_d_at_any_matched_alpha"
    ]:
        return "real_d_alpha_sensitive_repeat"
    if checks["best_real_d_beats_baseline"]:
        return "random_control_still_stronger"
    return "redesign_or_lower_alpha"


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


def format_alpha(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "_")


if __name__ == "__main__":
    main()
