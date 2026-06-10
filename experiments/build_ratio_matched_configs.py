"""Build Stable Base configs matched by observed geo_to_qk_ratio."""

from __future__ import annotations

import argparse
import copy
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
class CalibrationSpec:
    family: Family
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ERGT Stable Base configs with matched geo_to_qk_ratio."
    )
    parser.add_argument(
        "--target-ratio",
        action="append",
        type=float,
        required=True,
        help="Target geo_to_qk_ratio. Pass multiple times for multiple targets.",
    )
    parser.add_argument(
        "--calibration",
        action="append",
        required=True,
        help=(
            "Calibration run as family:path, e.g. "
            "real_d:runs/phase3_geo_attention/phase3_stable_base/"
            "real_d_alpha_0_1_warmup_cosine/metrics.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="configs/ergt_v1/phase3_ratio_matched",
        help="Directory where generated configs and manifest are saved.",
    )
    parser.add_argument(
        "--run-output-root",
        default="runs/phase3_geo_attention/phase3_ratio_matched",
        help="Root run directory assigned to generated configs.",
    )
    parser.add_argument(
        "--min-alpha",
        type=float,
        default=1e-4,
        help="Minimum allowed generated alpha.",
    )
    parser.add_argument(
        "--max-alpha",
        type=float,
        default=0.5,
        help="Maximum allowed generated alpha.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = [parse_calibration_spec(value) for value in args.calibration]
    manifest = build_ratio_matched_configs(
        target_ratios=[float(value) for value in args.target_ratio],
        calibrations=specs,
        output_dir=Path(args.output_dir),
        run_output_root=Path(args.run_output_root),
        min_alpha=float(args.min_alpha),
        max_alpha=float(args.max_alpha),
    )
    print(json.dumps(sanitize_for_json(manifest), indent=2, sort_keys=True))


def parse_calibration_spec(value: str) -> CalibrationSpec:
    parts = value.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"calibration spec must be family:path, got {value!r}")
    family_raw, path_raw = parts
    if family_raw not in {"real_d", "random_d", "shuffled_d"}:
        raise ValueError(f"unsupported family: {family_raw!r}")
    return CalibrationSpec(family=family_raw, path=Path(path_raw))  # type: ignore[arg-type]


def build_ratio_matched_configs(
    *,
    target_ratios: list[float],
    calibrations: list[CalibrationSpec],
    output_dir: Path,
    run_output_root: Path,
    min_alpha: float,
    max_alpha: float,
) -> dict[str, Any]:
    if not target_ratios:
        raise ValueError("at least one target ratio is required")
    if not calibrations:
        raise ValueError("at least one calibration run is required")
    if min_alpha <= 0 or max_alpha <= 0 or min_alpha > max_alpha:
        raise ValueError("alpha bounds must be positive and ordered")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "targets": {},
        "alpha_bounds": {"min_alpha": min_alpha, "max_alpha": max_alpha},
        "rule": "generated_alpha = calibration_alpha * target_ratio / observed_geo_to_qk_ratio",
    }
    calibration_runs = [load_calibration(spec) for spec in calibrations]

    for target_ratio in target_ratios:
        if target_ratio <= 0:
            raise ValueError("target ratios must be positive")
        target_label = format_ratio(target_ratio)
        target_dir = output_dir / f"target_{target_label}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_records = []

        for calibration in calibration_runs:
            generated = generate_config_for_target(
                calibration=calibration,
                target_ratio=target_ratio,
                target_label=target_label,
                target_dir=target_dir,
                run_output_root=run_output_root,
                min_alpha=min_alpha,
                max_alpha=max_alpha,
            )
            target_records.append(generated)

        manifest["targets"][target_label] = {
            "target_geo_to_qk_ratio": target_ratio,
            "configs": target_records,
        }

    save_json(output_dir / "ratio_matched_manifest.json", sanitize_for_json(manifest))
    return manifest


def load_calibration(spec: CalibrationSpec) -> dict[str, Any]:
    metrics = load_json(spec.path)
    condition = metrics.get("condition")
    if condition != spec.family:
        raise ValueError(f"expected condition {spec.family!r} in {spec.path}, found {condition!r}")
    config_path = spec.path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing calibration config: {config_path}")
    config = load_json(config_path)
    validate_stable_config(config, expected_distance_mode=spec.family)

    observed_ratio = extract_geo_to_qk_ratio(metrics)
    calibration_alpha = extract_alpha(metrics, config)
    if observed_ratio <= 0:
        raise ValueError(f"calibration ratio must be positive in {spec.path}")
    if calibration_alpha <= 0:
        raise ValueError(f"calibration alpha must be positive in {spec.path}")

    return {
        "family": spec.family,
        "metrics_path": spec.path,
        "config_path": config_path,
        "metrics": metrics,
        "config": config,
        "observed_geo_to_qk_ratio": observed_ratio,
        "calibration_alpha": calibration_alpha,
    }


def validate_stable_config(config: dict[str, Any], *, expected_distance_mode: str) -> None:
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
        raise ValueError("ratio-matched configs require gradient_mode=detached_d")
    if relational_graph.get("kernel") != "sigmoid_cosine":
        raise ValueError("ratio-matched configs require sigmoid_cosine")
    if relational_graph.get("normalize_hidden") is not True:
        raise ValueError("ratio-matched configs require normalize_hidden=true")
    if distance.get("normalization") != "offdiag_zscore_clamp":
        raise ValueError("ratio-matched configs require offdiag_zscore_clamp")
    if as_float_or_none(distance.get("clip_value")) is None:
        raise ValueError("ratio-matched configs require finite clip_value")
    if int(alpha.get("warmup_steps", 0)) <= 0:
        raise ValueError("ratio-matched configs require alpha warmup")


def generate_config_for_target(
    *,
    calibration: dict[str, Any],
    target_ratio: float,
    target_label: str,
    target_dir: Path,
    run_output_root: Path,
    min_alpha: float,
    max_alpha: float,
) -> dict[str, Any]:
    family = calibration["family"]
    observed_ratio = calibration["observed_geo_to_qk_ratio"]
    calibration_alpha = calibration["calibration_alpha"]
    scaled_alpha = calibration_alpha * target_ratio / observed_ratio
    generated_alpha = min(max(scaled_alpha, min_alpha), max_alpha)
    if not math.isclose(generated_alpha, scaled_alpha, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"generated alpha {scaled_alpha} for {family} target {target_ratio} "
            f"is outside bounds [{min_alpha}, {max_alpha}]"
        )

    config = copy.deepcopy(calibration["config"])
    seed = int(config.get("run", {}).get("seed", calibration["metrics"].get("seed", 0)))
    alpha_label = format_alpha(generated_alpha)
    run_id = f"{family}_target_{target_label}_alpha_{alpha_label}"
    config["run"]["output_dir"] = (run_output_root / f"target_{target_label}" / run_id).as_posix()
    config["run"]["ratio_match"] = {
        "target_geo_to_qk_ratio": target_ratio,
        "calibration_metrics": calibration["metrics_path"].as_posix(),
        "calibration_config": calibration["config_path"].as_posix(),
        "calibration_alpha": calibration_alpha,
        "calibration_geo_to_qk_ratio": observed_ratio,
        "generated_alpha": generated_alpha,
        "scaling_rule": "calibration_alpha * target_ratio / observed_geo_to_qk_ratio",
    }
    config["attention"]["alpha"]["initial_value"] = generated_alpha

    config_path = target_dir / f"{run_id}_seed{seed}.json"
    save_json(config_path, sanitize_for_json(config))
    return {
        "family": family,
        "target_geo_to_qk_ratio": target_ratio,
        "calibration_alpha": calibration_alpha,
        "calibration_geo_to_qk_ratio": observed_ratio,
        "generated_alpha": generated_alpha,
        "config_path": config_path.as_posix(),
        "expected_metrics_path": (Path(config["run"]["output_dir"]) / "metrics.json").as_posix(),
    }


def extract_geo_to_qk_ratio(metrics: dict[str, Any]) -> float:
    geometry = metrics.get("geometry", {})
    summary = geometry.get("summary", {}) if isinstance(geometry, dict) else {}
    ratio = as_float_or_none(summary.get("geo_to_qk_ratio"))
    if ratio is None:
        raise ValueError("metrics missing geometry.summary.geo_to_qk_ratio")
    return ratio


def extract_alpha(metrics: dict[str, Any], config: dict[str, Any]) -> float:
    alpha = metrics.get("attention", {}).get("alpha", {}).get("initial_value")
    if alpha is None:
        alpha = config.get("attention", {}).get("alpha", {}).get("initial_value")
    alpha_value = as_float_or_none(alpha)
    if alpha_value is None:
        raise ValueError("run missing alpha initial_value")
    return alpha_value


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
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def format_ratio(value: float) -> str:
    return f"{value:g}".replace(".", "_")


def format_alpha(value: float) -> str:
    return f"{value:.6g}".replace(".", "_")


if __name__ == "__main__":
    main()
