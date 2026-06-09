import json

from experiments.compare_phase3 import (
    build_ablation_report,
    build_comparison_results,
    sanitize_for_json,
)


def phase3_runs() -> dict:
    return {
        "baseline": {
            "condition": "baseline",
            "final_validation_loss": 1.25,
            "best_validation_loss": 1.2,
            "perplexity": 3.5,
            "average_tokens_per_second": 100,
        },
        "alpha_zero": {
            "condition": "alpha_zero",
            "final_validation_loss": 1.26,
            "best_validation_loss": 1.21,
            "perplexity": 3.55,
            "average_tokens_per_second": 90,
        },
        "real_d": {
            "condition": "real_d",
            "final_validation_loss": 1.15,
            "best_validation_loss": 1.1,
            "perplexity": 3.1,
            "average_tokens_per_second": 80,
            "geometry": {"summary": {"geo_to_qk_ratio": 0.1}},
        },
        "random_d": {
            "condition": "random_d",
            "final_validation_loss": 1.31,
            "best_validation_loss": 1.3,
            "perplexity": 3.7,
            "average_tokens_per_second": 82,
        },
        "shuffled_d": {
            "condition": "shuffled_d",
            "final_validation_loss": 1.29,
            "best_validation_loss": 1.28,
            "perplexity": 3.65,
            "average_tokens_per_second": 81,
        },
    }


def test_compare_phase3_detects_positive_ablation_pattern(tmp_path) -> None:
    runs = phase3_runs()
    paths = {condition: tmp_path / f"{condition}.json" for condition in runs}

    comparison = build_comparison_results(runs, paths)
    report = build_ablation_report(comparison, alpha_zero_tolerance=0.02)

    assert comparison["ranking"]["by_final_validation_loss"][0] == "real_d"
    assert report["checks"]["alpha_zero_matches_baseline"]
    assert report["checks"]["real_d_beats_baseline"]
    assert report["checks"]["real_d_beats_random_d"]
    assert report["checks"]["real_d_beats_shuffled_d"]
    assert report["summary"]["recommendation"] == "gate_ready_positive"


def test_compare_phase3_sanitizes_non_finite_values() -> None:
    payload = {"x": float("inf"), "nested": [float("nan"), 1.0]}

    sanitized = sanitize_for_json(payload)
    encoded = json.dumps(sanitized, allow_nan=False)

    assert sanitized == {"x": None, "nested": [None, 1.0]}
    assert encoded
