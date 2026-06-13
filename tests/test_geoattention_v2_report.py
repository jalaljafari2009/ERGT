import json

from evaluation.geoattention_v2_report import build_geoattention_v2_report


def test_geoattention_v2_report_passes_on_synthetic_memory_field() -> None:
    report = build_geoattention_v2_report(seed=2027)

    assert report["status"] == "pass"
    assert report["input_source"] == "synthetic_memory_smoke"
    assert report["checks"]["alpha_zero_matches_baseline"]
    assert report["checks"]["future_attention_forbidden"]
    assert report["checks"]["real_uses_memory_after_first_layer"]
    assert report["strict_gate_status"] == "needs_training_run"
    assert "real_stable_beats_random" in report["strict_gate_checks"]
    assert report["next_required_step"] == "run_geoattention_v2_training_controls"
    json.dumps(report)
