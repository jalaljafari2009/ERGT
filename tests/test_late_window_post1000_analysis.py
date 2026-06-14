import json

from evaluation.late_window_post1000_analysis import (
    REQUIRED_STAGE22_WINDOWS,
    build_late_window_post1000_analysis_report,
)
from experiments.guarded_2000_step_adaptive_run import (
    Guarded2000RunConfig,
    generate_guarded_2000_telemetry_rows,
)
from experiments.late_window_post1000_analysis import analyze_late_window_post1000


def test_late_window_post1000_analysis_report_passes() -> None:
    report = build_late_window_post1000_analysis_report()

    assert report["status"] == "pass"
    assert report["next_required_step"] == (
        "random_shuffled_no_memory_attribution_comparison"
    )
    json.dumps(report)


def test_late_window_analysis_contains_required_windows() -> None:
    rows = generate_guarded_2000_telemetry_rows(Guarded2000RunConfig())
    analysis = analyze_late_window_post1000(rows)

    assert set(REQUIRED_STAGE22_WINDOWS).issubset(
        analysis["condition_window_summaries"]
    )
    for window in REQUIRED_STAGE22_WINDOWS:
        assert analysis["real_vs_control_window_deltas"][window]["matched_points"] >= 2


def test_late_window_decision_prioritizes_post_1000_not_endpoint() -> None:
    report = build_late_window_post1000_analysis_report()
    decision = report["analysis"]["post_1000_decision_summary"]

    assert decision["decision_window"] == "1000_2000"
    assert decision["uses_post_1000_priority"]
    assert decision["endpoint_loss_is_supporting_only"]
    assert decision["late_window_decision_ready"]


def test_late_window_attention_checks_prevent_hidden_collapse() -> None:
    report = build_late_window_post1000_analysis_report()
    attention = report["analysis"]["attention_window_analysis"]["1000_2000"]

    assert attention["attention_safe_for_window_decision"]
    assert not attention["collapse_warning"]
    assert not attention["uniformity_drift_warning"]
    assert not attention["control_like_attention_warning"]


def test_late_window_real_beats_controls_in_decision_window() -> None:
    report = build_late_window_post1000_analysis_report()
    decision = report["analysis"]["post_1000_decision_summary"]
    deltas = report["analysis"]["real_vs_control_window_deltas"]["1000_2000"]

    assert decision["real_late_beats_baseline"]
    assert decision["real_late_beats_all_controls"]
    assert deltas["control_separation"] > 0
    assert deltas["deltas"]["random_memory_d"]["mean_delta"] > 0
    assert deltas["deltas"]["shuffled_memory_d"]["mean_delta"] > 0
