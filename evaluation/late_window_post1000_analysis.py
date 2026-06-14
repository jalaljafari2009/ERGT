"""Stage-22 late-window and post-1000 analysis contract."""

from __future__ import annotations

from typing import Any

from experiments.guarded_2000_step_adaptive_run import (
    Guarded2000RunConfig,
    generate_guarded_2000_telemetry_rows,
    summarize_guarded_replay,
)
from experiments.late_window_post1000_analysis import (
    LateWindowAnalysisConfig,
    analyze_late_window_post1000,
)

REQUIRED_STAGE22_WINDOWS = [
    "0_500",
    "500_1000",
    "1000_1500",
    "1500_2000",
    "1000_2000",
]

REQUIRED_STAGE22_OUTPUTS = [
    "condition_window_summaries",
    "real_vs_control_window_deltas",
    "attention_window_analysis",
    "post_1000_decision_summary",
    "endpoint_supporting_summary",
]


def build_late_window_post1000_analysis_report(
    *,
    telemetry_rows: list[dict[str, Any]] | None = None,
    config: LateWindowAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Build the stage-22 late-window analysis report."""

    active = config or LateWindowAnalysisConfig()
    active.validate()
    rows = telemetry_rows or generate_guarded_2000_telemetry_rows(
        Guarded2000RunConfig()
    )
    guarded_summary = summarize_guarded_replay(rows)
    analysis = analyze_late_window_post1000(rows, config=active)
    decision = analysis["post_1000_decision_summary"]
    attention = analysis["attention_window_analysis"]
    windows = analysis["condition_window_summaries"]
    decision_deltas = analysis["real_vs_control_window_deltas"][
        active.decision_window
    ]
    checks = {
        "guarded_run_contract_ready": (
            guarded_summary["all_conditions_present"]
            and guarded_summary["all_conditions_have_identical_steps"]
            and guarded_summary["all_conditions_reach_2000"]
        ),
        "required_windows_present": set(REQUIRED_STAGE22_WINDOWS).issubset(windows),
        "all_windows_have_condition_points": all(
            all(summary["points"] >= active.min_window_points for summary in window.values())
            for window in windows.values()
        ),
        "decision_prioritizes_post_1000": decision["uses_post_1000_priority"],
        "decision_window_is_1000_2000": decision["decision_window"] == "1000_2000",
        "real_late_window_trend_available": decision["real_late_slope"] is not None,
        "real_late_window_improving": decision["real_late_trend"] == "improving",
        "real_late_beats_baseline": decision["real_late_beats_baseline"],
        "real_late_beats_all_controls": decision["real_late_beats_all_controls"],
        "late_window_matched_controls_available": (
            decision_deltas["matched_points"] >= active.min_window_points
        ),
        "attention_analyzed_for_all_windows": set(REQUIRED_STAGE22_WINDOWS).issubset(
            attention
        ),
        "late_attention_not_collapsed": not attention[active.decision_window][
            "collapse_warning"
        ],
        "late_attention_no_uniformity_drift": not attention[active.decision_window][
            "uniformity_drift_warning"
        ],
        "late_attention_not_control_like": not attention[active.decision_window][
            "control_like_attention_warning"
        ],
        "endpoint_loss_is_supporting_only": analysis["endpoint_supporting_summary"][
            "endpoint_is_not_decision_source"
        ],
        "late_window_decision_ready": decision["late_window_decision_ready"],
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "stage": "stage22_late_window_post1000_analysis",
        "status": status,
        "scientific_scope": (
            "late-window analysis contract; prioritizes post-1000 windows and "
            "checks attention behavior in the same windows so endpoint loss cannot "
            "hide collapse, uniformity drift, or control-like attention"
        ),
        "input_source": (
            "provided_telemetry_rows"
            if telemetry_rows is not None
            else "guarded_2000_contract_replay"
        ),
        "required_windows": list(REQUIRED_STAGE22_WINDOWS),
        "required_outputs": list(REQUIRED_STAGE22_OUTPUTS),
        "checks": checks,
        "guarded_summary": {
            "condition_row_counts": guarded_summary["condition_row_counts"],
            "all_conditions_have_identical_steps": guarded_summary[
                "all_conditions_have_identical_steps"
            ],
            "all_conditions_reach_2000": guarded_summary[
                "all_conditions_reach_2000"
            ],
        },
        "analysis": analysis,
        "decision_table": _decision_table(analysis, active),
        "next_required_step": (
            "random_shuffled_no_memory_attribution_comparison"
            if status == "pass"
            else "fix_late_window_post1000_analysis"
        ),
    }


def _decision_table(
    analysis: dict[str, Any],
    config: LateWindowAnalysisConfig,
) -> list[dict[str, Any]]:
    table = []
    for window_name in REQUIRED_STAGE22_WINDOWS:
        real_summary = analysis["condition_window_summaries"][window_name][
            config.real_condition
        ]
        deltas = analysis["real_vs_control_window_deltas"][window_name]
        attention = analysis["attention_window_analysis"][window_name]
        table.append(
            {
                "window": window_name,
                "real_points": real_summary["points"],
                "real_mean_validation_loss": real_summary["mean_validation_loss"],
                "real_validation_slope": real_summary["validation_slope"],
                "real_trend": _trend_label(real_summary["validation_slope"]),
                "control_separation": deltas["control_separation"],
                "real_beats_all_controls": deltas["real_beats_all_controls"],
                "attention_safe": attention["attention_safe_for_window_decision"],
                "attention_advantage": attention["real_attention_advantage"],
            }
        )
    return table


def _trend_label(slope: float | None) -> str:
    if slope is None:
        return "insufficient_points"
    if slope < 0:
        return "improving"
    if slope > 0:
        return "worsening"
    return "flat"
