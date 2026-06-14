"""Stage-23 random/shuffled/no-memory attribution comparison contract."""

from __future__ import annotations

from typing import Any

from experiments.guarded_2000_step_adaptive_run import (
    Guarded2000RunConfig,
    generate_guarded_2000_telemetry_rows,
    summarize_guarded_replay,
)
from experiments.random_shuffled_no_memory_attribution import (
    REQUIRED_ATTRIBUTION_COMPARISON_OUTPUTS,
    AttributionComparisonConfig,
    compare_random_shuffled_no_memory_attribution,
)


def build_random_shuffled_no_memory_attribution_report(
    *,
    telemetry_rows: list[dict[str, Any]] | None = None,
    config: AttributionComparisonConfig | None = None,
) -> dict[str, Any]:
    """Build the stage-23 attribution comparison report."""

    active = config or AttributionComparisonConfig()
    active.validate()
    rows = telemetry_rows or generate_guarded_2000_telemetry_rows(
        Guarded2000RunConfig()
    )
    guarded_summary = summarize_guarded_replay(rows)
    comparison = compare_random_shuffled_no_memory_attribution(rows, config=active)
    random_analysis = comparison["random_advantage_analysis"]
    shuffled_analysis = comparison["shuffled_distribution_bias_analysis"]
    no_memory = comparison["no_memory_comparison"]
    instantaneous = comparison["instantaneous_comparison"]
    relation_specific = comparison["relation_specific_advantage_estimate"]
    attention = comparison["attention_behavior_comparison"]

    checks = {
        "guarded_run_contract_ready": (
            guarded_summary["all_conditions_present"]
            and guarded_summary["all_conditions_have_identical_steps"]
            and guarded_summary["all_conditions_reach_2000"]
        ),
        "required_outputs_present": set(REQUIRED_ATTRIBUTION_COMPARISON_OUTPUTS).issubset(
            comparison
        ),
        "decision_window_is_post1000": comparison["decision_window"] == "1000_2000",
        "random_advantage_analyzed": (
            random_analysis["real_advantage_over_random"] is not None
        ),
        "shuffled_distribution_bias_analyzed": (
            shuffled_analysis["real_advantage_over_shuffled"] is not None
        ),
        "random_does_not_dominate_real": not random_analysis["random_dominates_real"],
        "shuffled_does_not_dominate_real": not shuffled_analysis[
            "shuffled_dominates_real"
        ],
        "no_memory_does_not_match_or_beat_real": not no_memory[
            "control_dominates_real"
        ],
        "instantaneous_does_not_match_or_beat_stable_memory": not instantaneous[
            "control_dominates_real"
        ],
        "relation_specific_advantage_positive": relation_specific[
            "passes_relation_specific_threshold"
        ],
        "attention_behavior_separated_from_controls": attention[
            "attention_separated_from_controls"
        ],
        "generic_regularization_is_labeled": (
            comparison["generic_regularization_present"]
            and random_analysis["interpretation"]
            == "random_regularization_present_but_not_dominant"
            and shuffled_analysis["interpretation"]
            == "distribution_bias_present_but_not_dominant"
        ),
        "revision_trigger_empty_for_pass": not comparison["revision_triggers"],
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "stage": "stage23_random_shuffled_no_memory_attribution_comparison",
        "status": status,
        "scientific_scope": (
            "late-window attribution comparison; separates real geometry from "
            "generic random regularization, shuffled distribution bias, no-memory "
            "ablation, instantaneous geometry, and attention-behavior controls"
        ),
        "input_source": (
            "provided_telemetry_rows"
            if telemetry_rows is not None
            else "guarded_2000_contract_replay"
        ),
        "required_outputs": list(REQUIRED_ATTRIBUTION_COMPARISON_OUTPUTS),
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
        "comparison": comparison,
        "decision_table": _decision_table(comparison),
        "next_required_step": (
            "decision_gate_real_geometry_vs_generic_regularization"
            if status == "pass"
            else "revise_random_shuffled_no_memory_attribution"
        ),
    }


def _decision_table(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = comparison["control_profiles"]
    table = []
    for family in ["random", "shuffled", "no_memory", "instantaneous"]:
        profile = profiles[family]
        table.append(
            {
                "family": family,
                "condition": profile["condition"],
                "gain_vs_baseline": profile["gain_vs_baseline"],
                "gain_share_of_real_gain": profile[
                    "control_gain_share_of_real_gain"
                ],
                "real_advantage_over_control": profile[
                    "real_minus_control_loss_gain"
                ],
                "control_dominates_real": profile["control_dominates_real"],
                "matched_points": profile["matched_points"],
                "positive_delta_points": profile["positive_delta_points"],
            }
        )
    return table
