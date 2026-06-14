import json

from evaluation.random_shuffled_no_memory_attribution import (
    build_random_shuffled_no_memory_attribution_report,
)
from experiments.guarded_2000_step_adaptive_run import (
    Guarded2000RunConfig,
    generate_guarded_2000_telemetry_rows,
)
from experiments.random_shuffled_no_memory_attribution import (
    REQUIRED_ATTRIBUTION_COMPARISON_OUTPUTS,
    compare_random_shuffled_no_memory_attribution,
)


def test_random_shuffled_no_memory_attribution_report_passes() -> None:
    report = build_random_shuffled_no_memory_attribution_report()

    assert report["status"] == "pass"
    assert report["next_required_step"] == (
        "decision_gate_real_geometry_vs_generic_regularization"
    )
    json.dumps(report)


def test_required_attribution_outputs_are_present() -> None:
    rows = generate_guarded_2000_telemetry_rows(Guarded2000RunConfig())
    comparison = compare_random_shuffled_no_memory_attribution(rows)

    assert set(REQUIRED_ATTRIBUTION_COMPARISON_OUTPUTS).issubset(comparison)
    assert comparison["decision_window"] == "1000_2000"
    assert comparison["relation_specific_advantage_estimate"]["estimate"] > 0


def test_random_and_shuffled_are_labeled_but_do_not_dominate() -> None:
    report = build_random_shuffled_no_memory_attribution_report()
    comparison = report["comparison"]
    random_analysis = comparison["random_advantage_analysis"]
    shuffled_analysis = comparison["shuffled_distribution_bias_analysis"]

    assert comparison["generic_regularization_present"]
    assert random_analysis["random_has_generic_baseline_advantage"]
    assert not random_analysis["random_dominates_real"]
    assert random_analysis["interpretation"] == (
        "random_regularization_present_but_not_dominant"
    )
    assert shuffled_analysis["shuffled_has_distribution_bias_gain"]
    assert not shuffled_analysis["shuffled_dominates_real"]
    assert shuffled_analysis["interpretation"] == (
        "distribution_bias_present_but_not_dominant"
    )


def test_memory_and_stability_ablation_comparisons_support_real_geometry() -> None:
    report = build_random_shuffled_no_memory_attribution_report()
    comparison = report["comparison"]

    assert comparison["no_memory_comparison"]["memory_specific_gain"] > 0
    assert comparison["no_memory_comparison"]["supports_real_memory_or_stability"]
    assert comparison["instantaneous_comparison"]["stable_memory_gain"] > 0
    assert comparison["instantaneous_comparison"][
        "supports_real_memory_or_stability"
    ]


def test_attention_behavior_is_compared_against_controls() -> None:
    report = build_random_shuffled_no_memory_attribution_report()
    attention = report["comparison"]["attention_behavior_comparison"]

    assert attention["stage22_attention_safe"]
    assert attention["attention_separated_from_controls"]
    assert attention["minimum_attention_advantage"] > 0
    assert attention["real_minus_control_attention_score"]["random"] > 0
    assert attention["real_minus_control_attention_score"]["shuffled"] > 0


def test_random_dominance_blocks_claim_and_enters_revision() -> None:
    rows = generate_guarded_2000_telemetry_rows(Guarded2000RunConfig())
    for row in rows:
        if row["condition"] == "random_memory_d" and row["step"] >= 1000:
            row["validation_loss"] = row["validation_loss"] - 0.30

    report = build_random_shuffled_no_memory_attribution_report(telemetry_rows=rows)

    assert report["status"] == "fail"
    assert report["next_required_step"] == (
        "revise_random_shuffled_no_memory_attribution"
    )
    assert "random_dominates_real" in report["comparison"]["revision_triggers"]
    assert report["comparison"]["attribution_decision"] == "enter_revision"
