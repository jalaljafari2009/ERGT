import json

from evaluation.meta_control_attention_observer import (
    REQUIRED_META_CONTROL_OUTPUTS,
    build_meta_control_attention_observer_report,
)
from experiments.meta_control_attention_observer import (
    MetaControlAttentionObserver,
    observe_meta_control_attention,
)
from experiments.progress_logging import format_progress_line


def test_meta_control_masks_pending_control_signals_during_real_run() -> None:
    observation = observe_meta_control_attention(
        {
            "claim_eligibility": "not_eligible_pending_controls",
            "pending_control_families": ["random", "shuffled"],
            "real_vs_baseline_delta": 0.10,
            "loss_slope_gain": 0.08,
            "geo_to_qk_ratio": 0.06,
            "memory_stability": 0.40,
        }
    )

    assert observation["meta_control_mode"] == "online_partial"
    assert observation["pending_control_mask"]
    assert observation["offline_replay_required"]
    assert observation["meta_signal_status"]["control_separation"] == "masked_pending_control"
    assert observation["meta_attention_weights"]["control_separation"] == 0.0
    assert observation["meta_attention_weights"]["baseline_delta"] > 0
    assert observation["meta_observer_only"]


def test_meta_control_uses_control_separation_after_matched_replay() -> None:
    observation = observe_meta_control_attention(
        {
            "claim_eligibility": "eligible_complete_controls",
            "pending_control_families": [],
            "control_separation": 0.25,
            "real_vs_baseline_delta": 0.18,
            "loss_slope_gain": 0.12,
            "geo_to_qk_ratio": 0.09,
            "memory_stability": 0.50,
        }
    )

    assert observation["meta_control_mode"] == "offline_matched_replay"
    assert not observation["pending_control_mask"]
    assert not observation["offline_replay_required"]
    assert observation["meta_signal_status"]["control_separation"] == "available"
    assert observation["meta_attention_weights"]["control_separation"] > 0


def test_meta_control_parameter_allocation_sums_to_one() -> None:
    observation = observe_meta_control_attention(
        {
            "claim_eligibility": "eligible_complete_controls",
            "pending_control_families": [],
            "control_separation": 0.14,
            "loss_slope_gain": 0.08,
            "real_vs_baseline_delta": 0.11,
            "geo_to_qk_ratio": 0.18,
            "memory_stability": 0.42,
            "noise_risk": 0.20,
            "reach_starvation_score": 0.24,
            "distance_contrast_retention": 0.36,
        }
    )

    assert sum(observation["meta_attention_weights"].values()) == 1.0
    assert abs(sum(observation["meta_parameter_allocation"].values()) - 1.0) < 1e-9
    assert observation["meta_top_signal"] != "none"


def test_meta_control_conflict_reduces_confidence() -> None:
    observer = MetaControlAttentionObserver()
    clean = observer.summary(
        observer.observe(
            {
                "claim_eligibility": "eligible_complete_controls",
                "pending_control_families": [],
                "control_separation": 0.18,
                "loss_slope_gain": 0.10,
                "real_vs_baseline_delta": 0.12,
                "budget_conflict_score": 0.02,
            }
        )
    )
    conflict = observer.summary(
        observer.observe(
            {
                "claim_eligibility": "eligible_complete_controls",
                "pending_control_families": [],
                "control_separation": -0.05,
                "loss_slope_gain": 0.10,
                "real_vs_baseline_delta": 0.12,
                "budget_conflict_score": 0.75,
                "rigidity_risk": 0.80,
            }
        )
    )

    assert conflict["controller_conflict_score"] > clean["controller_conflict_score"]
    assert conflict["meta_control_confidence"] < clean["meta_control_confidence"]


def test_meta_control_attention_report_passes() -> None:
    report = build_meta_control_attention_observer_report()

    assert report["status"] == "pass"
    assert report["next_required_step"] == "open_adaptive_relational_control_trainer"
    assert report["checks"]["partial_run_masks_pending_control_signals"]
    assert report["checks"]["final_replay_uses_control_separation_when_available"]
    assert set(REQUIRED_META_CONTROL_OUTPUTS).issubset(report["partial_live_example"])
    json.dumps(report)


def test_progress_line_includes_meta_control_fields() -> None:
    line = format_progress_line(
        {
            "condition": "real_memory_d",
            "step": 1000,
            "validation_loss": 5.4,
            "meta_control_mode": "online_partial",
            "meta_observer_decision_summary": "observer_only_pending_control_replay_required",
            "meta_top_signal": "baseline_delta",
            "meta_suppressed_signal": "control_separation",
            "meta_control_confidence": 0.32,
            "meta_attention_entropy_normalized": 0.70,
            "controller_conflict_score": 0.10,
            "meta_alpha_weight": 0.25,
            "meta_memory_weight": 0.20,
            "meta_gate_weight": 0.10,
            "meta_reach_weight": 0.15,
            "meta_norm_weight": 0.30,
        }
    )

    assert "meta_mode=online_partial" in line
    assert "meta_top=baseline_delta" in line
    assert "metaConf=0.320" in line
    assert "metaAlpha=0.250" in line
