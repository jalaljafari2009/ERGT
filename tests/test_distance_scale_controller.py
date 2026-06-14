import json

from evaluation.distance_scale_controller import (
    REQUIRED_DISTANCE_SCALE_OUTPUTS,
    build_distance_scale_controller_report,
)
from experiments.distance_scale_controller import (
    DistanceScaleConfig,
    DistanceScaleController,
    DistanceScaleObservation,
)
from experiments.progress_logging import format_progress_line


def test_distance_scale_controller_increases_when_real_contrast_is_erased() -> None:
    controller = DistanceScaleController()

    decision = controller.update(
        DistanceScaleObservation(
            step=100,
            pre_norm_distance_contrast=0.60,
            post_norm_distance_contrast=0.12,
            distance_std_pre_norm=1.10,
            distance_std_post_norm=0.35,
            clipping_saturation_rate=0.03,
            geo_to_qk_ratio=0.015,
            attention_entropy_normalized=0.58,
            collapse_risk=0.05,
            real_distance_advantage=0.22,
            random_distance_advantage=0.03,
            shuffled_distance_advantage=0.02,
        )
    )

    assert decision.decision == "increase_distance_scale"
    assert decision.distance_norm_scale_delta > 0
    assert decision.contrast_evidence["contrast_erased_by_normalization"] is True
    assert decision.release_evidence["contrast_erasure_pressure"] > 0
    assert decision.parameter_trajectory["distance_norm_scale_next"] == (
        decision.current_distance_norm_scale
    )
    assert decision.decision_replay_record["decision"] == "increase_distance_scale"


def test_distance_scale_controller_decreases_when_noisy_or_clipped() -> None:
    controller = DistanceScaleController(
        DistanceScaleConfig(initial_distance_norm_scale=1.5)
    )

    decision = controller.update(
        DistanceScaleObservation(
            step=200,
            pre_norm_distance_contrast=0.08,
            post_norm_distance_contrast=0.04,
            distance_std_pre_norm=0.16,
            distance_std_post_norm=2.20,
            clipping_saturation_rate=0.46,
            geo_to_qk_ratio=0.26,
            attention_entropy_normalized=0.95,
            collapse_risk=0.55,
            real_distance_advantage=0.02,
            random_distance_advantage=0.12,
            shuffled_distance_advantage=0.10,
        )
    )

    assert decision.decision == "decrease_distance_scale"
    assert decision.distance_norm_scale_delta < 0
    assert decision.clipping_evidence["clipping_high"] is True
    assert decision.control_distance_evidence["control_dominates_real"] is True
    assert decision.restraint_evidence["control_dominance_pressure"] > 0


def test_distance_scale_controller_uses_future_leak_as_hard_stop() -> None:
    controller = DistanceScaleController()

    decision = controller.update(
        DistanceScaleObservation(
            step=300,
            pre_norm_distance_contrast=0.60,
            post_norm_distance_contrast=0.12,
            distance_std_pre_norm=1.10,
            distance_std_post_norm=0.35,
            clipping_saturation_rate=0.03,
            geo_to_qk_ratio=0.015,
            attention_entropy_normalized=0.58,
            collapse_risk=0.05,
            real_distance_advantage=0.22,
            future_leak_score=0.01,
        )
    )

    assert decision.decision == "hard_stop_hold"
    assert decision.distance_norm_scale_delta == 0
    assert decision.attention_safety_evidence["hard_stop_triggered"] is True


def test_distance_scale_controller_respects_bounds() -> None:
    controller = DistanceScaleController(
        DistanceScaleConfig(
            initial_distance_norm_scale=4.0,
            max_distance_norm_scale=4.0,
        )
    )

    decision = controller.update(
        DistanceScaleObservation(
            step=400,
            pre_norm_distance_contrast=0.60,
            post_norm_distance_contrast=0.12,
            distance_std_pre_norm=1.10,
            distance_std_post_norm=0.35,
            clipping_saturation_rate=0.03,
            geo_to_qk_ratio=0.015,
            attention_entropy_normalized=0.58,
            collapse_risk=0.05,
            real_distance_advantage=0.22,
        )
    )

    assert decision.current_distance_norm_scale <= controller.config.max_distance_norm_scale
    assert decision.distance_norm_scale_delta == 0


def test_distance_scale_controller_report_passes() -> None:
    report = build_distance_scale_controller_report()

    assert report["status"] == "pass"
    assert report["next_required_step"] == "joint_parameter_budget_allocator"
    assert report["checks"]["required_distance_scale_outputs_emitted"]
    assert report["checks"]["schema_declares_distance_scale_fields"]
    assert report["checks"]["can_increase_when_real_contrast_is_erased"]
    assert report["checks"]["can_decrease_when_scale_is_noisy_or_clipped"]
    assert set(REQUIRED_DISTANCE_SCALE_OUTPUTS).issubset(report["decisions"][-1])
    json.dumps(report)


def test_progress_line_includes_distance_scale_fields() -> None:
    line = format_progress_line(
        {
            "condition": "real_memory_d_adaptive",
            "step": 100,
            "validation_loss": 5.0,
            "distance_norm_scale": 1.0,
            "distance_norm_scale_next": 1.1,
            "distance_norm_scale_delta": 0.1,
            "distance_norm_scale_decision": "increase_distance_scale",
            "distance_norm_scale_credit": 0.50,
            "distance_norm_scale_risk_pressure": 0.08,
            "pre_norm_distance_contrast": 0.60,
            "post_norm_distance_contrast": 0.12,
            "distance_contrast_retention": 0.20,
            "clipping_saturation_rate": 0.03,
            "normalization_erasure_score": 0.53,
        }
    )

    assert "norm_decision=increase_distance_scale" in line
    assert "norm=1.000" in line
    assert "n_next=1.100" in line
    assert "d_norm=0.100" in line
    assert "preC=0.600" in line
    assert "postC=0.120" in line
    assert "ret=0.200" in line
    assert "clipSat=0.030" in line
