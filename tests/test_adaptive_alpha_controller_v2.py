import json

from evaluation.adaptive_alpha_controller_v2 import (
    build_adaptive_alpha_controller_v2_report,
)
from experiments.adaptive_alpha_v2 import (
    AdaptiveAlphaControllerV2,
    AdaptiveAlphaV2Config,
    AlphaV2Observation,
)


def test_adaptive_alpha_v2_report_passes_and_records_replay_data() -> None:
    report = build_adaptive_alpha_controller_v2_report()

    assert report["status"] == "pass"
    checks = report["checks"]
    assert checks["required_alpha_outputs_emitted"]
    assert checks["uses_pid_error_terms"]
    assert checks["release_growth_possible"]
    assert checks["restraint_shrink_possible"]
    assert checks["ordinary_risk_does_not_abort"]
    assert checks["control_family_can_restrain_growth"]
    assert checks["replay_records_present"]
    assert report["next_required_step"] == "adaptive_memory_controller"
    json.dumps(report)


def test_adaptive_alpha_v2_grows_when_release_evidence_dominates() -> None:
    controller = AdaptiveAlphaControllerV2(AdaptiveAlphaV2Config(initial_alpha=0.02))
    decision = controller.update(
        AlphaV2Observation(
            step=100,
            loss_slope_gain=0.2,
            ema_loss_delta=0.1,
            late_window_slope=-0.01,
            post_1000_trend="improving",
            rigidity_risk=0.0,
            collapse_risk=0.0,
            control_penalty=0.0,
            random_loss_slope_gain=0.01,
            shuffled_loss_slope_gain=0.01,
            geo_to_qk_ratio=0.05,
        )
    )

    assert decision.alpha_delta > 0
    assert decision.decision in {"grow_pid_release", "probe_up"}
    assert "decision_replay_record" in decision.__dataclass_fields__
    assert decision.parameter_trajectory["alpha_next"] == decision.next_alpha


def test_adaptive_alpha_v2_shrinks_when_controls_dominate() -> None:
    controller = AdaptiveAlphaControllerV2(AdaptiveAlphaV2Config(initial_alpha=0.08))
    decision = controller.update(
        AlphaV2Observation(
            step=100,
            loss_slope_gain=0.01,
            ema_loss_delta=0.0,
            late_window_slope=0.01,
            post_1000_trend="worsening",
            rigidity_risk=0.1,
            collapse_risk=0.1,
            control_penalty=0.2,
            random_loss_slope_gain=0.2,
            shuffled_loss_slope_gain=0.15,
            geo_to_qk_ratio=0.2,
        )
    )

    assert decision.alpha_delta < 0
    assert decision.control_family_evidence["control_dominates_real"] is True
    assert decision.decision in {"shrink_pid_restraint", "probe_down"}


def test_adaptive_alpha_v2_treats_ordinary_risk_as_pressure_not_abort() -> None:
    controller = AdaptiveAlphaControllerV2(AdaptiveAlphaV2Config(initial_alpha=0.08))
    decision = controller.update(
        AlphaV2Observation(
            step=100,
            loss_slope_gain=0.08,
            ema_loss_delta=0.03,
            late_window_slope=-0.01,
            post_1000_trend="improving",
            rigidity_risk=0.5,
            collapse_risk=0.4,
            control_penalty=0.0,
            geo_to_qk_ratio=0.2,
        )
    )

    assert decision.decision != "hard_stop_hold"
    assert decision.restraint_evidence["rigidity_pressure"] > 0
    assert decision.restraint_evidence["collapse_pressure"] > 0


def test_adaptive_alpha_v2_holds_on_explicit_hard_stop() -> None:
    controller = AdaptiveAlphaControllerV2(AdaptiveAlphaV2Config(initial_alpha=0.08))
    decision = controller.update(
        AlphaV2Observation(
            step=100,
            loss_slope_gain=0.2,
            hard_stop_triggered=True,
            hard_stop_reason="nan_or_inf",
        )
    )

    assert decision.decision == "hard_stop_hold"
    assert decision.alpha_delta == 0.0
    assert decision.next_alpha == 0.08
