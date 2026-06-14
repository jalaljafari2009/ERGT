import json

from evaluation.gate_floor_noise_controller import (
    REQUIRED_GATE_FLOOR_OUTPUTS,
    build_gate_floor_noise_controller_report,
)
from experiments.gate_floor_noise_controller import (
    GateFloorNoiseConfig,
    GateFloorNoiseController,
    GateFloorObservation,
)
from experiments.progress_logging import format_progress_line


def test_gate_floor_controller_raises_gate_when_edges_are_noisy() -> None:
    controller = GateFloorNoiseController()

    decision = controller.update(
        GateFloorObservation(
            step=100,
            memory_edge_density=0.45,
            edge_survival=0.60,
            random_edge_noise_score=0.75,
            shuffled_edge_noise_score=0.65,
            real_edge_starvation_score=0.05,
            attention_sparsity_0_01=0.40,
            attention_entropy_normalized=0.80,
            control_attention_separation=-0.10,
            collapse_risk=0.05,
        )
    )

    assert decision.decision == "raise_gate_floor"
    assert decision.gate_floor_delta > 0
    assert decision.edge_noise_evidence["noise_high"] is True
    assert decision.parameter_trajectory["gate_floor_next"] == decision.current_gate_floor
    assert decision.decision_replay_record["decision"] == "raise_gate_floor"


def test_gate_floor_controller_lowers_gate_when_real_edges_are_starved() -> None:
    controller = GateFloorNoiseController(GateFloorNoiseConfig(initial_gate_floor=0.08))

    decision = controller.update(
        GateFloorObservation(
            step=200,
            memory_edge_density=0.04,
            edge_survival=0.15,
            random_edge_noise_score=0.10,
            shuffled_edge_noise_score=0.12,
            real_edge_starvation_score=0.80,
            attention_sparsity_0_01=0.95,
            attention_entropy_normalized=0.45,
            control_attention_separation=0.20,
            collapse_risk=0.05,
        )
    )

    assert decision.decision == "lower_gate_floor"
    assert decision.gate_floor_delta < 0
    assert decision.starvation_evidence["starvation_high"] is True
    assert decision.release_evidence["real_edge_starvation_pressure"] > 0


def test_gate_floor_controller_uses_future_leak_as_hard_stop() -> None:
    controller = GateFloorNoiseController()

    decision = controller.update(
        GateFloorObservation(
            step=300,
            memory_edge_density=0.45,
            edge_survival=0.60,
            random_edge_noise_score=0.75,
            shuffled_edge_noise_score=0.65,
            real_edge_starvation_score=0.05,
            attention_sparsity_0_01=0.40,
            attention_entropy_normalized=0.80,
            control_attention_separation=-0.10,
            future_leak_score=0.01,
        )
    )

    assert decision.decision == "hard_stop_hold"
    assert decision.gate_floor_delta == 0


def test_gate_floor_controller_respects_bounds() -> None:
    controller = GateFloorNoiseController(
        GateFloorNoiseConfig(initial_gate_floor=0.34, max_gate_floor=0.35)
    )

    decision = controller.update(
        GateFloorObservation(
            step=400,
            memory_edge_density=0.50,
            edge_survival=0.60,
            random_edge_noise_score=0.90,
            shuffled_edge_noise_score=0.85,
            real_edge_starvation_score=0.02,
            attention_sparsity_0_01=0.30,
            attention_entropy_normalized=0.90,
            control_attention_separation=-0.20,
        )
    )

    assert decision.current_gate_floor <= controller.config.max_gate_floor


def test_gate_floor_noise_controller_report_passes() -> None:
    report = build_gate_floor_noise_controller_report()

    assert report["status"] == "pass"
    assert report["next_required_step"] == "causal_reachability_controller"
    assert report["checks"]["required_gate_floor_outputs_emitted"]
    assert report["checks"]["schema_declares_gate_floor_fields"]
    assert report["checks"]["can_raise_gate_floor_when_edges_are_noisy"]
    assert report["checks"]["can_lower_gate_floor_when_real_edges_are_starved"]
    assert set(REQUIRED_GATE_FLOOR_OUTPUTS).issubset(report["decisions"][-1])
    json.dumps(report)


def test_progress_line_includes_gate_floor_controller_fields() -> None:
    line = format_progress_line(
        {
            "condition": "real_memory_d_adaptive",
            "step": 100,
            "validation_loss": 5.0,
            "gate_floor": 0.05,
            "gate_floor_next": 0.07,
            "gate_floor_delta": 0.02,
            "gate_floor_decision": "raise_gate_floor",
            "gate_floor_credit": -0.30,
            "gate_floor_risk_pressure": 0.70,
            "edge_survival": 0.50,
            "random_edge_noise_score": 0.75,
            "real_edge_starvation_score": 0.05,
        }
    )

    assert "gate_decision=raise_gate_floor" in line
    assert "gate=0.050" in line
    assert "g_next=0.070" in line
    assert "d_gate=0.020" in line
    assert "gPress=0.700" in line
    assert "rNoise=0.750" in line
