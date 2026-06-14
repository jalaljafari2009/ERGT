import json

from evaluation.adaptive_memory_controller import (
    REQUIRED_MEMORY_CONTROLLER_OUTPUTS,
    build_adaptive_memory_controller_report,
)
from experiments.adaptive_memory_controller import (
    AdaptiveMemoryConfig,
    AdaptiveMemoryController,
    MemoryObservation,
)


def test_adaptive_memory_controller_injects_when_memory_is_starved() -> None:
    controller = AdaptiveMemoryController()

    decision = controller.update(
        MemoryObservation(
            step=100,
            memory_stability=0.40,
            memory_turnover=0.03,
            memory_persistence=0.34,
            memory_edge_density=0.04,
            memory_rigidity=0.10,
            noise_risk=0.08,
            real_memory_advantage=0.20,
            random_memory_advantage=0.02,
            shuffled_memory_advantage=0.01,
        )
    )

    assert decision.decision == "inject_memory"
    assert decision.eta_delta > 0
    assert decision.decay_delta >= 0
    assert decision.parameter_trajectory["eta_next"] == decision.current_eta
    assert decision.injected_evidence_ledger["release_evidence"]
    assert decision.decision_replay_record["decision"] == "inject_memory"


def test_adaptive_memory_controller_smooths_noisy_memory() -> None:
    controller = AdaptiveMemoryController()

    decision = controller.update(
        MemoryObservation(
            step=200,
            memory_stability=0.50,
            memory_turnover=0.24,
            memory_persistence=0.42,
            memory_edge_density=0.30,
            memory_rigidity=0.18,
            noise_risk=0.75,
            real_memory_advantage=0.02,
            random_memory_advantage=0.10,
            shuffled_memory_advantage=0.08,
        )
    )

    assert decision.decision == "smooth_memory"
    assert decision.eta_delta < 0
    assert decision.decay_delta > 0
    assert decision.noise_evidence["noise_high"] is True


def test_adaptive_memory_controller_releases_rigid_memory() -> None:
    controller = AdaptiveMemoryController()

    decision = controller.update(
        MemoryObservation(
            step=300,
            memory_stability=0.92,
            memory_turnover=0.01,
            memory_persistence=0.94,
            memory_edge_density=0.85,
            memory_rigidity=0.80,
            noise_risk=0.24,
            real_memory_advantage=0.08,
            random_memory_advantage=0.03,
            shuffled_memory_advantage=0.02,
        )
    )

    assert decision.decision == "release_rigid_memory"
    assert decision.eta_delta < 0
    assert decision.decay_delta < 0
    assert decision.rigidity_evidence["rigidity_high"] is True


def test_adaptive_memory_controller_uses_future_leak_as_hard_stop() -> None:
    controller = AdaptiveMemoryController()

    decision = controller.update(
        MemoryObservation(
            step=400,
            memory_stability=0.92,
            memory_turnover=0.01,
            memory_persistence=0.94,
            memory_edge_density=0.85,
            memory_rigidity=0.10,
            noise_risk=0.10,
            real_memory_advantage=0.20,
            future_leak_score=0.01,
        )
    )

    assert decision.decision == "hard_stop_hold"
    assert decision.eta_delta == 0
    assert decision.decay_delta == 0


def test_adaptive_memory_controller_respects_bounds() -> None:
    controller = AdaptiveMemoryController(
        AdaptiveMemoryConfig(initial_eta=0.03, initial_decay=0.94)
    )

    noisy = MemoryObservation(
        step=500,
        memory_stability=0.50,
        memory_turnover=0.25,
        memory_persistence=0.42,
        memory_edge_density=0.30,
        memory_rigidity=0.18,
        noise_risk=0.80,
        real_memory_advantage=0.01,
        random_memory_advantage=0.12,
        shuffled_memory_advantage=0.09,
    )
    decision = controller.update(noisy)

    assert decision.current_eta >= controller.config.min_eta
    assert decision.current_decay <= controller.config.max_decay


def test_adaptive_memory_controller_report_passes() -> None:
    report = build_adaptive_memory_controller_report()

    assert report["status"] == "pass"
    assert report["next_required_step"] == "gate_floor_and_noise_controller"
    assert report["checks"]["required_memory_controller_outputs_emitted"]
    assert report["checks"]["can_increase_eta_when_memory_is_starved"]
    assert report["checks"]["can_smooth_when_memory_is_noisy"]
    assert report["checks"]["can_release_rigid_memory"]
    assert set(REQUIRED_MEMORY_CONTROLLER_OUTPUTS).issubset(report["decisions"][-1])
    json.dumps(report)
