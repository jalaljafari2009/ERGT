from evaluation.gate_decision import decide_gate


def comparison() -> dict:
    return {
        "conditions": {
            "baseline": {"average_tokens_per_second": 100},
            "alpha_zero": {},
            "real_d": {"average_tokens_per_second": 80},
            "random_d": {},
            "shuffled_d": {},
        },
        "ranking": {"by_final_validation_loss": ["real_d", "baseline"]},
        "deltas": {
            "real_d_vs_baseline": {"final_validation_loss": {"absolute": -0.1, "relative": -0.05}}
        },
    }


def ablation_checks(**overrides) -> dict:
    checks = {
        "all_required_conditions_present": True,
        "alpha_zero_matches_baseline": True,
        "real_d_beats_baseline": True,
        "real_d_beats_random_d": True,
        "real_d_beats_shuffled_d": True,
        "real_d_validation_loss_finite": True,
    }
    checks.update(overrides)
    return {"summary": {"recommendation": "gate_ready_positive"}, "checks": checks, "deltas": {}}


def test_gate_decision_passes_when_all_required_checks_pass() -> None:
    decision = decide_gate(comparison(), ablation_checks())

    assert decision["decision"] == "pass"
    assert not decision["risks"]
    assert "anti_overclaim" in decision


def test_gate_decision_fails_when_alpha_zero_differs() -> None:
    decision = decide_gate(
        comparison(),
        ablation_checks(alpha_zero_matches_baseline=False),
    )

    assert decision["decision"] == "fail"
    assert any("alpha_zero" in risk for risk in decision["risks"])


def test_gate_decision_conditionally_passes_when_one_distance_control_passes() -> None:
    decision = decide_gate(
        comparison(),
        ablation_checks(real_d_beats_baseline=False, real_d_beats_shuffled_d=False),
    )

    assert decision["decision"] == "conditional_pass"
    assert decision["required_next_actions"]
