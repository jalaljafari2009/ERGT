import json

from evaluation.open_control_philosophy_contract import (
    ADAPTIVE_PARAMETERS,
    HARD_STOPS,
    SOFT_PRESSURES,
    build_open_control_philosophy_contract,
    validate_contract,
)


def test_open_control_contract_passes_and_is_json_serializable() -> None:
    report = build_open_control_philosophy_contract()

    assert report["status"] == "pass"
    assert all(report["checks"].values())
    assert report["next_required_step"] == "unified_telemetry_schema"
    json.dumps(report)


def test_open_control_contract_keeps_soft_pressures_out_of_hard_stops() -> None:
    report = build_open_control_philosophy_contract()

    assert set(report["hard_stops"]) == set(HARD_STOPS)
    assert not set(report["hard_stops"]).intersection(SOFT_PRESSURES)


def test_open_control_contract_declares_growth_and_restraint_for_parameters() -> None:
    report = build_open_control_philosophy_contract()

    assert set(report["adaptive_parameters"]) == set(ADAPTIVE_PARAMETERS)
    for parameter in report["adaptive_parameters"].values():
        assert parameter["growth_evidence"]
        assert parameter["restraint_evidence"]
        assert parameter["hard_ceiling_is_scientific_claim"] is False


def test_open_control_contract_requires_search_trajectory_and_replay_records() -> None:
    report = build_open_control_philosophy_contract()

    assert report["checks"]["telemetry_records_parameter_search_and_injected_evidence"]
    assert report["checks"]["controller_obligations_prevent_soft_flag_abort"]
    assert report["checks"]["controller_obligations_require_full_parameter_trajectory"]
    assert report["checks"]["controller_obligations_require_decision_replay"]
    assert report["checks"]["controller_obligations_require_search_not_static_gate"]
    assert "parameter_trajectory" in report["required_telemetry"]
    assert "injected_evidence_ledger" in report["required_telemetry"]
    assert "decision_replay_record" in report["required_telemetry"]


def test_open_control_contract_fails_if_geo_qk_is_made_hard_stop() -> None:
    report = build_open_control_philosophy_contract()
    report["hard_stops"]["geo_to_qk_ratio"] = {"reason": "bad hard ceiling"}

    checks = validate_contract(report)

    assert not checks["hard_stops_are_limited_to_safety_and_validity"]
    assert not checks["soft_pressures_are_not_hard_stops"]
