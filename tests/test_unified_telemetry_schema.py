import json

from evaluation.open_control_philosophy_contract import REQUIRED_TELEMETRY
from evaluation.unified_telemetry_schema import (
    FIELD_ALIASES,
    MINIMUM_LIVE_FIELDS,
    RUN02_COMPATIBILITY_FIELDS,
    build_unified_telemetry_schema_report,
    validate_telemetry_record,
    validate_unified_telemetry_schema,
)


def test_unified_telemetry_schema_passes_and_is_json_serializable() -> None:
    report = build_unified_telemetry_schema_report()

    assert report["status"] == "pass"
    assert all(report["checks"].values())
    assert report["next_required_step"] == "memory_state_instrumentation"
    json.dumps(report)


def test_unified_telemetry_schema_covers_open_control_required_fields() -> None:
    report = build_unified_telemetry_schema_report()

    assert set(REQUIRED_TELEMETRY).issubset(report["fields"])


def test_unified_telemetry_schema_accepts_run02_aliases() -> None:
    report = build_unified_telemetry_schema_report()
    fields_or_aliases = set(report["fields"])
    for aliases in FIELD_ALIASES.values():
        fields_or_aliases.update(aliases)

    assert set(RUN02_COMPATIBILITY_FIELDS).issubset(fields_or_aliases)


def test_unified_telemetry_record_validation_accepts_aliases() -> None:
    record = {
        "step": 100,
        "condition": "real_memory_d_adaptive",
        "validation_loss": 5.0,
        "alpha_effective": 0.02,
        "alpha_next": 0.03,
        "alpha_delta": 0.01,
        "alpha_decision": "grow",
        "adaptive_slope_gain": 0.0002,
        "base_centered": 0.001,
        "geo_to_qk_ratio": 0.02,
        "attention_entropy": 3.1,
        "mean_max_probability": 0.2,
        "rigidity_risk": 0.0,
        "control_penalty": 0.0,
    }

    result = validate_telemetry_record(record)

    assert result["status"] == "pass"
    assert result["missing_required_fields"] == []


def test_unified_telemetry_record_validation_reports_missing_fields() -> None:
    result = validate_telemetry_record({"step": 1, "condition": "x"})

    assert result["status"] == "fail"
    assert "validation_loss" in result["missing_required_fields"]


def test_unified_telemetry_schema_fails_on_duplicate_category_field() -> None:
    report = build_unified_telemetry_schema_report()
    report["categories"]["runtime"].append(MINIMUM_LIVE_FIELDS[0])

    checks = validate_unified_telemetry_schema(report)

    assert not checks["no_duplicate_category_fields"]
