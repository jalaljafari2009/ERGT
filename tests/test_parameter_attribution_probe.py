import json

from evaluation.parameter_attribution_probe import (
    ParameterAttributionConfig,
    build_parameter_attribution_probe_report,
)


def test_parameter_attribution_probe_report_passes_on_synthetic_telemetry() -> None:
    report = build_parameter_attribution_probe_report()

    assert report["status"] == "pass"
    checks = report["checks"]
    assert checks["required_attribution_outputs_emitted"]
    assert checks["major_decisions_have_attribution_or_uncertainty"]
    assert checks["contribution_estimate_fields_present"]
    assert checks["changed_parameters_have_estimates_or_uncertainty"]
    assert checks["ambiguous_interactions_flagged_when_present"]
    assert report["summary"]["alpha_contribution_estimate"] is not None
    assert report["summary"]["memory_eta_decay_contribution_estimate"] is not None
    assert report["summary"]["gate_floor_contribution_estimate"] is not None
    assert report["summary"]["normalization_contribution_estimate"] is not None
    assert report["summary"]["reachability_contribution_estimate"] is not None
    assert report["next_required_step"] == "adaptive_alpha_controller_v2"
    json.dumps(report)


def test_parameter_attribution_probe_estimates_individual_contributions() -> None:
    report = build_parameter_attribution_probe_report(
        telemetry_rows=[
            {
                "step": 100,
                "alpha_delta": 0.02,
                "alpha_decision": "grow",
                "alpha_credit": 0.5,
            },
            {
                "step": 200,
                "gate_floor_delta": 0.01,
                "gate_floor_decision": "raise",
                "gate_floor_credit": -0.2,
            },
        ]
    )

    summary = report["summary"]
    assert summary["alpha_contribution_estimate"] == 0.01
    assert summary["gate_floor_contribution_estimate"] == -0.002


def test_parameter_attribution_probe_flags_missing_credit_as_uncertain() -> None:
    report = build_parameter_attribution_probe_report(
        telemetry_rows=[
            {
                "step": 100,
                "alpha_delta": 0.01,
                "alpha_decision": "grow",
            }
        ]
    )

    assert report["status"] == "pass"
    row = report["row_attribution"][0]
    assert "alpha_changed_without_credit" in row["uncertainty_flags"]
    assert not row["has_attribution_evidence"]


def test_parameter_attribution_probe_flags_multi_parameter_interaction() -> None:
    report = build_parameter_attribution_probe_report(
        telemetry_rows=[
            {
                "step": 100,
                "alpha_delta": 0.01,
                "alpha_decision": "grow",
                "alpha_credit": 0.2,
                "memory_eta_delta": 0.02,
                "memory_eta_decision": "grow",
                "memory_eta_credit": 0.1,
                "gate_floor_delta": 0.01,
                "gate_floor_decision": "raise",
                "gate_floor_credit": 0.05,
            }
        ],
        config=ParameterAttributionConfig(),
    )

    assert report["checks"]["ambiguous_interactions_flagged_when_present"]
    assert "multi_parameter_interaction" in report["summary"]["uncertainty_flags"]
    assert report["status"] == "pass"
