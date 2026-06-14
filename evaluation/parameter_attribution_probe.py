"""Parameter attribution probe for open adaptive ERGT control."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterAttributionConfig:
    major_delta_threshold: float = 1e-9
    min_evidence_abs: float = 1e-12

    def validate(self) -> None:
        if self.major_delta_threshold < 0:
            raise ValueError("major_delta_threshold must be non-negative")
        if self.min_evidence_abs < 0:
            raise ValueError("min_evidence_abs must be non-negative")


PARAMETER_SPECS = {
    "alpha": {
        "delta": "alpha_delta",
        "decision": "alpha_decision",
        "credit": "alpha_credit",
        "aliases": ["adaptive_score", "loss_slope_gain"],
    },
    "memory_decay": {
        "delta": "memory_decay_delta",
        "decision": "memory_decay_decision",
        "credit": "memory_decay_credit",
        "aliases": ["memory_stability"],
    },
    "memory_eta": {
        "delta": "memory_eta_delta",
        "decision": "memory_eta_decision",
        "credit": "memory_eta_credit",
        "aliases": ["memory_turnover"],
    },
    "gate_floor": {
        "delta": "gate_floor_delta",
        "decision": "gate_floor_decision",
        "credit": "gate_floor_credit",
        "aliases": ["noise_risk"],
    },
    "distance_norm_scale": {
        "delta": "distance_norm_scale_delta",
        "decision": "distance_norm_scale_decision",
        "credit": "distance_norm_scale_credit",
        "aliases": ["distance_std"],
    },
    "causal_reachability": {
        "delta": "causal_reachability_delta",
        "decision": "causal_reachability_decision",
        "credit": "causal_reachability_credit",
        "aliases": ["causal_reachability"],
    },
}

REQUIRED_ATTRIBUTION_OUTPUTS = [
    "alpha_contribution_estimate",
    "memory_eta_decay_contribution_estimate",
    "gate_floor_contribution_estimate",
    "normalization_contribution_estimate",
    "reachability_contribution_estimate",
    "interaction_warnings",
    "uncertainty_flags",
]


def build_parameter_attribution_probe_report(
    *,
    telemetry_rows: list[dict[str, Any]] | None = None,
    config: ParameterAttributionConfig | None = None,
) -> dict[str, Any]:
    """Build the stage-8 parameter attribution report."""

    config = config or ParameterAttributionConfig()
    config.validate()
    input_source = "provided_telemetry"
    if telemetry_rows is None:
        telemetry_rows = _synthetic_telemetry()
        input_source = "synthetic_attribution_smoke"

    rows = _valid_rows(telemetry_rows)
    row_reports = [_attribute_row(row, config=config) for row in rows]
    summary = _summarize(row_reports)
    checks = {
        "required_attribution_outputs_emitted": all(
            field in summary for field in REQUIRED_ATTRIBUTION_OUTPUTS
        ),
        "major_decisions_have_attribution_or_uncertainty": all(
            not row["major_decision"]
            or row["has_attribution_evidence"]
            or row["uncertainty_flags"]
            for row in row_reports
        ),
        "contribution_estimate_fields_present": all(
            field in summary
            for field in [
                "alpha_contribution_estimate",
                "memory_eta_decay_contribution_estimate",
                "gate_floor_contribution_estimate",
                "normalization_contribution_estimate",
                "reachability_contribution_estimate",
            ]
        ),
        "changed_parameters_have_estimates_or_uncertainty": all(
            _changed_parameters_have_estimates_or_uncertainty(row)
            for row in row_reports
        ),
        "ambiguous_interactions_flagged_when_present": _ambiguous_interactions_flagged_when_present(
            row_reports
        ),
        "all_estimates_finite_or_none": _all_estimates_finite_or_none(summary),
    }
    status = "pass" if all(checks.values()) else "fail"

    return {
        "stage": "stage8_parameter_attribution_probe",
        "status": status,
        "input_source": input_source,
        "scientific_scope": (
            "mechanics gate only; estimates which adaptive parameters can be "
            "credited from telemetry and marks ambiguous interactions explicitly"
        ),
        "config": asdict(config),
        "required_outputs": list(REQUIRED_ATTRIBUTION_OUTPUTS),
        "checks": checks,
        "summary": summary,
        "row_attribution": row_reports,
        "next_required_step": (
            "adaptive_alpha_controller_v2" if status == "pass" else "fix_parameter_attribution"
        ),
    }


def _attribute_row(
    row: dict[str, Any],
    *,
    config: ParameterAttributionConfig,
) -> dict[str, Any]:
    changed = []
    contributions = {}
    evidence = {}
    uncertainty_flags = []
    for parameter, spec in PARAMETER_SPECS.items():
        delta = _as_float(row.get(spec["delta"]))
        decision = row.get(spec["decision"])
        credit = _credit_for(row, spec)
        if delta is not None and abs(delta) > config.major_delta_threshold:
            changed.append(parameter)
        if credit is not None:
            evidence[parameter] = credit
        if (
            delta is not None
            and abs(delta) > config.major_delta_threshold
            and credit is not None
            and abs(credit) >= config.min_evidence_abs
        ):
            contributions[parameter] = delta * credit
        elif _is_active_decision(decision):
            uncertainty_flags.append(f"{parameter}_changed_without_credit")

    interaction_warnings = []
    if len(changed) > 1:
        interaction_warnings.append(
            {
                "type": "multi_parameter_change",
                "step": int(row["step"]),
                "parameters": list(changed),
                "message": (
                    "simultaneous parameter changes make single-parameter "
                    "attribution ambiguous"
                ),
            }
        )
        uncertainty_flags.append("multi_parameter_interaction")

    return {
        "step": int(row["step"]),
        "condition": row.get("condition"),
        "changed_parameters": changed,
        "major_decision": bool(changed),
        "evidence": evidence,
        "contribution_estimates": contributions,
        "has_attribution_evidence": bool(contributions),
        "interaction_warnings": interaction_warnings,
        "uncertainty_flags": uncertainty_flags,
        "attribution_summary": _row_summary(contributions, uncertainty_flags),
    }


def _summarize(row_reports: list[dict[str, Any]]) -> dict[str, Any]:
    parameter_totals = {
        parameter: _sum_contributions(row_reports, parameter)
        for parameter in PARAMETER_SPECS
    }
    interaction_warnings = [
        warning
        for row in row_reports
        for warning in row["interaction_warnings"]
    ]
    uncertainty_flags = sorted(
        {
            flag
            for row in row_reports
            for flag in row["uncertainty_flags"]
        }
    )
    return {
        "alpha_contribution_estimate": parameter_totals["alpha"],
        "memory_eta_decay_contribution_estimate": _sum_optional(
            parameter_totals["memory_eta"],
            parameter_totals["memory_decay"],
        ),
        "memory_eta_contribution_estimate": parameter_totals["memory_eta"],
        "memory_decay_contribution_estimate": parameter_totals["memory_decay"],
        "gate_floor_contribution_estimate": parameter_totals["gate_floor"],
        "normalization_contribution_estimate": parameter_totals["distance_norm_scale"],
        "reachability_contribution_estimate": parameter_totals["causal_reachability"],
        "parameter_contribution_estimates": parameter_totals,
        "interaction_warnings": interaction_warnings,
        "uncertainty_flags": uncertainty_flags,
        "major_decision_count": sum(1 for row in row_reports if row["major_decision"]),
        "attribution_summary": _summary_label(parameter_totals, uncertainty_flags),
    }


def _sum_contributions(
    row_reports: list[dict[str, Any]],
    parameter: str,
) -> float | None:
    values = [
        row["contribution_estimates"][parameter]
        for row in row_reports
        if parameter in row["contribution_estimates"]
    ]
    if not values:
        return None
    return sum(values)


def _sum_optional(left: float | None, right: float | None) -> float | None:
    values = [value for value in [left, right] if value is not None]
    if not values:
        return None
    return sum(values)


def _credit_for(row: dict[str, Any], spec: dict[str, Any]) -> float | None:
    for field in [spec["credit"], *spec["aliases"]]:
        value = _as_float(row.get(field))
        if value is not None:
            return value
    return None


def _valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in rows if row.get("step") is not None]
    if not valid:
        raise ValueError("telemetry rows must include step")
    valid.sort(key=lambda row: int(row["step"]))
    return valid


def _is_active_decision(decision: Any) -> bool:
    if decision is None:
        return False
    text = str(decision)
    return bool(text and not text.startswith("hold"))


def _row_summary(
    contributions: dict[str, float],
    uncertainty_flags: list[str],
) -> str:
    if contributions and not uncertainty_flags:
        return "attribution_evidence_available"
    if contributions and uncertainty_flags:
        return "attribution_evidence_with_uncertainty"
    if uncertainty_flags:
        return "attribution_uncertain"
    return "no_major_parameter_change"


def _summary_label(
    parameter_totals: dict[str, float | None],
    uncertainty_flags: list[str],
) -> str:
    estimated = [name for name, value in parameter_totals.items() if value is not None]
    if uncertainty_flags:
        return f"estimated={','.join(estimated)}; uncertainty={','.join(uncertainty_flags)}"
    return f"estimated={','.join(estimated)}"


def _all_estimates_finite_or_none(summary: dict[str, Any]) -> bool:
    keys = [
        "alpha_contribution_estimate",
        "memory_eta_decay_contribution_estimate",
        "gate_floor_contribution_estimate",
        "normalization_contribution_estimate",
        "reachability_contribution_estimate",
    ]
    for key in keys:
        value = summary.get(key)
        if value is not None and not math.isfinite(float(value)):
            return False
    return True


def _changed_parameters_have_estimates_or_uncertainty(row: dict[str, Any]) -> bool:
    for parameter in row["changed_parameters"]:
        if parameter in row["contribution_estimates"]:
            continue
        if any(flag.startswith(f"{parameter}_") for flag in row["uncertainty_flags"]):
            continue
        if "multi_parameter_interaction" in row["uncertainty_flags"]:
            continue
        return False
    return True


def _ambiguous_interactions_flagged_when_present(
    row_reports: list[dict[str, Any]],
) -> bool:
    for row in row_reports:
        if len(row["changed_parameters"]) <= 1:
            continue
        if not any(
            warning["type"] == "multi_parameter_change"
            for warning in row["interaction_warnings"]
        ):
            return False
    return True


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _synthetic_telemetry() -> list[dict[str, Any]]:
    return [
        {
            "step": 100,
            "condition": "real_adaptive",
            "alpha_delta": 0.01,
            "alpha_decision": "grow",
            "alpha_credit": 0.20,
            "loss_slope_gain": 0.20,
        },
        {
            "step": 200,
            "condition": "real_adaptive",
            "memory_eta_delta": 0.02,
            "memory_eta_decision": "grow",
            "memory_eta_credit": 0.12,
        },
        {
            "step": 300,
            "condition": "real_adaptive",
            "memory_decay_delta": -0.01,
            "memory_decay_decision": "shrink",
            "memory_decay_credit": 0.08,
        },
        {
            "step": 400,
            "condition": "real_adaptive",
            "gate_floor_delta": 0.01,
            "gate_floor_decision": "raise",
            "gate_floor_credit": 0.10,
        },
        {
            "step": 500,
            "condition": "real_adaptive",
            "distance_norm_scale_delta": 0.05,
            "distance_norm_scale_decision": "expand",
            "distance_norm_scale_credit": 0.06,
        },
        {
            "step": 600,
            "condition": "real_adaptive",
            "causal_reachability_delta": 1.0,
            "causal_reachability_decision": "expand",
            "causal_reachability_credit": 0.04,
        },
        {
            "step": 700,
            "condition": "real_adaptive",
            "alpha_delta": 0.005,
            "alpha_decision": "grow",
            "alpha_credit": 0.11,
            "memory_eta_delta": 0.01,
            "memory_eta_decision": "grow",
            "memory_eta_credit": 0.05,
        },
    ]
