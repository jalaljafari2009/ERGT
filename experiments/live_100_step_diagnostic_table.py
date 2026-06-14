"""Live 100-step diagnostic table support for adaptive ERGT runs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from experiments.progress_logging import sanitize_for_json


@dataclass(frozen=True)
class LiveDiagnosticColumn:
    """One display column in the live diagnostic table."""

    key: str
    label: str
    sources: tuple[str, ...]
    precision: int | None = None


@dataclass(frozen=True)
class LiveDiagnosticTableConfig:
    """Configuration for compact live tables and plot payloads."""

    display_interval: int = 100
    max_table_rows: int = 12
    missing_value: str = "NA"

    def validate(self) -> None:
        if self.display_interval <= 0:
            raise ValueError("display_interval must be positive")
        if self.max_table_rows <= 0:
            raise ValueError("max_table_rows must be positive")
        if not self.missing_value:
            raise ValueError("missing_value must be non-empty")


LIVE_DIAGNOSTIC_COLUMNS: tuple[LiveDiagnosticColumn, ...] = (
    LiveDiagnosticColumn("step", "step", ("step",), 0),
    LiveDiagnosticColumn("condition", "condition", ("condition",)),
    LiveDiagnosticColumn("train_loss", "train", ("train_loss",), 4),
    LiveDiagnosticColumn("validation_loss", "val", ("validation_loss",), 4),
    LiveDiagnosticColumn(
        "delta_vs_baseline",
        "d_base",
        ("real_vs_baseline_delta", "baseline_centered_improvement"),
        4,
    ),
    LiveDiagnosticColumn(
        "rolling_slope",
        "slope",
        ("loss_slope", "loss_slope_gain", "adaptive_slope_gain"),
        6,
    ),
    LiveDiagnosticColumn("alpha", "alpha", ("alpha", "alpha_effective"), 4),
    LiveDiagnosticColumn("geo_to_qk_ratio", "geo_qk", ("geo_to_qk_ratio",), 4),
    LiveDiagnosticColumn("memory_stability", "m_stab", ("memory_stability",), 4),
    LiveDiagnosticColumn("memory_turnover", "m_turn", ("memory_turnover",), 4),
    LiveDiagnosticColumn("memory_persistence", "m_pers", ("memory_persistence",), 4),
    LiveDiagnosticColumn("memory_rigidity", "m_rigid", ("memory_rigidity",), 4),
    LiveDiagnosticColumn("noise_risk", "noise", ("noise_risk",), 4),
    LiveDiagnosticColumn(
        "attention_regime",
        "attn_regime",
        ("attention_behavior_regime",),
    ),
    LiveDiagnosticColumn(
        "attention_control_separation",
        "attn_sep",
        ("attention_behavior_separation", "control_attention_separation"),
        4,
    ),
    LiveDiagnosticColumn(
        "contrast_retention",
        "contrast",
        ("distance_contrast_retention",),
        4,
    ),
    LiveDiagnosticColumn(
        "future_leak",
        "f_leak",
        ("future_leak_score", "future_leakage_detected"),
        4,
    ),
    LiveDiagnosticColumn("meta_top_signal", "meta_top", ("meta_top_signal",)),
    LiveDiagnosticColumn(
        "meta_attention_entropy",
        "meta_ent",
        ("meta_attention_entropy",),
        4,
    ),
    LiveDiagnosticColumn("meta_alpha_weight", "w_alpha", ("meta_alpha_weight",), 4),
    LiveDiagnosticColumn(
        "meta_memory_weight",
        "w_memory",
        ("meta_memory_weight",),
        4,
    ),
    LiveDiagnosticColumn("meta_gate_weight", "w_gate", ("meta_gate_weight",), 4),
    LiveDiagnosticColumn("meta_reach_weight", "w_reach", ("meta_reach_weight",), 4),
    LiveDiagnosticColumn("meta_norm_weight", "w_norm", ("meta_norm_weight",), 4),
    LiveDiagnosticColumn(
        "controller_conflict_score",
        "conflict",
        ("controller_conflict_score",),
        4,
    ),
    LiveDiagnosticColumn(
        "meta_control_confidence",
        "meta_conf",
        ("meta_control_confidence",),
        4,
    ),
    LiveDiagnosticColumn("alpha_decision", "a_decision", ("alpha_decision",)),
    LiveDiagnosticColumn(
        "memory_eta_decision",
        "eta_decision",
        ("memory_eta_decision",),
    ),
    LiveDiagnosticColumn(
        "memory_decay_decision",
        "decay_decision",
        ("memory_decay_decision",),
    ),
    LiveDiagnosticColumn(
        "gate_floor_decision",
        "gate_decision",
        ("gate_floor_decision",),
    ),
    LiveDiagnosticColumn(
        "causal_reachability_decision",
        "reach_decision",
        ("causal_reachability_decision",),
    ),
    LiveDiagnosticColumn(
        "distance_norm_scale_decision",
        "norm_decision",
        ("distance_norm_scale_decision",),
    ),
    LiveDiagnosticColumn(
        "joint_budget_decision",
        "budget_decision",
        ("joint_budget_decision",),
    ),
    LiveDiagnosticColumn("trainer_status", "trainer", ("trainer_status",)),
    LiveDiagnosticColumn(
        "trainer_fail_fast_triggered",
        "fail_fast",
        ("trainer_fail_fast_triggered",),
        0,
    ),
)

REQUIRED_LIVE_DIAGNOSTIC_COLUMNS = tuple(column.key for column in LIVE_DIAGNOSTIC_COLUMNS)

PLOT_SERIES_GROUPS: dict[str, tuple[str, ...]] = {
    "loss": ("validation_loss", "delta_vs_baseline", "rolling_slope"),
    "geometry": ("alpha", "geo_to_qk_ratio", "contrast_retention"),
    "memory": (
        "memory_stability",
        "memory_turnover",
        "memory_persistence",
        "memory_rigidity",
        "noise_risk",
    ),
    "meta_control": (
        "meta_alpha_weight",
        "meta_memory_weight",
        "meta_gate_weight",
        "meta_reach_weight",
        "meta_norm_weight",
        "controller_conflict_score",
        "meta_control_confidence",
    ),
    "safety": ("future_leak", "trainer_fail_fast_triggered"),
}


class LiveDiagnosticTableBuilder:
    """Build live diagnostic rows, markdown tables, and plot-ready series."""

    def __init__(self, config: LiveDiagnosticTableConfig | None = None) -> None:
        self.config = config or LiveDiagnosticTableConfig()
        self.config.validate()

    def should_display(self, record: dict[str, Any]) -> bool:
        step = int(record.get("step", 0))
        return step == 1 or step % self.config.display_interval == 0

    def build_row(self, record: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for column in LIVE_DIAGNOSTIC_COLUMNS:
            row[column.key] = _first_present(record, column.sources)
        row["live_diagnostic_row_ready"] = True
        row["live_diagnostic_table_ready"] = True
        row["live_diagnostic_plot_ready"] = True
        return sanitize_for_json(row)

    def build_snapshot(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        table = self.format_markdown(rows)
        plot_payload = self.build_plot_payload(rows)
        return {
            "live_diagnostic_row_count": len(rows),
            "live_diagnostic_columns": [asdict(column) for column in LIVE_DIAGNOSTIC_COLUMNS],
            "live_diagnostic_table_markdown": table,
            "live_diagnostic_plot_payload": plot_payload,
            "live_diagnostic_table_ready": bool(rows),
            "live_diagnostic_plot_ready": bool(rows),
        }

    def format_markdown(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""
        visible_rows = rows[-self.config.max_table_rows :]
        labels = [column.label for column in LIVE_DIAGNOSTIC_COLUMNS]
        header = "| " + " | ".join(labels) + " |"
        divider = "| " + " | ".join("---" for _ in labels) + " |"
        body = [
            "| "
            + " | ".join(
                self._format_value(row.get(column.key), column)
                for column in LIVE_DIAGNOSTIC_COLUMNS
            )
            + " |"
            for row in visible_rows
        ]
        return "\n".join([header, divider, *body])

    def build_plot_payload(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        series_groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for group_name, fields in PLOT_SERIES_GROUPS.items():
            group: dict[str, list[dict[str, Any]]] = {}
            for field_name in fields:
                points = []
                for row in rows:
                    value = _numeric_or_none(row.get(field_name))
                    if value is None:
                        continue
                    points.append(
                        {
                            "step": int(row["step"]),
                            "condition": row.get("condition"),
                            "value": value,
                        }
                    )
                group[field_name] = points
            series_groups[group_name] = group
        return {
            "x_axis": "step",
            "series_groups": series_groups,
            "latest_step": int(rows[-1]["step"]) if rows else None,
            "plot_ready": bool(rows),
        }

    def _format_value(self, value: Any, column: LiveDiagnosticColumn) -> str:
        if value is None:
            return self.config.missing_value
        if isinstance(value, bool):
            return "1" if value else "0"
        numeric = _numeric_or_none(value)
        if numeric is not None and column.precision is not None:
            if column.precision == 0:
                return str(int(round(numeric)))
            return f"{numeric:.{column.precision}f}"
        return str(value)


def build_live_diagnostic_row(record: dict[str, Any]) -> dict[str, Any]:
    """Convenience wrapper for one diagnostic row."""

    return LiveDiagnosticTableBuilder().build_row(record)


def build_live_diagnostic_snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Convenience wrapper for a full table and plot snapshot."""

    return LiveDiagnosticTableBuilder().build_snapshot(rows)


def _first_present(record: dict[str, Any], sources: tuple[str, ...]) -> Any:
    for source in sources:
        if source in record and record[source] is not None:
            return record[source]
    return None


def _numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
