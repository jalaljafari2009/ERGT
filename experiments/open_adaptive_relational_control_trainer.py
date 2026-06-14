"""Open adaptive relational control trainer orchestration.

This module does not run the heavy language-model optimization itself. It
defines the controller-loop contract used by the adaptive trainer: telemetry in,
control separation plus meta-control observation out, with fail-fast safety and
replayable logs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from experiments.control_separation_scoring import (
    ControlSeparationConfig,
    ControlSeparationScorer,
)
from experiments.live_100_step_diagnostic_table import (
    LiveDiagnosticTableBuilder,
    LiveDiagnosticTableConfig,
)
from experiments.meta_control_attention_observer import (
    MetaControlAttentionConfig,
    MetaControlAttentionObserver,
)
from experiments.progress_logging import format_progress_line, sanitize_for_json

TrainerStatus = Literal["completed", "failed_fast", "empty"]

DECISION_FIELDS = (
    "alpha_decision",
    "memory_decay_decision",
    "memory_eta_decision",
    "gate_floor_decision",
    "distance_norm_scale_decision",
    "causal_reachability_decision",
    "joint_budget_decision",
    "control_separation_status",
    "meta_observer_decision_summary",
)

HARD_STOP_BOOLEAN_FIELDS = (
    "hard_stop_triggered",
    "nan_or_inf_detected",
    "loss_explosion_detected",
    "future_leakage_detected",
    "severe_attention_collapse_detected",
    "control_unfairness_detected",
)


@dataclass(frozen=True)
class OpenAdaptiveTrainerConfig:
    """Configuration for the stage-17 trainer orchestration contract."""

    real_condition: str = "real_memory_d"
    live_display_interval: int = 100
    fail_fast: bool = True
    artifact_bundle_name: str = "ergt_03_adaptive_control_report_bundle.zip"
    excluded_artifact_patterns: tuple[str, ...] = (
        "checkpoints/",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "optimizer_state*",
    )

    def validate(self) -> None:
        if not self.real_condition:
            raise ValueError("real_condition must be non-empty")
        if self.live_display_interval <= 0:
            raise ValueError("live_display_interval must be positive")
        if not self.artifact_bundle_name.endswith(".zip"):
            raise ValueError("artifact_bundle_name must be a .zip file")


@dataclass
class OpenAdaptiveTrainerRunResult:
    """Replayable trainer-loop run result."""

    trainer_status: TrainerStatus
    trainer_fail_fast_triggered: bool
    trainer_fail_fast_reason: str | None
    trainer_processed_rows: int
    trainer_processed_steps: list[int]
    progress_log: list[dict[str, Any]]
    live_display_rows: list[dict[str, Any]]
    live_display_lines: list[str]
    live_diagnostic_rows: list[dict[str, Any]]
    live_diagnostic_tables: list[str]
    live_diagnostic_plot_payloads: list[dict[str, Any]]
    live_diagnostic_row_count: int
    live_diagnostic_table_ready: bool
    live_diagnostic_plot_ready: bool
    controller_decision_log: list[dict[str, Any]]
    meta_control_observer_log: list[dict[str, Any]]
    control_separation_log: list[dict[str, Any]]
    safety_log: list[dict[str, Any]]
    lightweight_artifact_manifest: dict[str, Any]
    trainer_summary: dict[str, Any]
    trainer_replay_record: dict[str, Any]


class OpenAdaptiveRelationalControlTrainer:
    """Run the adaptive control loop over saved or live telemetry rows."""

    def __init__(
        self,
        config: OpenAdaptiveTrainerConfig | None = None,
        *,
        control_config: ControlSeparationConfig | None = None,
        meta_config: MetaControlAttentionConfig | None = None,
    ) -> None:
        self.config = config or OpenAdaptiveTrainerConfig()
        self.config.validate()
        self.control_scorer = ControlSeparationScorer(control_config)
        self.meta_observer = MetaControlAttentionObserver(meta_config)
        self.live_diagnostics = LiveDiagnosticTableBuilder(
            LiveDiagnosticTableConfig(display_interval=self.config.live_display_interval)
        )

    def run(self, telemetry_rows: list[dict[str, Any]]) -> OpenAdaptiveTrainerRunResult:
        progress_by_condition: dict[str, list[dict[str, Any]]] = {}
        progress_log: list[dict[str, Any]] = []
        live_rows: list[dict[str, Any]] = []
        live_lines: list[str] = []
        live_diagnostic_rows: list[dict[str, Any]] = []
        live_diagnostic_tables: list[str] = []
        live_diagnostic_plot_payloads: list[dict[str, Any]] = []
        decision_log: list[dict[str, Any]] = []
        meta_log: list[dict[str, Any]] = []
        separation_log: list[dict[str, Any]] = []
        safety_log: list[dict[str, Any]] = []
        status: TrainerStatus = "empty"
        fail_reason: str | None = None

        for index, raw_row in enumerate(telemetry_rows):
            row = _normalize_row(raw_row, index)
            progress_by_condition.setdefault(str(row["condition"]), []).append(row)
            record = dict(row)
            control_summary = self._score_control_separation(
                progress_by_condition,
                current_step=int(row["step"]),
            )
            if control_summary is not None:
                record.update(control_summary)
                separation_log.append(_log_entry(record, "control_separation"))

            meta_summary = self.meta_observer.summary(self.meta_observer.observe(record))
            record.update(meta_summary)
            meta_log.append(_log_entry(record, "meta_control_attention"))

            hard_stop_reason = _hard_stop_reason(record)
            if hard_stop_reason is not None:
                safety_log.append(_safety_entry(record, hard_stop_reason, True))
                record["trainer_status"] = "failed_fast"
                record["trainer_fail_fast_triggered"] = True
                record["trainer_fail_fast_reason"] = hard_stop_reason
                status = "failed_fast"
                fail_reason = hard_stop_reason
            else:
                safety_log.append(_safety_entry(record, None, False))
                record["trainer_status"] = "running"
                record["trainer_fail_fast_triggered"] = False
                record["trainer_fail_fast_reason"] = None
                status = "completed"

            decision_log.extend(_decision_entries(record))
            progress_log.append(sanitize_for_json(record))
            if (
                self.live_diagnostics.should_display(record)
                or status == "failed_fast"
            ):
                live_rows.append(sanitize_for_json(record))
                live_lines.append(format_progress_line(record))
                diagnostic_row = self.live_diagnostics.build_row(record)
                live_diagnostic_rows.append(diagnostic_row)
                snapshot = self.live_diagnostics.build_snapshot(live_diagnostic_rows)
                live_diagnostic_tables.append(snapshot["live_diagnostic_table_markdown"])
                live_diagnostic_plot_payloads.append(
                    snapshot["live_diagnostic_plot_payload"]
                )
            if status == "failed_fast" and self.config.fail_fast:
                break

        result_status = status
        if not progress_log:
            result_status = "empty"
        manifest = self._artifact_manifest()
        summary = {
            "trainer_status": result_status,
            "trainer_fail_fast_triggered": result_status == "failed_fast",
            "trainer_fail_fast_reason": fail_reason,
            "trainer_processed_rows": len(progress_log),
            "trainer_processed_steps": [int(row["step"]) for row in progress_log],
            "controller_decision_count": len(decision_log),
            "meta_observer_event_count": len(meta_log),
            "control_separation_event_count": len(separation_log),
            "safety_event_count": len(safety_log),
            "live_display_event_count": len(live_rows),
            "live_diagnostic_row_count": len(live_diagnostic_rows),
            "live_diagnostic_table_ready": bool(live_diagnostic_tables),
            "live_diagnostic_plot_ready": bool(live_diagnostic_plot_payloads),
            "progress_log_ready": bool(progress_log),
            "controller_decision_log_ready": bool(decision_log),
            "meta_control_observer_log_ready": bool(meta_log),
            "control_separation_log_ready": bool(separation_log),
            "safety_log_ready": bool(safety_log),
            "lightweight_artifact_bundle_ready": True,
            "checkpoint_artifacts_excluded": True,
            "artifact_bundle_name": self.config.artifact_bundle_name,
        }
        replay = {
            "trainer_status": result_status,
            "fail_fast": self.config.fail_fast,
            "live_display_interval": self.config.live_display_interval,
            "log_streams": [
                "progress_log",
                "controller_decision_log",
                "meta_control_observer_log",
                "control_separation_log",
                "safety_log",
                "live_diagnostic_rows",
                "live_diagnostic_tables",
                "live_diagnostic_plot_payloads",
            ],
            "sequential_run_rule": (
                "control separation and meta-control attention use current_step "
                "limits so later control rows are not read early"
            ),
        }
        return OpenAdaptiveTrainerRunResult(
            trainer_status=result_status,
            trainer_fail_fast_triggered=result_status == "failed_fast",
            trainer_fail_fast_reason=fail_reason,
            trainer_processed_rows=len(progress_log),
            trainer_processed_steps=[int(row["step"]) for row in progress_log],
            progress_log=progress_log,
            live_display_rows=live_rows,
            live_display_lines=live_lines,
            live_diagnostic_rows=live_diagnostic_rows,
            live_diagnostic_tables=live_diagnostic_tables,
            live_diagnostic_plot_payloads=live_diagnostic_plot_payloads,
            live_diagnostic_row_count=len(live_diagnostic_rows),
            live_diagnostic_table_ready=bool(live_diagnostic_tables),
            live_diagnostic_plot_ready=bool(live_diagnostic_plot_payloads),
            controller_decision_log=decision_log,
            meta_control_observer_log=meta_log,
            control_separation_log=separation_log,
            safety_log=safety_log,
            lightweight_artifact_manifest=manifest,
            trainer_summary=summary,
            trainer_replay_record=replay,
        )

    def summary(self, result: OpenAdaptiveTrainerRunResult) -> dict[str, Any]:
        return sanitize_for_json(asdict(result))

    def _score_control_separation(
        self,
        progress_by_condition: dict[str, list[dict[str, Any]]],
        *,
        current_step: int,
    ) -> dict[str, Any] | None:
        try:
            score = self.control_scorer.score(
                progress_by_condition,
                current_step=current_step,
            )
        except ValueError:
            return None
        return self.control_scorer.summary(score)

    def _artifact_manifest(self) -> dict[str, Any]:
        included = {
            "progress_log": "progress_log.jsonl",
            "controller_decision_log": "controller_decision_log.jsonl",
            "meta_control_observer_log": "meta_control_observer_log.jsonl",
            "control_separation_log": "control_separation_log.jsonl",
            "safety_log": "safety_log.jsonl",
            "live_diagnostic_rows": "live_diagnostic_rows.jsonl",
            "live_diagnostic_tables": "live_diagnostic_tables.md",
            "live_diagnostic_plot_payloads": "live_diagnostic_plot_payloads.json",
            "summary": "trainer_summary.json",
        }
        return {
            "artifact_bundle_name": self.config.artifact_bundle_name,
            "included_artifacts": included,
            "excluded_artifact_patterns": list(self.config.excluded_artifact_patterns),
            "checkpoint_artifacts_excluded": any(
                "checkpoint" in item or "*.pt" in item or "*.ckpt" in item
                for item in self.config.excluded_artifact_patterns
            ),
            "lightweight_only": True,
        }


def run_open_adaptive_control_trainer(
    telemetry_rows: list[dict[str, Any]],
    *,
    config: OpenAdaptiveTrainerConfig | None = None,
) -> dict[str, Any]:
    """Convenience wrapper returning a JSON-ready trainer run summary."""

    trainer = OpenAdaptiveRelationalControlTrainer(config)
    return trainer.summary(trainer.run(telemetry_rows))


def _normalize_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    if row.get("condition") is None:
        raise ValueError("telemetry row must include condition")
    if row.get("step") is None:
        raise ValueError("telemetry row must include step")
    normalized = dict(row)
    normalized["step"] = int(normalized["step"])
    normalized["trainer_row_index"] = index
    value = normalized.get("validation_loss")
    if value is not None:
        normalized["validation_loss"] = float(value)
    return normalized


def _hard_stop_reason(record: dict[str, Any]) -> str | None:
    validation_loss = record.get("validation_loss")
    if validation_loss is not None and not math.isfinite(float(validation_loss)):
        return "nan_or_inf_validation_loss"
    for field_name in HARD_STOP_BOOLEAN_FIELDS:
        if bool(record.get(field_name)):
            return field_name
    future_leak_score = record.get("future_leak_score")
    if future_leak_score is not None and float(future_leak_score) > 0:
        return "future_leak_score"
    hard_reason = record.get("hard_stop_reason")
    if hard_reason:
        return str(hard_reason)
    return None


def _decision_entries(record: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for field_name in DECISION_FIELDS:
        decision = record.get(field_name)
        if decision is None:
            continue
        entries.append(
            {
                "step": int(record["step"]),
                "condition": record["condition"],
                "controller": field_name,
                "decision": decision,
            }
        )
    return entries


def _log_entry(record: dict[str, Any], stream: str) -> dict[str, Any]:
    return sanitize_for_json(
        {
            "stream": stream,
            "step": record["step"],
            "condition": record["condition"],
            "mode": record.get("meta_control_mode")
            or record.get("control_separation_mode"),
            "status": record.get("control_separation_status")
            or record.get("meta_observer_decision_summary"),
            "replay_record": record.get("meta_replay_record")
            or record.get("decision_replay_record"),
        }
    )


def _safety_entry(
    record: dict[str, Any],
    reason: str | None,
    triggered: bool,
) -> dict[str, Any]:
    return sanitize_for_json(
        {
            "step": record["step"],
            "condition": record["condition"],
            "hard_stop_triggered": triggered,
            "hard_stop_reason": reason,
            "future_leak_score": record.get("future_leak_score"),
            "validation_loss": record.get("validation_loss"),
        }
    )


def _should_display(record: dict[str, Any], interval: int) -> bool:
    step = int(record["step"])
    return step == 1 or step % interval == 0
