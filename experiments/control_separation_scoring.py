"""Control separation scoring for sequential ERGT runs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ScoreMode = Literal["partial_live", "final_matched"]
ClaimEligibility = Literal[
    "not_eligible_pending_controls",
    "not_eligible_insufficient_matched_steps",
    "eligible_complete_controls",
]


@dataclass(frozen=True)
class ControlSeparationConfig:
    """Configuration for live and final control separation scoring."""

    real_condition: str = "real_memory_d"
    baseline_condition: str = "baseline"
    control_conditions: dict[str, str] = field(
        default_factory=lambda: {
            "alpha_zero": "alpha_zero",
            "random": "random_memory_d",
            "shuffled": "shuffled_memory_d",
            "no_memory": "no_memory_real_d",
            "instantaneous": "instantaneous_real_d",
        }
    )
    late_window_start: int = 1000
    min_matched_points: int = 2
    pass_margin: float = 0.0
    attention_pass_margin: float = 0.0

    def validate(self) -> None:
        if not self.real_condition:
            raise ValueError("real_condition must be non-empty")
        if not self.baseline_condition:
            raise ValueError("baseline_condition must be non-empty")
        if self.late_window_start < 0:
            raise ValueError("late_window_start must be non-negative")
        if self.min_matched_points < 1:
            raise ValueError("min_matched_points must be at least one")
        if "random" not in self.control_conditions:
            raise ValueError("random control must be configured")
        if "shuffled" not in self.control_conditions:
            raise ValueError("shuffled control must be configured")


@dataclass
class ControlSeparationScore:
    """One replayable control-separation decision."""

    mode: ScoreMode
    claim_eligibility: ClaimEligibility
    control_separation_status: str
    scientific_claim_credit: float
    available_control_families: list[str]
    pending_control_families: list[str]
    control_family_status: dict[str, str]
    matched_control_steps: list[int]
    matched_control_window: dict[str, int | None]
    real_vs_baseline_delta: float | None
    real_vs_alpha_zero_delta: float | None
    real_vs_random_delta: float | None
    real_vs_shuffled_delta: float | None
    real_vs_no_memory_delta: float | None
    real_vs_instantaneous_delta: float | None
    control_separation: float | None
    control_penalty: float
    generic_regularization_warning: str
    attention_behavior_separation: float | None
    partial_live_score: dict[str, Any]
    final_matched_score: dict[str, Any] | None
    decision_replay_record: dict[str, Any]


class ControlSeparationScorer:
    """Score real geometry against controls without using unavailable future runs."""

    def __init__(self, config: ControlSeparationConfig | None = None) -> None:
        self.config = config or ControlSeparationConfig()
        self.config.validate()

    def score(
        self,
        progress_by_condition: dict[str, list[dict[str, Any]]],
        *,
        current_step: int | None = None,
    ) -> ControlSeparationScore:
        rows_by_condition = _normalize_progress(progress_by_condition, current_step)
        real_rows = rows_by_condition.get(self.config.real_condition, [])
        if not real_rows:
            raise ValueError(f"missing real condition progress: {self.config.real_condition}")

        all_controls = self._control_condition_map()
        latest_real = real_rows[-1]
        latest_step = int(latest_real["step"])
        available, pending, statuses = self._control_availability(
            rows_by_condition,
            all_controls,
            latest_step,
        )
        deltas = self._partial_deltas(rows_by_condition, latest_real, all_controls)
        partial_separation = _min_present(
            [
                deltas["baseline"],
                deltas["alpha_zero"],
                deltas["random"],
                deltas["shuffled"],
                deltas["no_memory"],
                deltas["instantaneous"],
            ]
        )
        partial_attention = _attention_separation_at_step(
            rows_by_condition,
            latest_real,
            all_controls,
        )

        final_score = None
        matched_steps: list[int] = []
        matched_window = {"start_step": None, "end_step": None, "points": 0}
        mode: ScoreMode = "partial_live"
        claim_eligibility: ClaimEligibility = "not_eligible_pending_controls"
        status = "partial_signal_only"
        claim_credit = 0.0

        if not pending:
            matched_steps = self._matched_steps(rows_by_condition, all_controls)
            late_steps = [
                step for step in matched_steps if step >= self.config.late_window_start
            ]
            if len(late_steps) >= self.config.min_matched_points:
                mode = "final_matched"
                claim_eligibility = "eligible_complete_controls"
                matched_window = {
                    "start_step": late_steps[0],
                    "end_step": late_steps[-1],
                    "points": len(late_steps),
                }
                final_score = self._final_score(rows_by_condition, all_controls, late_steps)
                deltas.update(final_score["mean_deltas"])
                partial_separation = final_score["control_separation"]
                partial_attention = final_score["attention_behavior_separation"]
                if (
                    partial_separation is not None
                    and partial_separation > self.config.pass_margin
                    and (
                        partial_attention is None
                        or partial_attention >= self.config.attention_pass_margin
                    )
                ):
                    status = "pass_real_geometry_separated"
                    claim_credit = _bounded(partial_separation)
                else:
                    status = "fail_controls_not_separated"
            else:
                claim_eligibility = "not_eligible_insufficient_matched_steps"
                status = "not_eligible_insufficient_matched_steps"

        warning = _generic_regularization_warning(
            mode=mode,
            claim_eligibility=claim_eligibility,
            real_vs_baseline=deltas["baseline"],
            real_vs_random=deltas["random"],
            real_vs_shuffled=deltas["shuffled"],
            margin=self.config.pass_margin,
        )
        control_penalty = _control_penalty(partial_separation, self.config.pass_margin)
        partial = {
            "step": latest_step,
            "available_control_families": list(available),
            "pending_control_families": list(pending),
            "claim_eligibility": claim_eligibility,
            "real_vs_baseline_delta": deltas["baseline"],
            "real_vs_alpha_zero_delta": deltas["alpha_zero"],
            "real_vs_random_delta": deltas["random"],
            "real_vs_shuffled_delta": deltas["shuffled"],
            "real_vs_no_memory_delta": deltas["no_memory"],
            "real_vs_instantaneous_delta": deltas["instantaneous"],
            "control_separation": partial_separation,
            "scientific_claim_credit": claim_credit,
        }
        replay = {
            "mode": mode,
            "current_step": latest_step,
            "current_step_limit": current_step,
            "matched_control_steps": list(matched_steps),
            "pending_control_families": list(pending),
            "control_family_status": dict(statuses),
            "decision": status,
            "rule": (
                "baseline-only live gains are controller pressure; final claim "
                "requires matched late-window real-vs-control separation"
            ),
        }

        return ControlSeparationScore(
            mode=mode,
            claim_eligibility=claim_eligibility,
            control_separation_status=status,
            scientific_claim_credit=claim_credit,
            available_control_families=available,
            pending_control_families=pending,
            control_family_status=statuses,
            matched_control_steps=matched_steps,
            matched_control_window=matched_window,
            real_vs_baseline_delta=deltas["baseline"],
            real_vs_alpha_zero_delta=deltas["alpha_zero"],
            real_vs_random_delta=deltas["random"],
            real_vs_shuffled_delta=deltas["shuffled"],
            real_vs_no_memory_delta=deltas["no_memory"],
            real_vs_instantaneous_delta=deltas["instantaneous"],
            control_separation=partial_separation,
            control_penalty=control_penalty,
            generic_regularization_warning=warning,
            attention_behavior_separation=partial_attention,
            partial_live_score=partial,
            final_matched_score=final_score,
            decision_replay_record=replay,
        )

    def summary(self, score: ControlSeparationScore) -> dict[str, Any]:
        row = asdict(score)
        row.update(
            {
                "control_separation_mode": score.mode,
                "claim_eligibility": score.claim_eligibility,
            }
        )
        return row

    def _control_condition_map(self) -> dict[str, str]:
        return {
            "baseline": self.config.baseline_condition,
            **self.config.control_conditions,
        }

    def _control_availability(
        self,
        rows_by_condition: dict[str, list[dict[str, Any]]],
        controls: dict[str, str],
        latest_step: int,
    ) -> tuple[list[str], list[str], dict[str, str]]:
        available = []
        pending = []
        statuses = {}
        for family, condition in controls.items():
            row = _row_at_or_before(rows_by_condition.get(condition, []), latest_step)
            if row is None:
                pending.append(family)
                statuses[family] = "pending"
            else:
                available.append(family)
                statuses[family] = "available"
        return available, pending, statuses

    def _partial_deltas(
        self,
        rows_by_condition: dict[str, list[dict[str, Any]]],
        latest_real: dict[str, Any],
        controls: dict[str, str],
    ) -> dict[str, float | None]:
        real_loss = float(latest_real["validation_loss"])
        latest_step = int(latest_real["step"])
        deltas: dict[str, float | None] = {}
        for family, condition in controls.items():
            row = _row_at_or_before(rows_by_condition.get(condition, []), latest_step)
            deltas[family] = None if row is None else float(row["validation_loss"]) - real_loss
        return deltas

    def _matched_steps(
        self,
        rows_by_condition: dict[str, list[dict[str, Any]]],
        controls: dict[str, str],
    ) -> list[int]:
        required_conditions = [self.config.real_condition, *controls.values()]
        step_sets = []
        for condition in required_conditions:
            rows = rows_by_condition.get(condition, [])
            step_sets.append({int(row["step"]) for row in rows})
        if not step_sets:
            return []
        return sorted(set.intersection(*step_sets))

    def _final_score(
        self,
        rows_by_condition: dict[str, list[dict[str, Any]]],
        controls: dict[str, str],
        matched_steps: list[int],
    ) -> dict[str, Any]:
        real_by_step = _rows_by_step(rows_by_condition[self.config.real_condition])
        control_by_family = {
            family: _rows_by_step(rows_by_condition[condition])
            for family, condition in controls.items()
        }
        per_step = []
        deltas_by_family: dict[str, list[float]] = {family: [] for family in controls}
        attention_by_step = []
        for step in matched_steps:
            real_row = real_by_step[step]
            real_loss = float(real_row["validation_loss"])
            step_row: dict[str, Any] = {"step": step}
            for family, rows in control_by_family.items():
                delta = float(rows[step]["validation_loss"]) - real_loss
                step_row[f"real_vs_{family}_delta"] = delta
                deltas_by_family[family].append(delta)
            attention = _attention_separation_at_step(
                rows_by_condition,
                real_row,
                controls,
                exact_step=True,
            )
            step_row["attention_behavior_separation"] = attention
            if attention is not None:
                attention_by_step.append(attention)
            per_step.append(step_row)

        mean_deltas = {
            family: sum(values) / len(values) if values else None
            for family, values in deltas_by_family.items()
        }
        control_separation = _min_present(mean_deltas.values())
        return {
            "matched_steps": list(matched_steps),
            "mean_deltas": mean_deltas,
            "control_separation": control_separation,
            "control_penalty": _control_penalty(control_separation, self.config.pass_margin),
            "attention_behavior_separation": (
                sum(attention_by_step) / len(attention_by_step)
                if attention_by_step
                else None
            ),
            "per_step": per_step,
        }


def score_control_separation(
    progress_by_condition: dict[str, list[dict[str, Any]]],
    *,
    config: ControlSeparationConfig | None = None,
    current_step: int | None = None,
) -> dict[str, Any]:
    """Convenience wrapper returning a JSON-ready control-separation summary."""

    scorer = ControlSeparationScorer(config)
    return scorer.summary(scorer.score(progress_by_condition, current_step=current_step))


def _normalize_progress(
    progress_by_condition: dict[str, list[dict[str, Any]]],
    current_step: int | None,
) -> dict[str, list[dict[str, Any]]]:
    normalized = {}
    for condition, rows in progress_by_condition.items():
        clean_rows = []
        for row in rows:
            if row.get("step") is None or row.get("validation_loss") is None:
                continue
            step = int(row["step"])
            loss = float(row["validation_loss"])
            if not math.isfinite(loss):
                raise ValueError("validation_loss must be finite")
            if current_step is not None and step > current_step:
                continue
            clean_rows.append({**row, "step": step, "validation_loss": loss})
        clean_rows.sort(key=lambda item: int(item["step"]))
        normalized[condition] = clean_rows
    return normalized


def _rows_by_step(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["step"]): row for row in rows}


def _row_at_or_before(rows: list[dict[str, Any]], step: int) -> dict[str, Any] | None:
    candidates = [row for row in rows if int(row["step"]) <= step]
    if not candidates:
        return None
    return candidates[-1]


def _min_present(values: Any) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return min(present)


def _attention_separation_at_step(
    rows_by_condition: dict[str, list[dict[str, Any]]],
    real_row: dict[str, Any],
    controls: dict[str, str],
    *,
    exact_step: bool = False,
) -> float | None:
    step = int(real_row["step"])
    real_score = _first_numeric(
        real_row,
        ["attention_behavior_score", "attention_control_separation"],
    )
    control_deltas = []
    if real_score is not None:
        for condition in controls.values():
            rows = rows_by_condition.get(condition, [])
            control_row = (
                _rows_by_step(rows).get(step)
                if exact_step
                else _row_at_or_before(rows, step)
            )
            if control_row is None:
                continue
            control_score = _first_numeric(control_row, ["attention_behavior_score"])
            if control_score is not None:
                control_deltas.append(real_score - control_score)
    if control_deltas:
        return min(control_deltas)
    return _first_numeric(
        real_row,
        ["attention_behavior_separation", "attention_control_separation"],
    )


def _first_numeric(row: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def _control_penalty(control_separation: float | None, margin: float) -> float:
    if control_separation is None:
        return 0.0
    return max(0.0, margin - control_separation)


def _generic_regularization_warning(
    *,
    mode: ScoreMode,
    claim_eligibility: ClaimEligibility,
    real_vs_baseline: float | None,
    real_vs_random: float | None,
    real_vs_shuffled: float | None,
    margin: float,
) -> str:
    if claim_eligibility != "eligible_complete_controls":
        if real_vs_baseline is not None and real_vs_baseline > margin:
            return "baseline_only_signal_pending_controls"
        return "pending_controls"
    weak_controls = []
    if real_vs_random is not None and real_vs_random <= margin:
        weak_controls.append("random")
    if real_vs_shuffled is not None and real_vs_shuffled <= margin:
        weak_controls.append("shuffled")
    if weak_controls:
        return "real_not_separated_from_" + "_and_".join(weak_controls)
    if mode == "final_matched":
        return "none"
    return "pending_controls"


def _bounded(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))
