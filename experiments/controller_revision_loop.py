"""Controller revision loop for failed adaptive ERGT gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ControllerRevisionLoopConfig:
    require_specific_revision: bool = True
    require_validation_gate: bool = True
    require_replay_record: bool = True
    default_success_next_step: str = "longer_run_or_multi_seed_confirmation"
    default_failure_next_step: str = "apply_revision_and_rerun_short_smoke"

    def validate(self) -> None:
        if not self.default_success_next_step:
            raise ValueError("default_success_next_step must be non-empty")
        if not self.default_failure_next_step:
            raise ValueError("default_failure_next_step must be non-empty")


REVISION_CATALOG: dict[str, dict[str, Any]] = {
    "memory_starved": {
        "risk_track": "R1",
        "severity": "revision",
        "target_components": ["adaptive_memory_controller", "joint_budget_allocator"],
        "specific_revision": (
            "increase post-1000 memory_eta release, lower memory starvation penalty, "
            "and require edge-survival telemetry before re-entering the gate"
        ),
        "validation_gate": "real_memory must beat no_memory in the 1000-2000 window",
    },
    "memory_noisy": {
        "risk_track": "R1",
        "severity": "revision",
        "target_components": ["adaptive_memory_controller", "gate_floor_noise_controller"],
        "specific_revision": (
            "tighten noisy-edge filtering, decay unstable edges faster, and log "
            "noise-risk pressure in the injected-evidence ledger"
        ),
        "validation_gate": "noise risk decreases while memory stability does not collapse",
    },
    "memory_rigid": {
        "risk_track": "R1",
        "severity": "revision",
        "target_components": ["adaptive_memory_controller", "attention_rigidity_monitor"],
        "specific_revision": (
            "lower effective memory decay, add turnover pressure, and freeze memory "
            "growth when rigidity rises without validation gain"
        ),
        "validation_gate": "memory rigidity falls without losing late-window advantage",
    },
    "memory_not_stabilizing": {
        "risk_track": "R1",
        "severity": "revision",
        "target_components": ["adaptive_memory_controller"],
        "specific_revision": (
            "separate stable-memory gain from instantaneous geometry by increasing "
            "memory persistence only after attention and loss slope agree"
        ),
        "validation_gate": "stable memory beats instantaneous real geometry",
    },
    "future_leak_detected": {
        "risk_track": "R1",
        "severity": "hard_stop",
        "target_components": ["causal_reachability_controller", "causal_masking"],
        "specific_revision": (
            "abort claim path, repair causal reachability and mask enforcement, and "
            "rerun from smoke before any long run"
        ),
        "validation_gate": "future_leak_score remains zero for every late-window row",
    },
    "memory_or_causality_unresolved": {
        "risk_track": "R1",
        "severity": "revision",
        "target_components": ["memory_state_instrumentation", "causal_reachability_controller"],
        "specific_revision": (
            "add missing memory-scope and causal-validity telemetry before changing "
            "alpha or loss objectives"
        ),
        "validation_gate": "R1 audit reports memory gain and causal validity together",
    },
    "geometry_flattened": {
        "risk_track": "R2",
        "severity": "revision",
        "target_components": ["distance_scale_controller"],
        "specific_revision": (
            "relax distance normalization or clipping when contrast retention falls "
            "while keeping geo_to_qk inside the joint budget"
        ),
        "validation_gate": "distance_contrast_retention recovers above threshold",
    },
    "normalization_erased_contrast": {
        "risk_track": "R2",
        "severity": "revision",
        "target_components": ["distance_scale_controller", "joint_budget_allocator"],
        "specific_revision": (
            "revise z-score/clamp scale, log contrast before and after normalization, "
            "and block gates where geometry is numerically flattened"
        ),
        "validation_gate": "R2 audit clears contrast and geo_to_qk bounds",
    },
    "alpha_underpowered": {
        "risk_track": "R2",
        "severity": "revision",
        "target_components": ["adaptive_alpha_controller_v2", "joint_budget_allocator"],
        "specific_revision": (
            "release alpha growth only when multi-point loss slope and attention "
            "separation support geometry"
        ),
        "validation_gate": "geo_to_qk increases without attention collapse",
    },
    "alpha_overpowering": {
        "risk_track": "R2",
        "severity": "revision",
        "target_components": ["adaptive_alpha_controller_v2", "attention_rigidity_monitor"],
        "specific_revision": (
            "reduce alpha budget, add takeover pressure, and freeze alpha when "
            "attention max-probability or rigidity rises"
        ),
        "validation_gate": "attention remains noncollapsed while loss improves",
    },
    "causal_reach_too_tight": {
        "risk_track": "R1",
        "severity": "revision",
        "target_components": ["causal_reachability_controller"],
        "specific_revision": (
            "expand causal reach one level at a time when memory is stable and "
            "future leakage remains zero"
        ),
        "validation_gate": "reach expansion improves real over no-memory without leakage",
    },
    "causal_reach_too_loose": {
        "risk_track": "R1",
        "severity": "hard_stop",
        "target_components": ["causal_reachability_controller"],
        "specific_revision": (
            "tighten reachability threshold and rerun causal leak audit before "
            "training controls"
        ),
        "validation_gate": "future_leak_score is zero and locality remains interpretable",
    },
    "control_regularization_dominance": {
        "risk_track": "R2",
        "severity": "revision",
        "target_components": ["control_family_fairness_audit", "joint_budget_allocator"],
        "specific_revision": (
            "treat random/shuffled gains as generic regularization, reduce claim "
            "credit to zero, audit data/RNG isolation, and revise geometry-specific "
            "signals before another guarded run"
        ),
        "validation_gate": "real beats random and shuffled in the 1000-2000 window",
    },
    "random_dominates_real": {
        "risk_track": "R2",
        "severity": "revision",
        "target_components": ["control_family_fairness_audit", "joint_budget_allocator"],
        "specific_revision": (
            "treat the random-control win as claim blocking, audit RNG/data "
            "isolation, and reduce relation-claim credit until real beats random"
        ),
        "validation_gate": "real beats random_memory_d in the 1000-2000 window",
    },
    "shuffled_dominates_real": {
        "risk_track": "R2",
        "severity": "revision",
        "target_components": ["control_family_fairness_audit", "distance_scale_controller"],
        "specific_revision": (
            "treat the shuffled-control win as distribution-bias dominance and "
            "revise geometry so edge placement matters beyond distance statistics"
        ),
        "validation_gate": "real beats shuffled_memory_d in the 1000-2000 window",
    },
    "no_memory_matches_or_beats_real": {
        "risk_track": "R1",
        "severity": "revision",
        "target_components": ["adaptive_memory_controller"],
        "specific_revision": (
            "block memory credit and revise eta/decay until recurrent memory adds "
            "late-window value beyond no-memory real geometry"
        ),
        "validation_gate": "real memory beats no_memory_real_d",
    },
    "instantaneous_matches_or_beats_stable_memory": {
        "risk_track": "R1",
        "severity": "revision",
        "target_components": ["adaptive_memory_controller"],
        "specific_revision": (
            "block stable-memory credit and revise persistence/turnover until "
            "stable memory beats instantaneous real geometry"
        ),
        "validation_gate": "stable real memory beats instantaneous_real_d",
    },
    "attention_uniformity_drift": {
        "risk_track": "R3",
        "severity": "revision",
        "target_components": ["attention_rigidity_monitor", "distance_scale_controller"],
        "specific_revision": (
            "restore relational contrast and reduce uniform attention drift while "
            "preserving late-window loss trend"
        ),
        "validation_gate": "attention entropy no longer drifts toward uniformity",
    },
    "attention_control_like": {
        "risk_track": "R3",
        "severity": "revision",
        "target_components": ["meta_control_attention_observer", "joint_budget_allocator"],
        "specific_revision": (
            "lower scientific credit, reweight controller evidence toward relation-"
            "specific signals, and require attention separation before claim gates"
        ),
        "validation_gate": "real attention behavior separates from every control",
    },
    "attention_behavior_not_separated_from_controls": {
        "risk_track": "R3",
        "severity": "revision",
        "target_components": ["meta_control_attention_observer", "attention_rigidity_monitor"],
        "specific_revision": (
            "block attention-based claim support and revise controller weighting "
            "until real attention behavior separates from all controls"
        ),
        "validation_gate": "minimum real-vs-control attention advantage is positive",
    },
    "attention_head_lock_in": {
        "risk_track": "R3",
        "severity": "revision",
        "target_components": ["attention_rigidity_monitor"],
        "specific_revision": (
            "add head/layer diversity pressure and freeze geometry growth when one "
            "head or layer monopolizes attention"
        ),
        "validation_gate": "head/layer diversity recovers without losing real advantage",
    },
    "meta_control_attention_misweighted": {
        "risk_track": "R3",
        "severity": "revision",
        "target_components": ["meta_control_attention_observer"],
        "specific_revision": (
            "calibrate meta-control attention weights against loss slope, control "
            "separation, memory stability, and attention safety"
        ),
        "validation_gate": "meta-control top signals match observed successful revisions",
    },
    "controller_conflict_unresolved": {
        "risk_track": "R2",
        "severity": "revision",
        "target_components": ["joint_budget_allocator", "open_adaptive_relational_trainer"],
        "specific_revision": (
            "add conflict arbitration so alpha, memory, gate, reach, and normalization "
            "cannot request incompatible moves without a replayable decision"
        ),
        "validation_gate": "controller_conflict_score falls and replay records explain changes",
    },
    "baseline_only_evidence_insufficient": {
        "risk_track": "R2",
        "severity": "revision",
        "target_components": ["control_separation_scoring"],
        "specific_revision": (
            "block claim credit until random, shuffled, no-memory, instantaneous, "
            "alpha-zero, and baseline comparisons are all available"
        ),
        "validation_gate": "final matched controls exist before any claim decision",
    },
    "late_window_not_ready": {
        "risk_track": "R2",
        "severity": "revision",
        "target_components": ["guarded_2000_step_adaptive_run"],
        "specific_revision": (
            "rerun guarded profile until every required condition has matched "
            "100-step rows through step 2000"
        ),
        "validation_gate": "all required late windows have matched control points",
    },
    "relation_specific_advantage_not_established": {
        "risk_track": "R2",
        "severity": "revision",
        "target_components": ["random_shuffled_no_memory_attribution"],
        "specific_revision": (
            "revise relation-specific geometry source before adding auxiliary loss "
            "or longer runs"
        ),
        "validation_gate": "relation_specific_advantage_estimate becomes positive",
    },
}


REQUIRED_REVISION_LOOP_OUTPUTS = [
    "revision_mode",
    "failure_labels",
    "revision_plan",
    "unmapped_failure_labels",
    "controller_change_summary",
    "rerun_protocol",
    "decision_replay_record",
    "stage26_readiness",
]


def build_controller_revision_loop(
    decision_gate_report: dict[str, Any],
    *,
    config: ControllerRevisionLoopConfig | None = None,
) -> dict[str, Any]:
    """Map failed gate labels to concrete controller revisions."""

    active = config or ControllerRevisionLoopConfig()
    active.validate()
    gate = decision_gate_report["gate"]
    failure_labels = list(gate.get("failure_labels", []))
    if decision_gate_report.get("status") == "pass" and not failure_labels:
        return _noop_revision(decision_gate_report, active)

    revision_plan = [_revision_for_label(label) for label in failure_labels]
    unmapped = [
        label
        for label, revision in zip(failure_labels, revision_plan, strict=True)
        if revision is None
    ]
    mapped_revisions = [revision for revision in revision_plan if revision is not None]
    ordered = sorted(
        mapped_revisions,
        key=lambda item: _severity_rank(item["severity"]),
    )
    checks = {
        "every_failure_label_mapped": not unmapped,
        "specific_revisions_present": all(
            bool(revision["specific_revision"]) for revision in ordered
        ),
        "validation_gates_present": all(
            bool(revision["validation_gate"]) for revision in ordered
        ),
        "replay_record_present": True,
        "stage26_blocked_until_revision": bool(ordered),
    }
    return {
        "revision_mode": "apply_revisions",
        "failure_labels": failure_labels,
        "revision_plan": ordered,
        "unmapped_failure_labels": unmapped,
        "controller_change_summary": _controller_change_summary(ordered),
        "rerun_protocol": _rerun_protocol(ordered, active),
        "decision_replay_record": _decision_replay_record(
            decision_gate_report,
            failure_labels,
            ordered,
        ),
        "stage26_readiness": {
            "ready": False,
            "reason": "decision gate failed; apply revisions and rerun validation",
        },
        "checks": checks,
        "next_required_step": active.default_failure_next_step,
    }


def _noop_revision(
    decision_gate_report: dict[str, Any],
    config: ControllerRevisionLoopConfig,
) -> dict[str, Any]:
    checks = {
        "every_failure_label_mapped": True,
        "specific_revisions_present": True,
        "validation_gates_present": True,
        "replay_record_present": True,
        "stage26_blocked_until_revision": False,
    }
    return {
        "revision_mode": "noop_audit",
        "failure_labels": [],
        "revision_plan": [],
        "unmapped_failure_labels": [],
        "controller_change_summary": {},
        "rerun_protocol": {
            "required": False,
            "next_validation": config.default_success_next_step,
            "reason": "stage-24 gate passed with no active failure labels",
        },
        "decision_replay_record": {
            "source_stage": decision_gate_report["stage"],
            "source_status": decision_gate_report["status"],
            "source_decision": decision_gate_report["gate"]["decision"],
            "action": "no_controller_revision_required",
        },
        "stage26_readiness": {
            "ready": True,
            "reason": "no unresolved R1/R2/R3 blocker remains in the guarded contract",
        },
        "checks": checks,
        "next_required_step": config.default_success_next_step,
    }


def _revision_for_label(label: str) -> dict[str, Any] | None:
    item = REVISION_CATALOG.get(label)
    if item is None:
        return None
    return {
        "failure_label": label,
        "risk_track": item["risk_track"],
        "severity": item["severity"],
        "target_components": list(item["target_components"]),
        "specific_revision": item["specific_revision"],
        "validation_gate": item["validation_gate"],
        "requires_short_smoke": item["severity"] == "hard_stop",
    }


def _controller_change_summary(revisions: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for revision in revisions:
        for component in revision["target_components"]:
            summary.setdefault(
                component,
                {
                    "failure_labels": [],
                    "revision_count": 0,
                    "highest_severity": revision["severity"],
                },
            )
            summary[component]["failure_labels"].append(revision["failure_label"])
            summary[component]["revision_count"] += 1
            if _severity_rank(revision["severity"]) < _severity_rank(
                summary[component]["highest_severity"]
            ):
                summary[component]["highest_severity"] = revision["severity"]
    return summary


def _rerun_protocol(
    revisions: list[dict[str, Any]],
    config: ControllerRevisionLoopConfig,
) -> dict[str, Any]:
    hard_stop = any(revision["severity"] == "hard_stop" for revision in revisions)
    return {
        "required": True,
        "start_from": "short_smoke" if hard_stop else "guarded_2000_step_adaptive_run",
        "next_validation": config.default_failure_next_step,
        "hard_stop_present": hard_stop,
        "validation_gates": [revision["validation_gate"] for revision in revisions],
    }


def _decision_replay_record(
    decision_gate_report: dict[str, Any],
    labels: list[str],
    revisions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source_stage": decision_gate_report["stage"],
        "source_status": decision_gate_report["status"],
        "source_decision": decision_gate_report["gate"]["decision"],
        "failure_labels": list(labels),
        "revision_labels": [revision["failure_label"] for revision in revisions],
        "rule": "every failed gate label must map to a concrete revision before rerun",
    }


def _severity_rank(severity: str) -> int:
    ranks = {"hard_stop": 0, "revision": 1, "pressure": 2}
    return ranks.get(severity, 99)
