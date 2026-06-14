"""Stage-26 longer-run and multi-seed confirmation report."""

from __future__ import annotations

from typing import Any

from evaluation.controller_revision_loop import build_controller_revision_loop_report
from evaluation.decision_gate_real_geometry import build_decision_gate_real_geometry_report
from evaluation.guarded_2000_step_adaptive_run import (
    build_guarded_2000_step_adaptive_run_report,
)
from evaluation.late_window_post1000_analysis import (
    build_late_window_post1000_analysis_report,
)
from evaluation.short_smoke_failure_safety_validation import (
    build_short_smoke_failure_safety_validation_report,
)
from experiments.longer_run_multi_seed_confirmation import (
    REQUIRED_STAGE26_OUTPUTS,
    ConfirmationRunConfig,
    analyze_confirmation_replay,
    build_confirmation_run_manifest,
    confirmation_config_asdict,
    generate_confirmation_replay_rows,
)


def build_longer_run_multi_seed_confirmation_report(
    *,
    confirmation_rows: list[dict[str, Any]] | None = None,
    prerequisite_reports: dict[str, dict[str, Any]] | None = None,
    config: ConfirmationRunConfig | None = None,
) -> dict[str, Any]:
    """Build the stage-26 confirmation contract report."""

    active = config or ConfirmationRunConfig()
    active.validate()
    prerequisites = prerequisite_reports or _build_prerequisite_reports()
    rows = confirmation_rows or generate_confirmation_replay_rows(active)
    manifest = build_confirmation_run_manifest(active)
    analysis = analyze_confirmation_replay(rows, config=active)
    prerequisite_summary = _prerequisite_summary(prerequisites)
    aggregate = analysis["aggregate_confirmation_summary"]
    dominance = analysis["random_shuffled_dominance_audit"]
    profile_names = {profile["profile"] for profile in manifest["profiles"]}
    checks = {
        "required_outputs_present": set(REQUIRED_STAGE26_OUTPUTS).issubset(
            {
                "confirmation_run_manifest": manifest,
                "prerequisite_gate_summary": prerequisite_summary,
                **analysis,
            }
        ),
        "short_smoke_passed": prerequisite_summary["short_smoke"]["passed"],
        "guarded_2000_passed": prerequisite_summary["guarded_2000"]["passed"],
        "late_window_passed": prerequisite_summary["late_window"]["passed"],
        "decision_gate_passed": prerequisite_summary["decision_gate"]["passed"],
        "revision_loop_ready_for_stage26": prerequisite_summary["revision_loop"][
            "stage26_ready"
        ],
        "longer_profile_declared": active.longer_profile in profile_names,
        "multi_seed_profile_declared": active.primary_multi_seed_profile in profile_names,
        "multi_seed_has_required_seed_count": aggregate["profile_results"][
            active.primary_multi_seed_profile
        ]["seed_count"]
        >= active.min_seed_count,
        "all_required_conditions_in_manifest": all(
            set(active.conditions).issubset(profile["conditions"])
            for profile in manifest["profiles"]
        ),
        "matched_confirmation_windows_present": all(
            all(count >= 2 for count in summary["condition_points"].values())
            for summary in analysis["profile_seed_summaries"]
        ),
        "real_beats_all_controls_every_seed": all(
            summary["real_beats_all_controls"]
            for summary in analysis["profile_seed_summaries"]
        ),
        "no_random_or_shuffled_dominance": not dominance[
            "random_or_shuffled_dominance_detected"
        ],
        "longer_run_profile_passes": aggregate["longer_run_profile_passes"],
        "multi_seed_profile_passes": aggregate["multi_seed_profile_passes"],
        "lightweight_artifact_policy": (
            manifest["checkpoint_artifacts_excluded"]
            and manifest["lightweight_review_artifacts_only"]
        ),
        "anti_overclaim_boundary_present": bool(analysis["real_confirmation_boundary"]),
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "stage": "stage26_longer_run_or_multi_seed_confirmation",
        "status": status,
        "scientific_scope": (
            "confirmation contract and readiness gate; defines longer-run and "
            "multi-seed matched-control confirmation and blocks real scientific "
            "claims until actual telemetry satisfies the same rules"
        ),
        "config": confirmation_config_asdict(active),
        "required_outputs": list(REQUIRED_STAGE26_OUTPUTS),
        "checks": checks,
        "confirmation_run_manifest": manifest,
        "prerequisite_gate_summary": prerequisite_summary,
        "profile_seed_summaries": analysis["profile_seed_summaries"],
        "aggregate_confirmation_summary": aggregate,
        "random_shuffled_dominance_audit": dominance,
        "stage26_decision": analysis["stage26_decision"],
        "real_confirmation_boundary": analysis["real_confirmation_boundary"],
        "next_required_step": (
            "run_real_longer_or_multi_seed_confirmation"
            if status == "pass"
            else "fix_confirmation_prerequisites_or_contract"
        ),
    }


def _build_prerequisite_reports() -> dict[str, dict[str, Any]]:
    return {
        "short_smoke": build_short_smoke_failure_safety_validation_report(),
        "guarded_2000": build_guarded_2000_step_adaptive_run_report(),
        "late_window": build_late_window_post1000_analysis_report(),
        "decision_gate": build_decision_gate_real_geometry_report(),
        "revision_loop": build_controller_revision_loop_report(),
    }


def _prerequisite_summary(
    reports: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    revision = reports["revision_loop"]["revision"]
    return {
        "short_smoke": {
            "stage": reports["short_smoke"]["stage"],
            "status": reports["short_smoke"]["status"],
            "passed": reports["short_smoke"]["status"] == "pass",
        },
        "guarded_2000": {
            "stage": reports["guarded_2000"]["stage"],
            "status": reports["guarded_2000"]["status"],
            "passed": reports["guarded_2000"]["status"] == "pass",
        },
        "late_window": {
            "stage": reports["late_window"]["stage"],
            "status": reports["late_window"]["status"],
            "passed": reports["late_window"]["status"] == "pass",
        },
        "decision_gate": {
            "stage": reports["decision_gate"]["stage"],
            "status": reports["decision_gate"]["status"],
            "passed": reports["decision_gate"]["status"] == "pass",
        },
        "revision_loop": {
            "stage": reports["revision_loop"]["stage"],
            "status": reports["revision_loop"]["status"],
            "passed": reports["revision_loop"]["status"] == "pass",
            "stage26_ready": bool(revision["stage26_readiness"]["ready"]),
            "revision_mode": revision["revision_mode"],
        },
    }
