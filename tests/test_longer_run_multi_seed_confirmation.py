import json

from evaluation.longer_run_multi_seed_confirmation import (
    build_longer_run_multi_seed_confirmation_report,
)
from experiments.longer_run_multi_seed_confirmation import (
    REQUIRED_STAGE26_OUTPUTS,
    ConfirmationRunConfig,
    analyze_confirmation_replay,
    build_confirmation_run_manifest,
    generate_confirmation_replay_rows,
)


def test_longer_run_multi_seed_confirmation_report_passes() -> None:
    report = build_longer_run_multi_seed_confirmation_report()

    assert report["status"] == "pass"
    assert report["stage26_decision"] == "confirmation_contract_ready"
    assert report["next_required_step"] == "run_real_longer_or_multi_seed_confirmation"
    json.dumps(report)


def test_confirmation_manifest_declares_profiles_seeds_and_conditions() -> None:
    config = ConfirmationRunConfig()
    manifest = build_confirmation_run_manifest(config)
    profiles = {profile["profile"]: profile for profile in manifest["profiles"]}

    assert set(REQUIRED_STAGE26_OUTPUTS)
    assert "longer_single_seed_5000" in profiles
    assert "multi_seed_2000" in profiles
    assert profiles["multi_seed_2000"]["seeds"] == [2027, 2028, 2029]
    assert set(config.conditions).issubset(profiles["multi_seed_2000"]["conditions"])
    assert manifest["checkpoint_artifacts_excluded"]
    assert manifest["lightweight_review_artifacts_only"]


def test_confirmation_analysis_requires_real_to_beat_controls_every_seed() -> None:
    rows = generate_confirmation_replay_rows(ConfirmationRunConfig())
    analysis = analyze_confirmation_replay(rows)

    assert analysis["aggregate_confirmation_summary"]["multi_seed_profile_passes"]
    assert analysis["aggregate_confirmation_summary"]["longer_run_profile_passes"]
    assert analysis["aggregate_confirmation_summary"]["all_profiles_pass"]
    assert not analysis["random_shuffled_dominance_audit"][
        "random_or_shuffled_dominance_detected"
    ]
    assert all(
        summary["real_beats_all_controls"]
        for summary in analysis["profile_seed_summaries"]
    )


def test_random_dominance_in_one_seed_blocks_confirmation() -> None:
    rows = generate_confirmation_replay_rows(ConfirmationRunConfig())
    for row in rows:
        if (
            row["profile"] == "multi_seed_2000"
            and row["seed"] == 2028
            and row["condition"] == "random_memory_d"
            and row["step"] >= 1000
        ):
            row["validation_loss"] = row["validation_loss"] - 0.30

    report = build_longer_run_multi_seed_confirmation_report(confirmation_rows=rows)

    assert report["status"] == "fail"
    assert report["stage26_decision"] == "confirmation_contract_blocked"
    assert not report["checks"]["no_random_or_shuffled_dominance"]
    assert report["random_shuffled_dominance_audit"]["random_dominance_cases"]


def test_stage26_blocks_when_prerequisite_not_ready() -> None:
    reports = {
        "short_smoke": {"stage": "stage20", "status": "pass"},
        "guarded_2000": {"stage": "stage21", "status": "pass"},
        "late_window": {"stage": "stage22", "status": "pass"},
        "decision_gate": {"stage": "stage24", "status": "pass"},
        "revision_loop": {
            "stage": "stage25",
            "status": "pass",
            "revision": {
                "stage26_readiness": {"ready": False},
                "revision_mode": "apply_revisions",
            },
        },
    }

    report = build_longer_run_multi_seed_confirmation_report(
        prerequisite_reports=reports
    )

    assert report["status"] == "fail"
    assert not report["checks"]["revision_loop_ready_for_stage26"]
