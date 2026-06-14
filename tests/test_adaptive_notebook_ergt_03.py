import json

from evaluation.adaptive_notebook_ergt_03 import (
    EXPECTED_BUNDLE_NAME,
    EXPECTED_LOCAL_REVIEW_PATH,
    build_adaptive_notebook_ergt_03_report,
)


def test_adaptive_notebook_ergt_03_report_passes() -> None:
    report = build_adaptive_notebook_ergt_03_report()

    assert report["status"] == "pass"
    assert report["next_required_step"] == "short_smoke_and_failure_safety_validation"
    json.dumps(report)


def test_adaptive_notebook_uses_fixed_bundle_and_download_path() -> None:
    report = build_adaptive_notebook_ergt_03_report()

    assert report["expected_bundle_name"] == EXPECTED_BUNDLE_NAME
    assert report["expected_default_local_review_path"] == EXPECTED_LOCAL_REVIEW_PATH
    assert report["checks"]["fixed_bundle_name_declared"]
    assert report["checks"]["default_local_review_path_declared"]


def test_adaptive_notebook_bootstraps_repo_in_colab() -> None:
    report = build_adaptive_notebook_ergt_03_report()

    assert report["checks"]["colab_repo_bootstrap_present"]
    assert report["source_marker_checks"]["repo_clone_url"]
    assert report["source_marker_checks"]["git_clone_repo"]
    assert report["source_marker_checks"]["project_root_sys_path"]


def test_adaptive_notebook_has_failure_safety_controls() -> None:
    report = build_adaptive_notebook_ergt_03_report()

    assert report["checks"]["auto_shutdown_cell_present"]
    assert report["checks"]["fail_fast_report_present"]
    assert report["checks"]["preflight_contracts_present"]


def test_adaptive_notebook_exports_lightweight_live_artifacts_only() -> None:
    report = build_adaptive_notebook_ergt_03_report()

    assert report["checks"]["live_100_step_display_present"]
    assert report["checks"]["live_100_step_streaming_callback_present"]
    assert report["checks"]["live_parameter_columns_present"]
    assert report["checks"]["lightweight_zip_excludes_checkpoints"]
    assert report["source_marker_checks"]["live_rows"]
    assert report["source_marker_checks"]["live_tables"]
    assert report["source_marker_checks"]["live_plot_payloads"]
