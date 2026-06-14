"""Contract report for the ERGT-03 adaptive notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NOTEBOOK_PATH = Path("notebooks/ERGT_03_Adaptive_Relational_Control.ipynb")
EXPECTED_BUNDLE_NAME = "ergt_03_adaptive_control_report_bundle.zip"
EXPECTED_LOCAL_REVIEW_PATH = (
    r"C:\Users\Administrator\Downloads\ergt_03_adaptive_control_report_bundle.zip"
)

REQUIRED_SOURCE_MARKERS = {
    "adaptive_smoke_profile": 'RUN_PROFILE = "adaptive_smoke"',
    "guarded_2000_profile_declared": '"adaptive_2000_guarded"',
    "fixed_bundle_name": EXPECTED_BUNDLE_NAME,
    "default_local_review_path": EXPECTED_LOCAL_REVIEW_PATH,
    "auto_shutdown_flag": "AUTO_SHUTDOWN_COLAB_RUNTIME = True",
    "auto_shutdown_function": "shutdown_colab_runtime_if_requested",
    "lightweight_export_function": "export_report_bundle",
    "checkpoint_exclusion": '"checkpoints/"',
    "pt_exclusion": '"*.pt"',
    "ckpt_exclusion": '"*.ckpt"',
    "contract_tests": "tests/test_live_100_step_diagnostic_table.py",
    "trainer_contract_report": (
        "experiments/create_open_adaptive_relational_control_trainer_report.py"
    ),
    "live_table_contract_report": (
        "experiments/create_live_100_step_diagnostic_table_report.py"
    ),
    "adaptive_trainer": "run_open_adaptive_control_trainer",
    "live_rows": "live_diagnostic_rows",
    "live_tables": "live_diagnostic_tables",
    "live_plot_payloads": "live_diagnostic_plot_payloads",
    "fail_fast_report": "fail_fast_report.json",
    "future_leak_fail_fast": "future_leak_score",
    "stage_summary": "stage19_notebook_summary.json",
}


def build_adaptive_notebook_ergt_03_report(
    notebook_path: Path | str = NOTEBOOK_PATH,
) -> dict[str, Any]:
    """Validate the notebook-level execution contract for stage 19."""

    path = Path(notebook_path)
    notebook_exists = path.exists()
    notebook: dict[str, Any] = {}
    source = ""
    parse_error: str | None = None
    if notebook_exists:
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            source = _joined_source(notebook)
        except json.JSONDecodeError as exc:
            parse_error = str(exc)

    marker_checks = {
        name: marker in source for name, marker in REQUIRED_SOURCE_MARKERS.items()
    }
    cells = notebook.get("cells", []) if isinstance(notebook, dict) else []
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]
    kernelspec = notebook.get("metadata", {}).get("kernelspec", {})
    checks = {
        "notebook_exists": notebook_exists,
        "notebook_json_parses": notebook_exists and parse_error is None,
        "has_markdown_context": bool(markdown_cells),
        "has_multiple_code_cells": len(code_cells) >= 5,
        "python3_kernel_declared": kernelspec.get("name") == "python3",
        "fixed_bundle_name_declared": marker_checks["fixed_bundle_name"],
        "default_local_review_path_declared": marker_checks[
            "default_local_review_path"
        ],
        "auto_shutdown_cell_present": marker_checks["auto_shutdown_flag"]
        and marker_checks["auto_shutdown_function"],
        "lightweight_zip_excludes_checkpoints": marker_checks["lightweight_export_function"]
        and marker_checks["checkpoint_exclusion"]
        and marker_checks["pt_exclusion"]
        and marker_checks["ckpt_exclusion"],
        "fail_fast_report_present": marker_checks["fail_fast_report"]
        and marker_checks["future_leak_fail_fast"],
        "live_100_step_display_present": marker_checks["live_rows"]
        and marker_checks["live_tables"]
        and marker_checks["live_plot_payloads"],
        "preflight_contracts_present": marker_checks["contract_tests"]
        and marker_checks["trainer_contract_report"]
        and marker_checks["live_table_contract_report"],
        "short_smoke_profile_is_default": marker_checks["adaptive_smoke_profile"],
        "guarded_profile_is_declared": marker_checks["guarded_2000_profile_declared"],
        "adaptive_trainer_invoked": marker_checks["adaptive_trainer"],
        "stage_summary_written": marker_checks["stage_summary"],
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "stage": "stage19_adaptive_notebook_ergt_03",
        "status": status,
        "scientific_scope": (
            "notebook execution wrapper for adaptive relational control; validates "
            "safe smoke execution, live 100-step display, fail-fast export, "
            "fixed report bundle naming, and checkpoint exclusion"
        ),
        "notebook_path": str(path),
        "expected_bundle_name": EXPECTED_BUNDLE_NAME,
        "expected_default_local_review_path": EXPECTED_LOCAL_REVIEW_PATH,
        "checks": checks,
        "source_marker_checks": marker_checks,
        "parse_error": parse_error,
        "required_source_markers": REQUIRED_SOURCE_MARKERS,
        "next_required_step": (
            "short_smoke_and_failure_safety_validation"
            if status == "pass"
            else "fix_adaptive_notebook_ergt_03"
        ),
    }


def _joined_source(notebook: dict[str, Any]) -> str:
    parts: list[str] = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", [])
        if isinstance(source, list):
            parts.extend(str(line) for line in source)
        elif isinstance(source, str):
            parts.append(source)
    return "".join(parts)
