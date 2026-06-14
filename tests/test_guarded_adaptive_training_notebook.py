from __future__ import annotations

import json
from pathlib import Path

from experiments.guarded_adaptive_training_notebook import (
    PROFILE_SETTINGS,
    REPORT_BUNDLE_NAME,
    NotebookRuntime,
    ergt_config,
)
from layers.relational_graph import RelationalGraphConfig


def test_ergt04_notebook_json_and_contract_markers() -> None:
    notebook_path = Path("notebooks/ERGT_04_Guarded_Adaptive_Training.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert notebook["nbformat"] == 4
    assert "run_notebook" in source
    assert "guarded_2000_real_training" in source
    assert "real training, not synthetic telemetry" in source
    assert REPORT_BUNDLE_NAME in source


def test_guarded_profile_runs_real_controls_with_short_alpha_zero() -> None:
    profile = PROFILE_SETTINGS["guarded_2000_real_training"]

    assert profile["max_steps"] == 2000
    assert profile["alpha_zero_steps"] < profile["max_steps"]
    assert profile["conditions"] == [
        "baseline",
        "alpha_zero_short_check",
        "real_memory_d_adaptive",
        "random_memory_d_adaptive",
        "shuffled_memory_d_adaptive",
        "no_memory_real_d_adaptive",
        "instantaneous_real_d_adaptive",
    ]


def test_ergt04_relational_graph_config_matches_runtime_schema(tmp_path: Path) -> None:
    runtime = NotebookRuntime(
        project_root=tmp_path,
        run_profile="real_smoke_200",
        device="cpu",
        seed=2027,
        run_training=False,
        run_preflight_tests=False,
        prepare_data_if_missing=False,
        auto_shutdown_colab_runtime=False,
        auto_shutdown_after_success=False,
        auto_shutdown_on_failure=False,
        auto_shutdown_delay_seconds=0,
    )
    config = ergt_config(
        runtime,
        tmp_path,
        "alpha_zero_short_check",
        "zero_d",
        100,
        adaptive=False,
    )

    RelationalGraphConfig(**config["relational_graph"])
