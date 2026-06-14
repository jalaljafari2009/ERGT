from __future__ import annotations

import json
from pathlib import Path

from experiments.guarded_adaptive_training_notebook import (
    PROFILE_SETTINGS,
    REPORT_BUNDLE_NAME,
)


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
