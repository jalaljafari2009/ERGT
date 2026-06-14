"""Colab-facing real training runner for ERGT-04.

This module is intentionally notebook-friendly: it streams subprocess output
line by line, writes fixed lightweight review artifacts, and can disconnect a
Colab runtime after completion or handled failure.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - notebook display convenience
    from IPython.display import Markdown, display
except Exception:  # pragma: no cover
    Markdown = None
    display = None


REPORT_BUNDLE_NAME = "ergt_04_guarded_adaptive_training_report_bundle.zip"
DEFAULT_LOCAL_REVIEW_PATH = (
    r"C:\Users\Administrator\Downloads\ergt_04_guarded_adaptive_training_report_bundle.zip"
)
GIT_REPO_URL = "https://github.com/jalaljafari2009/ERGT.git"
LIGHTWEIGHT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".csv", ".png"}
MAX_BUNDLE_FILE_BYTES = 8 * 1024 * 1024
EXCLUDED_ARTIFACT_PATTERNS = [
    "checkpoints/",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "optimizer_state*",
]


PROFILE_SETTINGS: dict[str, dict[str, Any]] = {
    "real_smoke_200": {
        "context_length": 128,
        "max_steps": 200,
        "alpha_zero_steps": 100,
        "eval_interval": 100,
        "max_eval_batches": 8,
        "batch_size": 8,
        "n_layers": 4,
        "n_heads": 4,
        "hidden_dim": 256,
        "ffn_dim": 1024,
        "warmup_steps": 40,
        "alpha_warmup_steps": 100,
        "late_window": [100, 200],
        "per_run_timeout_minutes": 45,
        "precision": "auto",
        "dataloader_num_workers": 2,
        "conditions": [
            "baseline",
            "alpha_zero_short_check",
            "real_memory_d_adaptive",
            "random_memory_d_adaptive",
            "shuffled_memory_d_adaptive",
        ],
    },
    "guarded_2000_real_training": {
        "context_length": 256,
        "max_steps": 2000,
        "alpha_zero_steps": 200,
        "eval_interval": 100,
        "max_eval_batches": None,
        "batch_size": 16,
        "n_layers": 6,
        "n_heads": 8,
        "hidden_dim": 512,
        "ffn_dim": 2048,
        "warmup_steps": 200,
        "alpha_warmup_steps": 1000,
        "late_window": [1000, 2000],
        "per_run_timeout_minutes": 240,
        "precision": "auto",
        "dataloader_num_workers": 2,
        "conditions": [
            "baseline",
            "alpha_zero_short_check",
            "real_memory_d_adaptive",
            "random_memory_d_adaptive",
            "shuffled_memory_d_adaptive",
            "no_memory_real_d_adaptive",
            "instantaneous_real_d_adaptive",
        ],
    },
}


@dataclass(frozen=True)
class NotebookRuntime:
    project_root: Path
    run_profile: str
    device: str
    seed: int
    run_training: bool
    run_preflight_tests: bool
    prepare_data_if_missing: bool
    auto_shutdown_colab_runtime: bool
    auto_shutdown_after_success: bool
    auto_shutdown_on_failure: bool
    auto_shutdown_delay_seconds: int

    @property
    def profile(self) -> dict[str, Any]:
        if self.run_profile not in PROFILE_SETTINGS:
            raise ValueError(f"Unknown RUN_PROFILE: {self.run_profile}")
        return PROFILE_SETTINGS[self.run_profile]


def run_notebook(
    *,
    run_profile: str = "guarded_2000_real_training",
    device: str | None = None,
    seed: int = 2027,
    run_training: bool = True,
    run_preflight_tests: bool = True,
    prepare_data_if_missing: bool = True,
    auto_shutdown_colab_runtime: bool = True,
    auto_shutdown_after_success: bool = True,
    auto_shutdown_on_failure: bool = True,
    auto_shutdown_delay_seconds: int = 30,
) -> dict[str, Any]:
    """Run the ERGT-04 guarded real training notebook flow."""

    project_root = prepare_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    runtime = NotebookRuntime(
        project_root=project_root,
        run_profile=run_profile,
        device=device or default_device(),
        seed=seed,
        run_training=run_training,
        run_preflight_tests=run_preflight_tests,
        prepare_data_if_missing=prepare_data_if_missing,
        auto_shutdown_colab_runtime=auto_shutdown_colab_runtime,
        auto_shutdown_after_success=auto_shutdown_after_success,
        auto_shutdown_on_failure=auto_shutdown_on_failure,
        auto_shutdown_delay_seconds=auto_shutdown_delay_seconds,
    )

    run_root = make_run_root(project_root, runtime.run_profile)
    config_dir = run_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    git_commit = git_head(project_root)
    write_json(
        run_root / "notebook_runtime_config.json",
        {
            "run_profile": runtime.run_profile,
            "profile": runtime.profile,
            "device": runtime.device,
            "seed": runtime.seed,
            "git_commit": git_commit,
            "real_training": True,
            "synthetic_rows_used": False,
            "report_bundle_name": REPORT_BUNDLE_NAME,
            "default_local_review_path": DEFAULT_LOCAL_REVIEW_PATH,
            "auto_shutdown_colab_runtime": runtime.auto_shutdown_colab_runtime,
        },
    )

    print_startup(runtime, run_root, git_commit)
    try:
        preflight_records = run_preflight(runtime, run_root)
        write_json(run_root / "preflight_records.json", preflight_records)
        run_plan = build_run_plan(runtime, run_root, config_dir)
        training_records = run_training_plan(runtime, run_root, run_plan)
        flat_rows = write_review_artifacts(runtime, run_root, run_plan)
        summary = write_summary(
            runtime,
            run_root,
            run_plan,
            flat_rows,
            git_commit=git_commit,
            training_records=training_records,
        )
        bundle_path = export_report_bundle(runtime, run_root, "completed")
        summary["bundle_path"] = str(bundle_path)
        print(json.dumps(summary, indent=2, sort_keys=True))
        print("Fixed output bundle name:", REPORT_BUNDLE_NAME)
        print("Default local review path after Colab download:", DEFAULT_LOCAL_REVIEW_PATH)
        shutdown_colab_runtime_if_requested(runtime, "completed", failed=False)
        return summary
    except BaseException:
        export_report_bundle(runtime, run_root, "failure")
        shutdown_colab_runtime_if_requested(runtime, "failure", failed=True)
        raise


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def running_in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def prepare_project_root() -> Path:
    current_root = Path.cwd()
    if (current_root / "experiments").exists() and (current_root / "models").exists():
        return current_root
    colab_root = Path("/content/ERGT")
    if Path("/content").exists():
        if not colab_root.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", GIT_REPO_URL, str(colab_root)],
                check=True,
            )
        elif (colab_root / ".git").exists():
            subprocess.run(["git", "-C", str(colab_root), "pull", "--ff-only"], check=False)
        if (colab_root / "experiments").exists():
            os.chdir(colab_root)
            return colab_root
    raise RuntimeError(f"ERGT project root was not found. Expected {GIT_REPO_URL}.")


def make_run_root(project_root: Path, run_profile: str) -> Path:
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{run_profile}"
    run_root = project_root / "runs" / "notebook_04_guarded_adaptive_training" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def git_head(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip()
    except Exception:
        return "unknown"


def print_startup(runtime: NotebookRuntime, run_root: Path, git_commit: str) -> None:
    print(f"Repository URL: {GIT_REPO_URL}")
    print(f"Project root: {runtime.project_root}")
    print(f"Git commit: {git_commit}")
    print(f"Device: {runtime.device}")
    print(f"Profile: {runtime.run_profile}")
    print(f"Run root: {run_root}")
    print(f"Fixed bundle name: {REPORT_BUNDLE_NAME}")
    print(f"Default local review path: {DEFAULT_LOCAL_REVIEW_PATH}")
    print(json.dumps(runtime.profile, indent=2, sort_keys=True))


def run_preflight(runtime: NotebookRuntime, run_root: Path) -> list[dict[str, Any]]:
    data_dir = (
        runtime.project_root
        / "data"
        / "processed"
        / f"wikitext2_gpt2_ctx{runtime.profile['context_length']}"
    )
    records: list[dict[str, Any]] = []
    if runtime.prepare_data_if_missing and not (data_dir / "metadata.json").exists():
        records.append(
            run_streaming_cmd(
                [
                    sys.executable,
                    "scripts/prepare_wikitext2.py",
                    "--output-dir",
                    data_dir,
                    "--context-length",
                    str(runtime.profile["context_length"]),
                ],
                cwd=runtime.project_root,
                timeout_sec=60 * 60,
            )
        )
    if not (data_dir / "metadata.json").exists():
        raise FileNotFoundError(
            f"Prepared dataset metadata not found: {data_dir / 'metadata.json'}"
        )
    if runtime.run_preflight_tests:
        records.append(
            run_streaming_cmd(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_geo_attention.py",
                    "tests/test_adaptive_alpha.py",
                    "tests/test_ergt_v1_training.py",
                    "tests/test_control_separation_scoring.py",
                ],
                cwd=runtime.project_root,
                timeout_sec=15 * 60,
            )
        )
    print("Preflight complete.")
    write_json(run_root / "data_contract.json", {"data_dir": str(data_dir)})
    return records


def build_run_plan(
    runtime: NotebookRuntime, run_root: Path, config_dir: Path
) -> list[dict[str, Any]]:
    condition_specs = {
        "baseline": {
            "script": "experiments/train_baseline.py",
            "config": baseline_config(runtime, run_root, "baseline", runtime.profile["max_steps"]),
            "reference_progress": None,
        },
        "alpha_zero_short_check": {
            "script": "experiments/train_ergt_v1.py",
            "config": ergt_config(
                runtime,
                run_root,
                "alpha_zero_short_check",
                "zero_d",
                runtime.profile["alpha_zero_steps"],
                adaptive=False,
            ),
            "reference_progress": None,
        },
        "real_memory_d_adaptive": {
            "script": "experiments/train_ergt_adaptive_alpha.py",
            "config": ergt_config(
                runtime,
                run_root,
                "real_memory_d_adaptive",
                "real_stable_causal_d",
                runtime.profile["max_steps"],
                adaptive=True,
            ),
            "reference_progress": run_root / "baseline" / "progress_log.jsonl",
        },
        "random_memory_d_adaptive": {
            "script": "experiments/train_ergt_adaptive_alpha.py",
            "config": ergt_config(
                runtime,
                run_root,
                "random_memory_d_adaptive",
                "random_stable_causal_d",
                runtime.profile["max_steps"],
                adaptive=True,
            ),
            "reference_progress": run_root / "baseline" / "progress_log.jsonl",
        },
        "shuffled_memory_d_adaptive": {
            "script": "experiments/train_ergt_adaptive_alpha.py",
            "config": ergt_config(
                runtime,
                run_root,
                "shuffled_memory_d_adaptive",
                "shuffled_stable_causal_d",
                runtime.profile["max_steps"],
                adaptive=True,
            ),
            "reference_progress": run_root / "baseline" / "progress_log.jsonl",
        },
        "no_memory_real_d_adaptive": {
            "script": "experiments/train_ergt_adaptive_alpha.py",
            "config": ergt_config(
                runtime,
                run_root,
                "no_memory_real_d_adaptive",
                "no_memory_real_d",
                runtime.profile["max_steps"],
                adaptive=True,
            ),
            "reference_progress": run_root / "baseline" / "progress_log.jsonl",
        },
        "instantaneous_real_d_adaptive": {
            "script": "experiments/train_ergt_adaptive_alpha.py",
            "config": ergt_config(
                runtime,
                run_root,
                "instantaneous_real_d_adaptive",
                "instantaneous_real_d",
                runtime.profile["max_steps"],
                adaptive=True,
            ),
            "reference_progress": run_root / "baseline" / "progress_log.jsonl",
        },
    }
    run_plan: list[dict[str, Any]] = []
    for condition in runtime.profile["conditions"]:
        spec = condition_specs[condition]
        config_path = config_dir / f"{condition}.json"
        write_json(config_path, spec["config"])
        run_plan.append(
            {
                "condition": condition,
                "script": spec["script"],
                "config_path": config_path,
                "reference_progress": spec["reference_progress"],
                "max_steps": spec["config"]["training"]["max_steps"],
            }
        )
    serializable = [
        {
            **item,
            "config_path": item["config_path"].as_posix(),
            "reference_progress": (
                str(item["reference_progress"]) if item["reference_progress"] else None
            ),
        }
        for item in run_plan
    ]
    write_json(run_root / "run_plan.json", serializable)
    print(json.dumps(serializable, indent=2, sort_keys=True))
    return run_plan


def baseline_config(
    runtime: NotebookRuntime, run_root: Path, condition: str, max_steps: int
) -> dict[str, Any]:
    return {
        "run": run_section(runtime, run_root, condition),
        "dataset": common_dataset(runtime),
        "model": common_model(runtime, "transformer_baseline"),
        "training": common_training(runtime, max_steps),
        "logging": common_logging(runtime),
    }


def ergt_config(
    runtime: NotebookRuntime,
    run_root: Path,
    condition: str,
    distance_mode: str,
    max_steps: int,
    *,
    adaptive: bool,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "run": run_section(runtime, run_root, condition),
        "dataset": common_dataset(runtime),
        "model": common_model(runtime, "ergt_v1"),
        "relational_graph": {
            "kernel": "sigmoid_cosine",
            "temperature": 1.0,
            "normalize_hidden": True,
            "diagonal_policy": "zero",
            "causal": True,
            "clip_min": 0.0,
            "clip_max": 1.0,
        },
        "distance": {
            "formula": "-log(W + epsilon)",
            "epsilon": 1e-6,
            "diagonal_policy": "zero",
            "normalization": "offdiag_zscore_clamp",
            "clip_value": 5.0,
        },
        "attention": {
            "type": "geo_attention",
            "distance_mode": distance_mode,
            "gradient_mode": "detached_d",
            "head_sharing": "shared_d",
            "causal_runtime_distance": True,
            "max_causal_step": 1,
            "memory": {
                "decay": 0.7,
                "eta": 0.3,
                "gate_floor": 0.05,
                "min_context_edges": 2,
            },
            "alpha": {
                "mode": "fixed",
                "initial_value": 0.0,
                "non_negative": True,
                "warmup_steps": min(int(runtime.profile["alpha_warmup_steps"]), int(max_steps)),
            },
        },
        "training": common_training(runtime, max_steps),
        "logging": common_logging(runtime),
    }
    if adaptive:
        config["adaptive_alpha"] = {
            "initial_alpha": 0.0,
            "min_alpha": 0.0,
            "max_alpha": 0.18,
            "decision_interval_steps": int(runtime.profile["eval_interval"]),
            "min_points_for_slope": 4,
            "slope_window_points": 5,
            "exploration_points": 4,
            "exploration_alpha": 0.025,
            "exploration_step": 0.025,
            "growth_step": 0.035,
            "decay_step": 0.015,
            "max_change_per_decision": 0.01,
            "inertia": 0.75,
            "score_scale": 0.001,
            "positive_margin": 0.0002,
            "negative_margin": -0.0002,
            "slope_gain_weight": 1.0,
            "advantage_weight": 0.25,
            "geo_qk_target": 0.08,
            "geo_qk_risk_weight": 0.00035,
            "entropy_drop_weight": 0.0002,
            "max_probability_target": 0.35,
            "max_probability_risk_weight": 0.0003,
            "ema_beta": 0.7,
            "reference_progress": (run_root / "baseline" / "progress_log.jsonl").as_posix(),
        }
    return config


def run_section(runtime: NotebookRuntime, run_root: Path, condition: str) -> dict[str, Any]:
    return {
        "phase": "notebook_04_guarded_adaptive_training",
        "condition": condition,
        "seed": runtime.seed,
        "output_dir": (run_root / condition).as_posix(),
    }


def common_dataset(runtime: NotebookRuntime) -> dict[str, Any]:
    return {
        "name": "wikitext-2",
        "split": "standard",
        "tokenizer": "gpt2",
        "context_length": int(runtime.profile["context_length"]),
    }


def common_model(runtime: NotebookRuntime, model_type: str) -> dict[str, Any]:
    return {
        "type": model_type,
        "vocab_size": None,
        "n_layers": int(runtime.profile["n_layers"]),
        "n_heads": int(runtime.profile["n_heads"]),
        "hidden_dim": int(runtime.profile["hidden_dim"]),
        "ffn_dim": int(runtime.profile["ffn_dim"]),
        "dropout": 0.1,
        "bias": True,
        "positional_encoding": "learned_absolute",
    }


def common_training(runtime: NotebookRuntime, max_steps: int) -> dict[str, Any]:
    return {
        "optimizer": "adamw",
        "learning_rate": 0.0003,
        "betas": [0.9, 0.95],
        "weight_decay": 0.1,
        "batch_size": int(runtime.profile["batch_size"]),
        "max_steps": int(max_steps),
        "warmup_steps": min(int(runtime.profile["warmup_steps"]), int(max_steps)),
        "grad_clip": 1.0,
        "eval_interval": int(runtime.profile["eval_interval"]),
        "checkpoint_interval": int(max_steps),
        "save_checkpoints": False,
        "max_eval_batches": runtime.profile["max_eval_batches"],
        "precision": runtime.profile.get("precision", "auto"),
        "allow_tf32": True,
        "float32_matmul_precision": "high",
        "dataloader_num_workers": int(runtime.profile.get("dataloader_num_workers", 2)),
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }


def common_logging(runtime: NotebookRuntime) -> dict[str, Any]:
    return {
        "train_log": "train_log.jsonl",
        "progress_log": "progress_log.jsonl",
        "results": "metrics.json",
        "model_summary": "model_summary.json",
        "adaptive_alpha_log": "adaptive_alpha_log.jsonl",
        "log_geometry_diagnostics": True,
        "train_geometry_diagnostics_interval": int(runtime.profile["eval_interval"]),
    }


def run_training_plan(
    runtime: NotebookRuntime, run_root: Path, run_plan: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not runtime.run_training:
        print("RUN_TRAINING is False; configs were generated but no training was launched.")
        return records
    timeout_sec = int(runtime.profile["per_run_timeout_minutes"] * 60)
    for item in run_plan:
        condition = item["condition"]
        reference_progress = item["reference_progress"]
        if condition != "baseline" and reference_progress and not Path(reference_progress).exists():
            raise FileNotFoundError(
                f"Missing baseline reference progress before {condition}: {reference_progress}"
            )
        cmd: list[Any] = [
            sys.executable,
            "-u",
            item["script"],
            "--config",
            item["config_path"],
            "--device",
            runtime.device,
        ]
        if reference_progress and str(item["script"]).endswith("train_ergt_adaptive_alpha.py"):
            cmd.extend(["--reference-progress", reference_progress])
        record = run_streaming_cmd(cmd, cwd=runtime.project_root, timeout_sec=timeout_sec)
        record["condition"] = condition
        records.append(record)
        write_json(run_root / "training_records.json", records)
        cleanup_checkpoint_dirs(run_root)
    return records


def run_streaming_cmd(
    cmd: list[Any], *, cwd: Path, timeout_sec: int, check: bool = True
) -> dict[str, Any]:
    started = time.perf_counter()
    print("\n$ " + " ".join(str(part) for part in cmd), flush=True)
    process = subprocess.Popen(
        [str(part) for part in cmd],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    deadline = started + timeout_sec
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
            if time.perf_counter() > deadline:
                process.kill()
                raise TimeoutError(f"Command timed out after {timeout_sec}s: {cmd}")
        returncode = process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
    elapsed = time.perf_counter() - started
    print(f"returncode={returncode}, elapsed={elapsed / 60:.2f} min", flush=True)
    record = {
        "cmd": [str(part) for part in cmd],
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "stdout_tail": "".join(lines)[-8000:],
    }
    if check and returncode != 0:
        raise RuntimeError(f"Command failed with code {returncode}: {cmd}")
    return record


def write_review_artifacts(
    runtime: NotebookRuntime, run_root: Path, run_plan: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    flat_rows: list[dict[str, Any]] = []
    for item in run_plan:
        condition = item["condition"]
        rows = read_jsonl(run_root / condition / "progress_log.jsonl")
        flat_rows.extend(flatten_progress_row(row) for row in rows)

    baseline_by_step = {
        int(row["step"]): row
        for row in flat_rows
        if row.get("condition") == "baseline" and row.get("validation_loss") is not None
    }
    for row in flat_rows:
        step = row.get("step")
        baseline = baseline_by_step.get(int(step)) if step is not None else None
        if baseline and row.get("validation_loss") is not None:
            row["delta_vs_baseline"] = float(row["validation_loss"]) - float(
                baseline["validation_loss"]
            )
        else:
            row["delta_vs_baseline"] = None

    write_json(run_root / "flat_progress_rows.json", flat_rows)
    table = build_review_table(runtime, flat_rows)
    (run_root / "guarded_training_live_review_table.md").write_text(
        table, encoding="utf-8"
    )
    if Markdown is not None and display is not None:
        display(Markdown(table))
    else:
        print(table)
    write_plots(run_root, flat_rows)
    return flat_rows


def flatten_progress_row(row: dict[str, Any]) -> dict[str, Any]:
    geometry = row.get("validation_geometry") or row.get("geometry") or {}
    summary = geometry.get("summary", geometry) if isinstance(geometry, dict) else {}
    return {
        "condition": row.get("condition"),
        "step": row.get("step"),
        "train_loss": row.get("train_loss"),
        "validation_loss": row.get("validation_loss"),
        "best_validation_loss": row.get("best_validation_loss"),
        "alpha": row.get("alpha_effective") or row.get("alpha_next") or summary.get("alpha"),
        "alpha_next": row.get("alpha_next"),
        "alpha_delta": row.get("alpha_delta"),
        "alpha_decision": row.get("alpha_decision"),
        "geo_to_qk_ratio": row.get("geo_to_qk_ratio") or summary.get("geo_to_qk_ratio"),
        "attention_entropy": row.get("attention_entropy") or summary.get("attention_entropy"),
        "mean_max_probability": row.get("mean_max_probability")
        or summary.get("mean_max_probability"),
        "memory_decay": row.get("memory_decay") or summary.get("memory_decay"),
        "memory_eta": row.get("memory_eta") or summary.get("memory_eta"),
        "gate_floor": row.get("gate_floor") or summary.get("gate_floor"),
        "memory_stability": row.get("memory_stability") or summary.get("memory_stability"),
        "memory_turnover": row.get("memory_turnover") or summary.get("memory_turnover"),
        "memory_persistence": row.get("memory_persistence") or summary.get("memory_persistence"),
        "memory_rigidity": row.get("memory_rigidity") or summary.get("memory_rigidity"),
        "future_leak_score": row.get("future_leak_score") or summary.get("future_leak_score"),
        "tokens_per_second": row.get("tokens_per_second"),
        "gpu_memory_gb": row.get("gpu_memory_gb"),
        "elapsed_minutes": row.get("elapsed_minutes"),
    }


def build_review_table(runtime: NotebookRuntime, flat_rows: list[dict[str, Any]]) -> str:
    headers = [
        "step",
        "condition",
        "val",
        "d_vs_base",
        "alpha",
        "a_next",
        "geo/qk",
        "ent",
        "maxp",
        "m_eta",
        "m_stab",
        "m_pers",
        "m_rigid",
        "tok/s",
        "gpu",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in sorted(
        flat_rows, key=lambda item: (int(item.get("step") or 0), str(item.get("condition")))
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("step", "")),
                    str(row.get("condition", "")),
                    fmt(row.get("validation_loss")),
                    fmt(row.get("delta_vs_baseline")),
                    fmt(row.get("alpha")),
                    fmt(row.get("alpha_next")),
                    fmt(row.get("geo_to_qk_ratio"), 3),
                    fmt(row.get("attention_entropy"), 3),
                    fmt(row.get("mean_max_probability"), 3),
                    fmt(row.get("memory_eta"), 3),
                    fmt(row.get("memory_stability"), 3),
                    fmt(row.get("memory_persistence"), 3),
                    fmt(row.get("memory_rigidity"), 3),
                    fmt(row.get("tokens_per_second"), 0),
                    fmt(row.get("gpu_memory_gb"), 2),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def write_plots(run_root: Path, flat_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")
        return
    for key in [
        "validation_loss",
        "delta_vs_baseline",
        "geo_to_qk_ratio",
        "alpha",
        "memory_stability",
        "memory_rigidity",
    ]:
        series = [
            (row["step"], row.get(key), row["condition"])
            for row in flat_rows
            if row.get(key) is not None
        ]
        if not series:
            continue
        plt.figure(figsize=(9, 4))
        for condition in sorted({item[2] for item in series}):
            xs = [step for step, _, cond in series if cond == condition]
            ys = [value for _, value, cond in series if cond == condition]
            plt.plot(xs, ys, marker="o", label=condition)
        plt.title(key)
        plt.xlabel("step")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(run_root / f"review_{key}.png", dpi=120)
        if display is not None:
            plt.show()
        plt.close()


def write_summary(
    runtime: NotebookRuntime,
    run_root: Path,
    run_plan: list[dict[str, Any]],
    flat_rows: list[dict[str, Any]],
    *,
    git_commit: str,
    training_records: list[dict[str, Any]],
) -> dict[str, Any]:
    late_start, late_end = runtime.profile["late_window"]
    late_summary: dict[str, dict[str, Any]] = {}
    for condition in runtime.profile["conditions"]:
        late_summary[condition] = {
            "late_window_mean_validation_loss": window_mean(
                flat_rows, condition, late_start, late_end
            ),
            "final_metrics_path": (run_root / condition / "metrics.json").as_posix(),
            "progress_log_path": (run_root / condition / "progress_log.jsonl").as_posix(),
        }
    baseline_late = late_summary.get("baseline", {}).get(
        "late_window_mean_validation_loss"
    )
    for summary in late_summary.values():
        value = summary["late_window_mean_validation_loss"]
        summary["late_delta_vs_baseline"] = (
            None if baseline_late is None or value is None else value - baseline_late
        )
    summary = {
        "notebook": "ERGT_04_Guarded_Adaptive_Training",
        "run_profile": runtime.run_profile,
        "run_root": run_root.as_posix(),
        "git_commit": git_commit,
        "device": runtime.device,
        "profile": runtime.profile,
        "alpha_zero_policy": (
            "short wrapper-neutrality check only; not a full scientific condition"
        ),
        "real_training": True,
        "synthetic_rows_used": False,
        "late_window": runtime.profile["late_window"],
        "late_summary": late_summary,
        "conditions_completed": [
            item["condition"]
            for item in run_plan
            if (run_root / item["condition"] / "progress_log.jsonl").exists()
        ],
        "conditions_requested": runtime.profile["conditions"],
        "training_records": training_records,
        "report_bundle_name": REPORT_BUNDLE_NAME,
        "default_local_review_path": DEFAULT_LOCAL_REVIEW_PATH,
        "colab_url": (
            "https://colab.research.google.com/github/jalaljafari2009/ERGT/blob/"
            "main/notebooks/ERGT_04_Guarded_Adaptive_Training.ipynb"
        ),
        "next_required_step": (
            "analyze downloaded ERGT-04 bundle for late-window real-vs-control separation"
        ),
    }
    write_json(run_root / "guarded_adaptive_training_summary.json", summary)
    return summary


def window_mean(
    flat_rows: list[dict[str, Any]], condition: str, start: int, end: int
) -> float | None:
    rows = [
        row
        for row in flat_rows
        if row.get("condition") == condition
        and row.get("validation_loss") is not None
        and start <= int(row.get("step", -1)) <= end
    ]
    if not rows:
        return None
    return sum(float(row["validation_loss"]) for row in rows) / len(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def include_in_bundle(path: Path) -> bool:
    if not path.is_file():
        return False
    if "checkpoints" in path.parts or "__pycache__" in path.parts:
        return False
    if path.suffix.lower() not in LIGHTWEIGHT_SUFFIXES:
        return False
    return path.stat().st_size <= MAX_BUNDLE_FILE_BYTES


def cleanup_checkpoint_dirs(run_root: Path) -> None:
    for checkpoint_dir in run_root.rglob("checkpoints"):
        if checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


def export_report_bundle(
    runtime: NotebookRuntime, run_root: Path, reason: str
) -> Path:
    cleanup_checkpoint_dirs(run_root)
    write_json(
        run_root / "bundle_manifest.json",
        {
            "reason": reason,
            "run_profile": runtime.run_profile,
            "run_root": str(run_root),
            "bundle_name": REPORT_BUNDLE_NAME,
            "default_local_review_path": DEFAULT_LOCAL_REVIEW_PATH,
            "excluded_artifact_patterns": EXCLUDED_ARTIFACT_PATTERNS,
            "lightweight_only": True,
        },
    )
    bundle_path = (Path("/content") if running_in_colab() else run_root) / REPORT_BUNDLE_NAME
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in run_root.rglob("*"):
            if include_in_bundle(path):
                archive.write(path, path.relative_to(runtime.project_root).as_posix())
    print(f"\nBundle ready: {bundle_path}")
    print(f"Default local review path after download: {DEFAULT_LOCAL_REVIEW_PATH}")
    if running_in_colab():
        try:
            from google.colab import files  # type: ignore

            files.download(str(bundle_path))
        except Exception as exc:
            print(f"Automatic download failed: {exc}")
    return bundle_path


def shutdown_colab_runtime_if_requested(
    runtime: NotebookRuntime, reason: str, *, failed: bool
) -> None:
    should_shutdown = runtime.auto_shutdown_colab_runtime and (
        runtime.auto_shutdown_after_success or (failed and runtime.auto_shutdown_on_failure)
    )
    if not should_shutdown:
        print(f"Runtime shutdown skipped. Reason={reason}; requested=False")
        return
    if not running_in_colab():
        print(f"Runtime shutdown unavailable outside Colab. Reason={reason}")
        return
    print(
        f"Colab runtime shutdown requested after {runtime.auto_shutdown_delay_seconds}s. "
        f"Reason: {reason}",
        flush=True,
    )
    time.sleep(runtime.auto_shutdown_delay_seconds)
    from google.colab import runtime as colab_runtime  # type: ignore

    colab_runtime.unassign()


if __name__ == "__main__":
    run_notebook()
