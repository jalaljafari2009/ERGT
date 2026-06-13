import json
import tempfile
from pathlib import Path

from evaluation.run02_evidence_consolidation import (
    build_run02_evidence_consolidation_report,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_condition(
    root: Path,
    condition: str,
    losses: list[tuple[int, float]],
    *,
    adaptive: bool = False,
) -> None:
    rows = []
    alpha_rows = []
    best = float("inf")
    for index, (step, loss) in enumerate(losses):
        best = min(best, loss)
        row = {
            "step": step,
            "validation_loss": loss,
            "best_validation_loss": best,
            "train_loss": loss + 0.1,
            "geo_to_qk_ratio": 0.02 + index * 0.001,
            "attention_entropy": 3.0 - index * 0.01,
            "mean_max_probability": 0.2 + index * 0.005,
        }
        if adaptive:
            row.update(
                {
                    "alpha_effective": 0.01 * index,
                    "alpha_next": 0.01 * (index + 1),
                    "alpha_delta": 0.01,
                    "alpha_decision": "grow",
                    "adaptive_score": 0.001,
                    "adaptive_slope_gain": 0.0002,
                    "adaptive_advantage": 0.002,
                    "geo_qk_risk": 0.0,
                    "entropy_risk": 0.0,
                    "max_probability_risk": 0.0,
                    "adaptive_alpha": {
                        "step": step,
                        "decision": "grow",
                        "next_alpha": 0.01 * (index + 1),
                    },
                }
            )
            alpha_rows.append(row["adaptive_alpha"])
        rows.append(row)

    condition_dir = root / condition
    write_jsonl(condition_dir / "progress_log.jsonl", rows)
    (condition_dir / "metrics.json").write_text(
        json.dumps(
            {
                "final_validation_loss": losses[-1][1],
                "best_validation_loss": min(loss for _, loss in losses),
            }
        ),
        encoding="utf-8",
    )
    if adaptive:
        write_jsonl(condition_dir / "adaptive_alpha_log.jsonl", alpha_rows)


def test_run02_evidence_consolidation_ready_for_complete_adaptive_run() -> None:
    root = Path(tempfile.mkdtemp())
    steps = [100, 200, 300, 400]
    write_condition(root, "baseline", list(zip(steps, [5.0, 4.8, 4.6, 4.4], strict=True)))
    write_condition(root, "alpha_zero", list(zip(steps, [5.0, 4.8, 4.6, 4.4], strict=True)))
    write_condition(
        root,
        "real_memory_d_adaptive",
        list(zip(steps, [5.0, 4.79, 4.58, 4.37], strict=True)),
        adaptive=True,
    )
    write_condition(
        root,
        "random_memory_d_adaptive",
        list(zip(steps, [5.0, 4.81, 4.61, 4.41], strict=True)),
        adaptive=True,
    )
    write_condition(
        root,
        "shuffled_memory_d_adaptive",
        list(zip(steps, [5.0, 4.82, 4.62, 4.42], strict=True)),
        adaptive=True,
    )

    report = build_run02_evidence_consolidation_report(
        root,
        late_window=(200, 400),
    )

    assert report["status"] == "consolidated_ready_for_open_control_contract"
    assert report["checks"]["alpha_zero_matches_baseline"]
    assert report["checks"]["control_families_present"]
    assert report["checks"]["real_has_adaptive_telemetry"]
    real = report["condition_summaries"]["real_memory_d_adaptive"]
    assert real["late_window_eval_points"] == 3
    assert real["adaptive_decision_counts"]["grow"] == 4
    comparison = report["comparisons"]["real_vs_random_memory_d_adaptive"]
    assert comparison["final_validation_loss_delta"] > 0
    json.dumps(report)


def test_run02_evidence_consolidation_reports_missing_controls() -> None:
    root = Path(tempfile.mkdtemp())
    steps = [100, 200]
    write_condition(root, "baseline", list(zip(steps, [5.0, 4.8], strict=True)))
    write_condition(root, "alpha_zero", list(zip(steps, [5.0, 4.8], strict=True)))
    write_condition(
        root,
        "real_memory_d_adaptive",
        list(zip(steps, [5.0, 4.79], strict=True)),
        adaptive=True,
    )

    report = build_run02_evidence_consolidation_report(root, late_window=(100, 200))

    assert report["status"] == "incomplete_required_conditions_missing"
    assert not report["checks"]["required_conditions_present"]
