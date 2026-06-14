import json

from evaluation.live_100_step_diagnostic_table import (
    REQUIRED_LIVE_DIAGNOSTIC_OUTPUTS,
    build_live_100_step_diagnostic_table_report,
)
from experiments.live_100_step_diagnostic_table import (
    PLOT_SERIES_GROUPS,
    REQUIRED_LIVE_DIAGNOSTIC_COLUMNS,
    LiveDiagnosticTableBuilder,
)
from experiments.open_adaptive_relational_control_trainer import (
    OpenAdaptiveRelationalControlTrainer,
    OpenAdaptiveTrainerConfig,
)


def test_live_diagnostic_row_exposes_required_columns() -> None:
    row = LiveDiagnosticTableBuilder().build_row(
        {
            "step": 100,
            "condition": "real_memory_d",
            "validation_loss": 5.9,
            "real_vs_baseline_delta": 0.1,
            "loss_slope_gain": 0.02,
            "alpha_effective": 0.03,
            "alpha_next": 0.035,
            "alpha_delta": 0.005,
            "geo_to_qk_ratio": 0.04,
            "memory_eta_next": 0.32,
            "memory_eta_delta": 0.02,
            "memory_decay_next": 0.68,
            "memory_decay_delta": -0.02,
            "gate_floor_next": 0.05,
            "distance_norm_scale_next": 1.1,
            "causal_reachability_next": 2.0,
            "meta_top_signal": "memory_state",
            "trainer_status": "running",
            "trainer_fail_fast_triggered": False,
        }
    )

    assert set(REQUIRED_LIVE_DIAGNOSTIC_COLUMNS).issubset(row)
    assert row["delta_vs_baseline"] == 0.1
    assert row["rolling_slope"] == 0.02
    assert row["alpha"] == 0.03
    assert row["alpha_next"] == 0.035
    assert row["memory_eta"] == 0.32
    assert row["memory_decay"] == 0.68
    assert row["live_diagnostic_row_ready"]


def test_live_diagnostic_markdown_table_has_explicit_missing_values() -> None:
    table = LiveDiagnosticTableBuilder().format_markdown(
        [{"step": 100, "condition": "real_memory_d"}]
    )

    assert "| step | condition |" in table
    assert "real_memory_d" in table
    assert "NA" in table


def test_live_diagnostic_plot_payload_contains_series_groups() -> None:
    builder = LiveDiagnosticTableBuilder()
    rows = [
        builder.build_row(
            {
                "step": 100,
                "condition": "real_memory_d",
                "validation_loss": 5.9,
                "geo_to_qk_ratio": 0.04,
                "meta_alpha_weight": 0.2,
            }
        )
    ]
    payload = builder.build_plot_payload(rows)

    assert set(PLOT_SERIES_GROUPS).issubset(payload["series_groups"])
    assert payload["x_axis"] == "step"
    assert payload["series_groups"]["loss"]["validation_loss"][0]["value"] == 5.9
    assert payload["series_groups"]["geometry"]["geo_to_qk_ratio"][0]["value"] == 0.04


def test_trainer_emits_live_diagnostic_table_artifacts() -> None:
    trainer = OpenAdaptiveRelationalControlTrainer(
        OpenAdaptiveTrainerConfig(live_display_interval=100)
    )
    result = trainer.summary(
        trainer.run(
            [
                {"condition": "baseline", "step": 100, "validation_loss": 6.0},
                {
                    "condition": "real_memory_d",
                    "step": 100,
                    "validation_loss": 5.8,
                    "loss_slope_gain": 0.05,
                    "real_vs_baseline_delta": 0.2,
                },
            ]
        )
    )

    assert result["live_diagnostic_rows"]
    assert result["live_diagnostic_tables"]
    assert result["live_diagnostic_plot_payloads"]
    assert result["trainer_summary"]["live_diagnostic_table_ready"]
    assert result["trainer_summary"]["live_diagnostic_plot_ready"]


def test_live_diagnostic_fail_fast_row_is_displayed() -> None:
    trainer = OpenAdaptiveRelationalControlTrainer(
        OpenAdaptiveTrainerConfig(live_display_interval=100)
    )
    result = trainer.summary(
        trainer.run(
            [
                {"condition": "real_memory_d", "step": 100, "validation_loss": 5.8},
                {
                    "condition": "real_memory_d",
                    "step": 150,
                    "validation_loss": 5.7,
                    "future_leak_score": 0.2,
                },
                {"condition": "real_memory_d", "step": 200, "validation_loss": 5.6},
            ]
        )
    )

    assert result["trainer_status"] == "failed_fast"
    assert result["live_diagnostic_rows"][-1]["trainer_status"] == "failed_fast"
    assert result["live_diagnostic_rows"][-1]["future_leak"] == 0.2


def test_live_100_step_diagnostic_table_report_passes() -> None:
    report = build_live_100_step_diagnostic_table_report()

    assert report["status"] == "pass"
    assert report["next_required_step"] == "adaptive_notebook_ergt_03"
    assert set(REQUIRED_LIVE_DIAGNOSTIC_OUTPUTS).issubset(
        [
            *report["required_outputs"],
        ]
    )
    assert report["checks"]["trainer_emits_live_diagnostic_rows_during_run"]
    assert report["checks"]["fail_fast_row_is_displayed"]
    json.dumps(report)
