# Live 100-Step Diagnostic Table

## Purpose

Stage 18 turns live adaptive ERGT output into a readable table and plot-ready
payload. The previous trainer could emit long one-line progress strings. This
stage adds a structured display contract so Colab can show the run state every
100 steps without waiting for the end of a condition.

The table is an observability artifact. It does not create scientific claim
credit by itself.

## Display Rule

The table updates whenever the trainer emits a live display row:

```text
step == 1
step % live_display_interval == 0
fail_fast event
```

The default interval is:

```text
100 steps
```

## Required Columns

Every row exposes:

```text
step
condition
train_loss
validation_loss
delta_vs_baseline
rolling_slope
alpha
geo_to_qk_ratio
memory_stability
memory_turnover
memory_persistence
memory_rigidity
noise_risk
attention_regime
attention_control_separation
contrast_retention
future_leak
meta_top_signal
meta_attention_entropy
meta_alpha_weight
meta_memory_weight
meta_gate_weight
meta_reach_weight
meta_norm_weight
controller_conflict_score
meta_control_confidence
alpha_decision
memory_eta_decision
memory_decay_decision
gate_floor_decision
causal_reachability_decision
distance_norm_scale_decision
joint_budget_decision
trainer_status
trainer_fail_fast_triggered
```

Missing values are explicit as `NA` in markdown output. This matters because
sequential runs do not have all control-family evidence during the `real` run.

## Plot Payload

The live plot payload groups numeric series into:

```text
loss
geometry
memory
meta_control
safety
```

The x-axis is always:

```text
step
```

This lets the notebook render stable plots without guessing which fields exist.

## Trainer Integration

The open adaptive trainer now emits:

```text
live_diagnostic_rows
live_diagnostic_tables
live_diagnostic_plot_payloads
```

These streams are lightweight and suitable for the notebook report bundle.

## Machine Artifacts

This stage adds:

```text
experiments/live_100_step_diagnostic_table.py
evaluation/live_100_step_diagnostic_table.py
experiments/create_live_100_step_diagnostic_table_report.py
tests/test_live_100_step_diagnostic_table.py
```

Default report:

```text
runs/contracts/live_100_step_diagnostic_table.json
```

Usage:

```bash
python experiments/create_live_100_step_diagnostic_table_report.py
```

## Exit Criteria

This stage is complete when:

```text
all required columns are present
markdown table output is readable and explicit about missing values
plot payloads include loss, geometry, memory, meta-control, and safety series
trainer emits diagnostic rows, tables, and plot payloads during live checkpoints
fail-fast rows are displayed immediately
schema declares live diagnostic fields
```

Next stage:

```text
Adaptive Notebook ERGT-03, now specified in docs/39_adaptive_notebook_ergt_03.md
```
