# Open Adaptive Relational Control Trainer

## Purpose

Stage 17 defines the trainer-loop contract for adaptive ERGT. It is not a long
GPU run by itself. It is the orchestration layer that every later notebook and
adaptive run must follow.

The trainer receives telemetry rows and produces:

```text
progress log
controller decision log
meta-control observer log
control-separation log
safety log
live 100-step display rows
lightweight artifact manifest
trainer replay record
```

## Sequential Run Rule

The trainer preserves the stage-15 and stage-16 rule:

```text
baseline -> real -> random -> shuffled -> no_memory -> instantaneous
```

At every telemetry row, it passes `current_step` to Control Separation Scoring.
This prevents later control rows from being read too early. For example, a
`real` row at step 1000 can use baseline at or before 1000, but it cannot use a
future random row at 1500.

Meta-Control Attention then observes the merged telemetry:

```text
raw telemetry
+ control separation score when available
+ missing-aware meta-control attention
+ safety and trainer status
```

## Fail-Fast Safety

The trainer must stop immediately when a validity or safety hard stop appears:

```text
hard_stop_triggered
nan_or_inf_detected
loss_explosion_detected
future_leakage_detected
severe_attention_collapse_detected
control_unfairness_detected
future_leak_score > 0
non-finite validation_loss
```

Ordinary risk does not stop the run. It becomes controller pressure or a later
revision label.

## Required Logs

The trainer must expose:

```text
progress_log
controller_decision_log
meta_control_observer_log
control_separation_log
safety_log
```

These logs are replayable. They are not final scientific evidence by themselves;
they make later runs interpretable.

## Live Output

Every 100-step display row should include the merged telemetry fields already
defined by earlier stages:

```text
loss and slope fields
control separation fields
joint budget fields
meta-control attention fields
safety and trainer status
```

The trainer produces formatted live lines immediately rather than waiting for
the end of the run.

## Lightweight Artifact Rule

The default artifact bundle is:

```text
ergt_03_adaptive_control_report_bundle.zip
```

The bundle must include logs and summaries, and exclude heavy artifacts:

```text
checkpoints/
*.pt
*.pth
*.ckpt
optimizer_state*
```

## Machine Artifacts

This stage adds:

```text
experiments/open_adaptive_relational_control_trainer.py
evaluation/open_adaptive_relational_control_trainer.py
experiments/create_open_adaptive_relational_control_trainer_report.py
tests/test_open_adaptive_relational_control_trainer.py
```

Default report:

```text
runs/contracts/open_adaptive_relational_control_trainer.json
```

Usage:

```bash
python experiments/create_open_adaptive_relational_control_trainer_report.py
```

## Exit Criteria

This stage is complete when:

```text
trainer emits progress, controller, meta-control, control-separation, and safety logs
live 100-step rows are generated during the run
missing-aware control separation is preserved
meta-control observer log exists
fail-fast stops and records the reason
lightweight artifact manifest excludes checkpoints
trainer replay record is present
```

Next stage:

```text
Live 100-Step Diagnostic Table
```
