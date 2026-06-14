# Short Smoke and Failure-Safety Validation

## Purpose

Stage 20 is the mechanical gate before any guarded 2000-step adaptive run. It
does not claim scientific improvement. It checks whether the adaptive execution
path is safe enough to run longer.

The smoke uses a short 100/200-step telemetry sequence:

```text
baseline -> alpha_zero -> real_memory_d -> random_memory_d -> shuffled_memory_d
```

This preserves the sequential-run constraint: while `real_memory_d` is being
processed, later random/shuffled rows are not available and must not be read.

## Required Outputs

The report validates:

```text
100/200-step smoke run
live 100-step output
unified telemetry schema validation
controller decision log
meta-control observer log
ERGT-03 auto-shutdown path
future-leak fail-fast path
lightweight artifact contract
sequential no-peek behavior
```

## Safety Rule

The fail-fast smoke injects:

```text
future_leak_score > 0
```

The expected behavior is:

```text
trainer_status = failed_fast
trainer_fail_fast_reason = future_leak_score
later rows are not processed
the failing row is visible in the live diagnostic table
```

## Sequential Control Rule

During the live `real_memory_d` rows, the scorer may use baseline evidence, but
it must not use random or shuffled rows that have not happened yet.

The report therefore requires:

```text
control_separation_mode = partial_live
pending_control_mask = true
real_vs_random_delta = null
offline_replay_required = true
```

This keeps live controller pressure separate from final scientific claim credit.

## Machine Artifacts

This stage adds:

```text
evaluation/short_smoke_failure_safety_validation.py
experiments/create_short_smoke_failure_safety_validation_report.py
tests/test_short_smoke_failure_safety_validation.py
```

Default report:

```text
runs/contracts/short_smoke_failure_safety_validation.json
```

Usage:

```bash
python experiments/create_short_smoke_failure_safety_validation_report.py
```

## Exit Criteria

This stage is complete when:

```text
short smoke completed
live display exists at 100-step cadence
schema status is pass
controller decision log exists
meta-control observer log exists
ERGT-03 auto-shutdown hook exists
future-leak fail-fast stops early
checkpoint artifacts are excluded
real live rows do not peek at future controls
```

Next stage:

```text
Guarded 2000-Step Adaptive Run, now specified in
docs/41_guarded_2000_step_adaptive_run.md
```
