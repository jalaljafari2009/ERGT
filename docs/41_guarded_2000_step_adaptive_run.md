# Guarded 2000-Step Adaptive Run

## Purpose

Stage 21 defines the guarded 2000-step adaptive run contract. This is the first
full-length adaptive execution shape after the short smoke gate.

This stage is still not the final scientific decision. It verifies that every
required condition can expose comparable telemetry at the same cadence and that
late-window matched scoring is available for stage 22.

## Required Conditions

The guarded run contains:

```text
baseline
alpha_zero
real_memory_d
random_memory_d
shuffled_memory_d
no_memory_real_d
instantaneous_real_d
```

The run order is sequential:

```text
baseline -> alpha_zero -> real -> random -> shuffled -> no_memory -> instantaneous
```

This preserves the no-peek rule: live `real_memory_d` rows cannot use random,
shuffled, no-memory, or instantaneous rows that do not exist yet.

## Cadence

The guarded profile is:

```text
run_profile = adaptive_2000_guarded
max_steps = 2000
eval_interval = 100
live_display_interval = 100
late_window_start = 1000
```

Every condition must expose the exact same checkpoint steps:

```text
100, 200, ..., 2000
```

## Late-Window Readiness

Stage 21 verifies that the following windows are populated:

```text
0-500
500-1000
1000-1500
1500-2000
1000-2000
```

Final matched scoring is allowed only after all controls exist. In the replay
contract, final matched late-window scoring begins once at least two late-window
matched points exist.

## Important Boundary

The synthetic replay in this stage is a contract validation, not scientific
claim evidence.

Stage 21 answers:

```text
Can the guarded 2000-step adaptive run produce comparable telemetry?
```

Stage 22 answers:

```text
What do the post-1000 and late-window trajectories actually mean?
```

## Machine Artifacts

This stage adds:

```text
experiments/guarded_2000_step_adaptive_run.py
evaluation/guarded_2000_step_adaptive_run.py
experiments/create_guarded_2000_step_adaptive_run_report.py
tests/test_guarded_2000_step_adaptive_run.py
```

Default report:

```text
runs/contracts/guarded_2000_step_adaptive_run.json
```

Usage:

```bash
python experiments/create_guarded_2000_step_adaptive_run_report.py
```

## Exit Criteria

This stage is complete when:

```text
stage-20 smoke gate passed
all required conditions are present
all conditions have identical 100-step checkpoints
all conditions reach step 2000
live rows exist for every condition
schema validation passes
trainer completes without fail-fast
controller and meta-control logs exist
final matched late-window rows exist
late-window analysis is ready for stage 22
checkpoint artifacts remain excluded
```

Next stage:

```text
Late-Window and Post-1000 Analysis
```

Implemented by `docs/42_late_window_post1000_analysis.md`.
