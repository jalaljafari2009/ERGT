# Late-Window and Post-1000 Analysis

## Purpose

Stage 22 defines how ERGT reads a guarded 2000-step run after the model has
passed early instability.

The key rule is that endpoint loss is not enough. Decisions must be based on
post-1000 windows, matched control comparisons, and attention behavior over the
same intervals.

This stage answers:

```text
Does real stable causal geometry improve where memory and geometry should matter,
without hiding attention collapse or generic control-like behavior?
```

## Required Windows

The analysis always evaluates:

```text
0-500
500-1000
1000-1500
1500-2000
1000-2000
```

The decision window is:

```text
1000-2000
```

The endpoint at step 2000 is supporting evidence only. It cannot override the
late-window trend.

## Required Signals

For every condition and window, the analyzer tracks:

```text
validation loss mean
validation loss slope
baseline-centered improvement
geo_to_qk_ratio mean and max
attention entropy
attention entropy slope
mean max attention probability
attention rigidity risk
attention behavior score
memory stability
memory persistence
```

For real-vs-control scoring, it requires matched checkpoint steps before
computing deltas. This preserves the sequential no-peek contract from stage 21:
live real rows do not use future random, shuffled, no-memory, or instantaneous
rows that have not run yet.

## Attention Safety

Late-window scoring is valid only if attention behavior is checked over the same
windows used for loss decisions.

The stage blocks a clean decision if the decision window shows:

```text
attention collapse
uniformity drift
control-like attention behavior
missing attention telemetry
```

This protects against a false win where validation loss improves but the model
has become rigid, uniform, or indistinguishable from random/shuffled controls.

## Machine Artifacts

This stage adds:

```text
experiments/late_window_post1000_analysis.py
evaluation/late_window_post1000_analysis.py
experiments/create_late_window_post1000_analysis_report.py
tests/test_late_window_post1000_analysis.py
```

Default report:

```text
runs/contracts/late_window_post1000_analysis.json
```

Usage:

```bash
python experiments/create_late_window_post1000_analysis_report.py
```

## Current Contract Result

The current contract replay passes.

The report confirms:

```text
required windows present
post-1000 decision priority enforced
decision window is 1000-2000
late matched controls available
real late-window trend is improving
real beats baseline and all controls in the replay contract
attention is analyzed for every required window
late attention is not collapsed
late attention has no uniformity drift warning
late attention is not control-like
endpoint loss is supporting only
```

Important boundary:

```text
This is a mechanics and analysis-contract pass, not final scientific claim
evidence.
```

Real notebook telemetry can be passed through the same analyzer later.

## Exit Criteria

This stage is complete when:

```text
stage-21 guarded run contract passed
all required windows are populated
all required conditions have window summaries
matched control deltas exist for the decision window
post-1000 trend is the primary decision source
attention safety is checked on the same windows
endpoint-only claims are rejected
machine report passes
tests pass
```

Next stage:

```text
Random/Shuffled/No-Memory Attribution Comparison
```

Implemented by `docs/43_random_shuffled_no_memory_attribution.md`.
