# Control Separation Scoring

## Purpose

Stage 15 defines how ERGT scores real geometry against baseline and control
families without pretending that sequential runs are simultaneous.

The run order is usually:

```text
baseline -> real -> random -> shuffled -> no_memory -> instantaneous
```

During the `real` run, random and shuffled results do not exist yet. Therefore
the scorer has two modes:

```text
partial_live
final_matched
```

## Partial Live Score

`partial_live` is allowed to compare the current real run against controls that
already exist at the same or earlier step.

Most importantly:

```text
real_vs_baseline_delta = baseline_loss_at_or_before_step - real_loss
```

This can guide controllers, but it cannot create scientific claim credit.

If random or shuffled is missing:

```text
claim_eligibility = not_eligible_pending_controls
scientific_claim_credit = 0
```

This directly handles sequential Colab execution: the scorer must not require
future control data while real is still running, and it must not peek at future
control rows if a `current_step` limit is provided.

## Final Matched Score

`final_matched` is allowed only after all required control families exist:

```text
baseline
alpha_zero
random
shuffled
no_memory
instantaneous
```

The final score uses exact matched steps shared by all required conditions.
Claim credit is based on the late window only:

```text
step >= late_window_start
```

Default:

```text
late_window_start = 1000
min_matched_points = 2
```

## Delta Convention

Loss deltas are defined so positive means real is better:

```text
real_vs_control_delta = control_validation_loss - real_validation_loss
```

The main separation score is:

```text
control_separation = min(
  real_vs_baseline_delta,
  real_vs_alpha_zero_delta,
  real_vs_random_delta,
  real_vs_shuffled_delta,
  real_vs_no_memory_delta,
  real_vs_instantaneous_delta
)
```

## Pass Rule

Baseline improvement alone is never enough.

The pass condition is:

```text
real > baseline
real > alpha_zero
real > random
real > shuffled
real > no_memory
real > instantaneous
```

all in the matched late window.

If random or shuffled is not beaten, the scorer emits:

```text
generic_regularization_warning
scientific_claim_credit = 0
control_separation_status = fail_controls_not_separated
```

## Attention Behavior

Control separation also reports attention behavior separation. This is not
standalone proof, but it prevents a lower loss with control-like or collapsed
attention from being treated as clean evidence.

## Machine Artifacts

This stage adds:

```text
experiments/control_separation_scoring.py
evaluation/control_separation_scoring.py
experiments/create_control_separation_scoring_report.py
tests/test_control_separation_scoring.py
```

Default report:

```text
runs/contracts/control_separation_scoring.json
```

Usage:

```bash
python experiments/create_control_separation_scoring_report.py
```

## Live 100-Step Display

The progress line should expose:

```text
control_separation_mode
claim_eligibility
control_separation_status
real_vs_baseline_delta
real_vs_random_delta
real_vs_shuffled_delta
control_separation
control_penalty
generic_regularization_warning
attention_behavior_separation
```

## Exit Criteria

This stage is complete when:

```text
partial_live uses baseline without requiring unavailable controls
partial_live never grants claim credit
final_matched requires complete controls
final_matched uses matched late-window steps
random or shuffled dominance blocks the claim
attention behavior separation is reported
live progress logging exposes control separation fields
```

Next stage:

```text
Meta-Control Attention Layer - Observer First
```
