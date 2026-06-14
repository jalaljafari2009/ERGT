# Random/Shuffled/No-Memory Attribution Comparison

## Purpose

Stage 23 determines whether the late-window signal belongs to real relational
geometry or to generic regularization.

The core rule is:

```text
real must beat random, shuffled, no-memory, instantaneous, alpha-zero, and
baseline in the post-1000 decision window.
```

If random or shuffled dominates real, the result is not a real-geometry claim
and must enter revision.

## Decision Window

This stage inherits the stage-22 decision window:

```text
1000-2000
```

Endpoint loss is still supporting evidence only. Attribution is computed from
matched late-window summaries and matched real-vs-control deltas.

## Required Outputs

The stage emits:

```text
random_advantage_analysis
shuffled_distribution_bias_analysis
no_memory_comparison
instantaneous_comparison
relation_specific_advantage_estimate
attention_behavior_comparison
```

## Interpretation Rules

Random can improve over baseline. That is not automatically a failure. It means
generic regularization exists and must be labeled.

The claim fails only when:

```text
random >= real in the late window
shuffled >= real in the late window
no-memory real >= real memory
instantaneous real >= stable-memory real
attention behavior is not separated from controls
relation-specific advantage is not positive
```

The relation-specific estimate is the minimum real advantage over:

```text
random
shuffled
no-memory
instantaneous
```

This is intentionally conservative. The weakest required comparison determines
whether the signal can proceed to the decision gate.

## Attention Comparison

Attention behavior is compared against:

```text
baseline
alpha_zero
random
shuffled
no_memory
instantaneous
```

The stage requires the real attention behavior score to remain separated from
all controls in the decision window. This does not prove the physics claim by
itself, but it prevents a loss-only decision from hiding control-like attention.

## Machine Artifacts

This stage adds:

```text
experiments/random_shuffled_no_memory_attribution.py
evaluation/random_shuffled_no_memory_attribution.py
experiments/create_random_shuffled_no_memory_attribution_report.py
tests/test_random_shuffled_no_memory_attribution.py
```

Default report:

```text
runs/contracts/random_shuffled_no_memory_attribution.json
```

Usage:

```bash
python experiments/create_random_shuffled_no_memory_attribution_report.py
```

## Current Contract Result

The current guarded replay passes.

The report confirms:

```text
random has generic baseline advantage but does not dominate real
shuffled has distribution-bias gain but does not dominate real
real memory beats no-memory real geometry
stable real memory beats instantaneous real geometry
relation-specific advantage is positive
attention behavior is separated from controls
```

Important boundary:

```text
This is a mechanics and attribution-contract pass, not final scientific claim
evidence.
```

Real notebook telemetry can be passed through the same analyzer later.

## Exit Criteria

This stage is complete when:

```text
stage-22 late-window analysis passed
random advantage is analyzed
shuffled distribution bias is analyzed
no-memory ablation is analyzed
instantaneous ablation is analyzed
relation-specific advantage is positive
attention behavior is compared across controls
random/shuffled dominance enters revision instead of passing
machine report passes
tests pass
```

Next stage:

```text
Decision Gate: Real Geometry vs Generic Regularization
```

Implemented by `docs/44_decision_gate_real_geometry.md`.
