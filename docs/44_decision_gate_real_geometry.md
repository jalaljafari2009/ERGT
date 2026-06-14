# Decision Gate: Real Geometry vs Generic Regularization

## Purpose

Stage 24 is the guarded decision gate for the adaptive ERGT path.

It decides whether the current mechanics support a real stable causal geometry
signal, or whether the observed benefit is still explainable as generic
regularization, distribution bias, missing memory value, or control-like
attention.

The gate is intentionally strict:

```text
beating baseline alone is insufficient
```

## Required Comparisons

The real adaptive stable causal geometry condition must beat:

```text
baseline
alpha_zero
random_adaptive
shuffled_adaptive
no_memory_real
instantaneous_real
```

All comparisons are evaluated in the stage-22/stage-23 decision window:

```text
1000-2000
```

## Risk Audits

The gate also clears the three active risk tracks.

### R1: Memory and Causal Validity

Required:

```text
future_leak_score == 0
real memory beats no-memory real geometry
stable real memory beats instantaneous real geometry
```

### R2: Distance Contrast and Scale

Required:

```text
distance_contrast_retention remains above threshold
geo_to_qk_ratio is active but not takeover-scale
relation_specific_advantage is positive
```

### R3: Attention Behavior

Required:

```text
attention is separated from controls
attention is not collapsed
attention has no uniformity drift warning
attention is not control-like
```

## Failure Labels

If the gate fails, it emits labels for stage 25:

```text
control_regularization_dominance
baseline_only_evidence_insufficient
memory_starved
memory_not_stabilizing
future_leak_detected
memory_or_causality_unresolved
normalization_erased_contrast
attention_uniformity_drift
attention_control_like
attention_head_lock_in
```

These labels are not comments. They are the input contract for the controller
revision loop.

## Machine Artifacts

This stage adds:

```text
experiments/decision_gate_real_geometry.py
evaluation/decision_gate_real_geometry.py
experiments/create_decision_gate_real_geometry_report.py
tests/test_decision_gate_real_geometry.py
```

Default report:

```text
runs/contracts/decision_gate_real_geometry.json
```

Usage:

```bash
python experiments/create_decision_gate_real_geometry_report.py
```

## Current Contract Result

The current guarded replay passes.

The report confirms:

```text
real beats baseline
real beats alpha_zero
real beats random adaptive
real beats shuffled adaptive
real beats no-memory real
real beats instantaneous real
R1 memory/causality audit passes
R2 distance-scale audit passes
R3 attention-behavior audit passes
no failure labels are active
```

Important boundary:

```text
This is a guarded mechanics and decision-contract pass, not final scientific
claim evidence.
```

Real notebook telemetry, longer runs, and multi-seed confirmation are still
required before stronger claims.

## Exit Criteria

This stage is complete when:

```text
stage-22 late-window report passes
stage-23 attribution report passes
all required comparisons are positive
baseline-only evidence is rejected
R1/R2/R3 audits pass
attention gate passes
relation-specific gate passes
failure labels are empty on pass
machine report passes
tests pass
```

Next stage:

```text
Controller Revision Loop
```

Implemented by `docs/45_controller_revision_loop.md`.
