# Controller Revision Loop

## Purpose

Stage 25 turns a failed decision gate into concrete controller revisions.

The rule is:

```text
no failed run may remain as an unexplained failure
```

Every failure label must map to:

```text
target controller component
specific revision
validation gate
rerun protocol
decision replay record
```

If the stage-24 gate passes and no failure labels are active, this stage emits a
`noop_audit` and allows stage 26 to proceed.

## Failure Label Catalog

The revision loop covers the documented labels:

```text
memory_starved
memory_noisy
memory_rigid
geometry_flattened
alpha_underpowered
alpha_overpowering
causal_reach_too_tight
causal_reach_too_loose
control_regularization_dominance
normalization_erased_contrast
attention_uniformity_drift
attention_control_like
attention_head_lock_in
meta_control_attention_misweighted
controller_conflict_unresolved
```

It also covers gate-generated labels:

```text
baseline_only_evidence_insufficient
future_leak_detected
late_window_not_ready
memory_not_stabilizing
memory_or_causality_unresolved
relation_specific_advantage_not_established
random_dominates_real
shuffled_dominates_real
no_memory_matches_or_beats_real
instantaneous_matches_or_beats_stable_memory
attention_behavior_not_separated_from_controls
```

Unknown labels are not silently accepted.

## Revision Rules

Hard-stop failures, such as future leakage, require returning to short smoke
validation before any guarded or long run.

Regular revision failures require returning to the guarded adaptive run after the
targeted controller change.

Examples:

```text
control_regularization_dominance
-> audit data/RNG/control isolation and revise relation-specific signals

future_leak_detected
-> repair causal reachability and rerun from short smoke

attention_control_like
-> reweight controller evidence toward relation-specific attention separation
```

## Machine Artifacts

This stage adds:

```text
experiments/controller_revision_loop.py
evaluation/controller_revision_loop.py
experiments/create_controller_revision_loop_report.py
tests/test_controller_revision_loop.py
```

Default report:

```text
runs/contracts/controller_revision_loop.json
```

Usage:

```bash
python experiments/create_controller_revision_loop_report.py
```

## Current Contract Result

The current stage-24 gate passes, so the current stage-25 result is:

```text
revision_mode = noop_audit
stage26_readiness.ready = true
next_required_step = longer_run_or_multi_seed_confirmation
```

The report also validates synthetic failed gates so the revision loop is not
only a pass-through:

```text
random dominance -> control_regularization_dominance
future leak -> future_leak_detected hard stop
control-like attention -> attention_control_like
```

Important boundary:

```text
This is a revision mechanics contract, not a scientific confirmation.
```

## Exit Criteria

This stage is complete when:

```text
all documented failure labels have catalog entries
active failure labels are mapped
specific revisions are present
validation gates are present
decision replay record is present
noop audit allows stage 26 only when gate passed
failed gates block stage 26
synthetic failures map to revisions
machine report passes
tests pass
```

Next stage:

```text
Longer Run or Multi-Seed Confirmation
```

Implemented by `docs/46_longer_run_multi_seed_confirmation.md`.
