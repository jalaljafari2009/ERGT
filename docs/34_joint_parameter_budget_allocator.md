# Joint Parameter Budget Allocator

## 1. Purpose

This stage coordinates the adaptive degrees of freedom before a trainer applies
their requested deltas:

```text
alpha
memory_eta
memory_decay
gate_floor
distance_norm_scale
causal_reachability
```

The previous controllers can each recommend movement. The allocator prevents the
system from changing all knobs blindly in the same direction. It turns separate
controller requests into one replayable budget decision.

## 2. Required Outputs

Every budget decision must expose:

```text
change_budget
allocated_change_budget
geometry_budget
memory_budget
rigidity_budget
noise_budget
qk_competition_state
attention_behavior_regime
attention_derived_budget_pressure
budget_conflict_score
release_allocation
restraint_allocation
budget_allocation
budget_suppression_reasons
parameter_trajectory
injected_evidence_ledger
controller_state_snapshot
decision_replay_record
```

## 3. Allocation Policy

Geometry budget covers:

```text
alpha
distance_norm_scale
causal_reachability
```

Memory budget covers:

```text
memory_eta
memory_decay
gate_floor
```

When geometry is supported and attention is healthy, geometry can receive most
of the budget. When memory edges are noisy, the allocator shifts budget toward
`memory_decay`, `gate_floor`, or negative `memory_eta`. When attention or
controls disagree, geometry growth is suppressed even if individual geometry
controllers request growth.

## 4. Hard Stops and Soft Pressures

Hard stops:

```text
future_leak
explicit hard_stop_triggered
```

Soft pressures:

```text
geo/qk overpowered
attention rigidity
attention collapse pressure
control penalty
noise risk
attribution uncertainty
multi-controller conflict
```

Soft pressures do not invalidate the run. They reallocate, shrink, or freeze
the requested budget.

## 5. Machine Artifacts

This stage adds:

```text
experiments/joint_parameter_budget_allocator.py
evaluation/joint_parameter_budget_allocator.py
experiments/create_joint_parameter_budget_allocator_report.py
tests/test_joint_parameter_budget_allocator.py
```

It also extends:

```text
evaluation/unified_telemetry_schema.py
experiments/progress_logging.py
```

Default report:

```text
runs/contracts/joint_parameter_budget_allocator.json
```

Usage:

```bash
python experiments/create_joint_parameter_budget_allocator_report.py
```

## 6. Live Telemetry

Long runs should expose:

```text
budget
bUsed
bGeom
bMem
bRigid
bNoise
bAttn
bConflict
qk_state
attn_regime
budget_decision
```

These fields show whether the system is opening geometry, shifting into memory,
or suppressing movement because controllers conflict.

## 7. Exit Criteria

This stage is complete when:

```text
total allocation respects the declared change budget
geometry is prioritized only when attention and controls permit it
memory receives budget when edge noise dominates
geometry growth is suppressed under attention/control disagreement
future leak freezes all movement
every decision has a parameter trajectory
every decision has an injected evidence ledger
every decision has a replay record
live progress logging exposes joint budget decisions
```

Next stage:

```text
Control Separation Scoring
```
