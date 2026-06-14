# Adaptive Alpha Controller v2

## 1. Purpose

This stage turns alpha into an adaptive search parameter instead of a fixed cap
or a flag-driven gate.

The controller is PID-inspired:

```text
error = release_evidence - restraint_pressure
P = current evidence balance
I = persistent evidence balance
D = change in evidence balance
```

This is not a generic industrial PID loop copied blindly. It is a bounded
evidence controller where ordinary risks become pressure and only declared
safety or validity hard stops can hold alpha completely.

## 2. Required Outputs

Every alpha decision must expose:

```text
current_alpha
proposed_alpha
alpha_delta
release_evidence
restraint_evidence
slope_evidence
rigidity_evidence
control_family_evidence
parameter_trajectory
injected_evidence_ledger
controller_state_snapshot
decision_replay_record
```

## 3. Control Policy

Alpha may grow when release evidence dominates:

```text
positive loss-slope gain
positive EMA loss delta
late-window improvement
post-1000 improvement
real trend stronger than random and shuffled controls
```

Alpha is restrained when pressure dominates:

```text
rigidity risk
collapse risk
control penalty
random or shuffled trend matching or beating real
geo/qk pressure
```

These restraint signals do not abort the search. They alter the alpha delta and
are recorded in the replay record.

## 4. Attention-Behavior Correction

After the attention-as-search-surface revision, alpha decisions should be read
through attention behavior as well as loss trend. A useful alpha region is not
only a lower loss region; it should move real attention away from uniformity and
toward interpretable specialization without head lock-in, severe sparsity, or
geometry takeover.

Future alpha decisions should cite:

```text
attention behavior regime
real-vs-control attention separation
head/layer diversity pressure
uniformity or lock-in pressure
geometry takeover pressure
```

## 5. Machine Artifacts

This stage adds:

```text
experiments/adaptive_alpha_v2.py
evaluation/adaptive_alpha_controller_v2.py
experiments/create_adaptive_alpha_controller_v2_report.py
tests/test_adaptive_alpha_controller_v2.py
```

Default report:

```text
runs/contracts/adaptive_alpha_controller_v2.json
```

Usage:

```bash
python experiments/create_adaptive_alpha_controller_v2_report.py
```

## 6. Exit Criteria

This stage is complete when:

```text
alpha can grow from release evidence
alpha can shrink from restraint and control-family pressure
ordinary rigidity/collapse risk does not abort the search
explicit hard stops hold alpha
PID-style error, integral, and derivative terms are recorded
every decision has a parameter trajectory
every decision has an injected evidence ledger
every decision has a replay record
```

Next stage:

```text
Adaptive Memory Controller
```
