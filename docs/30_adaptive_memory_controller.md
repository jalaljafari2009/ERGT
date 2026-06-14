# Adaptive Memory Controller

## 1. Purpose

This stage makes memory injection adaptive. Memory is the central control path
for the ERGT claim, so `eta` and `decay` must not stay fixed while the system
is starved, noisy, or rigid.

The controller treats memory as an open search parameter:

```text
useful but weak memory -> increase eta and preserve more memory
noisy memory -> reduce eta and increase decay for smoothing
rigid memory -> reduce eta and reduce decay so stale structure can release
future leak -> validity hard stop
```

Ordinary memory risks do not abort the program. They become controller pressure
and are recorded in replayable decision records.

## 2. Required Outputs

Every memory decision must expose:

```text
current_eta
current_decay
proposed_eta
proposed_decay
eta_delta
decay_delta
memory_stability_trend
memory_turnover_trend
persistence_trend
noise_evidence
rigidity_evidence
release_evidence
restraint_evidence
parameter_trajectory
injected_evidence_ledger
controller_state_snapshot
decision_replay_record
```

## 3. Control Policy

The controller releases memory injection when:

```text
real memory advantage is positive
stability or persistence is below target but noise is low
edge density or persistence shows memory starvation
controls do not dominate real memory advantage
```

The controller restrains memory injection when:

```text
noise risk is high
turnover is above target
random or shuffled memory advantage matches or beats real memory
memory rigidity rises
future leak appears
```

Noise and rigidity are handled differently. Noise lowers `eta` and raises
`decay` to smooth new injections. Rigidity lowers both `eta` and `decay` so the
system can release stale memory instead of preserving a locked structure.

## 4. Attention-Behavior Correction

After the attention-as-search-surface revision, memory decisions should also
cite attention behavior. Good memory injection should not only improve memory
stability or loss; it should help attention form a real, control-separated
regime without uniformity drift, head lock-in, or collapse.

Future memory decisions should treat attention as evidence for:

```text
memory injection is helping real relational specialization
memory injection is only smoothing random or shuffled controls
memory is making attention rigid
memory is too weak to move attention away from uniformity
```

## 5. Machine Artifacts

This stage adds:

```text
experiments/adaptive_memory_controller.py
evaluation/adaptive_memory_controller.py
experiments/create_adaptive_memory_controller_report.py
tests/test_adaptive_memory_controller.py
```

Default report:

```text
runs/contracts/adaptive_memory_controller.json
```

Usage:

```bash
python experiments/create_adaptive_memory_controller_report.py
```

## 6. Exit Criteria

This stage is complete when:

```text
eta can grow when useful memory is starved
eta can shrink and decay can rise when memory is noisy
eta and decay can both shrink when memory is rigid
ordinary memory risk does not abort the search
future leak acts as a validity hard stop
every decision has a parameter trajectory
every decision has an injected evidence ledger
every decision has a replay record
```

Next stage:

```text
Gate-Floor and Noise Controller
```
