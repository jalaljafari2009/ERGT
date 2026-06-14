# Causal Reachability Controller

## 1. Purpose

This stage makes the finite-speed causal graph adaptive without weakening the
past-only contract:

```text
memory stable + reach too tight + attention still healthy -> expand reach
control/noise/collapse/uniformity pressure -> restrain reach
future leak -> validity hard stop
```

The controller does not assume that one fixed `max_causal_step` is correct. It
treats causal reach as a degree of freedom that can open when real memory needs
more past context, and can tighten when reach starts behaving like generic
regularization or destabilizes attention.

## 2. Required Outputs

Every causal-reach decision must expose:

```text
current_causal_reachability
proposed_causal_reachability
causal_reachability_delta
causal_reachability_credit
causal_reachability_risk_pressure
future_leak_evidence
memory_readiness_evidence
attention_locality_spread_evidence
control_reach_evidence
expansion_evidence
restraint_evidence
parameter_trajectory
injected_evidence_ledger
controller_state_snapshot
decision_replay_record
```

## 3. Control Policy

The controller expands causal reach only when:

```text
memory stability is above the release threshold
causal edge survival is too low or reach starvation is high
attention is too local but not uniform
head/layer diversity has not collapsed
real reach advantage exceeds random and shuffled controls
control reach noise is below the restraint threshold
```

The controller restrains causal reach when:

```text
memory is not ready
control reach noise is high
random or shuffled reach advantage dominates real
attention becomes uniform, low-diversity, or collapse-prone
control-like attention separation appears
```

Future leakage is never a soft pressure. Any nonzero future-leak score becomes a
hard validity hold.

## 4. Machine Artifacts

This stage adds:

```text
experiments/causal_reachability_controller.py
evaluation/causal_reachability_controller.py
experiments/create_causal_reachability_controller_report.py
tests/test_causal_reachability_controller.py
```

It also extends:

```text
evaluation/unified_telemetry_schema.py
experiments/progress_logging.py
```

Default report:

```text
runs/contracts/causal_reachability_controller.json
```

Usage:

```bash
python experiments/create_causal_reachability_controller_report.py
```

## 5. Live Telemetry

Long runs should expose:

```text
reach
r_next
d_reach
rCredit
rPress
cSurv
cStarv
cNoise
loc
spread
fLeak
```

These fields show whether finite-speed causal reach is opening, tightening, or
holding, and whether the decision came from real memory pressure or control/noise
pressure.

## 6. Exit Criteria

This stage is complete when:

```text
causal reach can expand when reach is too tight
causal reach can restrain when reach is noisy or collapse-prone
ordinary reach risk does not abort the search
future leak acts as a validity hard stop
every decision has a parameter trajectory
every decision has an injected evidence ledger
every decision has a replay record
live progress logging exposes causal-reach decisions
```

Next stage:

```text
Normalization and Distance-Scale Controller
```
