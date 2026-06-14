# Gate-Floor and Noise Controller

## 1. Purpose

This stage makes gate-floor filtering adaptive. The gate floor controls how
aggressively weak memory edges are filtered:

```text
noisy or control-like weak edges -> raise gate floor
starved real edges or over-sparse attention -> lower gate floor
future leak -> validity hard stop
```

The purpose is not to delete all weak edges. Weak edges may be the beginning of
real relational structure. The controller must distinguish edge noise from real
edge starvation.

## 2. Required Outputs

Every gate-floor decision must expose:

```text
current_gate_floor
proposed_gate_floor
gate_floor_delta
gate_floor_credit
gate_floor_risk_pressure
edge_noise_evidence
starvation_evidence
attention_pressure_evidence
control_attention_evidence
release_evidence
restraint_evidence
parameter_trajectory
injected_evidence_ledger
controller_state_snapshot
decision_replay_record
```

## 3. Control Policy

The controller raises `gate_floor` when:

```text
random/shuffled edge noise is high
control-like attention behavior appears
collapse pressure rises
weak edges look more like generic noise than real relation
```

The controller lowers `gate_floor` when:

```text
real edge starvation is high
edge survival is too low
edge density is below target
attention is over-sparse and real controls remain separated
```

Ordinary noise and starvation are controller pressure, not run-ending failures.
Future leakage remains a hard validity stop.

## 4. Machine Artifacts

This stage adds:

```text
experiments/gate_floor_noise_controller.py
evaluation/gate_floor_noise_controller.py
experiments/create_gate_floor_noise_controller_report.py
tests/test_gate_floor_noise_controller.py
```

It also extends:

```text
evaluation/unified_telemetry_schema.py
experiments/progress_logging.py
```

Default report:

```text
runs/contracts/gate_floor_noise_controller.json
```

Usage:

```bash
python experiments/create_gate_floor_noise_controller_report.py
```

## 5. Live Telemetry

Long runs should expose:

```text
gate
g_next
d_gate
gCredit
gPress
eSurv
rNoise
rStarv
```

These fields show whether the system is filtering noise, releasing starved real
edges, or holding the gate floor.

## 6. Exit Criteria

This stage is complete when:

```text
gate floor can rise when edge noise is high
gate floor can fall when real edges are starved
ordinary edge noise does not abort the search
future leak acts as a validity hard stop
every decision has a parameter trajectory
every decision has an injected evidence ledger
every decision has a replay record
live progress logging exposes gate-floor decisions
```

Next stage:

```text
Causal Reachability Controller
```
