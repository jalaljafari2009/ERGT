# Unified Telemetry Schema

## 1. Purpose

This stage defines one telemetry language for all open adaptive ERGT work.

The next stages will adapt memory, gate-floor, distance scale, causal
reachability, and alpha. Without a shared schema, every controller would invent
slightly different names for the same signals, making attribution and
comparison unreliable.

The schema defines:

```text
what must be logged
when it must be logged
which fields are canonical
which Run-02 names are accepted as aliases
which fields are required for live 100-step diagnosis
```

## 2. Design Rule

Telemetry must cover all mechanisms that can explain a result:

```text
loss trend
control separation
attention rigidity
memory state
geometry and distance scale
adaptive parameter changes
attribution
safety
runtime
```

If a controller changes a parameter without logging the evidence and risk behind
that change, the run is not interpretable.

## 3. Canonical Categories

The schema groups fields into:

```text
identity
data_model
loss_trend
control_separation
attention_rigidity
memory_state
geometry_distance
adaptive_parameters
attribution
safety
runtime
```

Every field belongs to exactly one category.

## 4. Minimum Live Fields

Every live 100-step display must include at least:

```text
step
condition
validation_loss
alpha
alpha_next
alpha_delta
alpha_decision
loss_slope_gain
baseline_centered_improvement
geo_to_qk_ratio
attention_entropy
mean_max_probability
rigidity_risk
control_penalty
```

Run-02 currently uses some compatibility names, for example
`alpha_effective` and `adaptive_slope_gain`. These are accepted as aliases, but
ERGT-03 should prefer canonical names.

## 5. Adaptive Parameter Fields

Every adaptive parameter must expose:

```text
<parameter>_next
<parameter>_delta
<parameter>_decision
<parameter>_credit
<parameter>_risk_pressure
```

For the next stages, this applies to:

```text
alpha
memory_decay
memory_eta
gate_floor
distance_norm_scale
causal_reachability
```

## 6. Memory Fields

Memory cannot be judged only from loss. Required memory fields include:

```text
memory_decay
memory_eta
gate_floor
memory_stability
memory_turnover
memory_edge_density
memory_persistence
memory_spectral_entropy
memory_effective_rank
memory_rigidity
noise_risk
```

These fields are the basis for the next stage:

```text
Memory State Instrumentation
```

## 7. Machine Schema

This stage adds:

```text
evaluation/unified_telemetry_schema.py
experiments/create_unified_telemetry_schema_report.py
```

Usage:

```bash
python experiments/create_unified_telemetry_schema_report.py
```

Default output:

```text
runs/contracts/unified_telemetry_schema.json
```

## 8. Exit Criteria

This stage is complete when the machine schema validates that:

```text
there are no duplicate field definitions
every field has category, type, cadence, and meaning
all Open Control required telemetry is covered
Run-02 aliases remain compatible
adaptive parameter change fields are declared
hard-stop fields are declared
minimum live fields are declared
```

Only after this schema exists should the project implement the memory-state
instrumentation layer.
