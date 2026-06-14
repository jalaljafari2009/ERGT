# Memory State Instrumentation

## 1. Purpose

This stage makes ERGT memory observable before adding any new memory controller.

The project now has an open-control philosophy and a unified telemetry schema.
The next requirement is to measure memory state itself:

```text
W_t is not useful just because it exists.
It must be stable, non-rigid, non-noisy, finite, and control-comparable.
```

This stage does not change the model objective and does not add cross-batch
memory. It instruments the memory already used by GeoAttention v2.

## 2. Scope

Current ERGT v2 memory is:

```text
layer-local memory inside one forward pass
```

It is passed from one layer to the next as `geometry_memory`. It is not yet:

```text
cross-batch memory
dataset-level memory
retrieval memory
long-term persistent history
```

The instrumentation must preserve that distinction. A positive result here
means only that the current memory state is measurable and usable for later
controller decisions.

## 3. Required Metrics

Every memory-capable run should expose:

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

Interpretation:

- `memory_stability`: alignment between memory and the current stable update.
- `memory_turnover`: average change from previous memory.
- `memory_edge_density`: valid edges above a declared threshold.
- `memory_persistence`: cosine similarity to previous memory.
- `memory_spectral_entropy`: normalized concentration of singular values.
- `memory_effective_rank`: effective dimensionality of the memory matrix.
- `memory_rigidity`: pressure from over-concentration or low rank.
- `noise_risk`: pressure from excessive turnover or rigidity.

## 4. Controls

Instrumentation must run over:

```text
real_memory
random_memory
shuffled_memory
instantaneous
no_memory
```

These controls do not prove memory is useful. They prove the measuring system
can observe memory and controls under the same field names.

## 5. Machine Artifacts

This stage adds:

```text
evaluation/memory_state_instrumentation.py
experiments/create_memory_state_instrumentation_report.py
tests/test_memory_state_instrumentation.py
```

Default report:

```text
runs/contracts/memory_state_instrumentation.json
```

Usage:

```bash
python experiments/create_memory_state_instrumentation_report.py
```

## 6. Live Telemetry

The training progress logger now flattens memory fields from geometry
diagnostics. Long runs should show at least:

```text
mStab
mTurn
mPers
mRigid
nRisk
```

These are diagnostic signals, not hard stops. Later controllers may use them as
growth and restraint evidence.

## 7. Exit Criteria

This stage is complete when:

```text
required memory fields are declared in the unified schema
required memory fields are emitted by instrumentation
real/random/shuffled/instantaneous/no-memory controls are instrumented
metrics are finite
the report states that memory scope is layer-local within a forward pass
GeoAttention diagnostics expose the memory metrics without changing attention
```

Next stage:

```text
Attention Rigidity and Collapse Monitor
```
