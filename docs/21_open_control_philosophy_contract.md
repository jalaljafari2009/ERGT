# Open Control Philosophy Contract

## 1. Purpose

This contract governs all post-Run-02 adaptive-control work.

The project should not pre-emptively choke geometry, memory, or causal
reachability with fixed scientific ceilings. The system must be allowed to grow
when the evidence supports growth and must be restrained when evidence shows
collapse, noise, unfair controls, or invalid causality.

The contract separates:

```text
hard stops     = safety and validity failures
soft pressures = risk signals that shape controller decisions
growth signals = evidence that a parameter deserves more budget
```

## 2. Core Rule

```text
Do not protect attention from geometry by default.
Let geometry and memory compete.
Require evidence before assigning scientific credit.
```

This means `geo/qk`, entropy, max probability, and memory turnover are not
scientific hard ceilings. They are pressure signals. If geometry grows while
loss slope, stability, and real-vs-control separation improve, the controller
may continue to grow geometry.

## 3. Hard Stops

Only these classes are hard stops:

```text
future leakage
NaN or Inf
loss explosion
severe attention collapse
control unfairness
```

These invalidate the run or make it uninterpretable. They are not ordinary
regularization pressures.

## 4. Soft Pressures

These are not hard stops:

```text
geo_to_qk_ratio
attention_entropy_drop
mean_max_probability
memory_turnover
memory_rigidity
control_penalty
```

They affect the controller score, but they do not automatically forbid growth.
The controller must report how much each pressure contributed to a decision.

## 5. Adaptive Parameters

The next open-control stages may adapt:

```text
alpha
memory_decay
memory_eta
gate_floor
distance_norm_scale
causal_reachability
```

Every adaptive parameter must have both:

```text
growth evidence
restraint evidence
```

No adaptive parameter may be changed without recording:

```text
previous value
next value
delta
decision
credit score
risk pressure
reason summary
```

## 6. Trend Requirement

The controller must not use one validation-loss delta as the main decision.

Allowed:

```text
rolling slope
EMA-smoothed advantage
multi-point attribution probe
late-window trend
control-separated trend
```

Disallowed:

```text
one point got better -> grow parameter
one point got worse  -> shrink parameter
```

## 7. Attribution Requirement

Before changing multiple parameters together, the system must estimate which
mechanism is likely responsible:

```text
alpha too weak
memory too noisy
memory too rigid
gate_floor starving relations
distance scale too compressed
causal reachability too narrow or too broad
```

Attribution may begin as a heuristic score, but it must be logged and auditable.

## 8. Control Requirement

A gain is not credited to real relational geometry unless real separates from
controls:

```text
real > random
real > shuffled
real memory > no-memory
real memory > instantaneous
```

If random or shuffled grows and improves at the same rate as real, the result is
classified as generic regularization until proven otherwise.

## 9. Runtime Bounds vs Scientific Bounds

Runtime bounds are allowed:

```text
max_alpha
min_decay
max_eta
max_runtime
max_memory
```

But runtime bounds are not scientific conclusions. A runtime bound only says the
experiment should not run away computationally. It does not say the theory
requires that bound.

## 10. Machine Contract

This stage adds:

```text
evaluation/open_control_philosophy_contract.py
experiments/create_open_control_philosophy_contract_report.py
```

Usage:

```bash
python experiments/create_open_control_philosophy_contract_report.py
```

Default output:

```text
runs/contracts/open_control_philosophy_contract.json
```

## 11. Exit Criteria

This stage is complete when the machine contract validates that:

```text
hard stops are limited to safety and validity
soft pressures are not hard stops
adaptive parameters have growth and restraint evidence
telemetry covers loss, attention, memory, and controls
controller obligations require trends instead of single-point deltas
controller obligations require random and shuffled controls
```

Only after this contract is in place should the project define the unified
telemetry schema for memory, attribution, and open parameter control.
