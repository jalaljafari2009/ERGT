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

The goal is adaptive optimization over unknown degrees of freedom, not a
sequence of brittle flags. Ordinary warning flags must not stop the program.
They should change controller pressure, parameter budget, search direction, or
revision labels while preserving a replayable decision history.

Attention behavior is part of the controller's observability surface. It should
be used to understand whether the system is finding a useful operating region,
but it must not be treated as the final scientific claim by itself. Attention
signals become behavioral evidence that is combined with loss trend, memory
state, attribution, and real-vs-control separation.

## 2. Core Rule

```text
Do not protect attention from geometry by default.
Let geometry and memory compete.
Require evidence before assigning scientific credit.
Use attention behavior to understand the search trajectory.
```

This means `geo/qk`, entropy, max probability, and memory turnover are not
scientific hard ceilings. They are pressure signals. If geometry grows while
loss slope, stability, and real-vs-control separation improve, the controller
may continue to grow geometry.

The controller should look for an interpretable attention regime, not only a
lower loss:

```text
not collapsed
not uniformly indifferent
real condition separated from random and shuffled controls
head/layer diversity preserved enough to avoid lock-in
geo/qk strong enough to affect attention without taking over blindly
```

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

For `random` and `shuffled`, control fairness also includes RNG isolation:

```text
random/shuffled control RNG must not consume training/dropout/sampler RNG
```

The control graph may be random, but generating it must not shift any other
stochastic path in the training run.

## 4. Soft Pressures

These are not hard stops:

```text
geo_to_qk_ratio
attention_entropy_drop
mean_max_probability
attention_sparsity
head_attention_diversity
layer_attention_diversity
geometry_takeover_score
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

Across a run, the system must also record:

```text
full parameter trajectory
injected evidence ledger
observed telemetry window
controller state snapshot
decision replay record
uncertainty and misdiagnosis labels
```

These records are required so later behavior analysis can determine where the
controller found useful regions, where it misread the state, and which injected
signals caused each action.

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

Random and shuffled controls must also be generated with isolated deterministic
generators. A control that advances global training RNG is not a fair control,
because it can alter dropout masks or later sampler randomness.

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
telemetry records parameter search and injected evidence
ordinary risk flags cannot abort optimization
controller obligations require replayable decisions
controller obligations require trends instead of single-point deltas
controller obligations require random and shuffled controls
controller obligations require RNG isolation for random and shuffled controls
```

Only after this contract is in place should the project define the unified
telemetry schema for memory, attribution, and open parameter control.
