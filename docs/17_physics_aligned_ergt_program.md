# Physics-Aligned ERGT Program

## 1. Purpose

This document defines the updated ERGT program after the Phase 3 evidence made
the original GeoAttention-only path insufficient.

The Phase 3 stable-base, confirm-seed, and ratio-matched results showed a useful
but incomplete signal:

- `real_d` can beat baseline or shuffled controls in some runs.
- `random_d` can still beat `real_d` under matched geometry strength.
- Gate 1 remains blocked until relational structure is separated from generic
  bias, smoothing, normalization, or control artifacts.

From this point forward, the project should treat this document as the movement
standard for post-Phase-3 work. Later implementation may refine details, but it
must preserve the core discipline:

```text
measure -> control -> observe -> validate -> inject -> regularize
```

For the reader-facing article that explains the same strengthened view in a
less procedural form, see `docs/18_ergt_position_paper.md`. This document is the
execution standard; the position paper is the conceptual orientation.

## 2. Claim Boundary

ERGT uses informational physics as an organizing inspiration, not as a direct
physical equivalence.

Required claim boundaries:

```text
hidden state != physical field
Phi != consciousness
causal mask != causal geometry
pairwise distance != full geometry
low entropy != good structure
baseline win != relational proof
EMA smoothing != meaningful memory
```

The permitted near-term claim is:

```text
Hidden states may carry a stable, compressible, causal, reconstructible
relational geometry that can improve or stabilize attention.
```

The forbidden near-term claims are:

```text
Transformer spacetime has emerged.
Phi is awareness.
ERGT proves general intelligence.
```

## 3. Mother Pipeline

The broad theory remains:

```text
Dynamics -> Relations -> Structure -> Compression -> Geometry -> Memory -> Reasoning -> Intelligence
```

The operational ERGT pipeline is:

```text
HiddenStates
-> W relations
-> local coherence / entropy / salience
-> Phi information potential
-> W_t stable memory
-> causal shortest-path geometry
-> GeoAttention
-> reasoning paths
-> intelligence-space evaluation
```

## 4. Physics-to-ERGT Map

The operational mapping is:

```text
Psi                         -> hidden states H_l(t)
rho = |Psi|^2               -> fixed salience definition
local coherence C           -> local cosine/neighborhood coherence
coherence gradient G        -> boundary sharpness
local entropy S_local       -> local relational entropy
awareness potential A       -> information potential Phi
w_ij                        -> relation weight W_ij
d_ij = -log(w_ij)           -> edge cost -log(W_ij + eps)
boundary data               -> allowed causal prefix/context
reconstruction deficit      -> information not reconstructible from allowed context
finite-speed propagation    -> causal reachability / no future edges
```

This mapping is operational and testable. It is not ontological identity.

## 5. Phase 0: Claim and Measurement Contracts

Goal: lock definitions before any new experiment.

Required contracts:

```text
valid_edge = causal_lower_triangular & non_diagonal & non_padding
diagonal policy fixed
causal policy fixed
normalization fixed
clipping fixed
rho/salience definition fixed
spectral operator fixed
reconstruction protocol fixed
Phi formula fixed
```

The salience contract must choose one definition or one fixed weighted
combination before a run:

```text
hidden_norm
graph_degree
attention_mass
fixed weighted combination
```

The spectral operator contract must choose one target before comparison:

```text
W
affinity exp(-D)
graph Laplacian L = Deg - W
normalized graph Laplacian
```

Deliverable:

```text
measurement_contract_report.json
```

Pass condition: all metrics are testable on small tensors, and no metric uses a
future token.

## 6. Phase 1: Strict W-Level Controls

Goal: make `real`, `random`, and `shuffled` differ only in relational
arrangement.

Correct pipeline:

```text
H -> W_family -> valid_edge_mask -> D_family -> normalization -> clipping -> alpha calibration
```

Invalid pipeline:

```text
H -> real D normalized -> random/shuffled D
```

Required controls:

```text
real_W
random_W with matched scale/distribution
shuffled_W only inside valid causal region
```

Required equality:

```text
same valid region
same diagonal policy
same causal policy
same normalization
same clipping
matched geo_to_qk_ratio
```

Separation must be reported:

```text
Sep(W_real, W_random) > 0
Sep(W_real, W_shuffled) > 0
```

`Sep` must be operationalized with effect size, bootstrap confidence interval,
or a permutation/randomization test.

## 7. Phase 2: Relational Field Observer

Goal: observe whether `H -> W -> D` creates non-trivial structure before any
intervention.

Required metrics:

```text
relational entropy
local relational entropy
spectral entropy
effective rank
coherence C
coherence gradient G
neighborhood overlap
layer-to-layer similarity
step-to-step stability
real vs random separation
real vs shuffled separation
```

Pass condition:

```text
real W/D separates from random and shuffled
real W/D is not uniform
real W/D is not diagonal-dominated
real W/D is not saturated
real neighborhoods are stable across nearby layers or steps
```

## 8. Phase 3: Resonant-Response Observer

Goal: probe the field without changing training.

Process:

```text
H_before -> W_before/D_before/Phi_before
controlled perturbation
H_after -> W_after/D_after/Phi_after
measure Delta C, Delta S, Delta neighborhoods, Delta Phi
reset to before
```

Pass condition:

```text
real response is stronger or more stable than random/shuffled response
probe does not alter the training trajectory
response is not explained by scale artifacts
```

## 9. Phase 4: Information Potential Phi

Goal: define the AI analogue of the physics awareness potential as an
operational stability selector, not as consciousness.

Formula:

```text
Phi =
coherence^a
* coherence_gradient^b
* low_local_entropy^c
* salience^d
* stability^e
* reconstruction_score^f
* causal_validity
* anti_collapse
```

`anti_collapse` must penalize:

```text
uniform W
diagonal domination
over-sparsity
single-token lock-in
entropy collapse
```

Pass condition:

```text
high Phi predicts better predictability, stability, or attention order
Phi_high is not equivalent to low entropy only
real Phi > random/shuffled Phi
```

## 10. Phase 5: Reconstruction Gate

Goal: require relational structure to be reconstructible from allowed causal
context.

Hidden-state deficit:

```text
reconstruction_deficit_h = ||h_i - R(H_<i)||^2
```

Relational deficit:

```text
reconstruction_deficit_W = ||W_i - R_W(H_<i)|| over allowed past positions only
```

The reconstructor must not receive the target state or any future-token
information. Otherwise the reconstruction gate becomes trivial.

Pass condition:

```text
deficit(real) < deficit(random/shuffled)
high-Phi regions reconstruct better
no future leakage
```

If a relation is only explainable with future-token access, it is invalid.

## 11. Phase 6: Phi-Gated Relational Memory Observer

Goal: observe memory as persistence of relational structure before using it in
attention.

Formula:

```text
W_t = decay * W_{t-1} + eta * Phi_gate * stable_update(H_t)
```

`stable_update` must be similarity-based:

```text
similarity high -> W_ij increases
dissimilarity/noise -> W_ij decays
```

Acceptable update kernels include:

```text
W_ij = sigmoid(gamma * cosine(h_i, h_j) + beta)
Delta W_ij proportional to exp(-||h_i - h_j||^2 / sigma^2)
```

Required controls:

```text
real W_t
random W_t
shuffled W_t
eta=1 instantaneous
generic smoothing
no-memory
```

Pass condition:

```text
real W_t > random W_t
real W_t > shuffled W_t
real W_t > instantaneous W
real W_t > generic smoothing
no future leakage
no collapse
```

If memory does not beat instantaneous or generic smoothing controls, it is not
meaningful relational memory.

## 12. Phase 7: Causal Shortest-Path Geometry

Goal: make geometry causal, not merely pairwise.

Definition:

```text
edge_cost_ij = -log(W_t_ij + eps)
D_causal(i,j) = shortest path over allowed causal edges
```

Allowed edges:

```text
j <= i
valid_edge == true
```

Pass condition:

```text
future edges forbidden
D_causal finite only where causally valid
real D_causal separates from random/shuffled
D_causal adds signal beyond pairwise D
```

## 13. Phase 8: GeoAttention v2

Goal: inject only stable, causal, validated geometry.

Formula:

```text
logits = QK^T / sqrt(d) - alpha * D_stable
```

Construction:

```text
H -> W_t -> D_causal -> normalized D_stable
```

Required controls:

```text
baseline
alpha_zero
real stable causal D
random stable causal D
shuffled stable causal D
instantaneous real D
pairwise real D
no-memory real D
```

Pass condition:

```text
real stable causal D > baseline
real stable causal D > random
real stable causal D > shuffled
real stable causal D > instantaneous
real stable causal D > pairwise/no-memory
alpha_zero ~= baseline
no future leakage
replicated across seeds
```

Beating baseline alone is not enough.

## 14. Phase 9: Auxiliary Physics-Inspired Loss

Goal: regularize only after the geometry has already passed intervention gates.

Loss:

```text
L = L_lm + lambda * regularizer
```

Allowed regularizers:

```text
spectral stability
neighborhood stability
reconstruction consistency
causal consistency
anti-collapse
```

Pass condition:

```text
validation or stability improves
model does not become rigid
random/shuffled do not receive the same gain
spectral entropy reduction does not cause collapse
```

## 15. Phase 10: Complete ERGT Architecture

Goal: integrate the validated components into a single architecture.

Pipeline:

```text
HiddenStates
-> RelationalGraph
-> Phi
-> DynamicGraphMemory
-> EmergentDistance
-> CausalGeometry
-> GeoAttention
```

Required comparisons:

```text
Transformer baseline
ERGT without memory
ERGT with memory
ERGT with causal geometry
ERGT with spectral regularization
matched-parameter GPT-style baseline
```

Pass condition: improvements must preserve the proven role of `W_t` and `D`,
not merely reflect extra capacity or a generic bias.

## 16. Phase 11: Reasoning Paths

Goal: test reasoning as traversal through stable relational geometry.

Definition:

```text
Reasoning = navigation of stable relational geometry
```

Candidate metrics:

```text
multi-hop path consistency
low-cost path alignment with correct answer
counterfactual edge removal sensitivity
long-range dependency score
path stability across paraphrases
compositional task performance
```

Pass condition:

```text
real geometry paths are more stable and explanatory than controls
removing high-Phi/high-stability edges harms task behavior
random/shuffled paths do not explain the same behavior
```

This can support reasoning-path analysis. It does not prove general reasoning.

## 17. Phase 12: Intelligence Space Evaluation

Goal: evaluate the long-term ERGT definition of intelligence.

Definition:

```text
Intelligence Space =
stable, compressible, causal, reconstructible relational geometry over hidden states
```

The broad theoretical definition is:

```text
Intelligence =
discovery + compression + stabilization + traversal
of relational structures
```

Four evaluation axes:

```text
Discovery: find relations that matter
Compression: reduce complexity without collapse
Stabilization: persist across layers, steps, or contexts
Traversal: use relational paths to solve tasks
```

Long-term pass condition:

```text
real relational structures self-organize
structures are reconstructible
structures improve reasoning-specific tasks
structures are not explained by random/shuffled controls
effects replicate across seeds and datasets
```

## 18. Final Gate

A strong claim requires all of the following:

```text
strict controls valid
real W separates from random/shuffled
real Phi is higher and anti-collapse-safe
reconstruction deficit is lower for real
real W_t is more stable than controls
causal shortest-path D beats pairwise/no-memory controls
GeoAttention v2 beats all required controls
auxiliary loss does not create collapse
reasoning paths are testable and control-sensitive
no future leakage
replication across seeds
```

## 19. Governing Rule

Every phase must be falsifiable.

```text
real > random/shuffled
```

must be demonstrated for relation, stability, reconstruction, memory, causal
geometry, attention, and later reasoning. If a phase fails that test, the next
phase pauses and the current operational definition must be redesigned.
