# Operational Definitions

## 1. Purpose

This document converts the ERGT theoretical vocabulary into measurable research
objects.

The project should not use terms such as relation, distance, geometry, memory,
concept, or reasoning as vague metaphors. Each term must have an operational
definition, a tensor-level representation, and a measurable criterion.

## 2. State

The ERGT state is:

```text
S = (X, W)
```

where:

- `X` is the hidden-state tensor.
- `W` is the induced relational graph.

For the first proof stage, `X` is the computational coordinate substrate and `W`
is the relational structure extracted from it.

Tensor shape:

```text
X: [batch, sequence, hidden_dim]
W: [batch, heads_or_1, sequence, sequence]
```

The exact head dimension is implementation-dependent. A first implementation may
use one shared graph per layer, then later test per-head graphs.

## 3. Relation

The operational relation between positions `i` and `j` is the strength of
interaction inferred from hidden-state correlation.

Initial kernel:

```text
K_ij = sigmoid((x_i^T x_j) / sqrt(d))
```

where:

- `x_i` and `x_j` are hidden states.
- `d` is the hidden dimension or projected relation dimension.
- `K_ij` is the instantaneous relation strength.

For Phase 1, the relation graph can be:

```text
W = K
```

For later dynamic memory phases:

```text
W(l+1) = (1 - eta)W(l) + etaK
```

## 4. Relation Validity

A relational graph is meaningful only if it is non-trivial.

Minimum checks:

- It is not uniform.
- It is not only the identity matrix.
- It is not numerically saturated near 0 or 1.
- It changes with input content.
- It has measurable structure across layers or examples.

Recommended metrics:

- Mean relation strength.
- Relation variance.
- Graph entropy.
- Sparsity under threshold.
- Degree distribution.
- Layer-to-layer similarity.
- Input perturbation stability.
- Difference from random and shuffled controls.

## 5. Structure

Structure is repeated, stable, or compressible organization inside `W`.

Operationally, structure exists when `W` shows at least one of:

- Non-random cluster organization.
- Stable neighborhoods.
- Recurrent high-strength pathways.
- Low effective rank relative to sequence length.
- Similar graph patterns across related inputs.
- Predictive value for attention behavior or model loss.

Suggested measurements:

```text
graph_entropy(W)
effective_rank(W)
community_score(W)
stability(W_layer_t, W_layer_t+1)
stability(W_input, W_perturbed_input)
```

## 6. Distance

Distance is the cost induced by relational strength.

Initial definition:

```text
D_ij = -log(W_ij + epsilon)
```

Interpretation:

```text
strong relation -> low distance
weak relation   -> high distance
```

Implementation requirements:

- `epsilon` must avoid `log(0)`.
- `D` should be normalized or clipped if it destabilizes attention logits.
- The diagonal should be handled explicitly.
- Causal masking must remain separate from distance.

Possible normalization:

```text
D_norm = (D - mean(D)) / (std(D) + epsilon)
```

or:

```text
D_norm = D / mean(D)
```

The chosen normalization must be recorded in experiment configs.

## 7. Distance Validity

A distance matrix is meaningful only if it carries structure beyond raw attention
duplication.

Minimum checks:

- Real `D` differs from random `D`.
- Real `D` differs from shuffled `D`.
- Real `D` affects attention differently than `QK^T` alone.
- Real `D` does not simply collapse to a diagonal penalty.

Suggested measurements:

- Correlation between `D` and standard attention logits.
- Correlation between `D` and final attention weights.
- Distance entropy.
- Distance variance.
- Triangle inequality violation rate, if metric-like behavior is claimed.
- Cluster quality under distance-based clustering.

The project does not need to prove that `D` is a perfect metric in Phase 2. It
only needs to show that `D` is a useful induced distance structure.

## 8. Geometry

Geometry is the organized structure induced by the distance matrix `D`.

Operational definition:

```text
Geometry = stable, non-trivial neighborhood and path structure induced by D
```

Geometry is present if `D` supports measurable:

- Neighborhoods.
- Clusters.
- Barriers.
- Low-dimensional tendencies.
- Stable relative distances.
- Predictive influence on attention.

Suggested metrics:

- Intrinsic dimension estimate.
- Distance-based clustering quality.
- Neighborhood overlap across layers.
- Neighborhood overlap under perturbation.
- Spectral profile of `W` or transformed `D`.
- Attention improvement when using real `D` compared with control `D`.

## 9. Concept

For the long-term theory:

```text
Concept = stable localized relational structure
```

Operationally, a candidate concept is a region of `W` or `D` that:

- Is internally coherent.
- Is separated from other regions.
- Persists across related contexts.
- Aligns with interpretable token groups, semantic units, or task-relevant
  features.

Version 1 should not claim concept discovery unless these criteria are directly
tested.

## 10. Memory

Memory is persistence of relational structure.

Operational definition:

```text
Memory = stability of W across layers, time, updates, or contexts
```

For Phase 4 and beyond, memory can be implemented as:

```text
W(l+1) = (1 - eta)W(l) + etaK
```

Measurements:

- Layer-to-layer persistence.
- Step-to-step persistence.
- Stability under paraphrase or perturbation.
- Recovery of previous relational neighborhoods.
- Effect of persistent `W` on loss or downstream behavior.

Version 1 may measure persistence, but it should not claim full graph memory
unless `W` is explicitly carried forward and shown to affect future computation.

## 11. Reasoning

Reasoning is traversal through stable relational geometry.

Operational candidates for later phases:

- Shortest or low-cost paths in `D`.
- Multi-hop movement between stable relational regions.
- Path consistency across equivalent prompts.
- Improved performance on tasks requiring compositional or long-range
  dependency traversal.

Version 1 should not claim reasoning improvement unless reasoning-specific tasks
are included.

## 12. Intelligence

The broad theoretical definition is:

```text
Intelligence = discovery, compression, stabilization, and traversal of
relational structures
```

This is not a Phase 1-3 metric.

For early stages, it should remain a guiding theory, not an experimental claim.

## 13. GeoAttention

GeoAttention uses induced distance as an attention bias:

```text
A = Softmax(QK^T / sqrt(d) - alphaD)
```

where:

- `QK^T / sqrt(d)` is the standard attention logit term.
- `D` is the induced relational distance.
- `alpha` is a learnable or configured scale.

Operational question:

```text
Does real D improve or stabilize attention compared with no D, random D, or
shuffled D?
```

This question defines the first proof stage.

## 14. Controls

Every experiment using `D` must include controls.

Required controls:

- `alpha = 0`.
- Random distance matrix.
- Shuffled distance matrix.
- Optional detached `D` vs trainable-through-`D`.
- Optional per-layer and per-head variants.

Without these controls, improvements cannot be attributed to relational
geometry.

## 15. Gate Definitions

The project may pass from Phase 3 to Phase 4 only if:

```text
GeoAttention(real D) > TransformerBaseline
GeoAttention(real D) > GeoAttention(random D)
GeoAttention(real D) > GeoAttention(shuffled D)
```

The comparison may be based on lower loss, lower perplexity, better stability,
or a clearly justified long-range metric.

If the result is ambiguous, the project should revise the operational definition
of `W`, `D`, or `alpha` before moving forward.

## 16. Documentation Requirements

Every experiment must record:

- Graph construction formula.
- Distance construction formula.
- Normalization method.
- Whether gradients pass through `W` or `D`.
- `alpha` initialization and schedule.
- Dataset and tokenizer.
- Model size.
- Training budget.
- Seed list.
- Baseline and ablation results.

This metadata is part of the scientific result. Without it, experiments are not
interpretable.
