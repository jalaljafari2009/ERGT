# ERGT Project Charter

## Purpose

ERGT exists to test a relational theory of representation and attention.

The project starts from the idea that informational systems are not primarily
collections of isolated objects or vectors. They are systems of interactions.
Stable structures, concepts, memory, geometry, and reasoning are treated as
emergent consequences of relational dynamics.

The first executable goal is deliberately narrower:

```text
Test whether correlation-induced distance can improve or stabilize attention in
a controlled transformer comparison.
```

## Core Research Question

Can a transformer derive a useful attention geometry from relations between
hidden states instead of relying only on direct dot-product attention?

The first proof stage is centered on this chain:

```text
Correlation -> Relational Graph -> Distance -> Geometry-Biased Attention
```

## Philosophical Position

ERGT uses a relational ontology:

```text
Relations are primary.
Objects, vectors, space, concepts, memory, and reasoning are derived or
stabilized structures.
```

In the implementation, vectors still appear as computational coordinates. This
does not make vectors ontologically primary. Hidden states are treated as a
coordinate substrate from which relational structure is extracted.

The theory should therefore privilege relational invariants over raw vector
coordinates.

## Scope of Version 1

Version 1 is limited to the minimum publishable mechanism:

```text
RelationalGraph + EmergentDistance + GeoAttention
```

The only required architectural change is the replacement or augmentation of
standard attention with a geometry-biased attention term:

```text
A = Softmax(QK^T / sqrt(d) - alpha * D)
```

where:

- `W` is a relational graph induced from hidden-state correlations.
- `D` is a distance matrix derived from `W`.
- `alpha` controls the strength of the geometry bias.

## In Scope

- Reproducing a controlled transformer baseline.
- Extracting relational graphs from hidden states.
- Converting relational strength into distance.
- Measuring whether the induced distance has stable structure.
- Testing GeoAttention against strict baselines.
- Running ablations with real, shuffled, random, and disabled distance.
- Logging graph and geometry metrics alongside language-model metrics.
- Defining gate conditions before moving to memory or causal geometry.

## Out of Scope for Version 1

The following are important to the long-term theory but should not be required
for the first proof:

- Persistent cross-batch memory.
- Full graph-memory architecture.
- Causal geometry.
- Spectral complexity loss.
- Continuous relational fields.
- Explicit reasoning benchmarks.
- Claims about general intelligence.
- Claims that ERGT fully explains concept formation or memory.

These topics can be developed only after the first proof stage produces a
credible result.

## Non-Negotiable Experimental Controls

All comparisons must hold these constant:

- Dataset
- Tokenizer
- Context length
- Model size or parameter budget
- Optimizer
- Learning-rate schedule
- Batch size or effective batch size
- Training steps or token budget
- Evaluation protocol
- Random seed protocol

The model comparison must isolate the effect of relational geometry. If multiple
things change at once, the result is not scientifically useful.

## First Proof Stage

The first proof stage contains four phases:

```text
Phase 0: Baseline Transformer
Phase 1: Relational Graph Observer
Phase 2: Emergent Distance
Phase 3: GeoAttention v1
```

The decisive phase is Phase 3.

ERGT-v1 must be compared against:

- Standard Transformer baseline.
- GeoAttention with real relational distance.
- GeoAttention with shuffled distance.
- GeoAttention with random distance.
- GeoAttention with `alpha = 0`.

The real distance condition must outperform or stabilize relative to the control
distance conditions for the geometric claim to remain credible.

## Success Criteria

ERGT-v1 is considered promising if it demonstrates at least one of the following
under controlled comparison:

- Lower validation loss.
- Lower perplexity.
- Better long-range dependency behavior.
- Better training stability.
- Comparable quality with improved attention sparsity or interpretability.

Any improvement must be checked against runtime and memory cost. A result that
is slightly better but computationally impractical should be treated as
incomplete rather than successful.

## Failure Criteria

The first proof stage fails or requires redesign if:

- Real relational distance performs no better than shuffled or random distance.
- The geometry term only duplicates standard attention without adding signal.
- Training becomes unstable across seeds.
- The graph collapses into trivial uniform or near-identity structure.
- The method is too expensive for the target context lengths.

Failure at this stage does not disprove the broader philosophy, but it does
invalidate the current operational mechanism.

## Gate to Later Phases

The project should not move to graph memory, causal geometry, spectral
complexity, or relational fields until Phase 3 shows a meaningful result.

The gate condition is:

```text
GeoAttention with real D must show a measurable advantage or stability benefit
over both the baseline and distance-control ablations.
```

If the gate is not passed, the next work should revise:

- The relational kernel.
- The distance transformation.
- The normalization of `D`.
- The schedule or parameterization of `alpha`.
- The layer at which relational geometry is injected.

## Research Discipline

The project should protect the depth of the theory without overclaiming early
results.

The philosophical claim is broad:

```text
Intelligence emerges from the discovery, compression, stabilization, and
traversal of relational structures.
```

The first experimental claim is narrow:

```text
Correlation-induced relational distance can improve or stabilize transformer
attention.
```

Keeping these levels separate is essential. The first paper should prove the
mechanism before claiming the full theory.
