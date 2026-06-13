# ERGT: Emergent Relational Geometry Transformer

ERGT is a research project for testing whether useful geometry can emerge from
relational dynamics inside a transformer-like architecture.

The central hypothesis is that geometry does not need to be predefined as an
external positional or metric structure. Instead, relational correlations between
hidden states can induce distance, distance can induce geometry, and that
geometry can improve or stabilize attention.

```text
Dynamics -> Relations -> Structure -> Compression -> Geometry -> Memory -> Reasoning -> Intelligence
```

## Updated Program After Phase 3 Evidence

The early Phase 3 runs produced a useful but incomplete signal: `real_d` can
improve over baseline or shuffled controls in some settings, but `random_d` can
still beat `real_d` under matched geometry strength. That means ERGT should not
continue as a simple GeoAttention extension.

From this point forward, post-Phase-3 work is governed by
`docs/17_physics_aligned_ergt_program.md`.

For a reader-facing position paper that explains the strengthened ERGT vision
and the meaning of "intelligence space", see
`docs/18_ergt_position_paper.md`.

The longer article manuscript artifacts are kept in
`docs/research_article/`.

The next post-evidence injection strategy, based on loss-slope feedback and
competitive alpha growth, is defined in
`docs/19_adaptive_competitive_alpha_plan.md`.

Run-02 evidence consolidation, which must happen before opening more adaptive
parameters, is defined in `docs/20_run02_evidence_consolidation.md`.

The updated movement standard is:

```text
measure -> control -> observe -> validate -> inject -> regularize
```

The updated operational path is:

```text
HiddenStates -> W -> Phi -> W_t -> D_causal -> D_stable -> GeoAttention -> Reasoning
```

## Core Claim

ERGT starts from a relational view of information:

```text
Information systems are fundamentally interaction systems.
Objects, vectors, concepts, memory, and reasoning are stabilized or compressed
forms of relational structure.
```

For the first proof stage, the large philosophical claim is intentionally reduced
to a smaller experimental claim:

```text
Correlation-induced relational distance can improve or stabilize transformer
attention under controlled comparison.
```

## Research Pipeline

The first executable path focuses on phases 0 through 3.

### Phase 0: Baseline Transformer

Reproduce a controlled transformer baseline using the same dataset, tokenizer,
training budget, optimizer, context length, and evaluation metrics that will be
used for ERGT.

The baseline is not meant to reinvent the transformer. It is a controlled
experimental reference.

### Phase 1: Relational Graph Observer

Construct a relational graph from hidden states without changing attention:

```text
K_ij = sigmoid((x_i^T x_j) / sqrt(d))
```

This phase tests whether non-trivial relational structure appears inside the
baseline model.

### Phase 2: Emergent Distance

Convert relational strength into distance:

```text
D_ij = -log(W_ij + epsilon)
```

This phase tests whether correlation-induced distance produces meaningful
geometric structure.

### Phase 3: GeoAttention v1

Replace standard attention with geometry-biased attention:

```text
Standard: A = Softmax(QK^T / sqrt(d))
ERGT:     A = Softmax(QK^T / sqrt(d) - alpha * D)
```

This is the first decisive experimental gate. ERGT should only move into memory,
causal geometry, spectral complexity, or relational field modeling if this stage
shows a meaningful advantage or stability benefit over the baseline.

## Long-Term Phases

Later phases extend the first proof into the deeper theory:

- Phase 4: Dynamic relational graph and graph memory.
- Phase 5: Complete ERGT architecture.
- Phase 6: Causal geometry for long-context structure.
- Phase 7: Spectral complexity minimization.
- Phase 8: Relational field model.
- Later: Reasoning paths over stable relational geometry.
- Later: Intelligence-space evaluation.

These phases remain downstream of the first proof stage.

## Minimum Publishable Version

The minimum publishable version is ERGT-v1:

```text
RelationalGraph + EmergentDistance + GeoAttention
```

Its core scientific claim is:

```text
Correlation -> Distance -> Geometry -> Attention
```

The first paper should avoid overclaiming about intelligence or reasoning. Those
ideas belong to the broader theory, but the initial experimental paper should
establish the measurable architectural mechanism first.

## Evaluation Principles

All comparisons must hold the following constant:

- Dataset
- Tokenizer
- Context length
- Parameter budget
- Optimizer
- Training steps
- Evaluation metrics
- Random seed protocol

The decisive comparison is between:

```text
TransformerBaseline
GeoAttention with real D
GeoAttention with shuffled D
GeoAttention with random D
GeoAttention with alpha = 0
```

If real relational distance does not outperform or stabilize relative to these
controls, the project should revise the graph, distance, or geometry definition
before continuing to later phases.

## Current Status

The project has Phase 3 evidence and control risks recorded in `docs/13` through
`docs/16`. The current design direction is no longer "proceed directly to graph
memory"; it is to tighten controls, observe relational structure, validate
reconstructible causal geometry, and only then re-enter GeoAttention v2.

The existing source notes are:

- `4.txt`: phased research roadmap.
- `5.txt`: operational architecture definition.
- `6.txt`: philosophical and theoretical foundation.

The next documents should define the project charter, theoretical foundation,
operational definitions, phase plan, and proof-stage protocol before
implementation begins.
