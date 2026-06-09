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

The project is currently in the documentation and research-design stage.

The existing source notes are:

- `4.txt`: phased research roadmap.
- `5.txt`: operational architecture definition.
- `6.txt`: philosophical and theoretical foundation.

The next documents should define the project charter, theoretical foundation,
operational definitions, phase plan, and proof-stage protocol before
implementation begins.
