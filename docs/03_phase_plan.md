# Phase Plan

## 1. Purpose

This document defines the ERGT execution path from theoretical preparation to
the first proof stage and later extensions.

The project has a broad theory, but implementation must proceed through narrow,
testable phases. Later phases are gated by the success of earlier phases.

## 2. Phase Overview

```text
Phase -1: Theoretical and Operational Grounding
Phase  0: Baseline Transformer
Phase  1: Relational Graph Observer
Phase  2: Emergent Distance and Geometry
Phase  3: GeoAttention v1
Gate  1: First Proof Decision
Phase  4: Dynamic Relational Graph Memory
Phase  5: Complete ERGT Architecture
Phase  6: Causal Geometry
Phase  7: Spectral Complexity
Phase  8: Relational Field Model
```

After the Phase 3 evidence, the post-Phase-3 execution path is strengthened by
`docs/17_physics_aligned_ergt_program.md`. The historical phase names remain
valid, but the movement standard is now:

```text
measure -> control -> observe -> validate -> inject -> regularize
```

The strengthened path is:

```text
Claim/measurement contracts
-> strict W-level controls
-> relational field observer
-> resonant-response observer
-> Phi information potential
-> reconstruction gate
-> Phi-gated relational memory observer
-> causal shortest-path geometry
-> GeoAttention v2
-> auxiliary loss
-> complete ERGT
-> reasoning paths
-> intelligence-space evaluation
```

The first implementation target is:

```text
Phase -1 through Phase 3
```

## 3. Phase -1: Theoretical and Operational Grounding

### Objective

Translate the philosophical foundation into a precise experimental contract.

### Required Documents

- `README.md`
- `docs/00_project_charter.md`
- `docs/01_theoretical_foundation.md`
- `docs/02_operational_definitions.md`
- `docs/03_phase_plan.md`
- `docs/18_ergt_position_paper.md`
- `.codex/project_context.md`

### Deliverables

- Clear distinction between broad theory and first experimental claim.
- Definitions for relation, structure, distance, geometry, memory, concept, and
  reasoning.
- Gate condition for moving beyond Phase 3.

### Success Criteria

- Every major theoretical term has an operational definition.
- The first proof claim is falsifiable.
- Codex has a stable documentation pack to follow.

## 4. Phase 0: Baseline Transformer

### Objective

Create a controlled transformer baseline for fair comparison.

The baseline should not be treated as a novelty contribution. It is the
experimental reference required to evaluate ERGT.

### Implementation Direction

Use an existing minimal GPT-style transformer pattern where possible. The
important requirement is modifiability, not originality.

Suggested implementation style:

```text
models/transformer_baseline.py
configs/baseline_wikitext2.json
experiments/train_baseline.py
experiments/eval_baseline.py
```

### Controls

The baseline must define:

- Dataset
- Tokenizer
- Context length
- Model size
- Optimizer
- Training steps or token budget
- Evaluation protocol
- Seed protocol

### Metrics

- Validation loss
- Perplexity
- Tokens per second
- GPU memory
- Training stability

### Deliverables

- `models/transformer_baseline.py`
- `configs/baseline_wikitext2.json`
- `runs/phase0_baseline/baseline_results.json`

### Success Criteria

- Baseline trains and evaluates reproducibly.
- Metrics are recorded in a machine-readable format.
- The code exposes an attention module that can later be replaced or extended.

## 5. Phase 1: Relational Graph Observer

### Objective

Construct a relational graph from hidden states without changing model behavior.

This phase asks:

```text
Do meaningful relation structures appear inside the transformer?
```

### Core Formula

```text
K_ij = sigmoid((x_i^T x_j) / sqrt(d))
W = K
```

### Implementation Direction

Add a graph observer that can attach to selected layers and record relation
matrices.

Suggested files:

```text
layers/relational_graph.py
experiments/analyze_relational_graph.py
```

### Metrics

- Graph entropy
- Relation variance
- Sparsity under threshold
- Degree distribution
- Layer-to-layer similarity
- Perturbation stability
- Difference from random or shuffled controls

### Deliverables

- `layers/relational_graph.py`
- `runs/phase1_graph/graph_stats.json`
- `runs/phase1_graph/W_heatmaps/`

### Success Criteria

- `W` is not uniform, saturated, or trivial.
- `W` changes with input content.
- Some non-random structure is measurable.

### Failure Modes

- `W` is nearly constant.
- `W` is dominated by the diagonal.
- `W` only duplicates standard attention without useful distinction.
- Metrics are unstable across examples or seeds.

## 6. Phase 2: Emergent Distance and Geometry

### Objective

Convert relational strength into distance and test whether a geometry-like
structure appears.

This phase asks:

```text
Can relation strength induce a useful distance structure?
```

### Core Formula

```text
D_ij = -log(W_ij + epsilon)
```

### Implementation Direction

Create a distance module that receives `W`, produces `D`, and applies explicit
normalization.

Suggested files:

```text
geometry/emergent_distance.py
experiments/analyze_emergent_geometry.py
```

### Metrics

- Distance entropy
- Distance variance
- Cluster quality
- Neighborhood overlap across layers
- Intrinsic dimension estimate
- Triangle inequality violation rate, if metric-like claims are made
- Correlation with standard attention logits

### Deliverables

- `geometry/emergent_distance.py`
- `runs/phase2_distance/geometry_report.json`
- `runs/phase2_distance/D_heatmaps/`

### Success Criteria

- `D` is numerically stable.
- `D` differs meaningfully from random and shuffled controls.
- `D` supports non-trivial neighborhoods or clusters.
- `D` is not merely a direct duplicate of standard attention logits.

### Failure Modes

- `D` is dominated by numerical artifacts.
- `D` collapses into uniform or diagonal structure.
- `D` destabilizes attention logits when used later.
- Geometry metrics show no meaningful organization.

## 7. Phase 3: GeoAttention v1

### Objective

Test whether induced relational distance improves or stabilizes attention.

This is the first decisive experimental phase.

### Core Formula

```text
Standard: A = Softmax(QK^T / sqrt(d))
ERGT:     A = Softmax(QK^T / sqrt(d) - alphaD)
```

### Implementation Direction

Replace or augment standard attention with `GeoAttention`.

Suggested files:

```text
attention/geo_attention.py
models/ergt_v1.py
experiments/train_ergt_v1.py
experiments/compare_phase3.py
```

### Required Comparisons

- Transformer baseline.
- GeoAttention with real `D`.
- GeoAttention with shuffled `D`.
- GeoAttention with random `D`.
- GeoAttention with `alpha = 0`.

### Metrics

- Validation loss
- Perplexity
- Training stability
- Tokens per second
- GPU memory
- Attention entropy
- Optional long-range dependency metric

### Deliverables

- `attention/geo_attention.py`
- `models/ergt_v1.py`
- `runs/phase3_geo_attention/comparison_results.json`
- `runs/phase3_geo_attention/ablation_report.json`

### Success Criteria

GeoAttention with real `D` should show at least one meaningful benefit:

- Lower validation loss.
- Lower perplexity.
- Better stability across seeds.
- Better long-range dependency behavior.
- Comparable quality with more interpretable or structured attention.

The improvement must survive distance-control ablations.

### Failure Modes

- Real `D` is no better than random or shuffled `D`.
- `alpha` collapses to zero or explodes.
- Training becomes unstable.
- Runtime or memory overhead is impractical.
- Gains disappear across seeds.

## 8. Gate 1: First Proof Decision

### Objective

Decide whether ERGT should proceed beyond Phase 3.

### Gate Condition

The project may proceed if:

```text
GeoAttention(real D) improves or stabilizes over TransformerBaseline
and
GeoAttention(real D) outperforms random/shuffled/disabled distance controls.
```

### Possible Outcomes

#### Pass

Move to Phase 4 and begin relational memory experiments.

#### Conditional Pass

Continue Phase 3 with more seeds, better normalization, or improved graph
construction.

#### Fail

Do not proceed to memory or causal geometry. Revisit:

- Relation kernel.
- Distance transformation.
- Distance normalization.
- `alpha` parameterization.
- Layer placement.
- Whether gradients pass through `W` and `D`.

The current Phase 3 evidence should be treated as requiring the strengthened
post-Phase-3 plan rather than direct implementation of graph memory. The next
work should begin with strict W-level controls, observer metrics, `Phi`, and
memory-as-observer as defined in `docs/17_physics_aligned_ergt_program.md`.

## 9. Phase 4: Dynamic Relational Graph Memory

### Objective

Introduce persistent relational structure.

### Core Formula

```text
W(l+1) = (1 - eta)W(l) + etaK
```

### Research Question

Can persistent graph structure act as memory?

### Metrics

- Graph stability
- Temporal persistence
- Layer persistence
- Perturbation recovery
- Effect on loss and long-context behavior

### Gate Requirement

Phase 4 should begin only after Phase 3 passes or conditionally passes.

## 10. Phase 5: Complete ERGT Architecture

### Objective

Integrate graph, distance, geometry attention, and memory into a complete
architecture.

### Pipeline

```text
HiddenStates
-> RelationalGraph
-> EmergentDistance
-> GeoAttention
-> FeedForward
```

### Comparisons

- Transformer baseline.
- ERGT-v1 without memory.
- ERGT with dynamic graph memory.
- GPT-style model with matched parameter budget.

## 11. Phase 6: Causal Geometry

### Objective

Introduce causal relational structure for long-context modeling.

### Core Formula

```text
C_ij = sigmoid(v * delta_t - D_ij)
A = Softmax(QK - alphaD + betaC)
```

### Required Safety Check

Causal masking must remain strict. The causal geometry term must not leak future
information.

### Metrics

- LongContext1024
- LongContext2048
- LongContext4096
- Attention noise
- Causal leakage checks
- Long-range dependency score

## 12. Phase 7: Spectral Complexity

### Objective

Introduce a compression principle over relational structure.

### Core Formula

```text
p_i = lambda_i / sum(lambda)
C(W) = -sum(p_i * log(p_i))
L = L_lm + lambdaC(W)
```

### Metrics

- Effective rank
- Spectral entropy
- Compression ratio
- Generalization gap
- Stability of relational geometry

## 13. Phase 8: Relational Field Model

### Objective

Generalize from discrete relational graphs to a dynamic relational field.

### Core Object

```text
W(i, j, t)
```

### Long-Term Pipeline

```text
DynamicField -> Geometry -> Attention -> Memory -> Reasoning
```

This phase is not part of the first publishable proof.

## 14. Recommended First Implementation Milestone

The first implementation milestone should include:

```text
Phase 0 complete
Phase 1 graph observer complete
Phase 2 distance analyzer complete
Phase 3 GeoAttention ablation complete
```

The corresponding scientific output should be:

```text
Does correlation-induced relational distance produce a useful attention bias?
```

## 15. Documentation Dependencies

Before implementation, complete:

- `docs/04_proof_stage_protocol.md`
- `docs/05_baseline_transformer_spec.md`
- `docs/06_relational_graph_spec.md`
- `docs/07_emergent_distance_spec.md`
- `docs/08_geo_attention_spec.md`
- `docs/09_metrics_and_ablation_plan.md`
- `docs/10_gate_conditions.md`
- `docs/17_physics_aligned_ergt_program.md`
- `docs/18_ergt_position_paper.md`
- `docs/11_repository_structure.md`

Implementation should not begin until the proof-stage protocol and repository
structure are defined.
