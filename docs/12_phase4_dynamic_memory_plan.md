# Phase 4 Dynamic Relational Graph Memory Plan

## 1. Purpose

This document defines the first executable plan for Phase 4 after Gate 1.

Phase 4 should test whether persistent relational graph structure can improve
or stabilize ERGT beyond GeoAttention v1.

The Phase 4 claim is narrower than the broad theory:

```text
A dynamically persisted relation graph can be a useful memory substrate for
GeoAttention.
```

This phase must not claim concept memory, reasoning, intelligence, or a full
relational field model.

## 2. Gate Lock

Phase 4 implementation is allowed only after Gate 1 produces:

```text
pass
```

or:

```text
conditional_pass
```

Required gate artifact:

```text
runs/gates/phase3_to_phase4_decision.json
```

If Gate 1 is missing or failed, Phase 4 work is limited to documentation and
planning.

## 3. Starting Point

Phase 4 builds on the Phase 3 pipeline:

```text
HiddenStates -> RelationalGraph -> EmergentDistance -> GeoAttention
```

Phase 4 inserts a persistent graph state:

```text
HiddenStates -> RelationalGraph -> DynamicGraphMemory -> EmergentDistance
-> GeoAttention
```

The minimum Phase 4 model should reuse the proven ERGT-v1 settings unless a
change is explicitly part of the memory condition.

Current candidate foundation:

```text
distance_mode: real_d
alpha: 0.2
distance normalization: offdiag_zscore_clamp
context_length: 256
dataset: WikiText-2 with GPT-2 tokenizer
```

## 4. Core Memory Formula

Let `K_l` be the current relation graph at layer `l`.

The memory graph is:

```text
W_mem_l = (1 - eta) * W_mem_(l-1) + eta * K_l
```

where:

- `K_l` is the current relation graph.
- `W_mem_l` is the persisted relation graph after update.
- `eta` is the update rate.

Interpretation:

```text
eta = 1.0 -> no persistence; current graph only
eta = 0.0 -> frozen memory
0 < eta < 1 -> persistent graph smoothing
```

The first implementation should use a fixed configured `eta`, not a learned
update rule.

## 5. Memory Scope

The first Phase 4 implementation should use layer-local memory inside a single
forward pass.

Allowed first scope:

```text
per_batch
per_sequence
per_layer_stack
```

Disallowed for the first implementation:

```text
cross_batch persistent memory
dataset-level memory bank
retrieval memory
external key-value cache
```

Reason:

Cross-batch memory introduces leakage and reproducibility risks. It can be
tested later only after layer-local memory is understood.

## 6. Tensor Contract

Expected graph shapes:

```text
hidden_states: [batch, sequence, hidden_dim]
K:             [batch, graph_heads_or_1, sequence, sequence]
W_mem:         [batch, graph_heads_or_1, sequence, sequence]
D_mem:         [batch, graph_heads_or_1, sequence, sequence]
```

`W_mem` must remain broadcast-compatible with GeoAttention.

The first implementation should preserve the shared graph setup:

```text
graph_heads: 1
head_sharing: shared_d
```

Per-head memory is a later variant.

## 7. Causal Safety

Phase 4 must preserve strict causal masking.

Rules:

- Memory must not allow attention from a token to future tokens.
- Causal masking must be applied after any geometry or memory bias.
- Any precomputed memory graph must be compatible with the same attention mask
  used by GeoAttention.
- Tests must verify future-token logits remain masked even when memory contains
  non-zero future-position entries.

The first implementation may compute full relation matrices, as Phase 3 does,
only if future positions are still unreachable after the final attention mask.

## 8. Default Phase 4 MVP

The first MVP should implement:

```text
memory/dynamic_graph.py
tests/test_dynamic_graph_memory.py
```

The module should be independent from language-model training.

Required behavior:

- Initialize memory from the first graph or zeros, depending on config.
- Update memory by EMA.
- Support explicit `eta`.
- Support optional detach policy.
- Preserve shape.
- Preserve finite values.
- Serialize diagnostics.

Suggested config:

```json
{
  "memory": {
    "type": "dynamic_graph",
    "enabled": true,
    "eta": 0.25,
    "initialization": "current",
    "detach_memory": true,
    "reset_policy": "per_forward",
    "diagonal_policy": "zero"
  }
}
```

## 9. Model Integration Plan

After the memory module passes isolated tests, add:

```text
models/ergt_memory.py
configs/ergt_memory/debug_memory.json
configs/ergt_memory/pilot_real_memory.json
configs/ergt_memory/pilot_random_memory.json
configs/ergt_memory/pilot_shuffled_memory.json
experiments/train_ergt_memory.py
experiments/compare_phase4.py
```

The model should compare against ERGT-v1, not only the baseline.

Core comparison:

```text
TransformerBaseline
ERGT-v1 real_d alpha=0.2
ERGT-memory real memory
ERGT-memory random memory
ERGT-memory shuffled memory
ERGT-memory disabled or eta=1.0
```

## 10. Controls

Required controls:

- `memory_disabled`: should match ERGT-v1 behavior within expected noise.
- `eta_1_current_only`: current graph only; no persistence.
- `random_memory`: memory graph with matched scale but random structure.
- `shuffled_memory`: memory graph with real values but disrupted structure.
- `alpha_zero_with_memory`: optional diagnostic; memory should not matter if
  geometry bias is disabled.

Without these controls, a gain could be caused by generic smoothing or random
regularization rather than relational memory.

## 11. Metrics

Phase 4 must record the Phase 3 metrics plus memory diagnostics.

Required model metrics:

- Validation loss.
- Perplexity.
- Training stability.
- Tokens per second.
- Peak GPU memory.

Required memory diagnostics:

- `eta`.
- Memory initialization policy.
- Memory detach policy.
- Mean and standard deviation of `W_mem`.
- Memory entropy.
- Memory sparsity under thresholds.
- Mean absolute difference between `W_mem` and `K`.
- Layer-to-layer memory similarity.
- Memory drift across layers.
- Ratio between geometry bias and standard QK logits.

Suggested diagnostics:

```text
memory_current_delta = mean(abs(W_mem - K))
memory_persistence = cosine_similarity(W_mem_l, W_mem_(l-1))
memory_entropy = graph_entropy(W_mem)
```

## 12. Pass Criteria

Phase 4 supports graph memory if:

```text
ERGT-memory real
  improves or stabilizes over ERGT-v1 real_d alpha=0.2
  and beats random/shuffled memory controls
  and preserves causal masking
  and has acceptable runtime and memory overhead
```

Acceptable evidence may include:

- Lower validation loss.
- Lower perplexity.
- Better stability across seeds.
- Comparable loss with stronger persistence diagnostics and bounded overhead.

## 13. Conditional Pass Criteria

Phase 4 receives a conditional pass if:

- Real memory beats ERGT-v1 on one seed but needs replication.
- Real memory beats one memory control but not all.
- Memory improves stability but not final loss.
- Runtime overhead is high but scientifically interpretable.

Action:

```text
repeat Phase 4 with more seeds, eta sweep, or improved memory normalization
```

## 14. Failure Criteria

Phase 4 fails if:

- Real memory does not beat ERGT-v1 or memory controls.
- Memory causes NaN or Inf.
- Causal masking fails.
- `eta=1` or random memory explains the gain.
- Runtime or memory overhead is impractical.
- Memory collapses to uniform, diagonal, or saturated structure.

Action:

```text
redesign memory update, memory normalization, eta policy, or injection point
```

## 15. Implementation Order

Recommended order:

1. Record confirm-seed Phase 3 results.
2. Produce Gate 1 decision.
3. Implement `memory/dynamic_graph.py`.
4. Add `tests/test_dynamic_graph_memory.py`.
5. Add memory diagnostics helpers.
6. Add `models/ergt_memory.py`.
7. Add debug and pilot configs.
8. Add `experiments/train_ergt_memory.py`.
9. Add `experiments/compare_phase4.py`.
10. Add Colab pilot notebook.

Do not skip steps 1 and 2.

## 16. First Pilot Budget

The first Phase 4 pilot should fit within a small Colab GPU budget.

Suggested pilot:

```text
context_length: 256
n_layers: 6
n_heads: 8
hidden_dim: 512
max_steps: 300 to 500
batch_size: 16 if memory allows, otherwise 8
eta: 0.25
alpha: 0.2
```

Run order:

```text
ERGT-v1 real_d alpha=0.2 reference, if not already available
ERGT-memory real eta=0.25
ERGT-memory eta=1.0
ERGT-memory random eta=0.25
```

Only expand to longer runs if the pilot is stable and non-trivial.

## 17. Anti-Overclaim Rule

A positive Phase 4 result supports only this claim:

```text
Persistent relational graph structure can be useful as a memory-like attention
bias.
```

It does not prove:

- Human-like memory.
- Concept formation.
- Reasoning.
- Intelligence.
- Relational field theory.

Those claims require later phases and separate measurements.
