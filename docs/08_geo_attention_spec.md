# GeoAttention Specification

## 1. Purpose

GeoAttention is the first architectural intervention in ERGT.

It tests whether an induced relational distance can act as a useful attention
bias inside a transformer.

The Phase 3 research question is:

```text
Does correlation-induced distance improve or stabilize attention under a
controlled comparison?
```

## 2. Core Formula

Standard causal self-attention computes:

```text
logits = QK^T / sqrt(head_dim)
A = Softmax(mask(logits))
```

GeoAttention adds a relational distance penalty:

```text
logits = QK^T / sqrt(head_dim) - alphaD
A = Softmax(mask(logits))
```

where:

- `D` is the normalized emergent distance matrix.
- `alpha` controls the strength of the geometry term.
- The causal mask remains mandatory for causal language modeling.

## 3. Scientific Role

GeoAttention is not a generic attention bias.

The distance term must be induced from the model's relational structure:

```text
X -> W -> D -> attention bias
```

The key claim is not that any extra bias helps. The key claim is that real
relational distance helps more than random, shuffled, or disabled distance.

## 4. Tensor Contract

Input hidden states:

```text
hidden_states: [batch, sequence, hidden_dim]
```

Attention projections:

```text
q: [batch, heads, sequence, head_dim]
k: [batch, heads, sequence, head_dim]
v: [batch, heads, sequence, head_dim]
```

Standard logits:

```text
qk_logits: [batch, heads, sequence, sequence]
```

Distance:

```text
D: [batch, 1, sequence, sequence]
```

or:

```text
D: [batch, heads, sequence, sequence]
```

Output:

```text
attention_output: [batch, sequence, hidden_dim]
attention_weights: [batch, heads, sequence, sequence]
```

If `D` has one graph head, it must broadcast across attention heads.

## 5. Default Phase 3 Design

The recommended first design is:

```text
shared D across heads
fixed alpha for initial sweep
D normalized before attention
causal mask applied after geometry bias
```

Rationale:

- Shared `D` is cheaper and easier to analyze.
- Fixed `alpha` makes initial ablations easier to interpret.
- Normalization prevents the geometry term from overwhelming `QK`.
- Masking after bias preserves the standard causal attention contract.

## 6. Forward Pass Order

The correct causal forward order is:

```text
q, k, v = projections(hidden_states)
qk_logits = q @ k.transpose(-2, -1) / sqrt(head_dim)
W = relational_graph(hidden_states)
D = emergent_distance(W)
D = normalize(D)
geo_logits = qk_logits - alpha * D
geo_logits = apply_causal_mask(geo_logits)
attention_weights = softmax(geo_logits)
attention_weights = dropout(attention_weights)
output = attention_weights @ v
output = output_projection(output)
```

The causal mask must not be replaced by `D`.

## 7. Causal Safety

For causal language modeling, token `i` must not attend to token `j > i`.

Required rule:

```text
No future-token information may affect current-token attention weights.
```

Safe default:

- Compute `D` from the same hidden states available at the layer.
- Apply the causal mask to final logits before softmax.
- Optionally mask `W` or `D` to lower-triangular form before use.

Strict causal runtime mode:

```text
D = causal_mask_distance(D)
geo_logits = qk_logits - alphaD
geo_logits = apply_causal_mask(geo_logits)
```

The experiment config must record whether `D` was full-analysis distance or
causal-runtime distance.

## 8. Distance Input Modes

GeoAttention should support multiple distance modes for ablations:

```text
real_d:      D induced from W
random_d:    random distance with matched shape and scale
shuffled_d:  real D with structure disrupted
zero_d:      D = 0
external_d:  precomputed D for analysis or replay
```

The same normalization policy should be applied consistently to all distance
modes unless the experiment explicitly tests normalization.

## 9. Alpha Parameter

`alpha` controls the geometry contribution:

```text
geo_term = alphaD
```

Initial fixed sweep:

```text
alpha in [0.01, 0.05, 0.1, 0.5, 1.0]
```

Recommended first proof run:

```text
alpha = 0.1
```

Only use this as a starting point. The correct value depends on the scale of
`D_norm` and `QK`.

Optional trainable form:

```text
alpha = softplus(raw_alpha)
```

This keeps `alpha` non-negative.

## 10. Alpha Scale Diagnostics

Every Phase 3 run must record:

```text
mean_abs_qk = mean(abs(QK^T / sqrt(d)))
mean_abs_geo = mean(abs(alphaD))
geo_to_qk_ratio = mean_abs_geo / (mean_abs_qk + epsilon)
```

If `geo_to_qk_ratio` is extremely high, geometry overwhelms attention.

If `geo_to_qk_ratio` is near zero, the experiment may be equivalent to the
baseline.

Recommended initial target:

```text
geo_to_qk_ratio roughly between 0.01 and 0.5
```

This range is a heuristic, not a theoretical law.

## 11. Distance Normalization

GeoAttention should consume normalized distance:

```text
D_norm = normalize(D)
```

Recommended default:

```text
D_norm = (D - mean(D_offdiag)) / (std(D_offdiag) + epsilon)
D_norm = clamp(D_norm, -5, 5)
```

The diagonal and masked future positions should be excluded from normalization
statistics when practical.

The exact policy must match `docs/07_emergent_distance_spec.md`.

## 12. Gradient Modes

GeoAttention should support two gradient modes:

```text
grad_d:       gradients flow through D and W
detached_d:   D is detached before attention use
```

Recommended sequence:

1. Start with `grad_d` if stable.
2. If unstable, test `detached_d`.
3. Report both if compute allows.

The gradient mode must be recorded because it changes the scientific
interpretation:

- `grad_d` means the model can shape its relational geometry for loss.
- `detached_d` means geometry acts as an observer-derived bias.

## 13. Head Sharing Modes

GeoAttention should support:

```text
shared_d:   one D shared across all attention heads
per_head_d: one D per attention head
```

Default:

```text
shared_d
```

`per_head_d` should be treated as a later ablation because it adds cost and
complexity.

## 14. Diagonal Policy

Self-distance affects self-attention behavior.

Recommended default:

```text
D_ii = 0 before attention use
```

Reason:

- A token should not be penalized for self-relation by a derived distance.
- Causal and attention masks already handle invalid positions.

However, experiments must record the diagonal policy. If the diagonal is left as
raw `-log(W_ii)`, it may distort the attention distribution.

## 15. Masked Positions

Future positions and padding positions must not influence softmax.

Masking rules:

```text
causal masked logits -> -inf
padding masked logits -> -inf
```

The distance term should not be used to represent invalidity. Invalidity belongs
to the mask.

## 16. Dropout

Attention dropout should be applied after softmax, matching the baseline:

```text
attention_weights = dropout(softmax(masked_geo_logits))
```

Dropout rate must match the baseline unless explicitly varied in a controlled
experiment.

## 17. Complexity

GeoAttention introduces or reuses sequence-pair matrices:

```text
W: [batch, graph_heads, sequence, sequence]
D: [batch, graph_heads, sequence, sequence]
```

This is `O(sequence^2)` memory and compute.

Record:

- Extra memory from `W`.
- Extra memory from `D`.
- Runtime overhead.
- Scaling with context length.

If the overhead is high, later phases may explore sparsity or low-rank
approximations. Do not start with those approximations in the first proof unless
compute makes dense testing impossible.

## 18. Required Conditions

Phase 3 must run these conditions:

```text
baseline
real_d
alpha_zero
random_d
shuffled_d
```

Definitions:

- `baseline`: standard attention, no geometry path.
- `real_d`: `D` induced from real `W`.
- `alpha_zero`: GeoAttention code path with `alpha = 0`.
- `random_d`: random `D` with matched shape and scale.
- `shuffled_d`: real `D` with relational arrangement disrupted.

`alpha_zero` is important because it detects accidental implementation changes
outside the geometry term.

## 19. Optional Conditions

Recommended later ablations:

```text
detached_d
trainable_alpha
per_head_d
raw_d_without_normalization
different_alpha_sweep
different_layer_injection_points
```

Only add these after the required comparison matrix is working.

## 20. Logging Requirements

Every GeoAttention run must log:

- Training loss.
- Validation loss.
- Perplexity.
- Attention entropy.
- `alpha` value.
- `geo_to_qk_ratio`.
- `D` mean and standard deviation.
- `D` min and max.
- Gradient mode.
- Distance mode.
- Normalization mode.
- Runtime and memory.

Recommended run files:

```text
runs/phase3_geo_attention/config.json
runs/phase3_geo_attention/train_log.jsonl
runs/phase3_geo_attention/comparison_results.json
runs/phase3_geo_attention/ablation_report.json
runs/phase3_geo_attention/model_summary.json
```

## 21. Suggested Config Schema

```json
{
  "attention_type": "geo_attention",
  "distance_mode": "real_d",
  "distance_normalization": "offdiag_zscore_clamp",
  "diagonal_policy": "zero",
  "head_sharing": "shared_d",
  "alpha": {
    "mode": "fixed",
    "initial_value": 0.1,
    "non_negative": true
  },
  "gradient_mode": "grad_d",
  "causal_runtime_distance": true,
  "log_geometry_diagnostics": true
}
```

## 22. Implementation Pseudocode

```python
def geo_attention(hidden_states, attention_mask=None):
    q, k, v = project_qkv(hidden_states)
    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)

    qk_logits = matmul(q, transpose_last_two(k)) / sqrt(head_dim)

    W = relational_graph(hidden_states)
    D = emergent_distance(W)
    D = normalize_distance(D)
    D = apply_distance_policy(D)

    if gradient_mode == "detached_d":
        D = D.detach()

    geo_logits = qk_logits - alpha * D
    geo_logits = apply_causal_and_padding_mask(geo_logits, attention_mask)

    attn = softmax(geo_logits)
    attn = dropout(attn)

    output = matmul(attn, v)
    output = merge_heads(output)
    output = output_projection(output)
    return output
```

This pseudocode is a contract, not final implementation.

## 23. Acceptance Criteria

GeoAttention implementation is acceptable if:

- It matches baseline behavior when `alpha = 0`.
- It preserves causal masking.
- It supports real, random, shuffled, and zero distance modes.
- It records geometry diagnostics.
- It keeps tensor shapes compatible with standard attention.
- It trains without NaN or Inf.
- It enables fair comparison against the baseline.

## 24. Scientific Success Criteria

Phase 3 supports ERGT-v1 if:

```text
real_d beats or stabilizes over baseline
and
real_d beats random_d and shuffled_d
and
alpha_zero behaves like the baseline
```

The exact success metric may be validation loss, perplexity, stability, or a
predefined long-range metric.

## 25. Failure Criteria

GeoAttention requires redesign if:

- `alpha_zero` does not match baseline behavior within expected noise.
- Real `D` performs like random or shuffled `D`.
- Training diverges or produces NaN.
- Geometry term overwhelms `QK`.
- `alpha` collapses to zero in all useful runs.
- Runtime or memory overhead is impractical.
- Causal masking is violated.

## 26. Redesign Order

If Phase 3 fails, change one variable at a time:

1. Check `alpha_zero` implementation equivalence.
2. Check causal and padding masks.
3. Adjust `D` normalization.
4. Sweep fixed `alpha`.
5. Detach `D`.
6. Change graph kernel normalization.
7. Change injection layer.
8. Test per-head `D`.
9. Test alternative distance transforms.

Do not proceed to memory or causal geometry until the GeoAttention mechanism is
credible.

## 27. Relationship to Later Phases

If Phase 3 passes, GeoAttention becomes the foundation for:

- Phase 4 graph memory.
- Phase 5 full ERGT.
- Phase 6 causal geometry.
- Phase 7 spectral complexity.

If Phase 3 fails, later phases should pause. A memory system built on weak
relational distance would not provide a clean test of the ERGT theory.
