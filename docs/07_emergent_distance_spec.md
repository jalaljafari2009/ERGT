# Emergent Distance Specification

## 1. Purpose

Emergent distance is the second operational object of ERGT.

It converts relational strength into a distance-like structure that can be
measured, visualized, and later used as an attention bias.

The Phase 2 research question is:

```text
Can relation strength induce a meaningful geometry-like distance structure?
```

## 2. Theoretical Role

ERGT reverses the usual order:

```text
Classical view: Space -> Distance -> Relation
ERGT view:      Relation -> Distance -> Geometry
```

Distance is not assumed as a primitive. It is derived from relation strength.

Strong relations imply low traversal cost. Weak relations imply high traversal
cost.

## 3. Core Formula

The default emergent distance is:

```text
D_ij = -log(W_ij + epsilon)
```

where:

- `W_ij` is the relation strength between positions `i` and `j`.
- `epsilon` prevents `log(0)`.
- `D_ij` is the induced distance or traversal cost.

Interpretation:

```text
W_ij high -> D_ij low
W_ij low  -> D_ij high
```

## 4. Tensor Contract

Input:

```text
W: [batch, graph_heads_or_1, sequence, sequence]
```

Output:

```text
D: [batch, graph_heads_or_1, sequence, sequence]
```

For Phase 3 attention use, `D` must be broadcastable to:

```text
attention_logits: [batch, attention_heads, sequence, sequence]
```

If `W` has one graph head and attention has multiple heads, then `D` is shared
across attention heads.

## 5. Numerical Safety

The distance transform can become unstable when `W` is near zero.

Required safeguards:

- Clamp or epsilon-protect `W`.
- Check for NaN and Inf.
- Record min, max, mean, and standard deviation of `D`.
- Normalize `D` before Phase 3 attention use.

Recommended epsilon:

```text
epsilon = 1e-6
```

If mixed precision is used, epsilon may need to be larger.

## 6. Diagonal Policy

The diagonal requires explicit handling.

Possible policies:

```text
keep:      D_ii = -log(W_ii + epsilon)
zero:      D_ii = 0
mask:      exclude diagonal from metrics
separate:  report diagonal and off-diagonal stats separately
```

Recommended Phase 2 policy:

```text
visualization: keep diagonal visible
metrics: report both diagonal-included and diagonal-excluded when practical
attention use: set or normalize diagonal according to the GeoAttention spec
```

The policy must be recorded in run metadata.

## 7. Normalization

Raw `D` should not be injected directly into attention unless explicitly tested.

Recommended first normalization:

```text
D_norm = (D - mean(D_offdiag)) / (std(D_offdiag) + epsilon)
```

Alternative normalization:

```text
D_norm = D / (mean(D_offdiag) + epsilon)
```

Optional clipping:

```text
D_norm = clamp(D_norm, -clip_value, clip_value)
```

Recommended first clip range:

```text
[-5, 5]
```

The same normalization must be used for real, random, and shuffled distance
conditions.

## 8. Causal Handling

Distance analysis and distance use must be distinguished.

```text
analysis_mode: may inspect full D if it does not affect model computation
causal_runtime_mode: future positions must be masked before D affects attention
```

For causal runtime attention, future-token distance must not influence current
token attention.

Correct Phase 3 order:

```text
logits = QK^T / sqrt(d) - alphaD
logits = apply_causal_mask(logits)
A = softmax(logits)
```

If `D` is computed from future-aware hidden states, it must not be used in a way
that leaks future information.

## 9. Geometry Definition

In Phase 2, geometry means:

```text
stable, non-trivial neighborhood and path structure induced by D
```

The project should not claim full metric geometry unless metric properties are
measured.

Minimum geometry evidence:

- Non-uniform neighborhoods.
- Non-random cluster or community structure.
- Stability across nearby layers or related inputs.
- Difference from random and shuffled distance.
- Potential predictive relation to attention behavior.

## 10. Distance Metrics

Required metrics:

- Distance mean.
- Distance variance.
- Distance entropy.
- Minimum and maximum distance.
- Diagonal vs off-diagonal statistics.
- Correlation with relation strength.
- Correlation with standard attention logits.
- Difference from random distance.
- Difference from shuffled distance.

Recommended additional metrics:

- Neighborhood overlap across layers.
- Distance-based cluster score.
- Intrinsic dimension estimate.
- Effective rank of transformed similarity.
- Triangle inequality violation rate.
- Spectral profile.

## 11. Distance Entropy

A simple first entropy can be computed by converting inverse distance or
normalized relation-derived weights into probabilities.

Option A, entropy over relation-derived weights:

```text
p_ij = W_ij / sum(W)
H_D_proxy = -sum(p_ij * log(p_ij + epsilon))
```

Option B, entropy over transformed distance affinity:

```text
A_D = exp(-D)
p_ij = A_D_ij / sum(A_D)
H_D = -sum(p_ij * log(p_ij + epsilon))
```

The chosen entropy definition must be recorded.

## 12. Neighborhoods

For each position `i`, define its distance neighborhood as the `k` nearest
positions under `D`.

```text
N_k(i) = top_k_smallest(D_i*)
```

Recommended `k` values:

```text
4, 8, 16
```

Measurements:

- Neighborhood overlap across layers.
- Neighborhood overlap between real and shuffled distance.
- Neighborhood stability under input perturbation.

Neighborhoods should be computed with causal constraints when used for causal
runtime interpretation.

## 13. Clustering

Distance-based clusters test whether `D` creates coherent regions.

Possible methods:

- Hierarchical clustering.
- Spectral clustering on `exp(-D)`.
- Simple threshold communities on `W`.

Required caution:

Clustering visualizations are not sufficient. Report numeric differences from
random and shuffled controls.

## 14. Intrinsic Dimension

Intrinsic dimension estimates are optional in early implementation but useful
for the geometry claim.

Possible approaches:

- Participation ratio from spectrum.
- PCA dimension on embeddings reconstructed from distance, if implemented.
- Nearest-neighbor intrinsic dimension estimators.

Do not overclaim from one estimate. Treat intrinsic dimension as a diagnostic,
not as proof of intelligence.

## 15. Triangle Inequality

The first `D` is not guaranteed to be a mathematical metric.

If the project claims metric-like geometry, measure triangle inequality
violations:

```text
violation(i,j,k) = D_ik > D_ij + D_jk
```

Report:

- Violation rate.
- Average violation magnitude.
- Whether violations differ from random controls.

If triangle inequality is not tested, describe `D` as distance-like or
geometry-inducing rather than as a strict metric.

## 16. Attention-Duplication Check

A critical risk is that `D` merely duplicates standard attention logits.

Required diagnostic:

```text
corr(flatten(D), flatten(QK^T / sqrt(d)))
```

Also compare:

```text
corr(flatten(-D), flatten(attention_weights))
```

Interpretation:

- Very high correlation may mean `D` adds little new signal.
- Very low correlation may mean `D` captures a different structure.
- The useful case must be judged by ablation results, not correlation alone.

## 17. Random and Shuffled Controls

Every Phase 2 report must compare real `D` with controls.

Controls:

```text
random_D: random distance with similar shape and scale
shuffled_D: real D with positions or entries shuffled
```

The control construction must preserve enough statistics to be fair.

Recommended shuffled variants:

- Shuffle entries within each matrix.
- Shuffle token positions consistently across rows and columns.

Record which shuffle was used.

## 18. Saved Output Format

Recommended `distance_stats.json`:

```json
{
  "run_id": "phase2_distance_example",
  "distance_formula": "-log(W + epsilon)",
  "epsilon": 0.000001,
  "normalization": "offdiag_zscore",
  "diagonal_policy": "reported_separately",
  "layers": [
    {
      "layer": 0,
      "mean": 0.0,
      "std": 0.0,
      "min": 0.0,
      "max": 0.0,
      "entropy": 0.0,
      "attention_logit_correlation": 0.0,
      "neighborhood_overlap_next_layer": 0.0
    }
  ],
  "controls": {
    "random_D": {},
    "shuffled_D": {}
  }
}
```

Recommended `geometry_report.json`:

```json
{
  "summary": {
    "non_trivial_geometry_detected": false,
    "main_evidence": [],
    "main_risks": []
  },
  "cluster_metrics": {},
  "intrinsic_dimension": {},
  "triangle_inequality": {},
  "notes": []
}
```

Use actual measured values in implementation. Placeholder values are schema
only.

## 19. Acceptance Criteria

Phase 2 passes if:

- `D` is numerically stable.
- `D` is not uniform or purely diagonal.
- Real `D` differs from random and shuffled controls.
- `D` has a documented normalization strategy.
- `D` supports non-trivial neighborhoods, clusters, or stability metrics.
- Attention-duplication diagnostics are reported.

## 20. Failure Criteria

Phase 2 requires redesign if:

- `D` contains NaN or Inf.
- `D` is dominated by numerical artifacts.
- `D` collapses to a trivial structure.
- Real `D` behaves like random or shuffled distance.
- `D` is nearly identical to standard attention logits and adds no measurable
  signal.
- Normalization is inconsistent across conditions.

## 21. Design Alternatives if Phase 2 Fails

If `D = -log(W + epsilon)` does not produce useful structure, test alternatives
one at a time:

1. Change `W` normalization before distance.
2. Use cosine-normalized relation strength.
3. Use a temperature-scaled kernel.
4. Try `D = 1 - W`.
5. Try `D = -logit(W)` with clipping.
6. Compute shortest-path distance over a thresholded graph.
7. Use symmetrized or causal-only variants of `W`.

Each alternative must be compared against the original definition and controls.

## 22. Relationship to GeoAttention

Phase 3 consumes normalized `D`:

```text
A = Softmax(QK^T / sqrt(d) - alphaD)
```

Therefore Phase 2 must provide:

- Stable raw `D`.
- Stable normalized `D`.
- Scale diagnostics.
- Control distance construction.
- Documentation of causal and diagonal policy.

If Phase 2 is weak, Phase 3 cannot produce a credible proof.
