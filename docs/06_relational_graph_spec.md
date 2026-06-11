# Relational Graph Specification

## 1. Purpose

The relational graph is the first operational object of ERGT.

It converts hidden-state correlations into an explicit relation matrix `W`.
Phase 1 uses this graph only as an observer. It must not change model behavior.

The Phase 1 research question is:

```text
Do non-trivial relational structures emerge inside a baseline transformer?
```

## 2. Theoretical Role

ERGT treats relations as more fundamental than isolated vectors.

The hidden states `X` are the computational coordinate substrate. The relational
graph `W` is the induced structure extracted from that substrate.

The first operational bridge is:

```text
Hidden states -> Correlation kernel -> Relational graph
```

## 3. Core Formula

The default relation kernel is:

```text
K_ij = sigmoid((x_i^T x_j) / sqrt(d))
```

For Phase 1:

```text
W = K
```

where:

- `x_i` is the hidden state at sequence position `i`.
- `x_j` is the hidden state at sequence position `j`.
- `d` is the hidden dimension or projected relation dimension.
- `K_ij` is the instantaneous relation strength.
- `W` is the relational graph used for analysis.

## 4. Tensor Contract

Input:

```text
hidden_states: [batch, sequence, hidden_dim]
```

Default output:

```text
W: [batch, 1, sequence, sequence]
```

Optional later output:

```text
W: [batch, heads, sequence, sequence]
```

The first implementation should prefer one shared graph per layer unless there
is a strong reason to begin with per-head graphs. A shared graph is cheaper,
simpler, and easier to analyze.

## 5. Layer Placement

The graph observer should be attachable to selected layers.

Recommended first placements:

- After token and position embedding.
- After each transformer block output.
- Optionally before attention inside a block.

For Phase 1, the preferred measurement point is:

```text
block output hidden_states
```

This gives a clean layer-to-layer sequence of relation graphs.

## 6. Causal Handling

For causal language modeling, Phase 1 may compute full pairwise relations for
analysis only if those relations do not feed back into model computation.

However, any relation graph used later inside attention must respect causality.

Two modes should be distinguished:

```text
analysis_mode: may inspect full W after the forward pass
causal_runtime_mode: masks future positions before W or D affects attention
```

Experiments must record which mode was used.

## 7. Diagonal Handling

The diagonal `W_ii` represents self-relation and can dominate analysis.

Required policy:

- Record whether the diagonal is included.
- For graph metrics, report both diagonal-included and diagonal-excluded
  variants when practical.
- For distance and attention use, define the diagonal behavior explicitly.

Recommended Phase 1 analysis:

```text
W_metrics = W with diagonal excluded where appropriate
W_visualization = W with diagonal visible and separately annotated
```

## 8. Normalization Options

The default kernel uses sigmoid. This bounds relation strength:

```text
W_ij in (0, 1)
```

Potential issue:

- Sigmoid can saturate if dot products are large.

Optional stabilizers:

```text
normalize hidden states before dot product
project hidden states before relation kernel
temperature-scale the dot product
```

Possible normalized kernel:

```text
z_i = layer_norm_or_l2_norm(x_i)
K_ij = sigmoid((z_i^T z_j) / tau)
```

Any normalization choice must be recorded.

## 9. Projection Option

A relation-specific projection may be introduced:

```text
r_i = x_i W_r
K_ij = sigmoid((r_i^T r_j) / sqrt(d_r))
```

This should not be the first default unless raw hidden-state correlations are
too unstable or uninformative.

If projection is used, record:

- Projection dimension.
- Whether projection is shared across layers.
- Whether projection parameters are trained.
- Whether graph gradients affect the base model.

## 10. Phase 1 Observer Mode

In Phase 1, the graph observer must not alter:

- Attention logits.
- Hidden states.
- Loss.
- Gradients.
- Optimizer updates.

This is essential. Phase 1 asks whether relation structure exists naturally
inside the baseline.

The observer may:

- Record selected `W` matrices.
- Aggregate graph metrics.
- Save samples for visualization.
- Compare real graphs to random and shuffled controls.

## 11. Metrics

Required graph metrics:

- Mean relation strength.
- Relation variance.
- Graph entropy.
- Sparsity under thresholds.
- Degree distribution.
- Diagonal dominance.
- Layer-to-layer similarity.
- Difference from shuffled graph.
- Difference from random graph.

Recommended additional metrics:

- Effective rank.
- Spectral entropy.
- Cluster or community score.
- Neighborhood overlap.
- Stability under input perturbation.
- Similarity across related examples.

## 12. Graph Entropy

Graph entropy should measure how concentrated or diffuse relation strengths are.

A simple first implementation may compute entropy over normalized edge weights:

```text
p_ij = W_ij / sum(W)
H(W) = -sum(p_ij * log(p_ij + epsilon))
```

When excluding the diagonal, compute `p_ij` only over off-diagonal entries.

The exact entropy definition must be stored in the report.

## 13. Sparsity

Although `W` is dense by construction, effective sparsity can be measured by
thresholding.

Example thresholds:

```text
0.50
0.75
0.90
0.95
```

Report:

```text
sparsity_tau = fraction of W_ij below threshold tau
```

Sparsity is not automatically good. The goal is to understand structure.

## 14. Degree Distribution

For a thresholded graph:

```text
degree_i = count_j(W_ij >= tau)
```

Report:

- Mean degree.
- Degree variance.
- Maximum degree.
- Minimum degree.
- Histogram if useful.

This helps detect whether the graph is uniform, hub-dominated, or locally
structured.

## 15. Layer-to-Layer Similarity

Layer-to-layer persistence is an early proxy for relational stability.

Possible metrics:

```text
cosine_similarity(flatten(W_l), flatten(W_l+1))
frobenius_distance(W_l, W_l+1)
neighborhood_overlap(W_l, W_l+1)
```

This does not yet prove memory, but it indicates whether relational structures
remain coherent across computation.

## 16. Perturbation Stability

A useful relational structure should not vanish under small non-semantic
perturbations.

Possible perturbations:

- Dropout-mode variation.
- Token masking for non-critical tokens.
- Paraphrase, in later datasets.
- Slight prompt extension.

For Phase 1, perturbation stability is optional but recommended.

## 17. Random and Shuffled Controls

Graph analysis must compare real `W` to controls.

Required controls:

```text
random_W: random matrix with similar shape and scale
shuffled_W: real W with entries or positions shuffled
```

Post-Phase-3 strict-control update:

Random and shuffled controls must be constructed at the `W` level before
distance normalization and attention use. They must preserve the same valid
region:

```text
valid_edge = causal_lower_triangular & non_diagonal & non_padding
```

Shuffling must not mix diagonal, padding, or future-invalid entries with valid
causal off-diagonal entries. The detailed movement standard is
`docs/17_physics_aligned_ergt_program.md`.

Purpose:

- `random_W` tests whether metrics are artifacts of dense matrices.
- `shuffled_W` tests whether structure depends on relational arrangement.

## 18. Visualization

Visualizations are useful but not sufficient.

Recommended outputs:

```text
runs/phase1_graph/W_heatmaps/
runs/phase1_graph/layer_comparison_plots/
runs/phase1_graph/degree_histograms/
```

Visualizations should be accompanied by numeric metrics in JSON.

## 19. Saved Output Format

Recommended `graph_stats.json` structure:

```json
{
  "run_id": "phase1_graph_example",
  "graph_kernel": "sigmoid_dot_sqrt_d",
  "diagonal_policy": "excluded_for_metrics",
  "layers": [
    {
      "layer": 0,
      "mean": 0.0,
      "variance": 0.0,
      "entropy": 0.0,
      "diagonal_dominance": 0.0,
      "sparsity": {
        "0.5": 0.0,
        "0.75": 0.0,
        "0.9": 0.0
      },
      "effective_rank": 0.0
    }
  ],
  "controls": {
    "random_W": {},
    "shuffled_W": {}
  }
}
```

Use actual values in implementation. Placeholder zeros are shown only as schema.

## 20. Acceptance Criteria

Phase 1 passes if:

- The graph observer runs without changing baseline behavior.
- `W` is numerically stable.
- `W` is not uniform, saturated, or purely diagonal.
- Real `W` differs from random and shuffled controls.
- At least one structure or stability metric is non-trivial.
- Outputs are saved in machine-readable form.

## 21. Failure Criteria

Phase 1 requires redesign if:

- `W` saturates near 0 or 1.
- `W` is nearly uniform across inputs.
- `W` is dominated entirely by self-relations.
- `W` does not differ from random or shuffled controls.
- Relation metrics are not reproducible.
- Observer mode accidentally affects model training or evaluation.

## 22. Design Alternatives if Phase 1 Fails

If raw hidden-state dot products do not produce meaningful graphs, test in this
order:

1. L2-normalize hidden states before dot product.
2. Add a temperature parameter.
3. Use cosine similarity instead of raw dot product.
4. Use a learned relation projection.
5. Compute relation from `Q` and `K` projections instead of raw hidden states.
6. Test layer-specific graph construction.

Only change one design variable at a time.

## 23. Relationship to Later Phases

Phase 2 consumes `W` to produce distance:

```text
D_ij = -log(W_ij + epsilon)
```

Phase 3 uses `D` inside attention:

```text
A = Softmax(QK^T / sqrt(d) - alphaD)
```

Therefore the relational graph must be:

- Stable enough to analyze.
- Numerically safe for logarithmic distance.
- Cheap enough for repeated training.
- Compatible with causal masking when used at runtime.

If `W` is weak, every later ERGT phase becomes weak.
