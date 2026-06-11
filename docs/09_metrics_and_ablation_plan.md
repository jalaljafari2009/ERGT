# Metrics and Ablation Plan

## 1. Purpose

This document defines the metrics and ablations required to evaluate ERGT-v1.

The project must distinguish between:

- Language-model performance.
- Training stability.
- Runtime and memory cost.
- Relational graph structure.
- Emergent distance structure.
- Attention behavior.
- Ablation evidence.

No single metric is enough to validate ERGT. The first proof stage requires a
consistent pattern across performance, controls, and diagnostics.

## 2. Primary Experimental Claim

The first proof stage tests:

```text
Correlation-induced relational distance can improve or stabilize transformer
attention.
```

This claim is supported only if:

```text
GeoAttention(real D)
  outperforms or stabilizes over TransformerBaseline
  and outperforms non-relational distance controls
```

After the Phase 3 ratio-matched evidence, post-Phase-3 work must also follow
`docs/17_physics_aligned_ergt_program.md`. That adds required diagnostics before
any stronger claim:

```text
strict W-level control equality
Sep(real, random/shuffled)
Phi with anti-collapse
reconstruction deficit from allowed context only
memory observer controls
causal shortest-path vs pairwise/no-memory
```

These diagnostics are movement criteria, not optional interpretation aids.

## 3. Required Model Conditions

The required Phase 3 conditions are:

| ID | Condition | Description | Required |
| --- | --- | --- | --- |
| `baseline` | Transformer baseline | Standard causal self-attention | Yes |
| `real_d` | ERGT-v1 | GeoAttention with induced distance | Yes |
| `alpha_zero` | Neutral GeoAttention | GeoAttention code path with `alpha = 0` | Yes |
| `random_d` | Random distance control | Distance sampled with matched shape and scale | Yes |
| `shuffled_d` | Shuffled distance control | Real distance with relational structure disrupted | Yes |

Optional conditions:

| ID | Condition | Description |
| --- | --- | --- |
| `detached_d` | Detached distance | `D` does not receive gradients |
| `trainable_alpha` | Learned alpha | `alpha = softplus(raw_alpha)` |
| `per_head_d` | Head-specific geometry | Separate distance per attention head |
| `raw_d` | Unnormalized distance | Tests normalization necessity |
| `alt_distance` | Alternative distance transform | Tests sensitivity to `-log(W)` |

## 4. Performance Metrics

Required:

- Training loss.
- Validation loss.
- Perplexity.
- Best validation loss.
- Final validation loss.
- Tokens processed.

Recommended:

- Validation-loss area under curve.
- Steps to reach target loss.
- Generalization gap.

Interpretation:

- Lower validation loss and perplexity are direct positive evidence.
- Faster convergence is useful only if final quality is not worse.
- Training loss alone is not sufficient.

## 5. Stability Metrics

Required:

- NaN or Inf occurrence.
- Gradient norm, if available.
- Loss spikes.
- Seed-to-seed variance.
- Run completion rate.

Recommended:

- Standard deviation of validation loss across seeds.
- Worst-seed performance.
- Sensitivity to `alpha`.

Interpretation:

- A model that wins only on one seed is preliminary.
- A model with slightly better mean but much worse worst-case behavior needs
  redesign.
- Stability gains can count as evidence even without large perplexity gains, if
  documented clearly.

## 6. Runtime and Memory Metrics

Required:

- Training tokens per second.
- Evaluation tokens per second.
- Peak memory.
- Wall-clock time, if available.
- Context length.

Recommended:

- Extra memory from `W`.
- Extra memory from `D`.
- Runtime overhead percentage relative to baseline.
- Scaling curve for multiple context lengths.

Interpretation:

- ERGT-v1 must report cost, not only quality.
- A small quality gain with extreme overhead is not a complete success.
- Overhead may be acceptable in Phase 3 if the mechanism is clearly validated,
  but it must be stated.

## 7. Relational Graph Metrics

Required:

- Mean relation strength.
- Relation variance.
- Graph entropy.
- Diagonal dominance.
- Sparsity under threshold.
- Degree distribution.
- Difference from random graph.
- Difference from shuffled graph.

Recommended:

- Effective rank.
- Spectral entropy.
- Cluster score.
- Neighborhood overlap.
- Layer-to-layer similarity.
- Perturbation stability.

Interpretation:

- A useful `W` should not be uniform, saturated, or purely diagonal.
- Real `W` must differ from random and shuffled controls.
- Graph metrics do not prove ERGT by themselves; they justify Phase 2 and 3.

## 8. Distance and Geometry Metrics

Required:

- Distance mean.
- Distance standard deviation.
- Distance min and max.
- Distance entropy.
- Diagonal vs off-diagonal statistics.
- Correlation with `W`.
- Correlation with standard attention logits.
- Difference from random distance.
- Difference from shuffled distance.

Recommended:

- Neighborhood overlap across layers.
- Cluster quality.
- Intrinsic dimension estimate.
- Triangle inequality violation rate, if metric claims are made.
- Spectral profile.

Interpretation:

- `D` must be numerically stable.
- `D` should not merely duplicate `QK`.
- Geometry claims require structure metrics, not only heatmaps.

## 9. Attention Metrics

Required:

- Attention entropy.
- Mean attention max probability.
- Distribution of attention weights.
- Geometry contribution scale:

```text
geo_to_qk_ratio = mean(abs(alphaD)) / (mean(abs(QK)) + epsilon)
```

Recommended:

- Attention-distance correlation.
- Attention sparsity.
- Per-head attention entropy.
- Layer-specific attention change.

Interpretation:

- If `geo_to_qk_ratio` is too small, ERGT is effectively baseline.
- If `geo_to_qk_ratio` is too large, `D` may dominate attention.
- Attention metrics are diagnostic; performance and ablations decide the claim.

## 10. Alpha Metrics

Required when `alpha` is used:

- Initial `alpha`.
- Final `alpha`, if trainable.
- Fixed or trainable mode.
- Non-negativity constraint.
- `geo_to_qk_ratio`.

Recommended:

- Per-layer alpha, if layer-specific.
- Per-head alpha, if head-specific.
- Alpha trajectory over training.

Interpretation:

- If trainable `alpha` collapses to zero, the model may reject geometry.
- If `alpha` explodes, normalization or constraints are wrong.
- Fixed alpha sweeps should precede complex alpha schedules.

## 11. Required Ablation: Alpha Zero

Condition:

```text
alpha = 0
```

Purpose:

Tests whether the GeoAttention implementation path is equivalent to baseline
when the geometry term is disabled.

Expected result:

```text
alpha_zero ~= baseline
```

If `alpha_zero` differs substantially from baseline, the implementation changed
something besides geometry and Phase 3 results are not clean.

## 12. Required Ablation: Random Distance

Condition:

```text
D = random matrix with matched shape and scale
```

Purpose:

Tests whether any additional attention bias improves the model.

Expected result for ERGT support:

```text
real_d > random_d
```

If `random_d` performs similarly to `real_d`, the relational structure claim is
weak.

## 13. Required Ablation: Shuffled Distance

Condition:

```text
D = shuffled real distance
```

Possible shuffle methods:

- Shuffle entries within each matrix.
- Shuffle token positions consistently across rows and columns.

Purpose:

Tests whether the arrangement of relational distances matters.

Expected result for ERGT support:

```text
real_d > shuffled_d
```

If shuffled distance performs similarly, the geometry may be acting only through
global statistics rather than relational structure.

## 14. Optional Ablation: Detached Distance

Condition:

```text
D = D.detach()
```

Purpose:

Tests whether ERGT works as an observer-derived bias or requires end-to-end
geometry shaping.

Interpretation:

- `detached_d` helps: relational structure is useful even without shaping.
- `grad_d` helps more: the model benefits from learning its geometry.
- `detached_d` is more stable: gradients through `D` may need redesign.

## 15. Optional Ablation: Trainable Alpha

Condition:

```text
alpha = softplus(raw_alpha)
```

Purpose:

Tests whether the model can learn geometry strength.

Interpretation:

- Useful if fixed-alpha sweeps show sensitivity.
- Not a replacement for fixed-alpha controls.
- Must log alpha trajectory.

## 16. Optional Ablation: Per-Head Distance

Condition:

```text
D: [batch, heads, sequence, sequence]
```

Purpose:

Tests whether different heads benefit from different relational geometries.

Interpretation:

- May improve expressivity.
- Increases compute and analysis complexity.
- Should not be first default.

## 17. Optional Ablation: Alternative Distance

Possible alternatives:

```text
D = 1 - W
D = -logit(W) with clipping
D = shortest_path_distance(thresholded_W)
```

Purpose:

Tests whether the first distance transform is the right operationalization.

Rule:

Only test alternatives after the default `-log(W + epsilon)` protocol is
implemented and measured.

## 18. Seed Protocol

Smoke tests:

```text
1 seed
```

Preliminary result:

```text
3 seeds
```

Preferred claim:

```text
5 seeds
```

Report:

- Mean.
- Standard deviation.
- Best seed.
- Worst seed.
- Ranking consistency.

Do not make strong claims from a single seed.

## 19. Result Tables

The Phase 3 report should include at least these tables.

### Performance Table

| Condition | Val Loss | Perplexity | Seeds | Notes |
| --- | --- | --- | --- | --- |
| baseline | | | | |
| alpha_zero | | | | |
| real_d | | | | |
| random_d | | | | |
| shuffled_d | | | | |

### Cost Table

| Condition | Tokens/sec | Peak Memory | Overhead vs Baseline |
| --- | --- | --- | --- |
| baseline | | | |
| real_d | | | |
| random_d | | | |
| shuffled_d | | | |

### Geometry Table

| Condition | Graph Entropy | Distance Entropy | Attention Corr | Notes |
| --- | --- | --- | --- | --- |
| real_d | | | | |
| random_d | | | | |
| shuffled_d | | | | |

## 20. Decision Rules

### Pass

All of the following hold:

```text
real_d improves or stabilizes over baseline
real_d outperforms random_d
real_d outperforms shuffled_d
alpha_zero matches baseline
runtime and memory overhead are documented
```

Action:

```text
Proceed to Phase 4.
```

### Conditional Pass

Some evidence supports `real_d`, but one issue remains:

- Too few seeds.
- High overhead.
- Alpha sensitivity.
- Weak but consistent gains.
- Missing optional diagnostics.

Action:

```text
Repeat Phase 3 with targeted fixes or more seeds.
```

### Fail

Any of the following hold:

```text
real_d does not beat controls
alpha_zero differs from baseline
D is unstable or trivial
training diverges
cost is impractical without evidence of mechanism
```

Action:

```text
Do not proceed to Phase 4. Redesign W, D, normalization, or injection point.
```

## 21. Reporting Discipline

Reports must separate:

```text
measured result
interpretation
theoretical implication
remaining uncertainty
```

Avoid claiming:

- ERGT proves intelligence.
- ERGT discovers concepts.
- ERGT has memory.
- ERGT reasons better.

unless the experiment directly measures those claims.

The first proof can only support:

```text
Induced relational distance is a useful attention bias.
```

That result would be enough to justify deeper phases.
