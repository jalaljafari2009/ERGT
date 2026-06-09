# First Proof Stage Protocol

## 1. Purpose

This document defines the controlled experimental protocol for the first ERGT
proof stage.

The goal is not to prove the full theory of relational intelligence. The goal is
to test one falsifiable claim:

```text
Correlation-induced relational distance can improve or stabilize transformer
attention.
```

The proof stage covers:

```text
Phase 0: Baseline Transformer
Phase 1: Relational Graph Observer
Phase 2: Emergent Distance
Phase 3: GeoAttention v1
```

## 2. Primary Hypothesis

The primary experimental hypothesis is:

```text
GeoAttention(real D) performs better or trains more stably than a matched
TransformerBaseline and distance-control variants.
```

where:

```text
D_ij = -log(W_ij + epsilon)
W_ij = sigmoid((x_i^T x_j) / sqrt(d))
```

## 3. What Counts as Evidence

The experiment produces evidence for ERGT-v1 only if real induced distance
outperforms or stabilizes relative to both:

- The standard transformer baseline.
- Non-relational distance controls.

An improvement over the baseline alone is not enough. The real distance must
also beat random, shuffled, or disabled distance controls.

## 4. Controlled Variables

All model comparisons must hold these constant:

- Dataset
- Tokenizer
- Train/validation split
- Context length
- Model width
- Number of layers
- Number of attention heads
- Parameter budget, within a documented tolerance
- Optimizer
- Learning-rate schedule
- Weight decay
- Dropout
- Batch size or effective batch size
- Training steps or token budget
- Evaluation interval
- Random seed list
- Hardware profile, when available

If a variable cannot be held constant, it must be recorded and justified.

## 5. Dataset Protocol

The first dataset should be small enough for repeated controlled experiments.

Recommended first dataset:

```text
WikiText-2
```

Optional later datasets:

```text
WikiText-103
OpenWebText subset
Synthetic long-range dependency tasks
```

The proof stage should not start with a very large dataset. The goal is to
establish mechanism and ablations before scaling.

## 6. Tokenizer Protocol

All compared models must use the same tokenizer.

Acceptable choices:

- A simple BPE tokenizer trained or reused consistently.
- GPT-2 tokenizer, if used identically for all models.
- Character-level tokenizer only for early debug experiments.

The tokenizer choice must be recorded in the experiment config.

## 7. Model Family

The baseline should be a small GPT-style causal transformer.

The implementation should be simple enough that the attention module can be
inspected and replaced.

Recommended baseline properties:

- Causal language model.
- Pre-norm transformer blocks.
- Standard multi-head self-attention.
- Feed-forward network.
- Residual connections.
- Layer normalization.

The baseline should be trained from scratch for the controlled comparison.
Pretrained GPT-2 may be used for reference analysis, but not as the primary
baseline unless ERGT is trained under an equivalent pretraining setup.

## 8. Compared Conditions

The required comparison matrix is:

| Condition | Description | Purpose |
| --- | --- | --- |
| `baseline` | Standard transformer attention | Main control |
| `real_d` | GeoAttention using induced `D` | ERGT-v1 candidate |
| `alpha_zero` | GeoAttention path with `alpha = 0` | Checks implementation neutrality |
| `random_d` | GeoAttention with random distance | Tests whether any bias helps |
| `shuffled_d` | GeoAttention with shuffled real distance | Tests whether structure matters |

Optional additional conditions:

| Condition | Description | Purpose |
| --- | --- | --- |
| `detached_d` | `D` does not receive gradients | Tests graph as observer-derived bias |
| `trainable_alpha` | `alpha` is learned | Tests adaptive geometry strength |
| `fixed_alpha` | `alpha` is fixed | Tests sensitivity to geometry scale |
| `per_head_d` | Separate `D` per head | Tests head-specific geometry |
| `shared_d` | One shared `D` per layer | Lower-cost default |

## 9. Phase 0 Protocol: Baseline

### Goal

Train and evaluate the standard transformer baseline.

### Required Outputs

```text
runs/phase0_baseline/config.json
runs/phase0_baseline/train_log.jsonl
runs/phase0_baseline/baseline_results.json
runs/phase0_baseline/model_summary.json
```

### Required Metrics

- Training loss curve
- Validation loss
- Perplexity
- Tokens per second
- Peak memory
- Seed-level variance, if multiple seeds are run

### Acceptance Criteria

- The model trains without numerical instability.
- Validation loss decreases meaningfully.
- Evaluation can be reproduced from config and checkpoint.
- Attention code path is accessible for Phase 3 replacement.

## 10. Phase 1 Protocol: Relational Graph Observer

### Goal

Measure relation graphs from hidden states without changing model outputs.

### Required Outputs

```text
runs/phase1_graph/config.json
runs/phase1_graph/graph_stats.json
runs/phase1_graph/sample_matrices/
```

### Required Metrics

- Mean relation strength
- Relation variance
- Graph entropy
- Sparsity at configured thresholds
- Degree distribution
- Layer-to-layer graph similarity
- Difference from random and shuffled controls

### Acceptance Criteria

- `W` is not uniform.
- `W` is not saturated.
- `W` is not dominated only by the diagonal.
- `W` changes with input content.
- At least one non-trivial structure metric is present.

## 11. Phase 2 Protocol: Emergent Distance

### Goal

Convert `W` into `D` and test whether distance induces geometry-like structure.

### Required Outputs

```text
runs/phase2_distance/config.json
runs/phase2_distance/geometry_report.json
runs/phase2_distance/distance_stats.json
runs/phase2_distance/sample_matrices/
```

### Required Metrics

- Distance mean and variance
- Distance entropy
- Distance normalization statistics
- Neighborhood overlap across layers
- Cluster quality or community score
- Intrinsic dimension estimate, if implemented
- Correlation between `D` and standard attention logits

### Acceptance Criteria

- `D` is numerically stable.
- `D` differs from random and shuffled distance controls.
- `D` is not only a diagonal or uniform penalty.
- `D` has a documented normalization strategy.

## 12. Phase 3 Protocol: GeoAttention

### Goal

Train and evaluate ERGT-v1 under controlled comparison.

### Core Equation

```text
A = Softmax(QK^T / sqrt(d) - alphaD)
```

### Required Outputs

```text
runs/phase3_geo_attention/config.json
runs/phase3_geo_attention/train_log.jsonl
runs/phase3_geo_attention/comparison_results.json
runs/phase3_geo_attention/ablation_report.json
runs/phase3_geo_attention/model_summary.json
```

### Required Metrics

- Validation loss
- Perplexity
- Training stability
- Tokens per second
- Peak memory
- Attention entropy
- `alpha` values over training, if trainable
- Geometry contribution scale relative to attention logits

### Acceptance Criteria

At least one of the following must hold:

- `real_d` has lower validation loss than baseline and controls.
- `real_d` has lower perplexity than baseline and controls.
- `real_d` has materially better stability across seeds.
- `real_d` improves a predefined long-range dependency metric.

The result is not accepted if the same improvement appears with random or
shuffled distance.

## 13. Statistical Protocol

A single seed is acceptable only for smoke tests.

For any claim, run multiple seeds:

```text
minimum: 3 seeds
preferred: 5 seeds
```

Report:

- Mean
- Standard deviation
- Best and worst seed
- Whether ranking is consistent across seeds

If compute is limited, the project may run one seed first, but the result must
be labeled as preliminary.

## 14. Runtime and Memory Protocol

ERGT adds `O(sequence^2)` relational objects, so runtime and memory must be
measured from the start.

Record:

- Training tokens per second
- Evaluation tokens per second
- Peak memory
- Extra memory from `W` and `D`
- Sequence length sensitivity

A small quality gain with extreme cost is not enough for a successful proof.

## 15. Gradient Protocol

Experiments must record whether gradients pass through:

- `W`
- `D`
- `alpha`

Recommended first setup:

```text
real_d: allow gradients unless unstable
detached_d: optional control
alpha: fixed first, trainable later
```

If training becomes unstable, the next controlled variant should detach `D` or
normalize it more aggressively before changing the broader architecture.

## 16. Alpha Protocol

`alpha` controls the strength of the geometry bias.

Initial options:

```text
fixed alpha: 0.01, 0.05, 0.1, 0.5, 1.0
trainable alpha: initialized small and constrained non-negative
```

The first run should avoid a large untested `alpha`. If `alphaD` overwhelms
`QK`, the experiment no longer tests a controlled geometry bias.

Record:

- Initial `alpha`
- Whether `alpha` is fixed or learned
- Final `alpha`
- Ratio of `alphaD` scale to `QK` scale

## 17. Normalization Protocol

`D` must be normalized before being used in attention unless an experiment
explicitly tests raw distance.

Acceptable first normalizations:

```text
D_norm = (D - mean(D)) / (std(D) + epsilon)
D_norm = D / (mean(D) + epsilon)
D_norm = clamp(D, min_value, max_value)
```

The chosen method must be fixed across compared conditions.

## 18. Causal Mask Protocol

For causal language modeling, the causal mask remains mandatory.

The geometry term must not permit future-token leakage.

Correct order:

```text
logits = QK^T / sqrt(d) - alphaD
logits = apply_causal_mask(logits)
A = softmax(logits)
```

Any future-aware construction of `W` or `D` must be masked or excluded for
causal experiments.

## 19. Result Interpretation

### Strong Positive Result

`real_d` beats baseline and all controls across multiple seeds with acceptable
cost.

Action:

```text
Proceed to Phase 4.
```

### Weak Positive Result

`real_d` sometimes helps, but results are seed-sensitive or cost-heavy.

Action:

```text
Improve normalization, alpha schedule, graph construction, and repeat Phase 3.
```

### Negative Result

`real_d` does not beat controls.

Action:

```text
Do not proceed to Phase 4. Revisit W, D, normalization, or injection point.
```

### Ambiguous Result

Metrics conflict or controls are incomplete.

Action:

```text
Run missing controls before making theoretical claims.
```

## 20. Reporting Requirements

The final Phase 3 report must include:

- Experimental config.
- Model sizes.
- Dataset and tokenizer details.
- Training budget.
- Comparison table.
- Ablation table.
- Runtime and memory table.
- Graph and distance diagnostics.
- Seed variance.
- Failure cases.
- Gate decision.

The report should separate:

```text
What was measured
What improved
What failed
What remains theoretical
```

## 21. Proof Boundary

Passing this protocol does not prove the full ERGT theory.

It only supports this claim:

```text
Induced relational distance can be a useful attention bias.
```

That result would justify moving toward relational memory and deeper geometry.
It would not yet prove concept emergence, reasoning, or intelligence.
