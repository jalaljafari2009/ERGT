# Phase 3 Stable Base Plan

## Purpose

The raw ERGT-v1 path is now treated as a diagnostic path, not the main
scientific candidate.

The Phase 3 candidate is:

```text
ERGT-v1 Stable Base =
detached_d + sigmoid_cosine W + clipped/normalized D + alpha warmup + controls
```

This change is motivated by the Phase 3 confirmation seed failure: the raw
`grad_d` route showed sensitivity to seed and numerical instability. Stable
Base keeps the relational-geometry hypothesis intact while applying standard
stabilizers used by attention-bias and cosine-attention literature.

## Stable Base Contract

All Stable Base ERGT runs must use:

```text
attention.gradient_mode = detached_d
relational_graph.kernel = sigmoid_cosine
relational_graph.normalize_hidden = true
distance.normalization = offdiag_zscore_clamp
distance.clip_value = finite positive value
attention.alpha.mode = fixed
attention.alpha.warmup_steps > 0 for non-zero alpha runs
```

The neutral control uses:

```text
condition = alpha_zero
attention.alpha.initial_value = 0.0
attention.alpha.warmup_steps = 0
```

## Why This Is Still a Valid Test

Stable Base does not claim that cosine or clipping are novel. Their role is
to make the emergent distance term testable.

The scientific claim remains:

```text
real_d should beat baseline and should also beat random_d and shuffled_d
under the same stable attention-bias machinery.
```

If random or shuffled distance performs similarly, the result is a generic
regularization/bias effect rather than evidence for emergent relational
geometry.

## Live Monitoring

Long Colab runs should not wait until the final archive to expose direction.
Each trainer now writes:

```text
runs/.../progress_log.jsonl
```

Only evaluation steps are logged to this file. Each record may include:

```text
step
condition
train_loss
validation_loss
best_validation_loss
perplexity
learning_rate
tokens_per_second
elapsed_seconds
elapsed_minutes
grad_norm
gpu_memory_gb
gpu_peak_memory_gb
```

ERGT records also include available geometry scalars:

```text
alpha_effective
target_alpha
alpha_warmup_factor
geo_to_qk_ratio
distance_mean
distance_std
attention_entropy
mean_max_probability
```

This is intentionally lightweight:

```text
no full attention matrix persistence
no per-batch plotting
no extra validation pass
```

## Initial Stable Base Run Set

Baseline:

```text
configs/baseline/phase3_stable_base_seed2027.json
```

ERGT:

```text
configs/ergt_v1/phase3_stable_base/alpha_zero_cosine_seed2027.json
configs/ergt_v1/phase3_stable_base/real_d_alpha_0_05_warmup_cosine_seed2027.json
configs/ergt_v1/phase3_stable_base/real_d_alpha_0_1_warmup_cosine_seed2027.json
configs/ergt_v1/phase3_stable_base/random_d_alpha_0_05_warmup_cosine_seed2027.json
configs/ergt_v1/phase3_stable_base/random_d_alpha_0_1_warmup_cosine_seed2027.json
configs/ergt_v1/phase3_stable_base/shuffled_d_alpha_0_1_warmup_cosine_seed2027.json
```

Colab notebook:

```text
notebooks/ergt_colab_phase3_stable_base.ipynb
```

Comparison:

```text
experiments/compare_phase3_stable_base.py
```

Output:

```text
runs/phase3_geo_attention/phase3_stable_base/phase3_stable_base_results.json
```

## Decision Rule

`stable_candidate_found_repeat_seeds`:

```text
all losses finite
alpha_zero close to baseline
best real_d beats baseline
best real_d beats best random_d
best real_d beats best shuffled_d
```

`stable_signal_partial_repeat_or_extend`:

```text
real_d beats baseline
real_d beats random_d at at least one matched alpha
but the full control set is not yet decisive
```

`control_confounded_signal`:

```text
real_d beats baseline
but random_d or shuffled_d is equally strong or stronger
```

`stable_base_needs_redesign`:

```text
losses are finite but real_d does not beat baseline
```

`unstable_or_incomplete`:

```text
missing result or non-finite loss
```

## Next Step After a Positive Stable Run

Do not move directly to Phase 4 after one positive Stable Base run.

Repeat only the best candidate across seeds:

```text
seed 2026
seed 2027
seed 2028
```

For each seed:

```text
baseline
best stable real_d
matched stable random_d
shuffled_d
```

Gate 1 can pass only after the stable candidate survives this seed
confirmation.
