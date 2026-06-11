# Phase 3 Stable Base Result

Date: 2026-06-11

## Run Scope

This result imports the Colab light Stable Base run at seed `2027`.

Stable Base contract:

- `gradient_mode = detached_d`
- `relational_graph.kernel = sigmoid_cosine`
- `relational_graph.normalize_hidden = true`
- `distance.normalization = offdiag_zscore_clamp`
- clipped distance with alpha warmup
- controls: `alpha_zero`, `real_d`, `random_d`, `shuffled_d`

## Primary Numbers

Baseline / alpha-zero:

- `baseline`: final validation loss `5.751527379150862`
- `alpha_zero_cosine`: final validation loss `5.751527379150862`

Best runs by family:

- `random_d alpha=0.1`: final validation loss `5.748521477751334`
- `real_d alpha=0.1`: final validation loss `5.749332700035968`
- `shuffled_d alpha=0.1`: final validation loss `5.755773684268877`

Matched-alpha checks:

- At alpha `0.05`, `real_d` beats `random_d` by `-0.0007686958745134476`.
- At alpha `0.1`, `random_d` beats `real_d` by `0.0008112222846339279`.

## Interpretation

The Stable Base run supports a stabilized geometry-injection pipeline:

- `alpha_zero` exactly matches the baseline, so the ERGT wrapper is not changing the model when the geometric term is disabled.
- `real_d` improves over baseline and alpha-zero.
- `shuffled_d` is clearly worse than `real_d`, which means corrupted geometry can hurt and the second term is not inert.

The run does not yet prove that real emergent distance is superior to random distance. The best `random_d` condition is slightly better than the best `real_d` condition, and the observed `geo_to_qk_ratio` values are not equal across families.

Current recommendation:

`stable_signal_partial_repeat_or_extend`

## Next Control

The next comparison must match `geo_to_qk_ratio`, not merely alpha. Generated config targets:

- `0.15`
- `0.20`
- `0.25`

Generated configs are stored under:

`configs/ergt_v1/phase3_ratio_matched/`

The ratio-matched comparison should be used before making a Gate 1 decision.
