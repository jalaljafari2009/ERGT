# Phase 3 Ratio-Matched T0.20 Result

Date: 2026-06-11

## Scope

This result imports the Colab light ratio-matched run for target
`geo_to_qk_ratio = 0.20` at seed `2027`.

The goal was to remove the main confound from the Stable Base comparison:
`real_d`, `random_d`, and `shuffled_d` should be judged at approximately equal
geometry contribution strength.

## Primary Numbers

Baseline:

- `baseline`: final validation loss `5.751527379150862`

Ratio-matched target `0.20`:

- `random_d`: final validation loss `5.748366042096878`, `geo_to_qk_ratio = 0.19899207426876317`
- `real_d`: final validation loss `5.749569392965456`, `geo_to_qk_ratio = 0.19839301611719234`
- `shuffled_d`: final validation loss `5.755869139603557`, `geo_to_qk_ratio = 0.20630582846456863`

Key deltas:

- `real_d - baseline`: `-0.001957986185406213`
- `real_d - random_d`: `+0.0012033508685780347`
- `real_d - shuffled_d`: `-0.006299746638100956`

## Interpretation

This run strengthens the evidence that the second attention term is active and
can matter:

- `real_d` beats the baseline.
- `shuffled_d` is substantially worse than `real_d`, so corrupted structure can
  harm the model.

However, this run does not support the strict Phase 3 success criterion:

- `random_d` beats `real_d` at the same target `geo_to_qk_ratio`.

Current recommendation:

`ratio_matched_partial_support`

## Design Risk Found

The next test should tighten the distance controls before concluding that
`random_d` is genuinely better.

Current implementation risk:

- `real_d` receives distance policy and normalization before attention use.
- `random_d` and `shuffled_d` are built from the already-processed distance.
- This can disturb policies such as diagonal handling after the control is
  generated.

The next corrective run should ensure that all families have identical:

- diagonal policy
- causal policy
- normalization policy
- clipping policy
- matched `geo_to_qk_ratio`

Only the relational arrangement should differ between `real_d`, `random_d`, and
`shuffled_d`.

If `random_d` still beats `real_d` after strict controls, Phase 3 should redesign
`W` rather than proceed to Phase 4.

## Program Update

This result changes the planning baseline. Post-Phase-3 work should follow
`docs/17_physics_aligned_ergt_program.md` rather than moving directly from
GeoAttention v1 to graph-memory implementation.

The next movement standard is:

```text
measure -> control -> observe -> validate -> inject -> regularize
```

Immediate next work:

```text
strict W-level controls
relational field observer
resonant-response observer
Phi and reconstruction gates
memory-as-observer before memory injection
causal shortest-path geometry before GeoAttention v2
```
