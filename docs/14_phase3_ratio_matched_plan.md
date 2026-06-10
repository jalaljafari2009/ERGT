# Phase 3 Ratio-Matched Geometry Plan

## Purpose

Matched alpha is not sufficient for a fair ERGT comparison.

The attention update is:

```text
logits = QK - alpha * D
```

Equal `alpha` does not guarantee equal geometry strength because `D` can have
different scale and distribution under:

```text
real_d
random_d
shuffled_d
```

The fairer comparison is:

```text
geo_to_qk_ratio(real_d) ~= geo_to_qk_ratio(random_d) ~= geo_to_qk_ratio(shuffled_d)
```

This stage tests whether the structure of `real_d` is useful after the
magnitude of the geometry term has been controlled.

## Contract

Ratio-matched runs must still use Stable Base:

```text
gradient_mode = detached_d
relational_graph.kernel = sigmoid_cosine
relational_graph.normalize_hidden = true
distance.normalization = offdiag_zscore_clamp
distance.clip_value = finite
alpha.warmup_steps > 0
```

The only thing ratio matching changes is the target alpha assigned to each
family.

## Calibration

Use completed Stable Base runs as calibration points.

For each family:

```text
observed_ratio = final geometry_summary.geo_to_qk_ratio
calibration_alpha = attention.alpha.initial_value
```

For a target ratio:

```text
generated_alpha = calibration_alpha * target_ratio / observed_ratio
```

This assumes local approximate linearity between `alpha` and
`geo_to_qk_ratio`. The final comparison validates the actual observed ratio
after training; if the actual ratio misses the target, the result is marked as
needing recalibration.

## Tools

Generate configs:

```powershell
python experiments/build_ratio_matched_configs.py `
  --target-ratio 0.15 `
  --target-ratio 0.30 `
  --calibration real_d:runs/phase3_geo_attention/phase3_stable_base/real_d_alpha_0_05_warmup_cosine/metrics.json `
  --calibration random_d:runs/phase3_geo_attention/phase3_stable_base/random_d_alpha_0_05_warmup_cosine/metrics.json `
  --calibration shuffled_d:runs/phase3_geo_attention/phase3_stable_base/shuffled_d_alpha_0_1_warmup_cosine/metrics.json
```

This writes:

```text
configs/ergt_v1/phase3_ratio_matched/
configs/ergt_v1/phase3_ratio_matched/ratio_matched_manifest.json
```

Compare completed ratio-matched runs:

```powershell
python experiments/compare_phase3_ratio_matched.py `
  --baseline runs/phase0_baseline/phase3_stable_base_seed2027/baseline_results.json `
  --run real_d:0.15:runs/.../metrics.json `
  --run random_d:0.15:runs/.../metrics.json `
  --run shuffled_d:0.15:runs/.../metrics.json `
  --ratio-tolerance 0.03
```

This writes:

```text
runs/phase3_geo_attention/phase3_ratio_matched/phase3_ratio_matched_results.json
```

## Interpretation

Strong support:

```text
actual ratios are within tolerance
real_d beats baseline
real_d beats random_d at the same target ratio
real_d beats shuffled_d at the same target ratio
```

Partial support:

```text
actual ratios are within tolerance
real_d beats baseline
but controls are not fully beaten
```

Recalibration:

```text
actual geo_to_qk_ratio misses the target tolerance
```

Failure:

```text
actual ratios are controlled
but real_d does not beat baseline or controls
```

## Why This Matters

If `random_d` only helps when its geometry term is weak, then matched-alpha
comparisons can confuse two effects:

```text
1. structure of D
2. magnitude of alpha * D
```

Ratio matching isolates the first effect. It asks:

```text
At equal geometry strength, is real relational distance better than random or shuffled distance?
```

That is a stronger test of the ERGT hypothesis than alpha matching alone.
