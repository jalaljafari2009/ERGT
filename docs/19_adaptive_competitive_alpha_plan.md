# Adaptive Competitive Alpha Plan

## 1. Why This Step Exists

The guarded 2000-step evidence run showed that `real_memory_d` was mechanically
healthy and slightly better than baseline, but the effect stayed small. The key
diagnostic was:

```text
alpha fixed at 0.025
geo/qk fell from about 0.022 near step 1000 to about 0.014 near step 2000
```

This means the geometry bias did not collapse attention, but it also did not
remain in strong competition with QK as the language model became stronger.

The next step should therefore not be a blind larger alpha. It should be a
competitive alpha policy driven by evidence from the training trajectory.

## 2. Principle

The model should not protect attention from geometry by construction. If stable
causal geometry improves the trajectory, alpha should be allowed to grow.

At the same time, geometry should not take over by force. Alpha growth must be
earned by loss slope, reference advantage, and non-collapse diagnostics.

The rule is:

```text
let geometry compete with QK,
but require evidence that the competition improves the model.
```

## 3. Control Signal

Alpha is updated only at evaluation points, for example every 100 steps. A
single validation point is not enough. The controller fits a line over a rolling
window of recent validation points:

```text
validation_loss ~= slope * step + intercept
```

The main signal is:

```text
slope_gain = baseline_slope - ergt_slope
```

Because lower loss is better, a positive `slope_gain` means ERGT is reducing
validation loss faster than the baseline reference over the window.

The controller also tracks:

```text
advantage = baseline_validation_loss - ergt_validation_loss
geo/qk risk
attention entropy drop
mean max attention probability
```

## 4. Score

The score is intentionally windowed and smoothed:

```text
score =
  slope_gain_weight * EMA(slope_gain)
+ advantage_weight  * EMA(advantage)
- geo_qk_risk_weight * geo_qk_risk
- entropy_drop_weight * entropy_risk
- max_probability_risk_weight * max_probability_risk
```

This prevents one noisy validation point from causing a large alpha move.

## 5. Alpha Update

Alpha is not a fixed warmup anymore. It is a controlled state variable:

```text
if score > positive_margin:
    alpha grows
elif score < negative_margin:
    alpha shrinks
else:
    alpha holds
```

The update uses inertia:

```text
alpha_next =
  alpha_current
+ bounded_change((1 - inertia) * proposed_delta)
```

This makes alpha responsive to a trend, not to a single point.

## 6. Exploration

The controller needs a small geometry signal before it can measure whether
geometry helps. Therefore the first few evaluation points may use a small
exploration growth toward a low alpha, unless risk diagnostics are already bad.

Exploration is not a claim. It is only a way to expose the model to a measurable
geometry signal.

## 7. What Is Not a Hard Stop

`geo/qk` is not a hard ceiling in this phase. It is a risk pressure. If geometry
can grow while validation slope improves and attention does not collapse, alpha
is allowed to keep growing.

There is still a numerical `max_alpha` to avoid runaway experiments, but this is
a runtime safety bound, not a scientific claim.

## 8. Required Controls

The adaptive policy must be run on the same control families:

```text
baseline
alpha_zero
real_memory_d_adaptive
random_memory_d_adaptive
shuffled_memory_d_adaptive
no_memory_real_d_adaptive
instantaneous_real_d_adaptive
```

The claim gate is stricter than "real beats baseline":

```text
real adaptive geometry must beat baseline
real adaptive geometry must beat random adaptive geometry
real adaptive geometry must beat shuffled adaptive geometry
real adaptive memory must beat no-memory and instantaneous variants
```

If random or shuffled grows alpha and improves as much as real, the result is a
generic regularization effect, not evidence for real relational geometry.

## 9. New Artifacts

This phase adds:

```text
experiments/adaptive_alpha.py
experiments/train_ergt_adaptive_alpha.py
notebooks/ERGT_02_Adaptive_Competitive_Alpha.ipynb
```

The previous attention evidence notebook remains unchanged as the record of the
fixed-alpha guarded run.
