# Loss-Slope and Trend Analyzer

## 1. Purpose

This stage makes adaptive decisions depend on validation-loss trends instead of
one noisy validation point.

The scientific concern is:

```text
controllers must cite windowed slope and EMA evidence before changing alpha,
memory, gate floor, reachability, or distance scale
```

This stage is diagnostic-only. It does not change any controller parameter.

## 2. Required Metrics

Every adaptive control run should expose:

```text
baseline_validation_loss
baseline_relative_validation_delta
ema_validation_loss
ema_baseline_validation_loss
ema_loss_delta
rolling_slope
baseline_rolling_slope
loss_slope_gain
late_window_slope
post_1000_trend
trend_window_points
```

## 3. Machine Artifacts

This stage adds:

```text
evaluation/loss_slope_trend_analyzer.py
experiments/create_loss_slope_trend_analyzer_report.py
tests/test_loss_slope_trend_analyzer.py
```

Default report:

```text
runs/contracts/loss_slope_trend_analyzer.json
```

Usage:

```bash
python experiments/create_loss_slope_trend_analyzer_report.py
```

## 4. Exit Criteria

This stage is complete when:

```text
baseline-relative validation delta is emitted
EMA loss delta is emitted
rolling slope uses at least the configured minimum window
late-window slope is emitted
post-1000 trend is emitted
controller evidence includes latest slope gain and EMA delta
```

Next stage:

```text
Parameter Attribution Probe
```
