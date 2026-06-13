# Run-02 Evidence Consolidation

## 1. Purpose

This stage consolidates what the project has learned before opening more
parameters. It prevents the next adaptive-control steps from becoming a blind
search over alpha, memory, normalization, and causal reachability.

The stage does not make a final ERGT claim. It prepares the evidence contract
for the next stage:

```text
Open Control Philosophy Contract
```

## 2. Fixed-Alpha Reference

The fixed-alpha guarded run established a conservative reference:

```text
alpha target = 0.025
warmup ends near step 1000
real_memory_d stayed slightly better than baseline
geo/qk fell from about 0.022 near step 1000 to about 0.014 near step 2000
```

The interpretation is:

```text
geometry did not destabilize attention
memory geometry produced a small positive signal
the signal did not grow strongly after alpha stopped increasing
the geometry term became relatively weaker as QK became stronger
```

This is the reason Run-02 introduced adaptive competitive alpha rather than a
larger fixed alpha.

## 3. Run-02 Evidence Inputs

Run-02 is expected to provide these condition families:

```text
baseline
alpha_zero
real_memory_d_adaptive
random_memory_d_adaptive
shuffled_memory_d_adaptive
no_memory_real_d_adaptive
instantaneous_real_d_adaptive
```

The minimum required set for consolidation is:

```text
baseline
alpha_zero
real_memory_d_adaptive
random_memory_d_adaptive
shuffled_memory_d_adaptive
```

The no-memory and instantaneous controls are not optional for a final claim, but
the consolidation script can still classify a partial run before they exist.

## 4. Required Signals

Every adaptive condition must expose:

```text
validation_loss
baseline_centered_improvement
alpha_effective
alpha_next
alpha_delta
alpha_decision
adaptive_score
adaptive_slope_gain
adaptive_advantage
geo_to_qk_ratio
attention_entropy
mean_max_probability
geo_qk_risk
entropy_risk
max_probability_risk
```

These signals are necessary because the next stages must decide whether a
parameter should grow, shrink, hold, or become part of a broader controller.

## 5. Consolidation Questions

The report answers these questions:

```text
Did alpha_zero still match baseline?
Did real adaptive geometry produce late-window evidence?
Did random and shuffled controls exist under the same protocol?
Did adaptive alpha telemetry exist?
Did rigidity telemetry exist?
Did real separate from random/shuffled, or was the gain generic?
Did alpha grow because of slope evidence or only exploration?
Did geo/qk, entropy, or max probability show collapse pressure?
```

## 6. Machine Report

This stage adds:

```text
evaluation/run02_evidence_consolidation.py
experiments/create_run02_evidence_consolidation_report.py
```

Usage:

```bash
python experiments/create_run02_evidence_consolidation_report.py \
  --run-root runs/notebook_02_adaptive_competitive_alpha/<run_id>
```

Default output:

```text
run02_evidence_consolidation_report.json
```

The Run-02 notebook also writes this report into its output bundle.

## 7. Status Meanings

```text
incomplete_needs_run02_bundle
  No usable Run-02 outputs were found.

incomplete_required_conditions_missing
  One or more required condition families are absent.

needs_investigation_alpha_zero_control
  The neutral alpha-zero control does not match baseline closely enough.

needs_investigation_missing_adaptive_telemetry
  The run does not expose alpha decision telemetry.

needs_investigation_missing_rigidity_telemetry
  The run cannot diagnose attention rigidity or collapse.

incomplete_needs_late_window
  Real adaptive geometry has not reached the analysis window.

consolidated_ready_for_open_control_contract
  Run-02 is sufficiently structured to move to the next planning stage.
```

## 8. Exit Criteria

This stage is complete when:

```text
the report is generated
alpha_zero is validated
real adaptive telemetry is present
random and shuffled controls are present
late-window fields exist
rigidity telemetry exists
```

If any of these fail, the next action is not to add more adaptive parameters.
The next action is to fix the Run-02 evidence pipeline.
