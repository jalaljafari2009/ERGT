# Normalization and Distance-Scale Controller

## 1. Purpose

This stage protects the relational distance signal from being erased by
normalization, clipping, or a fixed scale:

```text
real contrast exists before normalization
but post-normalization contrast is weak
and controls are not winning
and attention is still healthy
-> increase distance scale
```

The controller also prevents blind amplification:

```text
no pre-normalization signal
or clipping saturation is high
or random/shuffled distance advantage dominates
or attention is uniform/collapsing
-> decrease or hold distance scale
```

This directly addresses the earlier concern that normalization or fixed warmup
can flatten the geometry signal after the model representation grows.

## 2. Required Outputs

Every decision must expose:

```text
current_distance_norm_scale
proposed_distance_norm_scale
distance_norm_scale_delta
distance_norm_scale_credit
distance_norm_scale_risk_pressure
contrast_evidence
scale_evidence
clipping_evidence
control_distance_evidence
attention_safety_evidence
release_evidence
restraint_evidence
parameter_trajectory
injected_evidence_ledger
controller_state_snapshot
decision_replay_record
```

## 3. Control Policy

The controller increases `distance_norm_scale` only when:

```text
pre_norm_distance_contrast is present
post_norm_distance_contrast is too low
distance_contrast_retention is too low
geo_to_qk_ratio is underpowered
real distance advantage exceeds random and shuffled
clipping saturation is low
attention is not uniform or collapsed
```

The controller decreases `distance_norm_scale` when:

```text
pre-normalization contrast is absent
clipping saturation is high
geo_to_qk_ratio is overpowered
post-normalization std is too high or too low
random/shuffled controls dominate real
attention is uniform or collapse-prone
```

Future leakage remains a validity hard stop.

## 4. Machine Artifacts

This stage adds:

```text
experiments/distance_scale_controller.py
evaluation/distance_scale_controller.py
experiments/create_distance_scale_controller_report.py
tests/test_distance_scale_controller.py
```

It also extends:

```text
evaluation/unified_telemetry_schema.py
experiments/progress_logging.py
```

Default report:

```text
runs/contracts/distance_scale_controller.json
```

Usage:

```bash
python experiments/create_distance_scale_controller_report.py
```

## 5. Live Telemetry

Long runs should expose:

```text
norm
n_next
d_norm
nCredit
nPress
preC
postC
ret
preStd
postStd
clipSat
erase
nRel
nRest
realD
randD
shufD
```

These fields show whether the controller is preserving real distance contrast or
restraining scale because the signal is clipped, control-like, or unsafe.

## 6. Exit Criteria

This stage is complete when:

```text
distance scale can increase when real contrast is erased after normalization
distance scale can decrease when clipping/control/collapse pressure appears
pre-normalization absence is not amplified blindly
ordinary distance-scale risk does not abort the search
future leak acts as a validity hard stop
every decision has a parameter trajectory
every decision has an injected evidence ledger
every decision has a replay record
live progress logging exposes distance-scale decisions
```

Next stage:

```text
Joint Parameter Budget Allocator
```
