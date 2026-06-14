# Parameter Attribution Probe

## 1. Purpose

This stage makes adaptive parameter changes accountable before more controllers
are added.

The scientific concern is:

```text
when performance changes, the program must say which parameter change has
evidence behind it, or explicitly mark attribution as uncertain
```

This stage is diagnostic-only. It does not change alpha, memory, gate floor,
normalization, reachability, attention, or loss.

## 2. Required Outputs

Every major adaptive decision should expose:

```text
alpha_contribution_estimate
memory_eta_decay_contribution_estimate
gate_floor_contribution_estimate
normalization_contribution_estimate
reachability_contribution_estimate
interaction_warnings
uncertainty_flags
```

## 3. Interpretation

Contribution estimates are mechanics-level credit assignments from telemetry.
They are not causal proof. If multiple parameters change in the same decision
window, the report must include an interaction warning.

If a parameter changed but no credit/evidence field is available, the report
must set an uncertainty flag instead of silently assigning credit. That
uncertainty is not a stop signal; it is carried forward into budget allocation,
controller replay, and later behavior analysis.

## 4. Machine Artifacts

This stage adds:

```text
evaluation/parameter_attribution_probe.py
experiments/create_parameter_attribution_probe_report.py
tests/test_parameter_attribution_probe.py
```

Default report:

```text
runs/contracts/parameter_attribution_probe.json
```

Usage:

```bash
python experiments/create_parameter_attribution_probe_report.py
```

## 5. Exit Criteria

This stage is complete when:

```text
all required attribution outputs are emitted
major adaptive decisions include attribution evidence or an uncertainty flag
uncertainty flags do not abort adaptive search
alpha contribution can be estimated when alpha credit exists
memory eta and decay contribution can be estimated when memory credit exists
gate-floor contribution can be estimated when gate-floor credit exists
normalization contribution can be estimated when distance-scale credit exists
reachability contribution can be estimated when reachability credit exists
multi-parameter changes emit interaction warnings
```

Next stage:

```text
Adaptive Alpha Controller v2
```
