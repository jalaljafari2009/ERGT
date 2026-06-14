# Longer Run or Multi-Seed Confirmation

## Purpose

Stage 26 defines the confirmation contract after the guarded adaptive gate has
passed.

It does not turn replay data into a scientific claim. It defines what the next
real confirmation run must do before stronger claims are allowed.

## Entry Conditions

Confirmation is allowed only if:

```text
short smoke passed
guarded 2000-step adaptive run passed mechanics
late-window post-1000 analysis passed
decision gate passed
controller revision loop reports stage26 readiness
no unresolved R1/R2/R3 blocker remains
```

## Confirmation Profiles

The stage defines two confirmation profiles.

```text
longer_single_seed_5000
```

Purpose:

```text
check whether the real geometry advantage persists beyond 2000 steps
```

```text
multi_seed_2000
```

Purpose:

```text
check whether the real geometry advantage survives seed variation
```

The multi-seed profile uses:

```text
seeds = 2027, 2028, 2029
```

## Required Conditions

Every profile and seed must run:

```text
baseline
alpha_zero
real_memory_d
random_memory_d
shuffled_memory_d
no_memory_real_d
instantaneous_real_d
```

The run order remains sequential and no-peek:

```text
baseline -> alpha_zero -> real -> random -> shuffled -> no_memory -> instantaneous
```

Live real rows cannot use future control rows that have not run yet.

## Decision Rule

Every seed must show real stable causal geometry beating:

```text
baseline
alpha_zero
random_memory_d
shuffled_memory_d
no_memory_real_d
instantaneous_real_d
```

Random or shuffled dominance in any seed blocks confirmation.

The aggregate report also tracks:

```text
median relation-specific advantage
minimum relation-specific advantage
passing seed count
random/shuffled dominance cases
```

## Artifact Policy

The confirmation bundle remains lightweight:

```text
configs
metrics
progress logs
observer reports
summary reports
diagnostic tables
plot payloads
```

Checkpoints remain excluded from the review bundle.

## Machine Artifacts

This stage adds:

```text
experiments/longer_run_multi_seed_confirmation.py
evaluation/longer_run_multi_seed_confirmation.py
experiments/create_longer_run_multi_seed_confirmation_report.py
tests/test_longer_run_multi_seed_confirmation.py
```

Default report:

```text
runs/contracts/longer_run_multi_seed_confirmation.json
```

Usage:

```bash
python experiments/create_longer_run_multi_seed_confirmation_report.py
```

## Current Contract Result

The current contract replay passes.

It confirms:

```text
stage 20 passed
stage 21 passed
stage 22 passed
stage 24 passed
stage 25 reports stage26 readiness
longer-run profile is declared
multi-seed profile is declared
all required conditions are declared
matched confirmation windows exist
real beats all controls in every replay seed
random/shuffled dominance is absent
lightweight artifact policy is active
```

Important boundary:

```text
This is a confirmation-readiness contract. Real longer-run or multi-seed
telemetry is still required before stronger scientific claims.
```

## Exit Criteria

This stage is complete when:

```text
all prerequisite gates pass
confirmation manifest is complete
longer profile is declared
multi-seed profile has the required seed count
all required conditions are present for every seed/profile
real beats all controls in every replay seed/profile
random/shuffled dominance blocks the report
artifact policy excludes checkpoints
machine report passes
tests pass
```

Next action:

```text
Run real longer or multi-seed confirmation with actual telemetry.
```
