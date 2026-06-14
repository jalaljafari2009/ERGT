# Colab Notebook Execution Contract

## Purpose

Future ERGT notebooks must preserve the scientific controls while making long
Colab runs observable, recoverable, and cheap to review. This document collects
the notebook rules learned from ERGT-01, ERGT-02, ERGT-03, and the A100 runtime
optimization pass.

This is a notebook construction contract. It does not replace the telemetry
schema, the adaptive execution plan, or the A100 runtime contract. It tells every
new notebook how to expose those contracts to the operator.

## Required Language And Reader Contract

All notebook markdown, headings, printed status labels, table headers, and report
field names should be in English.

Rationale:

```text
notebooks are execution artifacts;
reports must be readable by external reviewers;
field names must remain stable across runs and scripts.
```

If Persian discussion notes are needed, keep them outside the machine-facing
cells or add them to a separate design note. Do not mix Persian labels into
exported telemetry keys, zip filenames, JSON fields, or Colab status messages.

## Repository Bootstrap

Every GitHub-to-Colab notebook must bootstrap the full repository, not only copy
one notebook file.

Required setup pattern:

```text
REPO_URL = "https://github.com/jalaljafari2009/ERGT.git"
REPO_DIR = "/content/ERGT"
clone or refresh REPO_DIR
cd /content/ERGT
add /content/ERGT to sys.path
print repository root, active commit, and notebook profile
```

The notebook must work when opened from:

```text
https://colab.research.google.com/github/jalaljafari2009/ERGT/blob/main/notebooks/<notebook>.ipynb
```

It must not assume that the user's browser has already uploaded local project
files.

## Fixed Output Bundle Contract

Every notebook must declare one fixed zip name before execution starts.

Example:

```text
ergt_03_adaptive_control_report_bundle.zip
```

Every notebook must also print the default local review path used by this
project:

```text
C:\Users\Administrator\Downloads\<fixed_bundle_name>.zip
```

The bundle should include only lightweight review artifacts:

```text
configs
metrics
progress logs
train logs
observer reports
controller logs
live diagnostic rows
live markdown tables
plot-ready payloads
summary JSON files
decision reports
environment/runtime metadata
```

The bundle must exclude heavy artifacts:

```text
checkpoints
model weights
optimizer states
large datasets
cache directories
temporary tensors
```

If a checkpoint is needed for a later experiment, store it through a separate
explicit checkpoint contract. Do not silently include it in the review bundle.

## Live Display Contract

Long runs must not leave the operator blind.

Every notebook that runs training or sequential controls must stream a stable
diagnostic view during execution at the declared cadence:

```text
default_display_interval_steps = 100
```

The display must happen during the run, not only after a condition finishes.
Buffered subprocess output should be avoided. Python commands should use
unbuffered output where possible:

```text
python -u ...
```

The live table should include, when available:

```text
step
condition
train_loss
validation_loss
delta_vs_baseline
late_window_delta
loss_slope
alpha
alpha_target
alpha_warmup_or_controller_state
geo_to_qk_ratio
mean_abs_geo
mean_abs_qk
memory_decay
memory_eta
gate_floor
memory_persistence
memory_update_norm
memory_stability
attention_entropy
attention_sparsity
attention_max_probability
rigidity_score
collapse_risk
noise_risk
future_leak_score
causal_reachability_status
normalization_status
control_separation_status
meta_control_attention_weights
controller_decisions
tokens_per_second
gpu_memory_gb
elapsed_minutes
```

The displayed row and the row written to disk must come from the same telemetry
snapshot. The notebook should also export plot-ready payloads so the final report
does not need to reconstruct curves from printed text.

## Sequential Control Awareness

Colab notebooks often execute controls sequentially:

```text
baseline -> alpha_zero -> real -> random -> shuffled -> no_memory -> instantaneous
```

Live scoring must therefore be missing-aware. During the `real` run, random and
shuffled telemetry may not exist yet. Controllers, meta-control attention, and
control separation scoring must use:

```text
available controls only for live decisions;
matched late-window controls only after all required controls have finished.
```

Do not let a live scorer pretend that future random/shuffled rows exist.

## Fail-Fast And Safety Contract

Every long notebook must run short preflight checks before training:

```text
notebook JSON parses
repository import works
required tests or contract checks pass
configuration writes successfully
output directory is writable
bundle exporter can run on an empty/smoke run
future leakage sentinel fails fast when forced
```

The run must stop early when hard safety gates fail:

```text
future_leak_score > 0
missing required telemetry keys
NaN or inf loss
NaN or inf geometry ratio
unbounded alpha/controller value
control family contract violation
bundle export failure
```

On failure, the notebook must still export the lightweight bundle with the
failure reason and all telemetry collected so far.

## Colab Runtime Shutdown

Every Colab-facing long-run notebook must expose a visible shutdown option:

```text
AUTO_SHUTDOWN_RUNTIME = True or False
```

Recommended behavior:

```text
default False for short smoke/debug profiles;
operator may set True before long unattended runs;
on completion or handled failure, export the bundle first;
then call the Colab runtime disconnect hook when running in Colab.
```

The shutdown cell must be robust to non-Colab environments and should print
whether shutdown was requested, skipped, or unavailable.

## Runtime Optimization Contract

New notebooks must use the shared A100 runtime fields unless a comparison
explicitly requires disabling them:

```json
{
  "training": {
    "precision": "auto",
    "allow_tf32": true,
    "float32_matmul_precision": "high",
    "dataloader_num_workers": 2,
    "pin_memory": true,
    "persistent_workers": true,
    "prefetch_factor": 2
  },
  "logging": {
    "train_geometry_diagnostics_interval": 100
  }
}
```

For strict numerical comparability against older runs, use:

```text
precision = "tf32"
```

For speed on A100/H100/L4, use:

```text
precision = "auto"
```

The runtime policy must be applied equally to baseline, real, random, shuffled,
alpha-zero, no-memory, and instantaneous controls.

## Geometry And Memory Preservation Rule

Runtime optimization must not remove the scientific mechanism.

Allowed:

```text
skip full diagnostics on non-display steps;
avoid materializing attention weights unless needed;
use vectorized equivalent computations;
use shared runtime precision policy across controls;
stream compact telemetry rows.
```

Not allowed:

```text
skip W_t memory updates;
reuse real memory or real distance for random/shuffled controls;
relax causal reachability;
remove future-leak checks;
change the loss target silently;
change the control family generation level silently;
normalize away real/random/shuffled contrast;
hide alpha, memory, or geometry-scale telemetry until the end.
```

If an optimization changes numerical behavior beyond precision policy, it must
be documented as a separate scientific change, not a notebook-speed change.

## Profile Contract

Every notebook must declare profiles explicitly:

```text
smoke
debug
guarded_2000
evidence
multi_seed
```

Each profile should declare:

```text
steps
context_length
conditions
seeds
eval_interval
live_display_interval
precision policy
shutdown default
expected wall-clock class
bundle name
decision rule
```

The default profile should be short enough to reveal wiring errors before a long
run. Long profiles must require an explicit operator change.

## Final Report Contract

At completion, the notebook must print:

```text
run_id
active git commit
profile
device
precision policy
conditions completed
conditions failed or skipped
late-window summary
decision label
bundle path in Colab
default local Downloads path
Colab notebook URL pattern
```

The final report should be clear enough that the next analysis step can start
from the downloaded zip without asking for filenames or paths again.

## Upstream References

This contract depends on:

```text
docs/22_unified_telemetry_schema.md
docs/24_active_adaptive_execution_plan.md
docs/37_open_adaptive_relational_control_trainer.md
docs/38_live_100_step_diagnostic_table.md
docs/39_adaptive_notebook_ergt_03.md
docs/47_a100_runtime_optimization.md
```

Future notebook-specific documents should reference this contract directly.

The first real guarded training notebook using this contract is documented in:

```text
docs/49_guarded_adaptive_training_notebook.md
```
