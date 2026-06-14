# Guarded Adaptive Training Notebook ERGT-04

## Purpose

ERGT-04 is the first Colab-facing notebook that runs real guarded adaptive
training after the ERGT-03 synthetic wrapper smoke.

```text
notebooks/ERGT_04_Guarded_Adaptive_Training.ipynb
```

ERGT-03 remains a smoke/contract notebook. ERGT-04 is the notebook to use when
the project needs actual baseline, real, random, shuffled, no-memory, and
instantaneous training telemetry.

## Execution Shape

The guarded profile runs:

```text
baseline: full 2000-step reference
alpha_zero_short_check: short 200-step wrapper-neutrality check
real_memory_d_adaptive: full adaptive-alpha run
random_memory_d_adaptive: full adaptive-alpha run
shuffled_memory_d_adaptive: full adaptive-alpha run
no_memory_real_d_adaptive: full adaptive-alpha run
instantaneous_real_d_adaptive: full adaptive-alpha run
```

The alpha-zero condition is intentionally short. It checks whether the ERGT
wrapper with zero geometry is neutral. It is not treated as a full scientific
geometry condition.

## Profiles

The notebook exposes two profiles:

```text
real_smoke_200
guarded_2000_real_training
```

The default is:

```text
guarded_2000_real_training
```

This is a real long run and should be launched on a suitable GPU, preferably
A100/H100 class.

## Live Execution Contract

Training commands are launched with:

```text
python -u
```

The notebook streams subprocess stdout line by line, so every 100-step progress
line from the training scripts is visible during execution. The progress line can
include loss, alpha, alpha decision, geometry/QK ratio, attention entropy, memory
fields, GPU memory, tokens/sec, and elapsed time when those fields are emitted by
the trainer.

## Artifact Contract

The fixed output bundle is:

```text
ergt_04_guarded_adaptive_training_report_bundle.zip
```

The default local review path after Colab download is:

```text
C:\Users\Administrator\Downloads\ergt_04_guarded_adaptive_training_report_bundle.zip
```

The bundle is lightweight. It includes configs, metrics, progress logs, adaptive
alpha logs, review tables, plot payloads, summaries, and runtime metadata.

The notebook and trainers use:

```json
{
  "save_checkpoints": false
}
```

Checkpoints and model weights are excluded and cleaned from the notebook run
root before bundle export.

## Runtime Policy

The notebook follows `docs/48_colab_notebook_execution_contract.md` and the A100
runtime policy in `docs/47_a100_runtime_optimization.md`.

Runtime policy is applied equally across control families:

```text
precision = auto
allow_tf32 = true
float32_matmul_precision = high
pin_memory = true
persistent_workers = true
prefetch_factor = 2
```

## Failure Safety

On preflight failure, training failure, or final completion, ERGT-04 attempts to
export the lightweight report bundle. When `AUTO_SHUTDOWN_COLAB_RUNTIME` is true,
it then requests Colab runtime shutdown after the configured delay.

## Output Interpretation

ERGT-04 itself is not the final scientific claim gate. It creates the real
training bundle required for:

```text
late-window real-vs-control separation
random/shuffled/no-memory attribution
controller revision
longer-run or multi-seed confirmation
```

The next step after a successful run is to inspect the downloaded bundle and run
late-window analysis against the actual progress logs.
