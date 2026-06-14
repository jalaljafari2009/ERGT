# A100 Runtime Optimization

## Purpose

This update reduces ERGT training wall-clock time on A100-class GPUs without
changing the control families, loss target, causal masking, or geometry claims.

## Optimization Scope

The expensive path was the real/random/shuffled stable-memory GeoAttention path,
especially during 2000-step runs with `context_length=256`.

The update optimizes four areas:

1. Geometry-state forwarding:
   memory-mode blocks now pass only `geometry_memory` during ordinary training.
   Full attention weights, distance tensors, QK logits, and geometry diagnostics
   are materialized only when diagnostics are requested.

2. Unit-step causal shortest path:
   `max_causal_step=1` now uses a direct prefix-sum shortest-path kernel instead
   of building a full direct edge-cost tensor and running a generic mask check.

3. Causal reconstruction prefix:
   prefix hidden reconstruction is vectorized with `cummax`/`gather`; it no
   longer loops over batch and position with CPU-GPU synchronizing `.item()`
   calls.

4. A100 runtime controls:
   trainers now support TF32, BF16/FP16 autocast, non-blocking device transfer,
   pinned DataLoader workers, and optional `torch_compile`.

## Runtime Config Fields

Training configs may include:

```json
{
  "training": {
    "precision": "auto",
    "allow_tf32": true,
    "float32_matmul_precision": "high",
    "dataloader_num_workers": 2,
    "pin_memory": true,
    "persistent_workers": true,
    "prefetch_factor": 2,
    "save_checkpoints": false
  },
  "logging": {
    "train_geometry_diagnostics_interval": 100
  }
}
```

`precision="auto"` uses BF16 on GPUs that support it, otherwise FP16 with a
GradScaler. `tf32` keeps full FP32 tensors while enabling TensorFloat-32 matmul
on CUDA.

`save_checkpoints=false` is recommended for Colab evidence notebooks when the
review bundle is the deliverable. It does not change training dynamics; it only
skips checkpoint writes and keeps the exported bundle lightweight.

## Notebook Impact

`ERGT_01_Attention_Evidence_Ladder.ipynb` and
`ERGT_02_Adaptive_Competitive_Alpha.ipynb` now write these runtime fields into
all generated baseline and ERGT configs. This keeps baseline, real, random,
shuffled, no-memory, and instantaneous controls under the same runtime policy.

Future notebooks must follow the broader Colab execution rules in
`docs/48_colab_notebook_execution_contract.md`, including live display,
lightweight bundles, auto-shutdown hooks, and equal runtime policy across control
families.

## Scientific Guard

These changes do not:

- change W-level control generation;
- reuse real distance or real memory across random/shuffled controls;
- relax causal masks or future leakage rules;
- change the language-model objective;
- remove 100-step validation/progress telemetry.

Diagnostics are still emitted at the configured evaluation interval. Ordinary
non-evaluation training steps avoid full diagnostic materialization to keep the
GPU doing model work instead of observer bookkeeping.
