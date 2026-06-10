# ERGT Implementation Backlog

## 1. Purpose

This backlog translates the ERGT documentation pack into executable tasks for
Codex.

The backlog is ordered by phase and should be followed conservatively. Do not
skip required controls, logging, or gate decisions.

## 2. Current Status

Documentation foundation is complete for the first implementation pass:

- `README.md`
- `.codex/project_context.md`
- `docs/00_project_charter.md`
- `docs/01_theoretical_foundation.md`
- `docs/02_operational_definitions.md`
- `docs/03_phase_plan.md`
- `docs/04_proof_stage_protocol.md`
- `docs/05_baseline_transformer_spec.md`
- `docs/06_relational_graph_spec.md`
- `docs/07_emergent_distance_spec.md`
- `docs/08_geo_attention_spec.md`
- `docs/09_metrics_and_ablation_plan.md`
- `docs/10_gate_conditions.md`
- `docs/11_repository_structure.md`
- `docs/12_phase4_dynamic_memory_plan.md`
- `docs/13_phase3_stable_base_plan.md`
- `docs/14_phase3_ratio_matched_plan.md`

The next work is implementation planning and source scaffolding.

## 3. Global Rules

Codex must preserve these rules during implementation:

- Read `.codex/project_context.md` before coding.
- Keep the broad ERGT theory separate from the first proof claim.
- Implement Phase 0 through Phase 3 before later mechanisms.
- Do not implement graph memory before Gate 1.
- Do not implement causal geometry before Gate 1.
- Do not implement spectral complexity before Gate 1.
- Keep all experiments reproducible from config files.
- Save machine-readable metrics for every run.
- Keep baseline and ERGT comparisons controlled.

## 4. Phase -1: Repository Scaffolding

### Task P-1.1: Create Source Tree

Status: complete

Create the initial directory structure:

```text
configs/
data/
models/
layers/
geometry/
attention/
experiments/
evaluation/
visualization/
runs/
tests/
scripts/
```

Acceptance criteria:

- Directories exist.
- No unrelated framework is introduced.
- Structure matches `docs/11_repository_structure.md`.

### Task P-1.2: Add Project Hygiene Files

Status: complete

Create project hygiene files as needed:

```text
.gitignore
pyproject.toml or requirements.txt
```

Acceptance criteria:

- Large run artifacts and checkpoints are ignored.
- Python dependencies are declared.
- Formatting and test commands are documented or discoverable.

### Task P-1.3: Add Config Skeletons

Status: complete

Create initial config directories and placeholder configs:

```text
configs/baseline/debug_wikitext2.json
configs/baseline/proof_wikitext2.json
configs/ergt_v1/real_d.json
configs/ergt_v1/alpha_zero.json
configs/ergt_v1/random_d.json
configs/ergt_v1/shuffled_d.json
configs/analysis/graph_observer.json
configs/analysis/distance_geometry.json
```

Acceptance criteria:

- Configs are valid JSON.
- Baseline and ERGT configs share controlled variables.
- ERGT configs differ only in intended attention or distance settings.

## 5. Phase 0: Baseline Transformer

### Task P0.1: Implement Baseline Model

Status: complete

Implement:

```text
models/transformer_baseline.py
```

Required components:

- Token embedding.
- Positional embedding.
- Pre-norm transformer blocks.
- Causal self-attention.
- Feed-forward network.
- Residual connections.
- Language-model head.

Acceptance criteria:

- Forward pass returns logits with shape `[batch, sequence, vocab_size]`.
- Loss can be computed for next-token prediction.
- Attention module is cleanly replaceable.
- Unit tests cover tensor shapes and causal mask behavior.

### Task P0.2: Implement Data Preparation

Status: complete

Implement the first dataset path for WikiText-2 or a local debug substitute.

Suggested files:

```text
scripts/prepare_wikitext2.py
experiments/data_utils.py
```

Acceptance criteria:

- Dataset split is deterministic.
- Tokenizer metadata is saved.
- Sequence packing is reproducible.
- All model variants can reuse the same data pipeline.

### Task P0.3: Implement Baseline Training

Status: complete

Implement:

```text
experiments/train_baseline.py
```

Acceptance criteria:

- Loads config.
- Sets seed.
- Trains baseline.
- Logs `train_log.jsonl`.
- Saves `baseline_results.json`.
- Saves `model_summary.json`.
- Saves checkpoints if configured.

### Task P0.4: Implement Baseline Evaluation

Status: complete

Implement:

```text
experiments/eval_baseline.py
```

Acceptance criteria:

- Computes validation loss.
- Computes perplexity.
- Loads saved config and checkpoint.
- Writes machine-readable evaluation output.

### Task P0.5: Phase 0 Tests

Status: complete

Implement:

```text
tests/test_transformer_baseline.py
tests/test_causal_masking.py
```

Acceptance criteria:

- Shape tests pass.
- Causal mask blocks future positions.
- One tiny training step runs without NaN or Inf.
- `alpha_zero` compatibility requirements are anticipated by attention design.

## 6. Phase 1: Relational Graph Observer

### Task P1.1: Implement Relational Graph

Status: complete

Implement:

```text
layers/relational_graph.py
```

Default formula:

```text
K_ij = sigmoid((x_i^T x_j) / sqrt(d))
W = K
```

Acceptance criteria:

- Input shape: `[batch, sequence, hidden_dim]`.
- Output shape: `[batch, 1, sequence, sequence]`.
- Optional diagonal policy is explicit.
- No NaN or Inf for normal hidden states.
- Observer mode does not alter model output.

### Task P1.2: Implement Graph Metrics

Status: complete

Implement:

```text
evaluation/graph_metrics.py
```

Required metrics:

- Mean.
- Variance.
- Graph entropy.
- Sparsity under thresholds.
- Diagonal dominance.
- Degree distribution.
- Random and shuffled comparisons.

Acceptance criteria:

- Metrics serialize to JSON.
- Metrics can exclude diagonal where configured.
- Random and shuffled controls are reproducible from seed.

### Task P1.3: Implement Graph Analysis Experiment

Status: complete

Implement:

```text
experiments/analyze_relational_graph.py
```

Acceptance criteria:

- Loads a baseline model or baseline checkpoint.
- Extracts `W` from configured layers.
- Writes `graph_stats.json`.
- Saves sample matrices under run artifacts.
- Does not change training or evaluation behavior.

### Task P1.4: Phase 1 Tests

Status: complete

Implement:

```text
tests/test_relational_graph.py
tests/test_metrics.py
```

Acceptance criteria:

- Graph shape is correct.
- Graph values are finite.
- Graph values are bounded when using sigmoid.
- Metrics produce valid JSON-compatible dictionaries.

## 7. Phase 2: Emergent Distance

### Task P2.1: Implement Emergent Distance

Status: complete

Implement:

```text
geometry/emergent_distance.py
```

Default formula:

```text
D_ij = -log(W_ij + epsilon)
```

Acceptance criteria:

- Input shape matches relational graph output.
- Output shape matches input graph shape.
- Epsilon handling prevents `log(0)`.
- Diagonal policy is explicit.
- Normalization options are implemented.
- Output is finite under expected inputs.

### Task P2.2: Implement Distance Metrics

Status: complete

Implement:

```text
evaluation/distance_metrics.py
```

Required metrics:

- Mean.
- Standard deviation.
- Min and max.
- Entropy.
- Diagonal vs off-diagonal statistics.
- Correlation with attention logits, when provided.
- Random and shuffled controls.

Acceptance criteria:

- Metrics serialize to JSON.
- Controls use the same normalization policy.
- Attention-duplication diagnostics are available.

### Task P2.3: Implement Geometry Analysis Experiment

Status: complete

Implement:

```text
experiments/analyze_emergent_geometry.py
```

Acceptance criteria:

- Loads or computes `W`.
- Produces `D`.
- Writes `distance_stats.json`.
- Writes `geometry_report.json`.
- Saves sample matrices under run artifacts.

### Task P2.4: Phase 2 Tests

Status: complete

Implement:

```text
tests/test_emergent_distance.py
```

Acceptance criteria:

- Distance is finite.
- Distance decreases as relation strength increases.
- Normalization is stable.
- Diagonal policy behaves as configured.

## 8. Phase 3: GeoAttention v1

### Task P3.1: Implement GeoAttention

Status: complete

Implement:

```text
attention/geo_attention.py
```

Core formula:

```text
logits = QK^T / sqrt(head_dim) - alphaD
```

Acceptance criteria:

- Supports `real_d`, `random_d`, `shuffled_d`, and `zero_d`.
- Supports fixed `alpha`.
- Supports optional trainable `alpha`.
- Supports detached and gradient-through-distance modes.
- Applies causal mask after geometry bias.
- Matches baseline behavior when `alpha = 0`, within expected numerical noise.

### Task P3.2: Implement ERGT-v1 Model

Status: complete

Implement:

```text
models/ergt_v1.py
```

Composition:

```text
RelationalGraph + EmergentDistance + GeoAttention
```

Acceptance criteria:

- Model config matches baseline config except ERGT-specific settings.
- Forward pass returns logits with standard language-model shape.
- Geometry diagnostics can be returned or logged.

### Task P3.3: Implement Attention Metrics

Status: complete

Implement:

```text
evaluation/attention_metrics.py
```

Required metrics:

- Attention entropy.
- Mean max attention probability.
- `geo_to_qk_ratio`.
- Alpha value.
- Distance mode.
- Distance normalization.

Acceptance criteria:

- Metrics serialize to JSON.
- Metrics can be aggregated across layers and heads.

### Task P3.4: Implement ERGT Training

Status: complete

Implement:

```text
experiments/train_ergt_v1.py
```

Acceptance criteria:

- Loads ERGT config.
- Trains `real_d`, `alpha_zero`, `random_d`, or `shuffled_d` condition.
- Logs geometry diagnostics.
- Saves train log, model summary, metrics, and checkpoint.

### Task P3.5: Implement Phase 3 Comparison

Status: complete

Implement:

```text
experiments/compare_phase3.py
```

Acceptance criteria:

- Reads baseline and ERGT run outputs.
- Produces `comparison_results.json`.
- Produces `ablation_report.json`.
- Includes pass, conditional pass, or fail recommendation inputs.

### Task P3.6: Phase 3 Tests

Status: complete

Implement:

```text
tests/test_geo_attention.py
tests/test_causal_masking.py
```

Acceptance criteria:

- `alpha_zero` path is equivalent to baseline attention.
- Causal mask prevents future-token attention.
- Random and shuffled distance modes run.
- GeoAttention output shape matches baseline attention output.
- No NaN or Inf under small synthetic inputs.

### Task P3.7: Implement Stable Base Candidate

Status: complete

Implement the post-confirmation-seed Stable Base candidate:

```text
docs/13_phase3_stable_base_plan.md
experiments/progress_logging.py
experiments/compare_phase3_stable_base.py
notebooks/ergt_colab_phase3_stable_base.ipynb
configs/baseline/phase3_stable_base_seed2027.json
configs/ergt_v1/phase3_stable_base/
```

Acceptance criteria:

- `GeoAttention` supports alpha warmup without changing old configs.
- Baseline and ERGT trainers write lightweight `progress_log.jsonl`.
- Trainers print one live progress line at each eval interval.
- Stable Base configs use `detached_d`, `sigmoid_cosine`, clipped distance,
  and alpha warmup for non-zero alpha runs.
- Stable Base comparison validates configs before interpreting results.
- Stable Base does not move the project to Phase 4 without seed confirmation.

### Task P3.8: Implement Ratio-Matched Geometry Control

Status: complete

Implement the stronger Phase 3 control that compares geometry at matched
`geo_to_qk_ratio`, not merely matched `alpha`:

```text
docs/14_phase3_ratio_matched_plan.md
experiments/build_ratio_matched_configs.py
experiments/compare_phase3_ratio_matched.py
tests/test_phase3_ratio_matched.py
```

Acceptance criteria:

- Generated configs preserve Stable Base settings.
- Generated alpha values are derived from completed calibration runs.
- Generated configs record `run.ratio_match` metadata.
- Ratio-matched comparison validates actual observed `geo_to_qk_ratio`.
- `real_d` is compared with `random_d` and `shuffled_d` only at matched
  geometry strength.
- If ratio tolerance is missed, the result is marked for recalibration rather
  than interpreted as evidence.

## 9. Gate 1: Phase 3 to Phase 4 Decision

### Task G1.1: Implement Gate Decision Utility

Status: complete

Implement:

```text
evaluation/gate_decision.py
```

Acceptance criteria:

- Reads `comparison_results.json` and `ablation_report.json`.
- Applies rules from `docs/10_gate_conditions.md`.
- Writes `runs/gates/phase3_to_phase4_decision.json`.
- Outputs one of: `pass`, `conditional_pass`, `fail`.

### Task G1.2: Produce Gate Decision

Run the gate utility after Phase 3 experiments.

Acceptance criteria:

- Decision file exists.
- Evidence and risks are listed.
- Next action is explicit.
- No later phase starts without a gate decision.

## 10. Deferred Phase 4: Dynamic Graph Memory

Do not implement memory code until Gate 1 passes or conditionally passes.

Gate prerequisite:

```text
GeoAttention(real D) must show credible benefit over baseline and controls.
```

### Task P4.1: Add Phase 4 Design Contract

Status: complete

Implement:

```text
docs/12_phase4_dynamic_memory_plan.md
```

Acceptance criteria:

- Defines the dynamic graph memory formula.
- Defines the Gate 1 prerequisite.
- Defines causal-safety constraints.
- Defines controls, metrics, pass criteria, and failure criteria.
- Does not implement memory before Gate 1.

### Task P4.2: Implement Dynamic Graph Memory Core

Status: deferred until Gate 1

Implement:

```text
memory/dynamic_graph.py
tests/test_dynamic_graph_memory.py
```

Acceptance criteria:

- Supports fixed `eta`.
- Supports explicit initialization policy.
- Preserves graph shape.
- Produces finite values.
- Records memory diagnostics.
- Does not introduce causal leakage.

### Task P4.3: Implement ERGT Memory Model

Status: deferred until Gate 1

Implement:

```text
models/ergt_memory.py
configs/ergt_memory/debug_memory.json
configs/ergt_memory/pilot_real_memory.json
configs/ergt_memory/pilot_random_memory.json
configs/ergt_memory/pilot_shuffled_memory.json
```

Acceptance criteria:

- Reuses ERGT-v1 settings unless memory-specific.
- Can disable memory for an ERGT-v1-equivalent control.
- Supports real, random, and shuffled memory conditions.
- Logs memory diagnostics.

### Task P4.4: Implement Phase 4 Training

Status: deferred until Gate 1

Implement:

```text
experiments/train_ergt_memory.py
```

Acceptance criteria:

- Loads memory configs.
- Saves config, metrics, train log, model summary, and checkpoints.
- Records runtime and peak memory.
- Fails fast on NaN or Inf.

### Task P4.5: Implement Phase 4 Comparison

Status: deferred until Gate 1

Implement:

```text
experiments/compare_phase4.py
```

Acceptance criteria:

- Compares ERGT-memory against ERGT-v1 real_d alpha=0.2.
- Includes random, shuffled, and disabled-memory controls.
- Writes machine-readable comparison and ablation reports.

### Task P4.6: Add Phase 4 Colab Pilot

Status: deferred until Gate 1

Implement:

```text
notebooks/ergt_colab_phase4_memory_pilot.ipynb
```

Acceptance criteria:

- Runs a small GPU pilot.
- Produces a light zip artifact with JSON and JSONL outputs.
- Avoids large checkpoint archives by default.

## 11. Deferred Phase 5 and Beyond

Do not implement these until explicitly authorized by gate decisions:

- Complete ERGT architecture.
- Causal geometry.
- Spectral complexity.
- Relational field model.

These mechanisms belong to the broader theory but are not part of the first
proof implementation.

## 12. First Implementation Order

When coding begins, use this order:

1. Repository scaffolding.
2. Config skeletons.
3. Baseline model.
4. Baseline tests.
5. Baseline training and evaluation scripts.
6. Relational graph module.
7. Graph metrics and analysis.
8. Emergent distance module.
9. Distance metrics and analysis.
10. GeoAttention.
11. ERGT-v1 model.
12. Phase 3 ablation configs.
13. Phase 3 training and comparison.
14. Gate decision utility.

## 13. Minimum Smoke Test Before Real Training

Before any real dataset training, run a synthetic smoke test:

- Tiny vocabulary.
- Tiny sequence length.
- Tiny batch.
- One or two transformer layers.
- One forward pass.
- One backward pass.
- One optimizer step.

Acceptance criteria:

- No NaN or Inf.
- Loss is finite.
- Shapes are correct.
- Metrics serialize.
- Causal mask works.

## 14. Definition of Done for First Proof Stage

The first proof stage is complete when:

- Phase 0 baseline run exists.
- Phase 1 graph report exists.
- Phase 2 distance report exists.
- Phase 3 comparison and ablation report exists.
- Gate 1 decision exists.
- All required configs are saved.
- All required metrics are machine-readable.
- The final interpretation does not overclaim beyond measured evidence.

The strongest possible first-stage conclusion is:

```text
Induced relational distance is a useful attention bias.
```

Any claim beyond that requires later phases and new measurements.
