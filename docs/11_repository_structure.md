# Repository Structure

## 1. Purpose

This document defines the recommended ERGT source tree.

The repository should keep theory, implementation, configs, experiments, and run
artifacts separate. This makes the first proof stage reproducible and prevents
later phases from mixing with the Phase 0-3 evidence.

## 2. Top-Level Layout

Recommended repository layout:

```text
.
├── README.md
├── docs/
├── .codex/
├── configs/
├── data/
├── models/
├── layers/
├── geometry/
├── attention/
├── losses/
├── experiments/
├── evaluation/
├── visualization/
├── runs/
├── tests/
└── scripts/
```

Only create directories as they become needed. The documentation may define the
full target structure before all folders exist.

## 3. Documentation

```text
docs/
├── 00_project_charter.md
├── 01_theoretical_foundation.md
├── 02_operational_definitions.md
├── 03_phase_plan.md
├── 04_proof_stage_protocol.md
├── 05_baseline_transformer_spec.md
├── 06_relational_graph_spec.md
├── 07_emergent_distance_spec.md
├── 08_geo_attention_spec.md
├── 09_metrics_and_ablation_plan.md
├── 10_gate_conditions.md
└── 11_repository_structure.md
```

The `docs/` directory is the active project contract.

Long-form article manuscript artifacts are kept in:

```text
docs/research_article/
```

This folder stores the paper-facing `.tex` source and compiled `.pdf`. It is a
research manuscript archive, while the executable implementation contract
remains in the numbered design documents.

Phase 4 planning is captured in:

```text
docs/12_phase4_dynamic_memory_plan.md
```

Phase 3 Stable Base planning is captured in:

```text
docs/13_phase3_stable_base_plan.md
```

Phase 3 ratio-matched geometry control is captured in:

```text
docs/14_phase3_ratio_matched_plan.md
```

Post-evidence adaptive control and Run-02 consolidation are captured in:

```text
docs/19_adaptive_competitive_alpha_plan.md
docs/20_run02_evidence_consolidation.md
docs/21_open_control_philosophy_contract.md
docs/22_unified_telemetry_schema.md
docs/23_memory_state_instrumentation.md
docs/24_active_adaptive_execution_plan.md
docs/25_attention_rigidity_and_collapse_monitor.md
docs/26_control_family_fairness_audit_v2.md
docs/27_loss_slope_and_trend_analyzer.md
docs/28_parameter_attribution_probe.md
docs/29_adaptive_alpha_controller_v2.md
docs/30_adaptive_memory_controller.md
docs/31_gate_floor_and_noise_controller.md
docs/32_causal_reachability_controller.md
docs/33_normalization_and_distance_scale_controller.md
docs/34_joint_parameter_budget_allocator.md
docs/35_control_separation_scoring.md
docs/36_meta_control_attention_observer.md
docs/37_open_adaptive_relational_control_trainer.md
docs/38_live_100_step_diagnostic_table.md
docs/39_adaptive_notebook_ergt_03.md
docs/40_short_smoke_failure_safety_validation.md
docs/41_guarded_2000_step_adaptive_run.md
docs/42_late_window_post1000_analysis.md
docs/43_random_shuffled_no_memory_attribution.md
docs/44_decision_gate_real_geometry.md
docs/45_controller_revision_loop.md
docs/46_longer_run_multi_seed_confirmation.md
```

The original source notes remain at the repository root:

```text
4.txt
5.txt
6.txt
```

These are source material, not implementation contracts.

## 4. Codex Context

```text
.codex/
├── project_context.md
└── implementation_backlog.md
```

`project_context.md` is the Codex entry point and reading order.

`implementation_backlog.md` should contain phase-ordered implementation tasks
and acceptance criteria.

Codex should read `.codex/project_context.md` before implementation work.

## 5. Configs

```text
configs/
├── baseline/
│   ├── debug_wikitext2.json
│   └── proof_wikitext2.json
├── ergt_v1/
│   ├── real_d.json
│   ├── alpha_zero.json
│   ├── random_d.json
│   └── shuffled_d.json
└── analysis/
    ├── graph_observer.json
    └── distance_geometry.json
```

Configs should be explicit and machine-readable.

Every run must save a copy of the exact config into its run directory.

## 6. Data

```text
data/
├── raw/
├── processed/
└── tokenizers/
```

The repository should avoid committing large datasets unless intentionally
configured. Dataset preparation should be reproducible from scripts and configs.

Recommended metadata:

```text
data/processed/<dataset_name>/metadata.json
```

Metadata should record:

- Dataset name.
- Split.
- Tokenizer.
- Sequence length.
- Preprocessing version.
- Number of tokens.

## 7. Models

```text
models/
├── transformer_baseline.py
└── ergt_v1.py
```

`transformer_baseline.py` should implement the controlled GPT-style baseline.

`ergt_v1.py` should compose:

```text
RelationalGraph + EmergentDistance + GeoAttention
```

Model files should not contain experiment-specific training loops.

## 8. Layers

```text
layers/
└── relational_graph.py
```

`relational_graph.py` owns:

- Relation kernel.
- Graph construction.
- Optional graph normalization.
- Observer-mode outputs.
- Runtime graph options needed by GeoAttention.

It should not own language-model loss or training logic.

## 9. Geometry

```text
geometry/
└── emergent_distance.py
```

`emergent_distance.py` owns:

- Distance transform.
- Distance normalization.
- Diagonal policy.
- Causal distance policy.
- Distance diagnostics helpers, if lightweight.

Complex analysis can live under `evaluation/` or `visualization/`.

## 10. Attention

```text
attention/
└── geo_attention.py
```

`geo_attention.py` owns:

- Standard attention-compatible tensor flow.
- Geometry bias application.
- `alpha` handling.
- Distance modes for ablations.
- Causal and padding mask integration.
- Attention diagnostics.

It should preserve baseline behavior when `alpha = 0`.

## 11. Losses

```text
losses/
└── spectral_complexity.py
```

This directory is for later phases.

Do not implement spectral complexity in ERGT-v1 unless Phase 3 passes and the
project explicitly moves to later phases.

## 12. Experiments

```text
experiments/
├── train_baseline.py
├── eval_baseline.py
├── analyze_relational_graph.py
├── analyze_emergent_geometry.py
├── train_ergt_v1.py
└── compare_phase3.py
```

Experiment scripts should:

- Load config.
- Set seeds.
- Build datasets.
- Build model.
- Run training or evaluation.
- Save logs and metrics.
- Avoid hidden defaults.

`experiments/progress_logging.py` owns lightweight progress JSONL writing,
GPU-memory scalar capture, and one-line eval progress formatting for long
Colab runs.

`experiments/build_ratio_matched_configs.py` and
`experiments/compare_phase3_ratio_matched.py` own the stronger Phase 3 control
where real, random, and shuffled distances are compared at matched
`geo_to_qk_ratio`.

They should not contain core model definitions.

## 13. Evaluation

```text
evaluation/
├── metrics.py
├── graph_metrics.py
├── distance_metrics.py
├── attention_metrics.py
└── gate_decision.py
```

Evaluation code should produce machine-readable outputs for the run directories.

`gate_decision.py` should implement the logic described in
`docs/10_gate_conditions.md` once experiments exist.

## 14. Visualization

```text
visualization/
├── plot_graph_heatmaps.py
├── plot_distance_heatmaps.py
├── plot_training_curves.py
└── plot_attention_diagnostics.py
```

Visualizations should be generated from saved metrics and sample matrices.

Visualizations are supportive evidence. They should not replace numeric reports.

## 15. Runs

```text
runs/
├── phase0_baseline/
├── phase1_graph/
├── phase2_distance/
├── phase3_geo_attention/
└── gates/
```

Run directories should be treated as experiment artifacts.

Recommended structure for a run:

```text
runs/<phase>/<run_id>/
├── config.json
├── train_log.jsonl
├── metrics.json
├── model_summary.json
├── checkpoints/
└── artifacts/
```

Gate decisions:

```text
runs/gates/phase3_to_phase4_decision.json
```

## 16. Tests

```text
tests/
├── test_transformer_baseline.py
├── test_relational_graph.py
├── test_emergent_distance.py
├── test_geo_attention.py
├── test_metrics.py
└── test_causal_masking.py
```

Minimum tests before serious experiments:

- Tensor shapes are correct.
- No NaN or Inf in graph and distance outputs.
- `alpha_zero` matches baseline attention behavior.
- Causal mask prevents future attention.
- Random and shuffled distance modes run.
- Metrics serialize to JSON.

## 17. Scripts

```text
scripts/
├── prepare_wikitext2.py
├── run_phase0_baseline.ps1
├── run_phase1_graph.ps1
├── run_phase2_distance.ps1
└── run_phase3_ablation.ps1
```

Scripts should be thin wrappers around experiment entry points.

Do not hide important parameters in scripts. Important values belong in configs.

## 18. Import Boundaries

Recommended dependency direction:

```text
models -> attention/layers/geometry
experiments -> models/evaluation/data
evaluation -> saved outputs and tensors
visualization -> saved outputs
```

Avoid circular dependencies.

Core model code should not import experiment scripts.

## 19. Phase Ownership

### Phase 0

Primary files:

```text
models/transformer_baseline.py
experiments/train_baseline.py
experiments/eval_baseline.py
configs/baseline/
```

### Phase 1

Primary files:

```text
layers/relational_graph.py
experiments/analyze_relational_graph.py
evaluation/graph_metrics.py
```

### Phase 2

Primary files:

```text
geometry/emergent_distance.py
experiments/analyze_emergent_geometry.py
evaluation/distance_metrics.py
```

### Phase 3

Primary files:

```text
attention/geo_attention.py
models/ergt_v1.py
experiments/train_ergt_v1.py
experiments/compare_phase3.py
evaluation/attention_metrics.py
```

## 20. Naming Rules

Use explicit names:

```text
relational_graph
emergent_distance
geo_attention
transformer_baseline
ergt_v1
```

Avoid vague names such as:

```text
module.py
utils2.py
new_model.py
test_exp.py
```

Research code becomes hard to interpret when names do not preserve the theory.

## 21. Artifact Rules

Every experiment artifact should include:

- Run ID.
- Phase.
- Config.
- Seed.
- Model condition.
- Dataset.
- Tokenizer.
- Metrics.

Large checkpoints and generated matrices may be excluded from version control,
but their paths and metadata should be recorded.

## 22. Version Control Guidance

Commit:

- Documentation.
- Source code.
- Config files.
- Lightweight metrics.
- Test files.

Usually do not commit:

- Large checkpoints.
- Full datasets.
- Large matrix dumps.
- Large generated plots, unless needed for a report.

Use `.gitignore` once implementation begins.

## 23. Implementation Backlog

The next Codex-facing task list should be:

```text
.codex/implementation_backlog.md
```

It should translate this repository structure into ordered tasks with acceptance
criteria.

Implementation should begin only after the backlog is created.
