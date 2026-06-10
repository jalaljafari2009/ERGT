# Codex Project Context: ERGT

This file is the Codex-facing entry point for the ERGT documentation pack.
Before implementing code, Codex should read the relevant documents listed here
in order and preserve the distinction between the broad theory and the first
controlled proof stage.

## Project Identity

- Project name: ERGT
- Full name: Emergent Relational Geometry Transformer
- Research type: architecture and theory of relational representation
- First proof target: ERGT-v1

## Required Reading Order

1. `README.md`
   - High-level project overview.
   - Research pipeline.
   - Minimum publishable version.

2. `docs/00_project_charter.md`
   - Project scope.
   - Version 1 boundaries.
   - Success and failure criteria.
   - Gate condition before later phases.

3. `docs/01_theoretical_foundation.md`
   - Philosophical and theoretical foundation.
   - Relations, structure, geometry, memory, reasoning, intelligence.

4. `docs/02_operational_definitions.md`
   - Measurement definitions for `W`, `D`, geometry, concept, memory, and
     reasoning.

5. `docs/03_phase_plan.md`
   - Full phase plan from theoretical grounding through ERGT-v1 and later
     extensions.

6. `docs/04_proof_stage_protocol.md`
   - Exact comparison protocol for the first proof stage.

## Implementation-Specific Documents

These documents should be read only when implementing the corresponding module:

- `docs/05_baseline_transformer_spec.md`
  - Baseline transformer specification.

- `docs/06_relational_graph_spec.md`
  - Relational graph design and metrics.

- `docs/07_emergent_distance_spec.md`
  - Emergent distance and geometry metrics.

- `docs/08_geo_attention_spec.md`
  - GeoAttention design and tensor contract.

- `docs/09_metrics_and_ablation_plan.md`
  - Metric and ablation matrix.

- `docs/10_gate_conditions.md`
  - Decision rules for moving beyond Phase 3.

- `docs/11_repository_structure.md`
  - Source tree and file ownership.

- `docs/12_phase4_dynamic_memory_plan.md`
  - Gate-locked plan for dynamic relational graph memory after Phase 3.

- `docs/13_phase3_stable_base_plan.md`
  - Stable Phase 3 candidate after confirmation-seed instability.
  - Defines detached distance, cosine graph construction, clipped distance,
    alpha warmup, live monitoring, and strict controls.

## Source Notes

The current source notes are:

- `4.txt`
  - Original phased research roadmap.

- `5.txt`
  - Operational architecture definition.

- `6.txt`
  - Philosophical and theoretical foundation.

These files are source material, not implementation specs. Codex should use the
curated documents in `docs/` as the active project contract once they are
created.

## Working Rule

Do not reduce ERGT to a generic attention tweak.

The broad theory is:

```text
Intelligence emerges from the discovery, compression, stabilization, and
traversal of relational structures.
```

The first experimental claim is narrower:

```text
Correlation-induced relational distance can improve or stabilize transformer
attention.
```

Implementation should prove the narrow claim first while preserving the broader
theoretical frame.
