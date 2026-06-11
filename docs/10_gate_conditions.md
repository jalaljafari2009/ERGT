# Gate Conditions

## 1. Purpose

This document defines the decision rules that control movement between ERGT
phases.

The most important gate is the transition from Phase 3 to Phase 4.

ERGT should not proceed to graph memory, causal geometry, spectral complexity,
or relational fields until the first proof stage gives credible evidence that
induced relational distance is useful for attention.

## 2. Gate Philosophy

The broad ERGT theory is deep, but execution must be disciplined.

The project should protect the theory by preventing premature architectural
expansion.

The rule is:

```text
Do not build later mechanisms on top of an unproven earlier mechanism.
```

For ERGT-v1, the earlier mechanism is:

```text
W -> D -> GeoAttention
```

After the Phase 3 evidence, the gate philosophy is strengthened by
`docs/17_physics_aligned_ergt_program.md`:

```text
measure -> control -> observe -> validate -> inject -> regularize
```

This means that later mechanisms must not be built directly on top of a weak
GeoAttention result. They must first pass strict control, observer,
reconstruction, memory-observer, and causal-geometry checks.

## 3. First Proof Claim

The first proof stage tests:

```text
Correlation-induced relational distance can improve or stabilize transformer
attention.
```

The gate evaluates whether this claim has enough support to justify Phase 4.

## 4. Phase 0 Completion Gate

Phase 0 is complete when the baseline transformer:

- Trains without numerical instability.
- Shows decreasing validation loss.
- Computes perplexity correctly.
- Saves config, logs, metrics, and checkpoints.
- Exposes attention code clearly enough for replacement.
- Uses the same dataset and tokenizer intended for ERGT-v1.

If Phase 0 fails, do not start Phase 1.

## 5. Phase 1 Completion Gate

Phase 1 is complete when the relational graph observer:

- Runs without changing baseline model behavior.
- Produces stable `W` matrices.
- Shows that `W` is not uniform, saturated, or purely diagonal.
- Compares real `W` against random and shuffled controls.
- Saves graph metrics in machine-readable form.

If `W` is trivial, do not proceed directly to Phase 2. Redesign graph
construction first.

## 6. Phase 2 Completion Gate

Phase 2 is complete when emergent distance:

- Produces numerically stable `D`.
- Defines and records normalization.
- Handles diagonal policy explicitly.
- Handles causal constraints explicitly.
- Shows real `D` differs from random and shuffled distance.
- Reports attention-duplication diagnostics.
- Provides at least one non-trivial geometry or neighborhood metric.

If `D` is unstable or trivial, do not proceed to Phase 3. Redesign `W`, `D`, or
normalization first.

## 7. Phase 3 Completion Gate

Phase 3 is complete only when all required conditions have run:

```text
baseline
alpha_zero
real_d
random_d
shuffled_d
```

and the report includes:

- Validation loss.
- Perplexity.
- Runtime.
- Memory.
- Attention diagnostics.
- Graph and distance diagnostics.
- Seed variance, unless explicitly labeled as preliminary.

## 8. Gate 1: Proceed to Phase 4

Gate 1 is passed when:

```text
GeoAttention(real D)
  improves or stabilizes over TransformerBaseline
  and outperforms GeoAttention(random D)
  and outperforms GeoAttention(shuffled D)
  and alpha_zero behaves like the baseline
```

The advantage may be based on:

- Lower validation loss.
- Lower perplexity.
- Better training stability.
- Better predefined long-range dependency behavior.
- Comparable performance with substantially better interpretability or
  structured attention, if clearly justified.

Runtime and memory overhead must be documented.

## 9. Strong Pass

A strong pass requires:

- `real_d` beats baseline and controls across multiple seeds.
- `alpha_zero` matches baseline within expected noise.
- Geometry diagnostics show non-trivial structure.
- Runtime and memory overhead are acceptable or clearly bounded.
- No causal leakage or masking issue is present.

Action:

```text
Proceed to Phase 4: Dynamic Relational Graph Memory.
```

## 10. Conditional Pass

A conditional pass occurs when the result is promising but incomplete.

Examples:

- `real_d` beats controls on one seed but needs more seeds.
- `real_d` improves stability but not perplexity.
- Gains are present but overhead is high.
- Alpha sensitivity is high but controllable.
- One optional diagnostic is missing.

Action:

```text
Repeat Phase 3 with targeted fixes before starting Phase 4.
```

Allowed fixes:

- More seeds.
- Better distance normalization.
- Fixed alpha sweep.
- Detached distance control.
- Improved logging.
- Runtime optimization that does not change the scientific condition.

## 11. Fail

Gate 1 fails if:

- `real_d` does not beat random or shuffled distance.
- `alpha_zero` does not match baseline behavior.
- Training diverges or produces NaN/Inf.
- `D` is trivial, unstable, or dominated by artifacts.
- Geometry term overwhelms standard attention.
- Causal masking is violated.
- Results are not reproducible.

Action:

```text
Do not proceed to Phase 4.
```

Revisit:

- Relation kernel.
- Hidden-state normalization.
- Distance transform.
- Distance normalization.
- Diagonal policy.
- Alpha parameterization.
- Injection layer.
- Gradient flow through `W` and `D`.

## 12. Later Phase Gates

The later phase gates below remain valid, but post-Phase-3 execution must use
the stricter program in `docs/17_physics_aligned_ergt_program.md` whenever there
is ambiguity about random/shuffled controls, reconstruction, causal leakage, or
collapse.

### Phase 4 to Phase 5

Proceed only if graph memory:

- Improves or stabilizes results compared with ERGT-v1.
- Shows measurable persistence in `W`.
- Does not introduce unacceptable leakage or cost.
- Beats instantaneous, no-memory, random-memory, shuffled-memory, and generic
  smoothing controls.
- Uses a similarity-based and `Phi`-gated update if the physics-aligned program
  is active.

### Phase 5 to Phase 6

Proceed only if full ERGT architecture:

- Remains competitive with baseline and ERGT-v1.
- Preserves the proven role of `D`.
- Has clear motivation for causal geometry.
- Preserves the proven role of `W_t`, `Phi`, and reconstruction controls when
  those components are active.

### Phase 6 to Phase 7

Proceed only if causal geometry:

- Improves long-context behavior.
- Passes strict causal leakage checks.
- Does not only add another uncontrolled bias.
- Adds signal beyond pairwise distance and no-memory controls.

### Phase 7 to Phase 8

Proceed only if spectral complexity:

- Produces measurable compression or stability.
- Does not collapse useful relational structure.
- Improves generalization or interpretability enough to justify complexity.
- Is paired with anti-collapse and reconstruction consistency checks.

## 13. Documentation Required for Any Gate Decision

Every gate decision must record:

- Date.
- Phase.
- Run IDs.
- Compared conditions.
- Metrics summary.
- Main evidence.
- Main failure modes.
- Decision: pass, conditional pass, or fail.
- Next action.

Recommended output:

```text
runs/gates/phase3_to_phase4_decision.json
```

Suggested schema:

```json
{
  "gate": "phase3_to_phase4",
  "decision": "conditional_pass",
  "run_ids": [],
  "evidence": [],
  "risks": [],
  "required_next_actions": []
}
```

## 14. Anti-Overclaim Rule

Passing Gate 1 does not prove:

- Concept emergence.
- Graph memory.
- Reasoning.
- Intelligence.
- A full relational field theory.

Passing Gate 1 only supports:

```text
Induced relational distance can be a useful attention bias.
```

This is enough to justify the next phase. It is not enough to claim the whole
theory is proven.

## 15. Default Decision Bias

When evidence is ambiguous, choose the more conservative decision.

```text
ambiguous pass -> conditional pass
ambiguous fail -> redesign current phase
missing controls -> no gate decision
```

For post-Phase-3 work, "missing controls" includes missing strict W-level
controls, missing `Phi` anti-collapse checks, missing reconstruction-deficit
checks, missing memory observer controls, or missing pairwise/no-memory causal
geometry controls.

The project should advance only when the current mechanism is clear enough to
support the next one.
