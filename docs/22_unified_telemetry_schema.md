# Unified Telemetry Schema

## 1. Purpose

This stage defines one telemetry language for all open adaptive ERGT work.

The next stages will adapt memory, gate-floor, distance scale, causal
reachability, and alpha. Without a shared schema, every controller would invent
slightly different names for the same signals, making attribution and
comparison unreliable.

The schema defines:

```text
what must be logged
when it must be logged
which fields are canonical
which Run-02 names are accepted as aliases
which fields are required for live 100-step diagnosis
```

## 2. Design Rule

Telemetry must cover all mechanisms that can explain a result:

```text
loss trend
control separation
attention rigidity
attention behavior
memory state
geometry and distance scale
adaptive parameter changes
attribution
adaptive_trainer
live_diagnostics
safety
runtime
```

If a controller changes a parameter without logging the evidence and risk behind
that change, the run is not interpretable.

Attention telemetry is not only a collapse alarm. It is also a behavioral probe
for the adaptive search. Controllers and later analysis should use attention
entropy, sparsity, max probability, head/layer diversity, and geometry takeover
to infer whether a degree of freedom is moving toward a useful operating region
or toward uniformity, lock-in, or generic regularization.

Control telemetry must also expose RNG isolation for random and shuffled
families. A control run is not fair if generating `W_random` or `W_shuffled`
advances the global training, dropout, or sampler RNG.

## 3. Canonical Categories

The schema groups fields into:

```text
identity
data_model
loss_trend
control_separation
attention_rigidity
attention_behavior
memory_state
geometry_distance
adaptive_parameters
attribution
adaptive_trainer
live_diagnostics
safety
runtime
```

Every field belongs to exactly one category.

## 4. Minimum Live Fields

Every live 100-step display must include at least:

```text
step
condition
validation_loss
alpha
alpha_next
alpha_delta
alpha_decision
loss_slope_gain
baseline_centered_improvement
geo_to_qk_ratio
attention_entropy
attention_entropy_normalized
attention_entropy_drop
mean_max_probability
valid_mean_max_probability
head_attention_diversity
layer_attention_diversity
geometry_takeover_score
rigidity_risk
collapse_risk
attention_behavior_regime
attention_control_separation
attention_search_pressure
control_penalty
control_rng_isolated
trainer_status
trainer_fail_fast_triggered
```

Run-02 currently uses some compatibility names, for example
`alpha_effective` and `adaptive_slope_gain`. These are accepted as aliases, but
ERGT-03 should prefer canonical names.

## 5. Adaptive Parameter Fields

Every adaptive parameter must expose:

```text
<parameter>_next
<parameter>_delta
<parameter>_decision
<parameter>_credit
<parameter>_risk_pressure
```

For the next stages, this applies to:

```text
alpha
memory_decay
memory_eta
gate_floor
gate_floor_next
gate_floor_delta
gate_floor_decision
gate_floor_credit
gate_floor_risk_pressure
edge_survival
random_edge_noise_score
shuffled_edge_noise_score
real_edge_starvation_score
distance_norm_scale
causal_reachability
```

## 6. Joint Budget Fields

Independent controllers are not enough once alpha, memory, gate floor, distance
scale, and causal reachability can all change. The live run must expose the
joint budget that coordinates those degrees of freedom:

```text
change_budget
allocated_change_budget
geometry_budget
memory_budget
rigidity_budget
noise_budget
qk_competition_state
attention_behavior_regime
attention_derived_budget_pressure
budget_conflict_score
joint_budget_decision
budget_allocation
budget_suppression_reasons
release_allocation
restraint_allocation
```

These fields are the basis for:

```text
Joint Parameter Budget Allocator
```

The allocator does not replace the individual controllers. It decides how much
of their requested change is allowed, suppressed, or shifted when evidence
conflicts across loss trend, memory, distance contrast, attention behavior, and
control-family separation.

## 7. Control Separation Fields

Control separation must support sequential execution. While `real` is running,
random and shuffled controls may not exist yet, so the schema must distinguish
partial live evidence from final matched claim evidence:

```text
control_separation_mode
claim_eligibility
control_separation_status
scientific_claim_credit
available_control_families
pending_control_families
control_family_status
matched_control_steps
matched_control_window
real_vs_baseline_delta
real_vs_alpha_zero_delta
real_vs_random_delta
real_vs_shuffled_delta
real_vs_no_memory_delta
real_vs_instantaneous_delta
control_separation
control_penalty
generic_regularization_warning
attention_behavior_separation
partial_live_score
final_matched_score
```

`partial_live` may guide controllers using `real_vs_baseline_delta`, but it must
emit:

```text
claim_eligibility = not_eligible_pending_controls
scientific_claim_credit = 0
```

until all required controls have matched late-window points.

## 8. Meta-Control Attention Fields

The meta-control attention observer must expose what it attends to, what it
masks, and how much confidence the observer has in its allocation:

```text
meta_control_mode
meta_observer_only
meta_attention_weights
meta_signal_status
meta_available_signal_count
meta_masked_signal_count
evidence_availability_score
pending_control_mask
offline_replay_required
meta_top_signal
meta_suppressed_signal
meta_attention_entropy
meta_attention_entropy_normalized
controller_agreement_score
controller_conflict_score
meta_control_confidence
meta_parameter_allocation
meta_alpha_weight
meta_memory_weight
meta_gate_weight
meta_reach_weight
meta_norm_weight
meta_observer_decision_summary
meta_replay_record
```

The first version is observer-only. It can explain signal weights and parameter
allocation pressure, but it must not directly change alpha, memory, gate floor,
reachability, or normalization.

## 9. Adaptive Trainer Fields

The open adaptive trainer is the orchestration layer for sequential runs,
controller decisions, meta-control observation, safety, and lightweight
artifacts. Required trainer fields include:

```text
trainer_status
trainer_event
trainer_fail_fast_triggered
trainer_fail_fast_reason
trainer_processed_rows
trainer_processed_steps
controller_decision_count
meta_observer_event_count
control_separation_event_count
safety_event_count
live_display_event_count
progress_log_ready
controller_decision_log_ready
meta_control_observer_log_ready
control_separation_log_ready
safety_log_ready
lightweight_artifact_bundle_ready
checkpoint_artifacts_excluded
artifact_bundle_name
lightweight_artifact_manifest
trainer_replay_record
```

These fields are the basis for:

```text
Open Adaptive Relational Control Trainer
```

The trainer must preserve sequential no-future-peek scoring. During `real`,
missing random/shuffled/no-memory/instantaneous controls are masked, and final
matched replay happens only after the controls exist.

## 10. Live Diagnostic Fields

The live 100-step display must be both human-readable and plot-ready. Required
live diagnostic fields include:

```text
live_diagnostic_row_ready
live_diagnostic_table_ready
live_diagnostic_plot_ready
live_diagnostic_row_count
live_diagnostic_columns
live_diagnostic_row
live_diagnostic_table_markdown
live_diagnostic_plot_payload
```

These fields are the basis for:

```text
Live 100-Step Diagnostic Table
```

The table is allowed to show `NA` for fields unavailable during sequential runs,
but it must not silently omit required columns.

## 11. Causal Reachability Fields

Causal reachability cannot be judged only from `max_causal_step`. Required
reachability fields include:

```text
causal_reachability
causal_reachability_next
causal_reachability_delta
causal_reachability_decision
causal_reachability_credit
causal_reachability_risk_pressure
causal_edge_survival
reach_starvation_score
reach_expansion_pressure
reach_restraint_pressure
control_reach_noise_score
attention_locality_score
attention_spread_score
future_leak_score
real_reach_advantage
random_reach_advantage
shuffled_reach_advantage
```

These fields are the basis for:

```text
Causal Reachability Controller
```

Future leakage remains a validity hard stop. Ordinary reachability risk is
controller pressure.

## 12. Memory Fields

Memory cannot be judged only from loss. Required memory fields include:

```text
memory_decay
memory_eta
gate_floor
memory_stability
memory_turnover
memory_edge_density
memory_persistence
memory_spectral_entropy
memory_effective_rank
memory_rigidity
noise_risk
```

These fields are the basis for the next stage:

```text
Memory State Instrumentation
```

## 13. Control RNG Fields

Random and shuffled controls must declare:

```text
control_rng_isolated
control_seed
control_seed_offset
control_step_seed
```

`control_rng_isolated=false` means the control comparison is not eligible for a
real-geometry claim.

## 14. Machine Schema

This stage adds:

```text
evaluation/unified_telemetry_schema.py
experiments/create_unified_telemetry_schema_report.py
```

Usage:

```bash
python experiments/create_unified_telemetry_schema_report.py
```

Default output:

```text
runs/contracts/unified_telemetry_schema.json
```

## 15. Exit Criteria

This stage is complete when the machine schema validates that:

```text
there are no duplicate field definitions
every field has category, type, cadence, and meaning
all Open Control required telemetry is covered
Run-02 aliases remain compatible
adaptive parameter change fields are declared
hard-stop fields are declared
minimum live fields are declared
```

Only after this schema exists should the project implement the memory-state
instrumentation layer.
