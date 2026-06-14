# Meta-Control Attention Observer

## Purpose

Stage 16 adds an observer-only attention layer over controller evidence. It does
not replace transparent controllers and it does not change parameters directly.

Its job is to answer:

```text
which controller signals is the adaptive system currently paying attention to?
which signals are unavailable, masked, or suppressed?
how should attention be allocated across alpha, memory, gate, reach, and norm?
how much confidence should we place in that allocation?
```

## Sequential Run Rule

ERGT runs are usually sequential:

```text
baseline -> real -> random -> shuffled -> no_memory -> instantaneous
```

During the `real` run, full control-family evidence does not exist yet. The
meta-control observer must therefore be missing-aware:

```text
pending controls -> control-family signals masked
available signals -> normal attention
claim credit -> still zero until matched controls exist
offline replay -> required after controls are complete
```

This preserves the usefulness of meta-control attention without pretending that
the observer has seen random or shuffled results during the real run.

## Observed Signals

The first observer reads:

```text
loss_slope
baseline_delta
control_separation
geo_qk
rigidity_collapse
memory_state
gate_noise
causal_reachability
distance_contrast
attribution_uncertainty
```

`control_separation` is masked until:

```text
claim_eligibility = eligible_complete_controls
pending_control_families = []
```

## Output Fields

Every observation emits:

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

## Modes

`online_partial`

Used during sequential runs when control evidence is incomplete. It can explain
available evidence and controller pressure, but cannot grant claim credit.

`offline_matched_replay`

Used after all control families are available. The same observer is replayed on
matched telemetry rows to see how attention would have changed with complete
evidence.

## Live 100-Step Display

The live line/table should include:

```text
meta_control_mode
meta_top_signal
meta_suppressed_signal
meta_control_confidence
meta_attention_entropy_normalized
controller_conflict_score
meta_alpha_weight
meta_memory_weight
meta_gate_weight
meta_reach_weight
meta_norm_weight
```

## Machine Artifacts

This stage adds:

```text
experiments/meta_control_attention_observer.py
evaluation/meta_control_attention_observer.py
experiments/create_meta_control_attention_observer_report.py
tests/test_meta_control_attention_observer.py
```

Default report:

```text
runs/contracts/meta_control_attention_observer.json
```

Usage:

```bash
python experiments/create_meta_control_attention_observer_report.py
```

## Exit Criteria

This stage is complete when:

```text
pending control-family evidence is masked
available non-control evidence still receives attention during real
final matched replay can attend to control separation
parameter allocation weights are reported and normalized
confidence drops under controller conflict
the observer remains observer-only
live progress logging exposes meta-control fields
```

Next stage:

```text
Open Adaptive Relational Control Trainer
```
