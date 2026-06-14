# Active Adaptive Relational Control Execution Plan

This document is the active execution ledger for the post-Run-02 adaptive ERGT
program. It is the source of truth for the current plan, completed stages,
remaining stages, and mid-run additions.

When the user asks "what is the plan now?", the answer should be derived from
the unfinished stages in this document. When a stage is completed, update its
status, evidence, and next action here. When the whole program is complete and a
new program is provided, replace this active plan with the new one, keeping the
old plan only if an archive is explicitly needed.

## Status Labels

- `DONE`: implemented, documented, and minimally validated.
- `NEXT`: the next stage to execute.
- `PENDING`: planned but not started.
- `BLOCKED`: cannot proceed until a named dependency is resolved.
- `REVISE`: evidence showed the stage or an earlier assumption must be changed.

## Operating Rules

1. This file is the active plan ledger.
2. Every completed stage must record the implementation evidence and validation.
3. Any user insight raised during execution must be added either to the relevant
   stage or to the inserted-notes section below before moving on.
4. Later stages must not hide failed gates. A failed gate becomes a revision
   item, not a reason to proceed blindly.
5. The program is diagnostic-driven: every controller must expose why it changed
   a parameter and what evidence restrained or released it.
6. Ordinary risk flags are not stop signals. They must become controller
   pressure, budget reallocation, shrink/freeze decisions, or revision labels.
   Only safety and validity failures may abort or invalidate a run.
7. The program is an adaptive search over unknown degrees of freedom. Every
   run must preserve a replayable trajectory of parameter values, injected
   evidence, observations, controller state, decisions, credits, and risk
   pressures so later behavior analysis can identify when the system acted well
   or misdiagnosed the state.
8. Attention behavior is a controller observability surface. Entropy,
   sparsity, max probability, head/layer diversity, and geometry takeover must
   be used to infer whether the search is moving toward a useful operating
   region, but attention alone is not scientific credit without loss trend,
   memory state, attribution, and real-vs-control separation.

## Global Risks Now Tracked Explicitly

### R1: Memory Temporal Scope and Causal Reachability Risk

The memory path must be checked for what it actually remembers. A layer-local
memory carried within one forward pass is not the same as historical relational
memory across training time or examples. The program must prove:

- memory is past-only for each sequence position;
- no future edge can enter through update, normalization, or reporting;
- no cross-batch or cross-family memory reuse exists;
- useful real edges survive long enough to become geometry;
- random or shuffled controls are not helped by a generic smoothing effect that
  real memory cannot use.

Primary stages covering this risk: 4, 10, 11, 12, 14, 16, 22, 23, 25.

### R2: Distance Signal Flattening and Geometry Strength Risk

The distance path must not erase the relational signal through normalization,
clipping, warmup, or a fixed alpha ceiling. The earlier guarded run showed
`geo_to_qk_ratio` declining after alpha reached its cap, so geometry may have
become relatively weaker as QK logits grew. The program must prove:

- real distance contrast survives normalization;
- alpha can keep growing when evidence supports it;
- geometry can compete with QK without collapsing attention;
- random and shuffled gains are not only generic regularization;
- late-window behavior is judged by trends, not isolated points.

Primary stages covering this risk: 5, 7, 9, 13, 14, 16, 18, 22, 23, 25.

### R3: Attention Behavior as Optimization-Surface Risk

Attention is the clearest live view into whether adaptive degrees of freedom
are producing a useful regime or only changing loss indirectly. The program
must prove:

- attention is neither collapsed nor uniformly indifferent;
- head and layer diversity do not disappear as alpha, memory, gate floor, or
  reachability adapt;
- real attention behavior separates from random and shuffled controls;
- geometry has enough influence to move attention but does not blindly take
  over QK;
- controller decisions cite attention behavior when searching for the optimal
  region of degrees of freedom.

Primary stages covering this risk: 5, 9, 10, 11, 12, 14, 15, 16, 18, 21, 22,
23, 24, 25.

## Stage Ledger

### 1. Run-02 Evidence Consolidation

Status: `DONE`

Reference: `docs/20_run02_evidence_consolidation.md`

Purpose: consolidate the guarded fixed-run evidence before opening more
adaptive parameters.

Evidence: Run-02 criteria and diagnostics were documented. The result showed
that fixed alpha and fixed memory controls are not enough for a strong real
geometry claim.

### 2. Open Control Philosophy Contract

Status: `DONE`

Reference: `docs/21_open_control_philosophy_contract.md`

Purpose: define the rule that growth and restraint should come from evidence
rather than from arbitrary fixed ceilings.

Evidence: adaptive parameters must report growth evidence, restraint evidence,
and decision reasons.

### 3. Unified Telemetry Schema

Status: `DONE`

Reference: `docs/22_unified_telemetry_schema.md`

Purpose: define one telemetry language for loss, geometry, memory, controls,
attribution, safety, and runtime fields.

Evidence: telemetry contracts and aliases were documented for adaptive runs.

### 4. Memory State Instrumentation

Status: `DONE`

Reference: `docs/23_memory_state_instrumentation.md`

Purpose: expose memory state instead of treating memory as an invisible
mechanism.

Evidence: memory diagnostics were added for stability, turnover, persistence,
rigidity, and noise risk. This stage measures R1; it does not by itself solve
R1.

Required follow-up: stages 10, 11, and 12 must use these fields to control
memory update, gate floor, and causal reachability.

### 5. Attention Rigidity and Collapse Monitor

Status: `DONE`

Reference: `docs/25_attention_rigidity_and_collapse_monitor.md`

Purpose: detect whether geometry makes attention too rigid, too sparse, too
uniform, or dominated by a small number of heads/layers.

Required outputs:

- attention entropy by condition, layer, and step;
- mean max attention probability;
- sparsity at fixed thresholds;
- head diversity;
- collapse score;
- geometry takeover score.

Pass condition: collapse and rigidity must be exposed as controller pressure.
They must not act as ordinary stop flags unless they reach a declared safety
hard-stop condition such as severe attention collapse.

Evidence: diagnostic-only rigidity and collapse metrics were added for
normalized entropy, valid-edge max probability, valid-edge sparsity, head
diversity, layer diversity, geometry takeover, rigidity risk, collapse risk, and
severe collapse detection. Live progress logging now exposes these fields, and
the contract report `runs/contracts/attention_rigidity_monitor.json` passes.

Post-stage correction: these attention metrics are no longer interpreted only
as collapse diagnostics. They are also behavioral search evidence for later
controllers. Later stages must use them to identify useful, uniform, rigid,
takeover, and control-like attention regimes.

Risk coverage: R2 and R3.

### 6. Control-Family Fairness Audit v2

Status: `DONE`

Reference: `docs/26_control_family_fairness_audit_v2.md`

Purpose: prove that real, random, shuffled, no-memory, and instantaneous
conditions differ only in the intended relational mechanism.

Required outputs:

- same data path and batch path;
- isolated control RNG;
- no cross-family real distance reuse;
- no cross-family real memory reuse;
- matched normalization policy;
- matched clipping and diagonal policy;
- comparable `geo_to_qk_ratio`.

Pass condition: random and shuffled are valid controls only if all fairness
checks pass.

Evidence: stage-6 fairness diagnostics were added for same data path, same
batch path, isolated control RNG, no cross-family real distance reuse, no
cross-family real memory reuse, matched normalization, matched clipping, matched
diagonal policy, and comparable `geo_to_qk_ratio` against the same QK
denominator. The contract report
`runs/contracts/control_family_fairness_audit_v2.json` passes.

Risk coverage: R1 and R2.

### 7. Loss-Slope and Trend Analyzer

Status: `DONE`

Reference: `docs/27_loss_slope_and_trend_analyzer.md`

Purpose: make controller decisions from trends, not from one noisy loss point.

Required outputs:

- baseline-relative validation delta;
- EMA loss delta;
- rolling slope;
- late-window slope;
- post-1000 trend;
- stability of advantage.

Pass condition: controller decisions must cite windowed slope and EMA evidence.

Evidence: stage-7 trend diagnostics were added for baseline-relative validation
delta, EMA validation delta, rolling slope, baseline rolling slope, late-window
slope, post-1000 trend, and controller evidence fields. The contract report
`runs/contracts/loss_slope_trend_analyzer.json` passes.

Risk coverage: R2.

### 8. Parameter Attribution Probe

Status: `DONE`

Reference: `docs/28_parameter_attribution_probe.md`

Purpose: estimate which parameter changes are responsible for observed gains or
failures.

Required outputs:

- alpha contribution estimate;
- memory eta and decay contribution estimate;
- gate-floor contribution estimate;
- normalization contribution estimate;
- reachability contribution estimate;
- interaction warnings when attribution is ambiguous.

Pass condition: major adaptive decisions must include attribution evidence or an
explicit uncertainty flag. Uncertainty is not a stop signal; it is logged for
later behavior analysis and budget allocation.

Evidence: stage-8 attribution diagnostics were added for alpha, memory eta and
decay, gate floor, normalization scale, and causal reachability contribution
estimates. The probe emits explicit uncertainty flags and interaction warnings
for simultaneous parameter changes. The contract report
`runs/contracts/parameter_attribution_probe.json` passes.

Risk coverage: R1 and R2.

### 9. Adaptive Alpha Controller v2

Status: `DONE`

Reference: `docs/29_adaptive_alpha_controller_v2.md`

Purpose: allow alpha to grow when real geometry is useful and restrained when
controls or collapse signals indicate generic regularization or rigidity.

Required outputs:

- current alpha;
- proposed alpha;
- alpha delta;
- release evidence;
- restraint evidence;
- slope evidence;
- rigidity evidence;
- control-family evidence.

Pass condition: alpha growth must be justified by real late-window trend and
must not be driven by random or shuffled advantage.

Evidence: stage-9 alpha controller v2 was added as a PID-inspired evidence
balance controller rather than a blind fixed-threshold gate. The controller
tracks release evidence, restraint evidence, slope evidence, rigidity evidence,
control-family evidence, controller state snapshots, parameter trajectories,
injected-evidence ledgers, and decision replay records. Ordinary risk signals
become restraint pressure or shrink/freeze decisions; only explicit hard-stop
signals hold alpha without pretending the search failed. The contract report
`runs/contracts/adaptive_alpha_controller_v2.json` passes.

Risk coverage: R2.

### 10. Adaptive Memory Controller

Status: `DONE`

Reference: `docs/30_adaptive_memory_controller.md`

Purpose: make memory eta and decay respond to memory state instead of staying
fixed.

Required outputs:

- memory stability trend;
- memory turnover trend;
- persistence trend;
- noise risk;
- rigidity risk;
- eta and decay decisions.

Pass condition: memory parameters must adapt when memory is starved, noisy, or
rigid.

Evidence: stage-10 memory controller was added as an adaptive search controller
over `eta` and `decay`. It increases memory injection when useful real memory
is starved, smooths noisy memory by reducing `eta` and raising `decay`, and
releases rigid memory by reducing both `eta` and `decay`. The controller emits
memory stability, turnover, persistence, noise, rigidity, release, restraint,
controller-state, parameter-trajectory, injected-evidence-ledger, and
decision-replay records. Future leak remains a validity hard stop; ordinary
memory risk becomes controller pressure. The contract report
`runs/contracts/adaptive_memory_controller.json` passes.

Risk coverage: R1.

### 11. Gate-Floor and Noise Controller

Status: `DONE`

Reference: `docs/31_gate_floor_and_noise_controller.md`

Purpose: prevent weak random edges from dominating while avoiding premature
deletion of weak but forming real edges.

Required outputs:

- gate floor;
- edge density;
- edge survival;
- random-edge noise score;
- real-edge starvation score.
- attention sparsity and entropy pressure;
- real-vs-control attention separation pressure.

Pass condition: gate floor must respond differently to noise dominance and real
edge starvation, and must not create attention uniformity or collapse while
filtering noise.

Evidence: stage-11 gate-floor/noise control was added as a replayable
controller over `gate_floor`. It raises the floor when random/shuffled edge
noise or control-like attention dominates, lowers the floor when real edges are
starved or attention is over-sparse, and treats future leak as a validity hard
stop. The controller emits edge-noise evidence, starvation evidence,
attention-pressure evidence, control-attention evidence, parameter trajectory,
injected-evidence ledger, controller state, and decision replay records. The
contract report `runs/contracts/gate_floor_noise_controller.json` passes.

Risk coverage: R1 and R3.

### 12. Causal Reachability Controller

Status: `DONE`

Reference: `docs/32_causal_reachability_controller.md`

Purpose: keep geometry past-only while allowing causal reach to open when
evidence shows the graph is too locally constrained.

Required outputs:

- max causal step;
- future leak score;
- causal edge survival;
- reach expansion decision;
- reach restraint decision.
- attention locality and spread evidence;
- head/layer diversity under reach changes.

Pass condition: future leak must remain zero; reach may expand only when memory
is stable enough and attention behavior is not collapsing, uniformly
indifferent, or control-like.

Evidence: stage-12 causal reachability control was added as a replayable
controller over finite-speed causal reach. It expands reach when stable memory,
edge starvation, and local attention show the graph is too tight; restrains reach
when memory is unready, controls dominate, attention becomes uniform or
low-diversity, or collapse pressure rises; and treats future leak as a validity
hard stop. The controller emits future-leak evidence, memory-readiness evidence,
attention locality/spread evidence, control-reach evidence, expansion/restraint
evidence, parameter trajectory, injected-evidence ledger, controller state, and
decision replay records. Live progress logging now exposes causal-reach fields,
and the contract report `runs/contracts/causal_reachability_controller.json`
passes.

Risk coverage: R1 and R3.

### 13. Normalization and Distance-Scale Controller

Status: `DONE`

Reference: `docs/33_normalization_and_distance_scale_controller.md`

Purpose: prevent distance normalization or clipping from flattening real
relational contrast.

Required outputs:

- pre-normalization distance contrast;
- post-normalization distance contrast;
- contrast retention;
- distance std before and after normalization;
- clipping saturation rate;
- geometry scale recommendation.

Pass condition: if real contrast exists before normalization but disappears
after it, normalization must be revised before longer runs.

Evidence: stage-13 normalization and distance-scale control was added as a
replayable controller over `distance_norm_scale`. It increases scale only when
pre-normalization contrast exists, post-normalization contrast or retention is
weak, geo/QK is underpowered, controls are not dominant, clipping is low, and
attention is not uniform or collapsed. It decreases or holds scale when
pre-normalization signal is absent, clipping saturation is high, geo/QK is
overpowered, random/shuffled distance advantage dominates, post-normalization
std is unsafe, or attention becomes uniform/collapse-prone. The controller emits
contrast evidence, scale evidence, clipping evidence, control-distance evidence,
attention-safety evidence, release/restraint evidence, parameter trajectory,
injected-evidence ledger, controller state, and decision replay records. Live
progress logging now exposes distance-scale fields, and the contract report
`runs/contracts/distance_scale_controller.json` passes.

Risk coverage: R2.

### 14. Joint Parameter Budget Allocator

Status: `DONE`

Reference: `docs/34_joint_parameter_budget_allocator.md`

Purpose: coordinate alpha, memory, gate floor, reachability, and distance scale
as one budget instead of independent knobs.

Required outputs:

- geometry budget;
- memory budget;
- rigidity budget;
- noise budget;
- QK competition state;
- release or restraint allocation.
- attention behavior regime;
- attention-derived budget pressure.

Pass condition: no parameter should grow in a way that contradicts another
controller's safety evidence or pushes attention into collapse, uniformity, or
control-like behavior.

Evidence: stage-14 joint budget allocation was added as a replayable
coordination layer across `alpha`, `distance_norm_scale`, `causal_reachability`,
`memory_eta`, `memory_decay`, and `gate_floor`. It allocates a finite change
budget across geometry, memory, rigidity restraint, and noise control; shifts
budget toward memory smoothing when edge noise dominates; suppresses geometry
growth when attention pressure or control penalties show conflict; and treats
future leak or explicit hard-stop signals as validity stops. The allocator emits
release/restraint allocation, budget-allocation maps, suppression reasons,
attention-derived budget pressure, QK competition state, parameter trajectory,
injected-evidence ledger, controller state, and decision replay records. Live
progress logging now exposes joint budget fields, and the contract report
`runs/contracts/joint_parameter_budget_allocator.json` passes.

Risk coverage: R1, R2, and R3.

### 15. Control Separation Scoring

Status: `DONE`

Reference: `docs/35_control_separation_scoring.md`

Purpose: score whether real geometry separates from random, shuffled,
no-memory, instantaneous, alpha-zero, and baseline.

Required outputs:

- real vs random score;
- real vs shuffled score;
- real vs no-memory score;
- real vs instantaneous score;
- real vs baseline score;
- generic regularization warning.
- real-vs-control attention behavior separation.

Pass condition: beating baseline alone is not enough.

Sequential-run rule: during the `real` run, `random` and `shuffled` may not exist
yet. Stage 15 therefore has two scoring modes:

```text
partial_live = use available baseline/alpha-zero evidence only as controller pressure
final_matched = score real against all controls on matched late-window steps
```

`partial_live` must keep:

```text
claim_eligibility = not_eligible_pending_controls
scientific_claim_credit = 0
```

until all required controls are present.

Evidence: stage-15 control separation scoring was added with explicit
`partial_live` and `final_matched` modes. The scorer can compare real against
baseline at the current step without peeking at future control runs, marks
missing random/shuffled/no-memory/instantaneous conditions as pending, and
forbids scientific claim credit until all controls have matched late-window
points. Final scoring uses positive deltas where real has lower validation loss,
requires real to beat baseline, alpha-zero, random, shuffled, no-memory, and
instantaneous, and emits a generic-regularization warning when random or
shuffled is not beaten. Live progress logging now exposes control-separation
fields, and the contract report `runs/contracts/control_separation_scoring.json`
passes.

Risk coverage: R1, R2, and R3.

### 16. Meta-Control Attention Layer - Observer First

Status: `DONE`

Reference: `docs/36_meta_control_attention_observer.md`

Purpose: let the system attend over all controller signals before the trainer
turns those signals into parameter changes.

This is not a replacement for the transparent controllers. It is an
observer-first attention layer over their evidence:

```text
loss_slope
baseline_delta
control_separation
geo/qk
rigidity/collapse
memory_stability/turnover/persistence
gate_noise
causal_reachability
distance_contrast
attribution_uncertainty
```

Required outputs:

- meta-control attention weights per controller signal;
- per-parameter attention allocation for alpha, memory eta, memory decay, gate
  floor, causal reachability, and distance scale;
- top attended evidence source;
- suppressed evidence source;
- attention entropy over controller signals;
- controller agreement score;
- controller conflict score;
- meta-control confidence;
- observer-only decision summary.

Live 100-step display requirements:

- the meta-attention layer may read controller telemetry every step;
- the notebook/trainer should print and plot the meta-control summary every 100
  steps;
- the 100-step table must include at least `meta_top_signal`,
  `meta_alpha_weight`, `meta_memory_weight`, `meta_gate_weight`,
  `meta_reach_weight`, `meta_norm_weight`, `meta_attention_entropy`,
  `controller_conflict_score`, and `meta_control_confidence`.

Pass condition: the first version must remain observer-only. It can explain how
it would weight controllers, but it must not directly alter parameters until a
later revision gate promotes it from observer to actuator.

Sequential-run rule: this stage inherits the stage-15 missing-control constraint.
During the `real` run, control-family signals that depend on random/shuffled or
other unavailable controls must be masked, while available signals such as
baseline delta, loss slope, geo/QK, memory, gate/noise, reachability, and
distance contrast remain visible. After all controls exist, the same observer is
replayed on matched rows.

Evidence: stage-16 meta-control attention was added as an observer-only,
missing-aware attention layer over controller evidence. It emits per-signal
attention weights, signal status, pending-control masks, evidence availability,
top and suppressed signal, attention entropy, controller agreement/conflict,
confidence, per-parameter allocation weights for alpha/memory/gate/reach/norm,
and replay records. Pending control-family evidence is masked during sequential
real runs, and offline matched replay can attend to control separation after the
controls exist. Live progress logging now exposes meta-control fields, and the
contract report `runs/contracts/meta_control_attention_observer.json` passes.

Risk coverage: R1, R2, and R3.

### 17. Open Adaptive Relational Control Trainer

Status: `DONE`

Reference: `docs/37_open_adaptive_relational_control_trainer.md`

Purpose: implement the trainer that runs the controller loop and records every
decision.

Required outputs:

- unbuffered live logging;
- per-100-step telemetry;
- controller decision log;
- meta-control attention observer log;
- fail-fast safety checks;
- lightweight artifact bundle.

Pass condition: trainer must expose decisions and meta-control attention
summaries while running, not only at the end.

Evidence: stage-17 trainer orchestration was added as a replayable, sequential
controller-loop contract. It consumes live telemetry rows, performs
current-step-limited control separation, applies missing-aware meta-control
attention, emits progress/controller/meta-control/control-separation/safety
logs, stops on validity or safety hard stops, and declares a lightweight artifact
bundle that excludes checkpoints. The trainer preserves the baseline -> real ->
random -> shuffled sequential rule without peeking at future controls, and the
contract report `runs/contracts/open_adaptive_relational_control_trainer.json`
passes.

Risk coverage: R1, R2, and R3.

### 18. Live 100-Step Diagnostic Table

Status: `DONE`

Reference: `docs/38_live_100_step_diagnostic_table.md`

Purpose: make each 100-step checkpoint readable during Colab execution.

Required columns:

- step;
- condition;
- train loss;
- validation loss;
- delta vs baseline;
- rolling slope;
- alpha;
- geo/qk;
- memory stability;
- memory turnover;
- memory persistence;
- memory rigidity;
- noise risk;
- attention regime;
- attention control separation;
- contrast retention;
- future leak.
- meta top signal;
- meta attention entropy;
- meta alpha/memory/gate/reach/norm weights;
- controller conflict score;
- meta-control confidence.

Pass condition: table and plots must update during the run.

Evidence: stage-18 live diagnostics were added as structured 100-step display
artifacts. The trainer now emits `live_diagnostic_rows`,
`live_diagnostic_tables`, and `live_diagnostic_plot_payloads` at display
checkpoints and fail-fast events. The table includes loss, baseline delta,
slope, alpha, geo/QK, memory state, noise, attention regime, control separation,
distance contrast, future leak, meta-control attention allocation, controller
conflict, confidence, decision columns, and trainer status. Missing values are
explicit in markdown output, and plot payloads are grouped by loss, geometry,
memory, meta-control, and safety. The contract report
`runs/contracts/live_100_step_diagnostic_table.json` passes.

Risk coverage: R1, R2, and R3.

### 19. Adaptive Notebook ERGT-03

Status: `DONE`

Purpose: create a new notebook for adaptive relational control without changing
the earlier evidence notebook.

Required outputs:

- Colab-ready notebook;
- GitHub-to-Colab repository bootstrap;
- fixed output bundle name;
- default local review path;
- auto GPU shutdown cell;
- fail-fast report;
- 100-step meta-control attention display;
- lightweight zip excluding checkpoints.

Pass condition: notebook can be run safely on a short smoke profile before a
2000-step run.

Implementation summary: ERGT-03 now exists as
`notebooks/ERGT_03_Adaptive_Relational_Control.ipynb`. It defaults to the
`adaptive_smoke` profile, declares an `adaptive_2000_guarded` profile, runs
contract preflights, automatically clones the full repository to `/content/ERGT`
when opened from GitHub in Colab, adds the project root to `sys.path`, invokes
the open adaptive trainer, writes live 100-step rows/tables/plot payloads,
streams every 100-step diagnostic table during the notebook run through
`display_live_diagnostic_event`, verifies a future-leak fail-fast path, exports
the fixed lightweight bundle
`ergt_03_adaptive_control_report_bundle.zip`, prints the default review path
`C:\Users\Administrator\Downloads\ergt_03_adaptive_control_report_bundle.zip`,
and includes Colab runtime shutdown hooks. The stage contract is documented in
`docs/39_adaptive_notebook_ergt_03.md`; the machine report is
`runs/contracts/adaptive_notebook_ergt_03.json`.

Runtime update: A100-class runs are optimized in
`docs/47_a100_runtime_optimization.md`. The implementation separates
geometry-memory state forwarding from full diagnostics, vectorizes causal
prefix reconstruction, adds a unit-step causal shortest-path fast path, and
uses shared runtime config fields for TF32/BF16/DataLoader settings across
baseline and ERGT controls.

Risk coverage: R1, R2, and R3.

### 20. Short Smoke and Failure-Safety Validation

Status: `DONE`

Purpose: validate mechanics before a long run.

Required outputs:

- 100- or 200-step smoke;
- live output confirmed;
- schema validation;
- controller decision log exists;
- meta-control observer log exists;
- auto-shutdown path exists;
- fail-fast path tested.

Pass condition: no 2000-step run until smoke passes.

Implementation summary: the stage-20 gate is implemented in
`evaluation/short_smoke_failure_safety_validation.py` and executed by
`experiments/create_short_smoke_failure_safety_validation_report.py`. The smoke
uses 100/200-step telemetry over baseline, alpha-zero, real, random, and
shuffled conditions; confirms live 100-step output, schema compatibility,
controller and meta-control logs, ERGT-03 auto-shutdown availability,
future-leak fail-fast behavior, lightweight artifact exclusion, and sequential
no-peek behavior for live real rows. The stage contract is documented in
`docs/40_short_smoke_failure_safety_validation.md`; the machine report is
`runs/contracts/short_smoke_failure_safety_validation.json`.

Risk coverage: R1, R2, and R3.

### 21. Guarded 2000-Step Adaptive Run

Status: `DONE`

Purpose: run a short but meaningful adaptive evidence test.

Required outputs:

- baseline;
- alpha-zero;
- real adaptive memory geometry;
- random adaptive memory geometry;
- shuffled adaptive memory geometry;
- no-memory real geometry;
- instantaneous real geometry.

Pass condition: all conditions must expose comparable telemetry and late-window
analysis.

Implementation summary: the guarded 2000-step adaptive run contract is
implemented in `experiments/guarded_2000_step_adaptive_run.py`, evaluated by
`evaluation/guarded_2000_step_adaptive_run.py`, and executed by
`experiments/create_guarded_2000_step_adaptive_run_report.py`. It defines the
`adaptive_2000_guarded` profile with 100-step cadence, all seven required
conditions, sequential no-peek ordering, comparable telemetry, final matched
late-window readiness, and checkpoint-excluding artifact policy. This is a
contract/replay gate, not scientific claim evidence. The stage contract is
documented in `docs/41_guarded_2000_step_adaptive_run.md`; the machine report is
`runs/contracts/guarded_2000_step_adaptive_run.json`.

Risk coverage: R1, R2, and R3.

### 22. Late-Window and Post-1000 Analysis

Status: `DONE`

Purpose: judge the run where memory and geometry should begin to matter, not
only from early training noise.

Required windows:

- 0-500;
- 500-1000;
- 1000-1500;
- 1500-2000;
- 1000-2000.

Pass condition: decisions must prioritize post-1000 and late-window trends.
Attention behavior must be analyzed across the same windows so endpoint loss
cannot hide a late attention collapse, uniformity drift, or control-like
attention pattern.

Implementation summary: the late-window analyzer is implemented in
`experiments/late_window_post1000_analysis.py`, evaluated by
`evaluation/late_window_post1000_analysis.py`, and executed by
`experiments/create_late_window_post1000_analysis_report.py`. It enforces the
required windows, makes `1000-2000` the decision window, treats endpoint loss as
supporting evidence only, requires matched control deltas, and checks attention
collapse, uniformity drift, and control-like attention behavior over the same
windows. The stage contract is documented in
`docs/42_late_window_post1000_analysis.md`; the machine report is
`runs/contracts/late_window_post1000_analysis.json`. The current report passes
as a mechanics and analysis-contract gate, not final scientific claim evidence.

Risk coverage: R1, R2, and R3.

### 23. Random/Shuffled/No-Memory Attribution Comparison

Status: `DONE`

Purpose: determine whether gains come from real relational geometry or generic
regularization.

Required outputs:

- random advantage analysis;
- shuffled distribution-bias analysis;
- no-memory comparison;
- instantaneous comparison;
- relation-specific advantage estimate.
- attention-behavior comparison across controls.

Pass condition: if random or shuffled dominates in the late window, the claim is
not real geometry and must enter revision.

Implementation summary: the attribution comparison is implemented in
`experiments/random_shuffled_no_memory_attribution.py`, evaluated by
`evaluation/random_shuffled_no_memory_attribution.py`, and executed by
`experiments/create_random_shuffled_no_memory_attribution_report.py`. It compares
real memory geometry against random, shuffled, no-memory, instantaneous,
alpha-zero, and baseline in the `1000-2000` decision window. It labels generic
random/shuffled gains when controls beat baseline but fail to dominate real, and
it enters revision if random, shuffled, no-memory, or instantaneous controls
match or beat real. It also compares attention behavior across controls. The
stage contract is documented in
`docs/43_random_shuffled_no_memory_attribution.md`; the machine report is
`runs/contracts/random_shuffled_no_memory_attribution.json`. The current report
passes as a mechanics and attribution-contract gate, not final scientific claim
evidence.

Risk coverage: R1, R2, and R3.

### 24. Decision Gate: Real Geometry vs Generic Regularization

Status: `DONE`

Purpose: decide whether the adaptive program supports a real geometry claim.

Pass condition:

```text
real adaptive stable causal geometry > random adaptive
real adaptive stable causal geometry > shuffled adaptive
real adaptive stable causal geometry > no-memory real
real adaptive stable causal geometry > instantaneous real
real adaptive stable causal geometry > alpha-zero
real adaptive stable causal geometry > baseline
```

If only baseline is beaten, the result is insufficient.

The decision gate must also check that real attention behavior is interpretable
and separated from controls. A lower loss with collapsed, uniform, or
control-like attention is not enough for a real geometry claim.

Implementation summary: the decision gate is implemented in
`experiments/decision_gate_real_geometry.py`, evaluated by
`evaluation/decision_gate_real_geometry.py`, and executed by
`experiments/create_decision_gate_real_geometry_report.py`. It requires real
stable causal geometry to beat baseline, alpha-zero, random, shuffled,
no-memory, and instantaneous controls in the `1000-2000` decision window. It
also clears R1 memory/causal validity, R2 distance contrast and scale, and R3
attention-behavior audits. Failure cases emit revision labels for stage 25. The
stage contract is documented in `docs/44_decision_gate_real_geometry.md`; the
machine report is `runs/contracts/decision_gate_real_geometry.json`. The current
report passes as a guarded mechanics and decision-contract gate, not final
scientific claim evidence.

### 25. Controller Revision Loop

Status: `DONE`

Purpose: classify failures and revise the controller rather than adding more
complexity blindly.

Failure labels:

- `memory_starved`;
- `memory_noisy`;
- `memory_rigid`;
- `geometry_flattened`;
- `alpha_underpowered`;
- `alpha_overpowering`;
- `causal_reach_too_tight`;
- `causal_reach_too_loose`;
- `control_regularization_dominance`;
- `normalization_erased_contrast`.
- `attention_uniformity_drift`;
- `attention_control_like`;
- `attention_head_lock_in`;
- `meta_control_attention_misweighted`;
- `controller_conflict_unresolved`;

Pass condition: every failed run must map to one or more failure labels and a
specific revision.

Implementation summary: the controller revision loop is implemented in
`experiments/controller_revision_loop.py`, evaluated by
`evaluation/controller_revision_loop.py`, and executed by
`experiments/create_controller_revision_loop_report.py`. It maps every documented
failure label to a target controller component, specific revision, validation
gate, rerun protocol, and replay record. Because the current stage-24 gate
passes, the current report emits `revision_mode = noop_audit` and marks stage 26
ready. The report also validates synthetic failed gates for random dominance,
future leakage, and control-like attention. The stage contract is documented in
`docs/45_controller_revision_loop.md`; the machine report is
`runs/contracts/controller_revision_loop.json`.

### 26. Longer Run or Multi-Seed Confirmation

Status: `DONE`

Purpose: confirm only after the guarded adaptive evidence gate passes.

Allowed only if:

- short smoke passed;
- 2000-step adaptive run passed mechanics;
- real geometry beat controls in late-window analysis;
- no unresolved R1, R2, or R3 blocker remains.

Implementation summary: the confirmation contract is implemented in
`experiments/longer_run_multi_seed_confirmation.py`, evaluated by
`evaluation/longer_run_multi_seed_confirmation.py`, and executed by
`experiments/create_longer_run_multi_seed_confirmation_report.py`. It defines the
`longer_single_seed_5000` and `multi_seed_2000` confirmation profiles, requires
all matched controls for every seed/profile, blocks random or shuffled dominance
in any seed, preserves the no-peek order, and keeps checkpoint artifacts out of
the review bundle. The stage contract is documented in
`docs/46_longer_run_multi_seed_confirmation.md`; the machine report is
`runs/contracts/longer_run_multi_seed_confirmation.json`. The current report
passes as a confirmation-readiness contract, not final scientific claim evidence.

## Inserted Notes During Execution

Use this section when the user raises a new concern or insight mid-run. Add the
note, the target stage, and the intended handling.

| Date | Note | Target stage | Handling |
| --- | --- | --- | --- |
| 2026-06-14 | Memory may be hurt by how it sees past/future or by too-tight causal reach. | 4, 10, 12, 24 | Track as R1 and require explicit memory scope, future leak, edge survival, and reachability diagnostics. |
| 2026-06-14 | Normalization, warmup, or fixed alpha may flatten the distance signal and make geometry fade after step 1000. | 5, 9, 13, 14, 24 | Track as R2 and require contrast-retention, geo/qk, adaptive alpha, and distance-scale diagnostics. |
| 2026-06-14 | The system has unknown degrees of freedom and must optimize them intelligently rather than stop on ordinary flags. It must log the full injected-evidence and parameter trajectory for later behavior analysis. | 9, 10, 11, 12, 13, 14, 16, 17, 18, 21, 22, 25 | Treat risk flags as controller pressure unless they are safety/validity hard stops. Require replayable decision logs, parameter trajectories, evidence ledgers, and misdiagnosis labels. |
| 2026-06-14 | Attention should be used to understand system behavior and locate the optimal region of adaptive degrees of freedom. | 5, 9, 10, 11, 12, 14, 15, 16, 18, 21, 22, 23, 24, 25 | Track as R3. Treat attention entropy, sparsity, max probability, head/layer diversity, geometry takeover, and real-vs-control attention separation as behavioral search evidence. Do not assign scientific credit from attention alone. |
| 2026-06-14 | Add a meta-control attention layer over controller signals, but keep it observer-only at first. It may read every step, while live display should update every 100 steps. | 16, 17, 18, 19, 20, 21, 25 | Insert Meta-Control Attention after Control Separation Scoring and before the adaptive trainer. Require 100-step display of top signal, parameter weights, entropy, controller conflict, and confidence. |

## Current Next Action

Run real longer or multi-seed confirmation with actual telemetry.

Stages 6 through 26 now provide the required mechanics, telemetry,
failure-safety validation, guarded 2000-step comparable-run contract, and
post-1000 decision-gate, revision-loop, and confirmation-readiness analysis.

## Revised Remaining Program Standard

From stage 9 onward, every controller and run artifact must satisfy the
adaptive-search contract:

```text
ordinary risk flag -> pressure / budget shift / shrink / freeze / revision label
safety or validity hard stop -> abort, rewind, or invalidate the run
every decision -> replayable record
every degree of freedom -> full parameter trajectory
every injected signal -> evidence ledger entry
every bad decision -> misdiagnosis or uncertainty label
attention behavior -> search evidence, not standalone proof
```

The remaining stages keep their order, but their interpretation changes:

| Stage | Revised role |
| --- | --- |
| 9 | Build alpha controller v2 as a search controller with release/restraint evidence, not a cap-based alpha gate. |
| 10 | Build memory eta/decay controller as adaptive search over memory stability, turnover, persistence, noise, and rigidity. |
| 11 | Build gate-floor/noise controller as pressure-based filtering informed by edge noise and attention sparsity/uniformity. |
| 12 | Build causal reachability controller with future leak as hard validity stop and attention locality/spread as search pressure. |
| 13 | Build normalization/distance-scale controller to preserve contrast and log when scale choices erase signal. |
| 14 | Allocate joint parameter budget across all degrees of freedom using attribution, uncertainty, replay records, and attention behavior regime. |
| 15 | Score control separation as claim credit and control pressure, including attention-behavior separation. |
| 16 | Add observer-only meta-control attention over controller signals, read every step, and display a 100-step attention summary. |
| 17 | Implement trainer with controller state snapshots, meta-control observer logs, injected evidence ledger, decision replay records, and full parameter trajectories. |
| 18 | Make live diagnostics show both current values, decision reasons, attention behavior regime, and meta-control attention allocation for each degree of freedom. |
| 19 | Notebook must preserve artifacts needed for behavior analysis, not only final metrics. |
| 20 | Smoke validation checks mechanics, meta-control logs, and replayability before long runs. |
| 21 | Guarded 2000-step run performs adaptive search while logging controller decisions, meta-control attention, and attention regimes. |
| 22 | Late-window analysis evaluates trajectories, meta-control attention, attention regimes, and misdiagnoses, not just endpoint loss. |
| 23 | Attribution comparison separates real geometry from generic regularization using replay records, meta-control attention, and attention behavior. |
| 24 | Decision gate judges the evidence claim, including attention interpretability; it does not erase the optimization trajectory. |
| 25 | Revision loop maps bad decisions, bad meta-control attention, and bad attention regimes to failure labels and controller updates. |
| 26 | Longer/multi-seed confirmation happens only after the replayable adaptive search path is interpretable. |
