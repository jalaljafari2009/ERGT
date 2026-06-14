# Attention Rigidity and Collapse Monitor

## 1. Purpose

This stage makes attention collapse observable before alpha, memory, gate floor,
reachability, or distance scale are allowed to adapt more aggressively.

The scientific concern is:

```text
geometry must be allowed to compete, and collapse signals must become visible
controller pressure instead of silent failure
```

This stage is diagnostic-only at implementation time. It does not change
attention logits, loss, gradients, alpha, memory, or normalization.

After the adaptive-search revision, its outputs also become behavioral evidence
for later controllers. The metrics still do not directly change attention in
this stage, but later stages must use them to infer whether the search is
moving toward a useful operating region or toward collapse, uniformity, lock-in,
or control-like behavior.

## 2. Risk Covered

Primary risk:

```text
R2: Distance Signal Flattening and Geometry Strength Risk
```

Secondary risk:

```text
an adaptive controller may mistake attention collapse for useful geometry
an adaptive controller may miss the useful region because it sees only loss
```

## 3. Required Metrics

Every geometry-diagnostic run should expose:

```text
attention_entropy
attention_entropy_normalized
attention_entropy_drop
mean_max_probability
valid_mean_max_probability
attention_sparsity_0_01
attention_sparsity_0_001
valid_attention_sparsity_0_01
valid_attention_sparsity_0_001
head_attention_diversity
head_collapse_risk
layer_attention_diversity
layer_collapse_risk
geometry_takeover_score
geo_qk_risk
entropy_risk
max_probability_risk
valid_sparsity_risk
rigidity_risk
collapse_risk
severe_attention_collapse_detected
```

Interpretation:

- `attention_entropy_normalized`: entropy relative to causal row capacity.
- `attention_entropy_drop`: loss of normalized entropy.
- `valid_mean_max_probability`: max attention probability on causal-valid edges.
- `head_attention_diversity`: dissimilarity among heads.
- `layer_attention_diversity`: dissimilarity among layers.
- `geometry_takeover_score`: how close geometry is to dominating QK scale.
- `rigidity_risk`: pressure from concentrated, sparse, or head-collapsed attention.
- `collapse_risk`: max of rigidity and geometry-takeover risk.

Behavioral interpretation:

- useful regime: loss trend improves while attention remains diverse and real
  differs from random and shuffled controls;
- uniformity drift: entropy remains high but attention does not specialize and
  geometry has little effect;
- lock-in: max probability, sparsity, or head/layer collapse rises while
  diversity falls;
- geometry takeover: `geo/qk` and takeover score rise faster than loss and
  control separation can justify;
- control-like behavior: random or shuffled attention regimes match the real
  regime.

## 4. Machine Artifacts

This stage adds:

```text
evaluation/attention_rigidity_monitor.py
experiments/create_attention_rigidity_monitor_report.py
tests/test_attention_rigidity_monitor.py
```

It also extends:

```text
evaluation/attention_metrics.py
evaluation/unified_telemetry_schema.py
experiments/progress_logging.py
```

Default report:

```text
runs/contracts/attention_rigidity_monitor.json
```

Usage:

```bash
python experiments/create_attention_rigidity_monitor_report.py
```

## 5. Live Telemetry

Long runs should now print and store enough attention-collapse evidence for
controller decisions:

```text
ent
nEnt
maxp
hDiv
geo/qk
gTake
gRisk
rigid
collapse
```

These fields are restraint evidence. They should not prevent geometry growth by
themselves; they change controller pressure, budget, shrink/freeze decisions, or
revision labels. Only declared safety hard stops such as severe attention
collapse may invalidate or abort a run.

From stage 11 onward, these fields are also search evidence. Controllers should
use them to choose between increasing injection, smoothing noise, releasing
rigidity, changing gate floor, changing reach, or reallocating joint budget.

## 6. Exit Criteria

This stage is complete when:

```text
required attention-collapse fields are declared in the unified schema
required fields are emitted by the monitor
rigid synthetic attention has higher collapse risk than spread attention
geometry-takeover synthetic attention has higher geo/qk risk
identical heads have higher head-collapse risk than diverse heads
live progress logging exposes the key risks
GeoAttention diagnostics expose these metrics without changing attention
```

Next stage:

```text
Control-Family Fairness Audit v2
```
