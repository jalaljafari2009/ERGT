# Control-Family Fairness Audit v2

## 1. Purpose

This stage proves that control families differ only in the intended relational
mechanism before any adaptive training run is interpreted.

The scientific concern is:

```text
real geometry must not win or lose because controls used a different data path,
batch path, random source, memory source, or distance policy
```

This stage is diagnostic-only. It does not train a model and does not change
attention, loss, alpha, memory, reachability, or normalization.

## 2. Families Audited

```text
real_stable_causal_d
random_stable_causal_d
shuffled_stable_causal_d
no_memory_real_d
instantaneous_real_d
```

## 3. Required Checks

Every control-family fairness report must verify:

```text
same_data_path
same_batch_path
same_model_shape
same_graph_policy
same_distance_policy
same_memory_policy_for_memory_modes
control_rng_isolated_random
control_rng_isolated_shuffled
control_rng_does_not_touch_global_rng
no_cross_family_real_distance_reuse
no_cross_family_real_memory_reuse
random_and_shuffled_built_before_distance
finite_distance_regions_match
random_distance_not_real_distance
shuffled_distance_not_real_distance
matched_normalization_policy
matched_clipping_policy
matched_distance_diagonal_policy
matched_graph_diagonal_policy
geo_to_qk_ratio_finite
geo_to_qk_ratio_same_qk_denominator
control_isolation_gate_passed
```

## 4. Machine Artifacts

This stage adds:

```text
evaluation/control_family_fairness_audit.py
experiments/create_control_family_fairness_audit_report.py
tests/test_control_family_fairness_audit.py
```

Default report:

```text
runs/contracts/control_family_fairness_audit_v2.json
```

Usage:

```bash
python experiments/create_control_family_fairness_audit_report.py
```

## 5. Exit Criteria

This stage is complete when:

```text
all five control families share data and batch fingerprints
random and shuffled controls use isolated local RNG
global torch RNG is unchanged by control construction
real distance and real memory are not reused across control families
normalization, clipping, and diagonal policies match
geo_to_qk_ratio is computed against the same QK denominator
the previous GeoAttention v2 control-isolation gate still passes
```

Next stage:

```text
Loss-Slope and Trend Analyzer
```
