# Baseline Transformer Specification

## 1. Purpose

The baseline transformer is the experimental control for ERGT-v1.

It is not a novelty contribution and it should not be implemented as a large
research project by itself. Its purpose is to provide a clean, reproducible, and
modifiable transformer reference so that GeoAttention can be compared under
controlled conditions.

The baseline answers one question:

```text
How does a standard transformer perform under the exact same training and
evaluation setup used for ERGT-v1?
```

## 2. Baseline Principle

Do not compare ERGT against an unfair or uncontrolled model.

The baseline must match ERGT on:

- Dataset
- Tokenizer
- Context length
- Model size
- Number of layers
- Number of heads
- Hidden dimension
- Feed-forward dimension
- Optimizer
- Training budget
- Evaluation protocol
- Seed protocol

The only intended difference in Phase 3 is the attention mechanism.

## 3. Recommended Implementation Strategy

Use a small GPT-style causal transformer pattern.

Acceptable sources:

- A nanoGPT-style implementation.
- A minimal PyTorch GPT implementation.
- A HuggingFace GPT-style model initialized from scratch, if attention can be
  modified cleanly.

Avoid using a pretrained model as the primary baseline. Pretrained GPT-2 may be
used for reference analysis, but not for the main controlled comparison unless
ERGT receives an equivalent pretraining setup.

## 4. Model Type

The Phase 0 baseline is:

```text
small causal language model
```

Required components:

- Token embedding.
- Positional encoding or positional embedding.
- Stacked transformer blocks.
- Causal multi-head self-attention.
- Feed-forward network.
- Residual connections.
- Layer normalization.
- Final language-model head.

Recommended block style:

```text
X -> LayerNorm -> CausalSelfAttention -> Residual
  -> LayerNorm -> FeedForward -> Residual
```

This pre-norm structure is stable and easy to extend.

## 5. Attention Contract

The attention module must expose a clean replacement point for Phase 3.

Standard attention:

```text
logits = QK^T / sqrt(head_dim)
logits = apply_causal_mask(logits)
A = softmax(logits)
Y = A V
```

Expected tensor shapes:

```text
hidden_states: [batch, sequence, hidden_dim]
q, k, v:       [batch, heads, sequence, head_dim]
logits:        [batch, heads, sequence, sequence]
attention:     [batch, heads, sequence, sequence]
output:        [batch, sequence, hidden_dim]
```

The implementation should make it possible to replace:

```text
logits = QK^T / sqrt(head_dim)
```

with:

```text
logits = QK^T / sqrt(head_dim) - alphaD
```

without changing unrelated parts of the model.

## 6. Positional Information

The baseline may use learned positional embeddings or a standard positional
method, but the choice must remain constant across baseline and ERGT-v1.

Recommended first version:

```text
learned absolute positional embeddings
```

Reason:

- Simple.
- Easy to reproduce.
- Does not introduce another geometry method that confounds the first proof.

Later comparisons may include RoPE, ALiBi, or relative position bias as stronger
baselines, but the first proof should start with the simplest controlled
condition.

## 7. Suggested Initial Config

The exact values may be adjusted for hardware, but a first debug configuration
should be small enough for repeated ablations.

Example debug config:

```json
{
  "model_type": "gpt_causal_baseline",
  "vocab_size": "tokenizer_defined",
  "context_length": 128,
  "n_layers": 4,
  "n_heads": 4,
  "hidden_dim": 256,
  "ffn_dim": 1024,
  "dropout": 0.1,
  "bias": true
}
```

Example proof-stage config:

```json
{
  "model_type": "gpt_causal_baseline",
  "vocab_size": "tokenizer_defined",
  "context_length": 256,
  "n_layers": 6,
  "n_heads": 8,
  "hidden_dim": 512,
  "ffn_dim": 2048,
  "dropout": 0.1,
  "bias": true
}
```

Do not treat these numbers as fixed theory. They are starting points. The final
choice must fit available compute and be reused exactly for ERGT-v1.

## 8. Dataset

Recommended first dataset:

```text
WikiText-2
```

Dataset requirements:

- Fixed train/validation split.
- Deterministic preprocessing.
- Same tokenization for all model variants.
- Same sequence packing strategy for all model variants.

The baseline run must save dataset metadata into the run directory.

## 9. Tokenizer

Recommended first choices:

- GPT-2 tokenizer for convenience and comparability.
- A project-local BPE tokenizer if reproducibility is preferred.

The tokenizer must be identical for all Phase 0 and Phase 3 comparisons.

Record:

- Tokenizer name or path.
- Vocabulary size.
- Special tokens.
- Whether tokenization is cached.

## 10. Training Objective

Use standard next-token prediction:

```text
L = cross_entropy(logits[:, :-1], tokens[:, 1:])
```

The baseline objective should not include any graph, geometry, memory, or
spectral loss.

## 11. Optimizer and Schedule

Recommended optimizer:

```text
AdamW
```

Recommended recorded parameters:

- Learning rate.
- Betas.
- Weight decay.
- Gradient clipping.
- Warmup steps.
- Max steps or token budget.
- Scheduler type.

All of these must be reused for ERGT-v1 unless a documented stability issue
requires a separate controlled run.

## 12. Logging Requirements

Every baseline run must produce:

```text
runs/phase0_baseline/config.json
runs/phase0_baseline/train_log.jsonl
runs/phase0_baseline/baseline_results.json
runs/phase0_baseline/model_summary.json
```

Required `train_log.jsonl` fields:

- Step.
- Training loss.
- Validation loss, when evaluated.
- Learning rate.
- Tokens processed.
- Tokens per second.
- Peak memory, when available.
- Seed.

Required `baseline_results.json` fields:

- Final training loss.
- Best validation loss.
- Final validation loss.
- Perplexity.
- Total training tokens.
- Total wall-clock time, if available.
- Average tokens per second.
- Peak memory, if available.
- Config hash or run identifier.

## 13. Checkpoint Requirements

Save at least:

```text
runs/phase0_baseline/checkpoints/best.pt
runs/phase0_baseline/checkpoints/last.pt
```

Checkpoints should include:

- Model state dict.
- Optimizer state dict.
- Scheduler state dict, if used.
- Step.
- Config.
- Random seed.

## 14. Reproducibility Requirements

A baseline run is not complete unless it can be reproduced from saved
configuration.

Required:

- Set Python, NumPy, and framework seeds where applicable.
- Save the exact config used.
- Save dependency versions if practical.
- Save the git commit hash if the project is inside git.
- Avoid hidden defaults that are not recorded.

## 15. Acceptance Criteria

Phase 0 is complete when:

- The baseline trains without numerical failure.
- Validation loss decreases from the initial value.
- Perplexity is computed correctly.
- Runtime and memory are recorded.
- The run can be reproduced from config.
- The attention module can be replaced or augmented for GeoAttention.

## 16. Failure Criteria

Phase 0 must be revised if:

- Training diverges.
- Evaluation is not reproducible.
- Dataset or tokenizer differs from later ERGT runs.
- The model implementation hides attention in a way that makes Phase 3
  comparison difficult.
- Metrics are missing or not machine-readable.

## 17. Non-Goals

The baseline phase should not attempt to:

- Beat published GPT-2 results.
- Train a production-scale model.
- Tune architecture extensively.
- Introduce memory mechanisms.
- Introduce graph or geometry losses.
- Use pretrained weights for the primary comparison.

The baseline should be strong enough to be fair and simple enough to be
understood.

## 18. Relationship to Later Phases

Phase 1 attaches a relational graph observer to the baseline.

Phase 2 derives distance from the observed graph.

Phase 3 replaces or augments the attention logits with the geometry term.

Therefore the baseline code must be written with these extension points in mind:

```text
hidden states available for graph extraction
attention logits available for geometry bias
attention module replaceable without changing the rest of the model
```

The cleaner this baseline is, the more credible the ERGT comparison will be.
