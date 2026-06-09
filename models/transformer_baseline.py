"""Controlled GPT-style transformer baseline for ERGT experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformerBaselineConfig:
    vocab_size: int
    context_length: int
    n_layers: int = 4
    n_heads: int = 4
    hidden_dim: int = 256
    ffn_dim: int = 1024
    dropout: float = 0.1
    bias: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")


class CausalSelfAttention(nn.Module):
    """Standard causal multi-head self-attention with an explicit replacement point."""

    def __init__(self, config: TransformerBaselineConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.hidden_dim = config.hidden_dim
        self.head_dim = config.hidden_dim // config.n_heads

        self.qkv_proj = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(hidden_dim, dim=-1)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        logits = self.compute_attention_logits(q, k)
        logits = self.apply_attention_mask(logits, attention_mask)
        attention_weights = F.softmax(logits, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        output = attention_weights @ v
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, hidden_dim)
        output = self.resid_dropout(self.out_proj(output))

        if need_weights:
            return output, attention_weights
        return output

    def compute_attention_logits(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute the replaceable standard attention logit term."""
        return (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

    def apply_attention_mask(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sequence_length = logits.size(-1)
        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=logits.device,
        ).tril()
        logits = logits.masked_fill(
            ~causal_mask.view(1, 1, sequence_length, sequence_length), -torch.inf
        )

        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError("attention_mask must have shape [batch, sequence]")
            padding_mask = attention_mask[:, None, None, :].to(
                dtype=torch.bool, device=logits.device
            )
            logits = logits.masked_fill(~padding_mask, -torch.inf)

        return logits

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, config: TransformerBaselineConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_dim, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.hidden_dim, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerBaselineConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_dim, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_dim, bias=config.bias)
        self.ffn = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attn_input = self.ln_1(hidden_states)
        attn_result = self.attn(
            attn_input, attention_mask=attention_mask, need_weights=need_weights
        )

        if need_weights:
            attn_output, attention_weights = attn_result
        else:
            attn_output = attn_result
            attention_weights = None

        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.ffn(self.ln_2(hidden_states))

        if need_weights:
            return hidden_states, attention_weights
        return hidden_states


class TransformerBaseline(nn.Module):
    """Small causal language model used as the controlled ERGT baseline."""

    def __init__(self, config: TransformerBaselineConfig | dict) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = TransformerBaselineConfig(**config)
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_ln = nn.LayerNorm(config.hidden_dim, bias=config.bias)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        return_attention_weights: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")

        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context_length "
                f"{self.config.context_length}"
            )

        positions = torch.arange(0, sequence_length, dtype=torch.long, device=input_ids.device)
        hidden_states = (
            self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        )
        hidden_states = self.dropout(hidden_states)

        all_hidden_states: list[torch.Tensor] = []
        all_attention_weights: list[torch.Tensor] = []

        for block in self.blocks:
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
            if return_attention_weights:
                hidden_states, attention_weights = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    need_weights=True,
                )
                all_attention_weights.append(attention_weights)
            else:
                hidden_states = block(hidden_states, attention_mask=attention_mask)

        hidden_states = self.final_ln(hidden_states)
        if return_hidden_states:
            all_hidden_states.append(hidden_states)

        logits = self.lm_head(hidden_states)
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))

        output: dict[str, torch.Tensor | list[torch.Tensor]] = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        if return_hidden_states:
            output["hidden_states"] = all_hidden_states
        if return_attention_weights:
            output["attention_weights"] = all_attention_weights
        return output

    def model_summary(self) -> dict[str, int | float | str]:
        total_params = sum(param.numel() for param in self.parameters())
        trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return {
            "type": "transformer_baseline",
            "vocab_size": self.config.vocab_size,
            "context_length": self.config.context_length,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
            "hidden_dim": self.config.hidden_dim,
            "ffn_dim": self.config.ffn_dim,
            "dropout": self.config.dropout,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
