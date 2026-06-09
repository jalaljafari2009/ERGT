"""ERGT-v1 model: RelationalGraph + EmergentDistance + GeoAttention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.geo_attention import GeoAttention, GeoAttentionConfig
from models.transformer_baseline import FeedForward, TransformerBaselineConfig


@dataclass(frozen=True)
class ERGTV1Config:
    vocab_size: int
    context_length: int
    n_layers: int = 4
    n_heads: int = 4
    hidden_dim: int = 256
    ffn_dim: int = 1024
    dropout: float = 0.1
    bias: bool = True
    attention: dict[str, Any] | None = None
    relational_graph: dict[str, Any] | None = None
    distance: dict[str, Any] | None = None

    def baseline_config(self) -> TransformerBaselineConfig:
        return TransformerBaselineConfig(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_dim=self.hidden_dim,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
            bias=self.bias,
        )


class ERGTBlock(nn.Module):
    def __init__(self, config: ERGTV1Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_dim, bias=config.bias)
        self.attn = GeoAttention(
            _geo_attention_config(config),
            relational_graph_config=config.relational_graph,
            distance_config=_distance_config(config),
        )
        self.ln_2 = nn.LayerNorm(config.hidden_dim, bias=config.bias)
        self.ffn = FeedForward(config.baseline_config())

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | dict[str, Any]:
        attn_input = self.ln_1(hidden_states)
        if return_diagnostics:
            attn_result = self.attn(
                attn_input,
                attention_mask=attention_mask,
                return_diagnostics=True,
            )
            hidden_states = hidden_states + attn_result["output"]
            hidden_states = hidden_states + self.ffn(self.ln_2(hidden_states))
            return {
                "hidden_states": hidden_states,
                "attention_weights": attn_result["attention_weights"],
                "distance": attn_result["distance"],
                "qk_logits": attn_result["qk_logits"],
                "diagnostics": attn_result["diagnostics"],
            }

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


class ERGTV1(nn.Module):
    """Causal language model used for Phase 3 ERGT-v1 comparisons."""

    def __init__(self, config: ERGTV1Config | dict[str, Any]) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = _coerce_project_config(config)
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([ERGTBlock(config) for _ in range(config.n_layers)])
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
        return_geometry_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor | list[Any]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")

        _, sequence_length = input_ids.shape
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
        all_geometry_diagnostics: list[dict[str, Any]] = []

        for block in self.blocks:
            if return_hidden_states:
                all_hidden_states.append(hidden_states)

            if return_geometry_diagnostics:
                block_result = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    return_diagnostics=True,
                )
                hidden_states = block_result["hidden_states"]
                if return_attention_weights:
                    all_attention_weights.append(block_result["attention_weights"])
                all_geometry_diagnostics.append(
                    {
                        "diagnostics": block_result["diagnostics"],
                        "attention_weights": block_result["attention_weights"],
                        "distance": block_result["distance"],
                        "qk_logits": block_result["qk_logits"],
                    }
                )
            elif return_attention_weights:
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

        output: dict[str, torch.Tensor | list[Any]] = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        if return_hidden_states:
            output["hidden_states"] = all_hidden_states
        if return_attention_weights:
            output["attention_weights"] = all_attention_weights
        if return_geometry_diagnostics:
            output["geometry_diagnostics"] = all_geometry_diagnostics
        return output

    def model_summary(self) -> dict[str, int | float | str | dict[str, Any] | None]:
        total_params = sum(param.numel() for param in self.parameters())
        trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return {
            "type": "ergt_v1",
            "vocab_size": self.config.vocab_size,
            "context_length": self.config.context_length,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
            "hidden_dim": self.config.hidden_dim,
            "ffn_dim": self.config.ffn_dim,
            "dropout": self.config.dropout,
            "attention": self.config.attention,
            "relational_graph": self.config.relational_graph,
            "distance": self.config.distance,
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


def _coerce_project_config(config: dict[str, Any]) -> ERGTV1Config:
    model = dict(config.get("model", config))
    model.pop("type", None)
    model.pop("positional_encoding", None)
    context_length = model.get("context_length")
    if context_length is None:
        context_length = config.get("dataset", {}).get("context_length")
    if context_length is None:
        raise ValueError("context_length must be provided in model or dataset config")
    return ERGTV1Config(
        vocab_size=int(model["vocab_size"]),
        context_length=int(context_length),
        n_layers=int(model.get("n_layers", 4)),
        n_heads=int(model.get("n_heads", 4)),
        hidden_dim=int(model.get("hidden_dim", 256)),
        ffn_dim=int(model.get("ffn_dim", 1024)),
        dropout=float(model.get("dropout", 0.1)),
        bias=bool(model.get("bias", True)),
        attention=config.get("attention"),
        relational_graph=config.get("relational_graph"),
        distance=config.get("distance"),
    )


def _geo_attention_config(config: ERGTV1Config) -> GeoAttentionConfig:
    attention = config.attention or {}
    alpha = attention.get("alpha", {})
    return GeoAttentionConfig(
        n_heads=config.n_heads,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        bias=config.bias,
        distance_mode=attention.get("distance_mode", "real_d"),
        head_sharing=attention.get("head_sharing", "shared_d"),
        alpha_mode=alpha.get("mode", "fixed"),
        alpha_initial_value=float(alpha.get("initial_value", 0.1)),
        alpha_non_negative=bool(alpha.get("non_negative", True)),
        gradient_mode=attention.get("gradient_mode", "grad_d"),
    )


def _distance_config(config: ERGTV1Config) -> dict[str, Any]:
    attention = config.attention or {}
    distance = dict(config.distance or {})
    distance["causal_runtime_distance"] = attention.get("causal_runtime_distance", False)
    return distance
