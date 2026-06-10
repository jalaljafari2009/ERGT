"""Geometry-biased attention for ERGT-v1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry.emergent_distance import (
    EmergentDistance,
    make_random_distance_like,
    make_shuffled_distance,
)
from layers.relational_graph import RelationalGraph
from models.transformer_baseline import TransformerBaselineConfig

DistanceMode = Literal["real_d", "random_d", "shuffled_d", "zero_d"]
AlphaMode = Literal["fixed", "trainable"]
GradientMode = Literal["grad_d", "detached_d"]
HeadSharing = Literal["shared_d", "per_head_d"]


@dataclass(frozen=True)
class GeoAttentionConfig:
    n_heads: int
    hidden_dim: int
    dropout: float = 0.1
    bias: bool = True
    distance_mode: DistanceMode = "real_d"
    head_sharing: HeadSharing = "shared_d"
    alpha_mode: AlphaMode = "fixed"
    alpha_initial_value: float = 0.1
    alpha_non_negative: bool = True
    alpha_warmup_steps: int = 0
    gradient_mode: GradientMode = "grad_d"

    def __post_init__(self) -> None:
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if self.distance_mode not in {"real_d", "random_d", "shuffled_d", "zero_d"}:
            raise ValueError(f"unsupported distance_mode: {self.distance_mode}")
        if self.head_sharing not in {"shared_d", "per_head_d"}:
            raise ValueError(f"unsupported head_sharing: {self.head_sharing}")
        if self.alpha_mode not in {"fixed", "trainable"}:
            raise ValueError(f"unsupported alpha_mode: {self.alpha_mode}")
        if self.alpha_warmup_steps < 0:
            raise ValueError("alpha_warmup_steps must be non-negative")
        if self.gradient_mode not in {"grad_d", "detached_d"}:
            raise ValueError(f"unsupported gradient_mode: {self.gradient_mode}")


class GeoAttention(nn.Module):
    """Causal multi-head self-attention with an induced distance penalty."""

    def __init__(
        self,
        attention_config: GeoAttentionConfig | dict | TransformerBaselineConfig,
        relational_graph_config: dict[str, Any] | None = None,
        distance_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = self._coerce_config(attention_config)
        self.n_heads = self.config.n_heads
        self.hidden_dim = self.config.hidden_dim
        self.head_dim = self.hidden_dim // self.n_heads

        graph_config = dict(relational_graph_config or {})
        if graph_config.get("diagonal_policy") == "keep_for_distance":
            graph_config["diagonal_policy"] = "keep"

        self.relational_graph = RelationalGraph(graph_config)
        self.emergent_distance = EmergentDistance(distance_config or {})

        self.qkv_proj = nn.Linear(self.hidden_dim, 3 * self.hidden_dim, bias=self.config.bias)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.config.bias)
        self.attn_dropout = nn.Dropout(self.config.dropout)
        self.resid_dropout = nn.Dropout(self.config.dropout)

        if self.config.alpha_mode == "trainable":
            initial = max(float(self.config.alpha_initial_value), 1e-8)
            if self.config.alpha_non_negative:
                raw_initial = math.log(math.exp(initial) - 1.0)
            else:
                raw_initial = initial
            self.raw_alpha = nn.Parameter(torch.tensor(raw_initial, dtype=torch.float32))
        else:
            self.register_buffer(
                "fixed_alpha",
                torch.tensor(float(self.config.alpha_initial_value), dtype=torch.float32),
                persistent=True,
            )
        self.register_buffer(
            "training_step",
            torch.tensor(0, dtype=torch.long),
            persistent=False,
        )

    @classmethod
    def from_project_config(cls, project_config: dict[str, Any]) -> GeoAttention:
        model = project_config["model"]
        attention = project_config["attention"]
        alpha = attention.get("alpha", {})
        attention_config = GeoAttentionConfig(
            n_heads=int(model["n_heads"]),
            hidden_dim=int(model["hidden_dim"]),
            dropout=float(model.get("dropout", 0.1)),
            bias=bool(model.get("bias", True)),
            distance_mode=attention.get("distance_mode", "real_d"),
            head_sharing=attention.get("head_sharing", "shared_d"),
            alpha_mode=alpha.get("mode", "fixed"),
            alpha_initial_value=float(alpha.get("initial_value", 0.1)),
            alpha_non_negative=bool(alpha.get("non_negative", True)),
            alpha_warmup_steps=int(alpha.get("warmup_steps", 0)),
            gradient_mode=attention.get("gradient_mode", "grad_d"),
        )
        return cls(
            attention_config,
            relational_graph_config=project_config.get("relational_graph"),
            distance_config={
                **project_config.get("distance", {}),
                "causal_runtime_distance": attention.get("causal_runtime_distance", False),
            },
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | dict[str, Any]:
        if hidden_states.dim() != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden_dim]")

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(hidden_dim, dim=-1)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        qk_logits = self.compute_attention_logits(q, k)
        distance = self.compute_distance(hidden_states, attention_mask=attention_mask)
        if self.config.gradient_mode == "detached_d":
            distance = distance.detach()

        distance = self._broadcast_distance(distance, qk_logits)
        alpha = self.alpha()
        geo_logits = self.apply_geometry_bias(qk_logits, distance, alpha)
        masked_logits = self.apply_attention_mask(geo_logits, attention_mask)
        attention_weights = F.softmax(masked_logits, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        output = attention_weights @ v
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, hidden_dim)
        output = self.resid_dropout(self.out_proj(output))

        if return_diagnostics:
            return {
                "output": output,
                "attention_weights": attention_weights.detach(),
                "distance": distance.detach(),
                "qk_logits": qk_logits.detach(),
                "alpha": alpha.detach(),
                "diagnostics": self.diagnostics(qk_logits, distance, alpha),
            }
        if need_weights:
            return output, attention_weights
        return output

    def compute_attention_logits(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

    def compute_distance(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.distance_mode == "zero_d":
            batch_size, sequence_length, _ = hidden_states.shape
            return torch.zeros(
                batch_size,
                1,
                sequence_length,
                sequence_length,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        graph = self.relational_graph(hidden_states, attention_mask=attention_mask)
        distance = self.emergent_distance(graph, attention_mask=attention_mask)

        if self.config.distance_mode == "real_d":
            return distance

        if self.config.distance_mode == "random_d":
            return make_random_distance_like(distance)
        if self.config.distance_mode == "shuffled_d":
            return make_shuffled_distance(distance)
        raise ValueError(f"unsupported distance_mode: {self.config.distance_mode}")

    def apply_geometry_bias(
        self,
        qk_logits: torch.Tensor,
        distance: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        if torch.all(alpha == 0):
            return qk_logits
        safe_distance = torch.where(torch.isfinite(distance), distance, torch.zeros_like(distance))
        geo_term = alpha * safe_distance
        return qk_logits - geo_term

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

    def alpha(self) -> torch.Tensor:
        alpha = self.target_alpha()
        if self.config.alpha_warmup_steps <= 0:
            return alpha
        return alpha * self.alpha_warmup_factor()

    def target_alpha(self) -> torch.Tensor:
        if self.config.alpha_mode == "trainable":
            if self.config.alpha_non_negative:
                return F.softplus(self.raw_alpha)
            return self.raw_alpha
        alpha = self.fixed_alpha
        if self.config.alpha_non_negative:
            alpha = torch.clamp(alpha, min=0.0)
        return alpha

    def alpha_warmup_factor(self) -> torch.Tensor:
        if self.config.alpha_warmup_steps <= 0:
            return torch.tensor(1.0, dtype=torch.float32, device=self.training_step.device)
        step = self.training_step.to(dtype=torch.float32)
        return torch.clamp(step / float(self.config.alpha_warmup_steps), min=0.0, max=1.0)

    def set_training_step(self, step: int) -> None:
        self.training_step.fill_(max(int(step), 0))

    def diagnostics(
        self,
        qk_logits: torch.Tensor,
        distance: torch.Tensor,
        alpha: torch.Tensor,
    ) -> dict[str, float]:
        finite_distance = distance[torch.isfinite(distance)]
        finite_qk = qk_logits[torch.isfinite(qk_logits)]
        mean_abs_qk = finite_qk.abs().mean() if finite_qk.numel() else torch.tensor(float("nan"))
        if torch.all(alpha == 0) or finite_distance.numel() == 0:
            mean_abs_geo = torch.tensor(0.0, device=qk_logits.device)
        else:
            mean_abs_geo = (alpha * finite_distance).abs().mean()
        target_alpha = self.target_alpha()
        warmup_factor = self.alpha_warmup_factor()
        return {
            "alpha": float(alpha.detach().cpu().item()),
            "target_alpha": float(target_alpha.detach().cpu().item()),
            "alpha_warmup_steps": float(self.config.alpha_warmup_steps),
            "alpha_warmup_factor": float(warmup_factor.detach().cpu().item()),
            "mean_abs_qk": float(mean_abs_qk.detach().cpu().item()),
            "mean_abs_geo": float(mean_abs_geo.detach().cpu().item()),
            "geo_to_qk_ratio": float(
                (mean_abs_geo / (mean_abs_qk.abs() + 1e-12)).detach().cpu().item()
            ),
            "distance_mean": float(finite_distance.mean().detach().cpu().item())
            if finite_distance.numel()
            else float("nan"),
            "distance_std": float(finite_distance.std(unbiased=False).detach().cpu().item())
            if finite_distance.numel()
            else float("nan"),
        }

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)

    def _broadcast_distance(self, distance: torch.Tensor, qk_logits: torch.Tensor) -> torch.Tensor:
        if distance.shape == qk_logits.shape:
            return distance
        if distance.dim() != 4:
            raise ValueError("distance must have shape [batch, heads, sequence, sequence]")
        if distance.size(0) != qk_logits.size(0) or distance.shape[-2:] != qk_logits.shape[-2:]:
            raise ValueError("distance shape must match attention batch and sequence dimensions")
        if distance.size(1) == 1:
            return distance.expand(qk_logits.shape)
        if distance.size(1) != qk_logits.size(1):
            raise ValueError("distance heads must be 1 or match attention heads")
        return distance

    def _coerce_config(
        self,
        config: GeoAttentionConfig | dict | TransformerBaselineConfig,
    ) -> GeoAttentionConfig:
        if isinstance(config, GeoAttentionConfig):
            return config
        if isinstance(config, TransformerBaselineConfig):
            return GeoAttentionConfig(
                n_heads=config.n_heads,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                bias=config.bias,
            )
        if isinstance(config, dict):
            allowed = {field.name for field in GeoAttentionConfig.__dataclass_fields__.values()}
            return GeoAttentionConfig(
                **{key: value for key, value in config.items() if key in allowed}
            )
        raise TypeError(f"unsupported config type: {type(config)!r}")
