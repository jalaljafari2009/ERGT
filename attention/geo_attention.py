"""Geometry-biased attention for ERGT-v1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.causal_shortest_path_geometry import (
    causal_shortest_path_distance,
    finite_speed_edge_mask,
    pairwise_edge_distance,
)
from evaluation.relational_memory_observer import MemoryConfig, stable_memory_update
from geometry.emergent_distance import (
    EmergentDistance,
)
from layers.relational_graph import (
    RelationalGraph,
    make_random_graph_like,
    make_shuffled_graph,
    make_valid_edge_mask_like,
)
from models.transformer_baseline import TransformerBaselineConfig

DistanceMode = Literal[
    "real_d",
    "random_d",
    "shuffled_d",
    "zero_d",
    "real_stable_causal_d",
    "random_stable_causal_d",
    "shuffled_stable_causal_d",
    "instantaneous_real_d",
    "pairwise_real_d",
    "no_memory_real_d",
]
AlphaMode = Literal["fixed", "trainable"]
GradientMode = Literal["grad_d", "detached_d"]
HeadSharing = Literal["shared_d", "per_head_d"]

V1_DISTANCE_MODES = {"real_d", "random_d", "shuffled_d", "zero_d"}
V2_DISTANCE_MODES = {
    "real_stable_causal_d",
    "random_stable_causal_d",
    "shuffled_stable_causal_d",
    "instantaneous_real_d",
    "pairwise_real_d",
    "no_memory_real_d",
}
STABLE_MEMORY_DISTANCE_MODES = {
    "real_stable_causal_d",
    "random_stable_causal_d",
    "shuffled_stable_causal_d",
    "pairwise_real_d",
}


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
    memory_decay: float = 0.7
    memory_eta: float = 0.3
    memory_gate_floor: float = 0.05
    memory_min_context_edges: int = 2
    max_causal_step: int | None = 1

    def __post_init__(self) -> None:
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if self.distance_mode not in V1_DISTANCE_MODES | V2_DISTANCE_MODES:
            raise ValueError(f"unsupported distance_mode: {self.distance_mode}")
        if self.head_sharing not in {"shared_d", "per_head_d"}:
            raise ValueError(f"unsupported head_sharing: {self.head_sharing}")
        if self.alpha_mode not in {"fixed", "trainable"}:
            raise ValueError(f"unsupported alpha_mode: {self.alpha_mode}")
        if self.alpha_warmup_steps < 0:
            raise ValueError("alpha_warmup_steps must be non-negative")
        if self.gradient_mode not in {"grad_d", "detached_d"}:
            raise ValueError(f"unsupported gradient_mode: {self.gradient_mode}")
        MemoryConfig(
            decay=self.memory_decay,
            eta=self.memory_eta,
            gate_floor=self.memory_gate_floor,
            min_context_edges=self.memory_min_context_edges,
        ).validate()
        if self.max_causal_step is not None and self.max_causal_step <= 0:
            raise ValueError("max_causal_step must be positive or None")


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
        self.relational_graph_config = graph_config

        self.relational_graph = RelationalGraph(graph_config)
        self.emergent_distance = EmergentDistance(distance_config or {})
        self.memory_config = MemoryConfig(
            decay=self.config.memory_decay,
            eta=self.config.memory_eta,
            gate_floor=self.config.memory_gate_floor,
            min_context_edges=self.config.memory_min_context_edges,
        )

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
            memory_decay=float(attention.get("memory", {}).get("decay", 0.7)),
            memory_eta=float(attention.get("memory", {}).get("eta", 0.3)),
            memory_gate_floor=float(attention.get("memory", {}).get("gate_floor", 0.05)),
            memory_min_context_edges=int(
                attention.get("memory", {}).get("min_context_edges", 2)
            ),
            max_causal_step=attention.get("max_causal_step", 1),
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
        geometry_memory: torch.Tensor | None = None,
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
        distance_result = self.compute_distance(
            hidden_states,
            attention_mask=attention_mask,
            geometry_memory=geometry_memory,
            return_memory=True,
        )
        distance = distance_result["distance"]
        updated_geometry_memory = distance_result["geometry_memory"]
        geometry_metadata = distance_result["metadata"]
        if self.config.gradient_mode == "detached_d":
            distance = distance.detach()
            if updated_geometry_memory is not None:
                updated_geometry_memory = updated_geometry_memory.detach()

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
                "geometry_memory": updated_geometry_memory.detach()
                if updated_geometry_memory is not None
                else None,
                "diagnostics": self.diagnostics(
                    qk_logits,
                    distance,
                    alpha,
                    geometry_metadata=geometry_metadata,
                ),
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
        geometry_memory: torch.Tensor | None = None,
        return_memory: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        if self.config.distance_mode == "zero_d":
            batch_size, sequence_length, _ = hidden_states.shape
            distance = torch.zeros(
                batch_size,
                1,
                sequence_length,
                sequence_length,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            result = {
                "distance": distance,
                "geometry_memory": None,
                "metadata": self._geometry_metadata(
                    pipeline="zero_distance",
                    memory_used=False,
                    shortest_path=False,
                ),
            }
            return result if return_memory else distance

        if self.config.distance_mode in V2_DISTANCE_MODES:
            result = self.compute_v2_distance(
                hidden_states,
                attention_mask=attention_mask,
                geometry_memory=geometry_memory,
            )
            return result if return_memory else result["distance"]

        graph = self.compute_control_graph(hidden_states, attention_mask=attention_mask)
        distance = self.emergent_distance(graph, attention_mask=attention_mask)
        result = {
            "distance": distance,
            "geometry_memory": None,
            "metadata": self._geometry_metadata(
                pipeline="v1_pairwise_distance",
                memory_used=False,
                shortest_path=False,
            ),
        }
        return result if return_memory else distance

    def compute_v2_distance(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        geometry_memory: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Build stable causal geometry for GeoAttention v2."""

        graph = self.compute_control_graph(hidden_states, attention_mask=attention_mask)
        valid_edge_mask = make_valid_edge_mask_like(graph, attention_mask=attention_mask)
        valid_edge_mask = self._prepare_mask(valid_edge_mask, graph)
        direct_edge_mask = finite_speed_edge_mask(
            valid_edge_mask,
            max_causal_step=self.config.max_causal_step,
        )
        distance_mode = self.config.distance_mode
        memory_used = False
        updated_memory = None

        if distance_mode == "no_memory_real_d":
            source_graph = self._valid_only(graph, valid_edge_mask)
            distance = causal_shortest_path_distance(
                source_graph,
                direct_edge_mask,
                epsilon=self.emergent_distance.config.epsilon,
            )
            pipeline = "H -> W_real -> finite-speed D_causal"
            shortest_path = True
        else:
            update = stable_memory_update(
                hidden_states,
                graph,
                valid_edge_mask,
                attention_mask=attention_mask,
                graph_config=self.relational_graph_config,
                memory_config=self.memory_config,
            )
            stable_update = update["stable_update"]
            if distance_mode == "instantaneous_real_d":
                source_graph = stable_update
                pipeline = "H -> stable_update -> finite-speed D_causal"
                shortest_path = True
            else:
                source_graph, memory_used = self._memory_step(
                    stable_update,
                    geometry_memory,
                    valid_edge_mask,
                )
                updated_memory = source_graph
                if distance_mode == "pairwise_real_d":
                    pipeline = "H -> W_t memory -> finite-speed pairwise D"
                    shortest_path = False
                else:
                    pipeline = "H -> W_t memory -> finite-speed D_causal"
                    shortest_path = True

            if distance_mode == "pairwise_real_d":
                distance = pairwise_edge_distance(
                    source_graph,
                    direct_edge_mask,
                    epsilon=self.emergent_distance.config.epsilon,
                )
            else:
                distance = causal_shortest_path_distance(
                    source_graph,
                    direct_edge_mask,
                    epsilon=self.emergent_distance.config.epsilon,
                )

        return {
            "distance": self.emergent_distance.normalize_precomputed(distance),
            "geometry_memory": updated_memory,
            "metadata": self._geometry_metadata(
                pipeline=pipeline,
                memory_used=memory_used,
                shortest_path=shortest_path,
            ),
        }

    def compute_control_graph(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return real or W-level control graph before distance construction."""

        graph = self.relational_graph(hidden_states, attention_mask=attention_mask)
        if self.config.distance_mode in {
            "real_d",
            "real_stable_causal_d",
            "instantaneous_real_d",
            "pairwise_real_d",
            "no_memory_real_d",
        }:
            return graph
        valid_edge_mask = make_valid_edge_mask_like(graph, attention_mask=attention_mask)
        if self.config.distance_mode in {"random_d", "random_stable_causal_d"}:
            return make_random_graph_like(graph, valid_edge_mask=valid_edge_mask)
        if self.config.distance_mode in {"shuffled_d", "shuffled_stable_causal_d"}:
            return make_shuffled_graph(graph, valid_edge_mask=valid_edge_mask)
        raise ValueError(f"unsupported distance_mode: {self.config.distance_mode}")

    def apply_geometry_bias(
        self,
        qk_logits: torch.Tensor,
        distance: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        if torch.all(alpha == 0):
            return qk_logits
        safe_distance = self._distance_for_bias(distance)
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
        geometry_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        finite_distance = distance[torch.isfinite(distance)]
        finite_qk = qk_logits[torch.isfinite(qk_logits)]
        mean_abs_qk = finite_qk.abs().mean() if finite_qk.numel() else torch.tensor(float("nan"))
        if torch.all(alpha == 0) or finite_distance.numel() == 0:
            mean_abs_geo = torch.tensor(0.0, device=qk_logits.device)
        else:
            mean_abs_geo = (alpha * finite_distance).abs().mean()
        target_alpha = self.target_alpha()
        warmup_factor = self.alpha_warmup_factor()
        result: dict[str, Any] = {
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
        if geometry_metadata is not None:
            result.update(geometry_metadata)
        return result

    def uses_geometry_memory(self) -> bool:
        return self.config.distance_mode in STABLE_MEMORY_DISTANCE_MODES

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

    def _memory_step(
        self,
        stable_update: torch.Tensor,
        previous_memory: torch.Tensor | None,
        valid_edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        previous_memory_valid = (
            previous_memory is not None
            and previous_memory.shape == stable_update.shape
            and previous_memory.device == stable_update.device
        )
        if not previous_memory_valid:
            return stable_update.clamp(0.0, 1.0), False

        memory = (
            self.memory_config.decay * previous_memory.to(dtype=stable_update.dtype)
            + self.memory_config.eta * stable_update
        )
        memory = memory.clamp(0.0, 1.0)
        return self._valid_only(memory, valid_edge_mask), True

    def _geometry_metadata(
        self,
        *,
        pipeline: str,
        memory_used: bool,
        shortest_path: bool,
    ) -> dict[str, Any]:
        return {
            "distance_mode": self.config.distance_mode,
            "geometry_pipeline": pipeline,
            "geometry_version": "v2" if self.config.distance_mode in V2_DISTANCE_MODES else "v1",
            "geometry_memory_used": memory_used,
            "causal_shortest_path": shortest_path,
            "max_causal_step": self.config.max_causal_step,
        }

    def _distance_for_bias(self, distance: torch.Tensor) -> torch.Tensor:
        finite_mask = torch.isfinite(distance)
        if not bool(finite_mask.any().item()):
            return torch.zeros_like(distance)
        finite_values = distance[finite_mask]
        fill_value = finite_values.max().detach()
        return torch.where(finite_mask, distance, fill_value)

    def _valid_only(self, graph: torch.Tensor, valid_edge_mask: torch.Tensor) -> torch.Tensor:
        valid_edge_mask = self._prepare_mask(valid_edge_mask, graph)
        return torch.where(valid_edge_mask, graph, torch.zeros_like(graph))

    def _prepare_mask(self, valid_edge_mask: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        valid_edge_mask = valid_edge_mask.to(dtype=torch.bool, device=graph.device)
        if valid_edge_mask.shape == graph.shape:
            return valid_edge_mask
        if valid_edge_mask.size(1) == 1:
            return valid_edge_mask.expand_as(graph)
        raise ValueError("valid_edge_mask must match graph shape or be head-shared")

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
