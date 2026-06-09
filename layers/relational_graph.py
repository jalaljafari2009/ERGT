"""Relational graph construction for ERGT.

Phase 1 uses this module as an observer: it extracts relation matrices from
hidden states without changing the baseline model's computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

DiagonalPolicy = Literal["keep", "zero", "mask"]


@dataclass(frozen=True)
class RelationalGraphConfig:
    kernel: str = "sigmoid_dot_sqrt_d"
    graph_heads: int = 1
    normalize_hidden: bool = False
    temperature: float | None = None
    diagonal_policy: DiagonalPolicy = "keep"

    def __post_init__(self) -> None:
        if self.kernel not in {"sigmoid_dot_sqrt_d", "sigmoid_cosine"}:
            raise ValueError(f"unsupported relational graph kernel: {self.kernel}")
        if self.graph_heads != 1:
            raise ValueError("Phase 1 supports only graph_heads=1")
        if self.temperature is not None and self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.diagonal_policy not in {"keep", "zero", "mask"}:
            raise ValueError(f"unsupported diagonal_policy: {self.diagonal_policy}")


class RelationalGraph(nn.Module):
    """Build a dense relational graph from hidden-state correlations."""

    def __init__(self, config: RelationalGraphConfig | dict | None = None) -> None:
        super().__init__()
        if config is None:
            config = RelationalGraphConfig()
        elif isinstance(config, dict):
            config = RelationalGraphConfig(**config)
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return relation graph `W` with shape `[batch, 1, sequence, sequence]`."""
        if hidden_states.dim() != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden_dim]")

        relation_states = hidden_states
        if self.config.normalize_hidden or self.config.kernel == "sigmoid_cosine":
            relation_states = F.normalize(relation_states, p=2, dim=-1)

        logits = relation_states @ relation_states.transpose(-2, -1)
        logits = logits / self._scale(hidden_states.size(-1))

        graph = torch.sigmoid(logits).unsqueeze(1)
        graph = self._apply_attention_mask(graph, attention_mask)
        graph = self._apply_diagonal_policy(graph)
        return graph

    def _scale(self, hidden_dim: int) -> float:
        if self.config.temperature is not None:
            return self.config.temperature
        if self.config.kernel == "sigmoid_cosine":
            return 1.0
        return math.sqrt(hidden_dim)

    def _apply_attention_mask(
        self,
        graph: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return graph
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if attention_mask.shape[0] != graph.shape[0] or attention_mask.shape[1] != graph.shape[-1]:
            raise ValueError("attention_mask shape must match graph batch and sequence dimensions")

        valid = attention_mask.to(dtype=torch.bool, device=graph.device)
        pair_mask = valid[:, None, :, None] & valid[:, None, None, :]
        return graph.masked_fill(~pair_mask, 0.0)

    def _apply_diagonal_policy(self, graph: torch.Tensor) -> torch.Tensor:
        if self.config.diagonal_policy == "keep":
            return graph

        sequence_length = graph.size(-1)
        diagonal = torch.eye(sequence_length, dtype=torch.bool, device=graph.device).view(
            1,
            1,
            sequence_length,
            sequence_length,
        )
        if self.config.diagonal_policy == "zero":
            return graph.masked_fill(diagonal, 0.0)
        if self.config.diagonal_policy == "mask":
            return graph.masked_fill(diagonal, torch.nan)
        raise ValueError(f"unsupported diagonal_policy: {self.config.diagonal_policy}")


def make_shuffled_graph(
    graph: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Shuffle graph entries within each `[sequence, sequence]` matrix."""
    if graph.dim() != 4:
        raise ValueError("graph must have shape [batch, heads, sequence, sequence]")

    flat = graph.reshape(*graph.shape[:2], -1)
    shuffled = torch.empty_like(flat)
    for batch_idx in range(flat.size(0)):
        for head_idx in range(flat.size(1)):
            permutation = torch.randperm(flat.size(-1), device=graph.device, generator=generator)
            shuffled[batch_idx, head_idx] = flat[batch_idx, head_idx, permutation]
    return shuffled.view_as(graph)


def make_random_graph_like(
    graph: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create a random graph with the same shape and approximate value range."""
    if graph.dim() != 4:
        raise ValueError("graph must have shape [batch, heads, sequence, sequence]")

    finite_graph = graph[torch.isfinite(graph)]
    if finite_graph.numel() == 0:
        return torch.rand(graph.shape, device=graph.device, dtype=graph.dtype, generator=generator)

    min_value = finite_graph.min()
    max_value = finite_graph.max()
    random_unit = torch.rand(
        graph.shape, device=graph.device, dtype=graph.dtype, generator=generator
    )
    return min_value + random_unit * (max_value - min_value)
