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


def make_valid_edge_mask_like(
    graph: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    causal: bool = True,
    include_diagonal: bool = False,
) -> torch.Tensor:
    """Return the valid relation region for a graph-shaped tensor.

    Rows are current positions `i`, columns are context positions `j`. For the
    strict post-Phase-3 controls, only causal non-diagonal non-padding edges are
    valid for control generation.
    """

    if graph.dim() != 4:
        raise ValueError("graph must have shape [batch, heads, sequence, sequence]")
    if graph.size(-1) != graph.size(-2):
        raise ValueError("graph must be square in the last two dimensions")

    batch_size, heads, sequence_length, _ = graph.shape
    positions = torch.arange(sequence_length, device=graph.device)
    current = positions.view(sequence_length, 1)
    context = positions.view(1, sequence_length)

    valid = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=graph.device)
    if causal:
        valid &= context <= current
    if not include_diagonal:
        valid &= context != current

    valid = valid.view(1, 1, sequence_length, sequence_length).expand(
        batch_size,
        heads,
        sequence_length,
        sequence_length,
    )

    if attention_mask is None:
        return valid.clone()

    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if attention_mask.shape != (batch_size, sequence_length):
        raise ValueError("attention_mask shape must match graph batch and sequence dimensions")

    nonpadding = attention_mask.to(dtype=torch.bool, device=graph.device)
    pair_mask = nonpadding[:, None, :, None] & nonpadding[:, None, None, :]
    return (valid & pair_mask).clone()


def make_shuffled_graph(
    graph: torch.Tensor,
    generator: torch.Generator | None = None,
    valid_edge_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shuffle graph entries within each `[sequence, sequence]` matrix.

    When `valid_edge_mask` is provided, shuffling happens only inside that
    region and all invalid entries are preserved exactly.
    """

    if graph.dim() != 4:
        raise ValueError("graph must have shape [batch, heads, sequence, sequence]")
    if valid_edge_mask is None:
        flat = graph.reshape(*graph.shape[:2], -1)
        shuffled = torch.empty_like(flat)
        for batch_idx in range(flat.size(0)):
            for head_idx in range(flat.size(1)):
                permutation = torch.randperm(
                    flat.size(-1),
                    device=graph.device,
                    generator=generator,
                )
                shuffled[batch_idx, head_idx] = flat[batch_idx, head_idx, permutation]
        return shuffled.view_as(graph)

    valid_edge_mask = _prepare_valid_edge_mask(graph, valid_edge_mask)
    shuffled = graph.clone()
    for batch_idx in range(graph.size(0)):
        for head_idx in range(graph.size(1)):
            mask = valid_edge_mask[batch_idx, head_idx] & torch.isfinite(graph[batch_idx, head_idx])
            values = graph[batch_idx, head_idx][mask]
            if values.numel() <= 1:
                continue
            permutation = torch.randperm(values.numel(), device=graph.device, generator=generator)
            shuffled[batch_idx, head_idx][mask] = values[permutation]
    return shuffled


def make_random_graph_like(
    graph: torch.Tensor,
    generator: torch.Generator | None = None,
    valid_edge_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create a random graph with matched shape and value scale.

    When `valid_edge_mask` is provided, randomization happens only inside that
    region by resampling valid real values with replacement. This preserves the
    marginal scale while destroying relation arrangement.
    """

    if graph.dim() != 4:
        raise ValueError("graph must have shape [batch, heads, sequence, sequence]")
    if valid_edge_mask is None:
        finite_graph = graph[torch.isfinite(graph)]
        if finite_graph.numel() == 0:
            return torch.rand(
                graph.shape,
                device=graph.device,
                dtype=graph.dtype,
                generator=generator,
            )

        min_value = finite_graph.min()
        max_value = finite_graph.max()
        random_unit = torch.rand(
            graph.shape, device=graph.device, dtype=graph.dtype, generator=generator
        )
        return min_value + random_unit * (max_value - min_value)

    valid_edge_mask = _prepare_valid_edge_mask(graph, valid_edge_mask)
    random_graph = graph.clone()
    for batch_idx in range(graph.size(0)):
        for head_idx in range(graph.size(1)):
            mask = valid_edge_mask[batch_idx, head_idx] & torch.isfinite(graph[batch_idx, head_idx])
            values = graph[batch_idx, head_idx][mask]
            if values.numel() == 0:
                continue
            sample_indices = torch.randint(
                values.numel(),
                (values.numel(),),
                device=graph.device,
                generator=generator,
            )
            random_graph[batch_idx, head_idx][mask] = values[sample_indices]
    return random_graph


def _prepare_valid_edge_mask(graph: torch.Tensor, valid_edge_mask: torch.Tensor) -> torch.Tensor:
    valid_edge_mask = valid_edge_mask.to(dtype=torch.bool, device=graph.device)
    if valid_edge_mask.shape == graph.shape:
        return valid_edge_mask
    if valid_edge_mask.dim() != 4:
        raise ValueError("valid_edge_mask must have shape [batch, heads, sequence, sequence]")
    if (
        valid_edge_mask.size(0) != graph.size(0)
        or valid_edge_mask.shape[-2:] != graph.shape[-2:]
    ):
        raise ValueError("valid_edge_mask must match graph batch and sequence dimensions")
    if valid_edge_mask.size(1) == 1:
        return valid_edge_mask.expand_as(graph)
    if valid_edge_mask.size(1) != graph.size(1):
        raise ValueError("valid_edge_mask heads must be 1 or match graph heads")
    return valid_edge_mask
