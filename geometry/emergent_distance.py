"""Emergent distance construction for ERGT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

DiagonalPolicy = Literal["keep", "zero", "mask"]
NormalizationMode = Literal["none", "offdiag_zscore", "offdiag_zscore_clamp", "mean_scale"]


@dataclass(frozen=True)
class EmergentDistanceConfig:
    epsilon: float = 1e-6
    normalization: NormalizationMode = "offdiag_zscore_clamp"
    clip_value: float = 5.0
    diagonal_policy: DiagonalPolicy = "zero"
    causal_runtime_distance: bool = False

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.normalization not in {
            "none",
            "offdiag_zscore",
            "offdiag_zscore_clamp",
            "mean_scale",
        }:
            raise ValueError(f"unsupported normalization: {self.normalization}")
        if self.clip_value <= 0:
            raise ValueError("clip_value must be positive")
        if self.diagonal_policy not in {"keep", "zero", "mask"}:
            raise ValueError(f"unsupported diagonal_policy: {self.diagonal_policy}")


class EmergentDistance(nn.Module):
    """Convert relation strength `W` into induced distance `D`."""

    def __init__(self, config: EmergentDistanceConfig | dict | None = None) -> None:
        super().__init__()
        if config is None:
            config = EmergentDistanceConfig()
        elif isinstance(config, dict):
            allowed = {field.name for field in EmergentDistanceConfig.__dataclass_fields__.values()}
            config = EmergentDistanceConfig(
                **{key: value for key, value in config.items() if key in allowed}
            )
        self.config = config

    def forward(
        self,
        graph: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return distance `D` with the same shape as `graph`."""
        self._validate_graph(graph)
        safe_graph = torch.clamp(graph, min=self.config.epsilon)
        distance = -torch.log(safe_graph)
        distance = self._apply_padding_mask(distance, attention_mask)
        if self.config.causal_runtime_distance:
            distance = self._apply_causal_distance_mask(distance)
        distance = self._apply_diagonal_policy(distance)
        distance = self.normalize(distance)
        distance = self._apply_diagonal_policy(distance)
        return distance

    def normalize(self, distance: torch.Tensor) -> torch.Tensor:
        self._validate_graph(distance)
        mode = self.config.normalization
        if mode == "none":
            return distance

        valid_mask = torch.isfinite(distance)
        offdiag_mask = self._offdiag_mask(distance)
        stats_mask = valid_mask & offdiag_mask

        if mode in {"offdiag_zscore", "offdiag_zscore_clamp"}:
            mean, std = self._masked_mean_std(distance, stats_mask)
            safe_values = torch.where(stats_mask, distance, mean.expand_as(distance))
            normalized = (safe_values - mean) / (std + self.config.epsilon)
            normalized = torch.where(valid_mask, normalized, distance)
            if mode == "offdiag_zscore_clamp":
                clamped = torch.clamp(
                    normalized, -self.config.clip_value, self.config.clip_value
                )
                normalized = torch.where(valid_mask, clamped, distance)
            return normalized

        if mode == "mean_scale":
            mean, _ = self._masked_mean_std(distance, stats_mask)
            safe_values = torch.where(valid_mask, distance, torch.zeros_like(distance))
            normalized = safe_values / (mean.abs() + self.config.epsilon)
            return torch.where(valid_mask, normalized, distance)

        raise ValueError(f"unsupported normalization: {mode}")

    def normalize_precomputed(self, distance: torch.Tensor) -> torch.Tensor:
        """Normalize a distance tensor that was already built outside `forward`."""

        self._validate_graph(distance)
        distance = self._apply_diagonal_policy(distance)
        distance = self.normalize(distance)
        return self._apply_diagonal_policy(distance)

    def _apply_padding_mask(
        self,
        distance: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return distance
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if (
            attention_mask.shape[0] != distance.shape[0]
            or attention_mask.shape[1] != distance.shape[-1]
        ):
            raise ValueError(
                "attention_mask shape must match distance batch and sequence dimensions"
            )

        valid = attention_mask.to(dtype=torch.bool, device=distance.device)
        pair_mask = valid[:, None, :, None] & valid[:, None, None, :]
        return distance.masked_fill(~pair_mask, torch.inf)

    def _apply_causal_distance_mask(self, distance: torch.Tensor) -> torch.Tensor:
        sequence_length = distance.size(-1)
        causal = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=distance.device,
        ).tril()
        return distance.masked_fill(~causal.view(1, 1, sequence_length, sequence_length), torch.inf)

    def _apply_diagonal_policy(self, distance: torch.Tensor) -> torch.Tensor:
        if self.config.diagonal_policy == "keep":
            return distance

        diagonal = self._diagonal_mask(distance)
        if self.config.diagonal_policy == "zero":
            return distance.masked_fill(diagonal, 0.0)
        if self.config.diagonal_policy == "mask":
            return distance.masked_fill(diagonal, torch.nan)
        raise ValueError(f"unsupported diagonal_policy: {self.config.diagonal_policy}")

    def _masked_mean_std(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe_values = torch.where(mask, values, torch.zeros_like(values))
        counts = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
        mean = safe_values.sum(dim=(-2, -1), keepdim=True) / counts
        centered = torch.where(mask, safe_values - mean, torch.zeros_like(values))
        variance = (centered**2).sum(
            dim=(-2, -1),
            keepdim=True,
        ) / counts
        variance = variance.clamp_min(self.config.epsilon)
        std = torch.sqrt(variance)
        return mean, std

    def _diagonal_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        sequence_length = tensor.size(-1)
        return torch.eye(sequence_length, dtype=torch.bool, device=tensor.device).view(
            1,
            1,
            sequence_length,
            sequence_length,
        )

    def _offdiag_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        return ~self._diagonal_mask(tensor).expand_as(tensor)

    def _validate_graph(self, graph: torch.Tensor) -> None:
        if graph.dim() != 4:
            raise ValueError("graph must have shape [batch, heads, sequence, sequence]")
        if graph.size(-1) != graph.size(-2):
            raise ValueError("graph must be square in the last two dimensions")


def make_random_distance_like(
    distance: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create random distance with matched finite min/max range."""
    if distance.dim() != 4:
        raise ValueError("distance must have shape [batch, heads, sequence, sequence]")
    finite_values = distance[torch.isfinite(distance)]
    if finite_values.numel() == 0:
        return torch.rand(
            distance.shape,
            device=distance.device,
            dtype=distance.dtype,
            generator=generator,
        )
    min_value = finite_values.min()
    max_value = finite_values.max()
    random_unit = torch.rand(
        distance.shape,
        device=distance.device,
        dtype=distance.dtype,
        generator=generator,
    )
    random_distance = min_value + random_unit * (max_value - min_value)
    return torch.where(torch.isfinite(distance), random_distance, distance)


def make_shuffled_distance(
    distance: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Shuffle finite distance entries within each `[sequence, sequence]` matrix."""
    if distance.dim() != 4:
        raise ValueError("distance must have shape [batch, heads, sequence, sequence]")

    shuffled = distance.clone()
    for batch_idx in range(distance.size(0)):
        for head_idx in range(distance.size(1)):
            matrix = distance[batch_idx, head_idx]
            finite_mask = torch.isfinite(matrix)
            finite_values = matrix[finite_mask]
            if finite_values.numel() == 0:
                continue
            permutation = torch.randperm(
                finite_values.numel(), device=distance.device, generator=generator
            )
            shuffled[batch_idx, head_idx][finite_mask] = finite_values[permutation]
    return shuffled
