"""Measurement and claim contracts for the post-Phase-3 ERGT program."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch

ContractStatus = Literal["pass", "fail"]


CLAIM_BOUNDARIES: tuple[str, ...] = (
    "hidden state != physical field",
    "Phi != consciousness",
    "causal mask != causal geometry",
    "pairwise distance != full geometry",
    "low entropy != good structure",
    "baseline win != relational proof",
    "EMA smoothing != meaningful memory",
)


@dataclass(frozen=True)
class MeasurementContract:
    """Machine-readable contract that must be fixed before later ERGT phases."""

    contract_version: str = "measurement_contract_v1"
    valid_edge_policy: str = "causal_lower_triangular_non_diagonal_non_padding"
    graph_diagonal_policy: str = "keep_for_distance"
    distance_diagonal_policy: str = "zero"
    causal_policy: str = "lower_triangular_no_future_edges"
    padding_policy: str = "exclude_padding_pairs"
    normalization: str = "offdiag_zscore_clamp"
    clip_value: float = 5.0
    salience_definition: str = "hidden_norm"
    spectral_operator: str = "symmetrized_normalized_laplacian"
    reconstruction_protocol: str = "causal_prefix_excludes_target"
    phi_formula: str = "fixed_log_linear_anti_collapse_v1"
    geo_to_qk_ratio_policy: str = "report_and_match_before_comparison"
    control_generation_level: str = "W_level_before_distance_normalization"
    forbidden_metric_inputs: tuple[str, ...] = field(
        default_factory=lambda: ("future_tokens", "target_hidden_state", "future_relations")
    )

    def validate(self) -> dict[str, bool]:
        """Return validation checks for the contract itself."""

        return {
            "valid_edge_policy_fixed": (
                self.valid_edge_policy == "causal_lower_triangular_non_diagonal_non_padding"
            ),
            "graph_diagonal_policy_fixed": self.graph_diagonal_policy
            in {"keep_for_distance", "keep", "zero", "mask"},
            "distance_diagonal_policy_fixed": self.distance_diagonal_policy
            in {"zero", "mask", "keep"},
            "causal_policy_forbids_future": self.causal_policy
            == "lower_triangular_no_future_edges",
            "padding_policy_excludes_pairs": self.padding_policy == "exclude_padding_pairs",
            "normalization_fixed": self.normalization
            in {"none", "offdiag_zscore", "offdiag_zscore_clamp", "mean_scale"},
            "clip_value_positive": self.clip_value > 0,
            "salience_definition_fixed": bool(self.salience_definition),
            "spectral_operator_fixed": bool(self.spectral_operator),
            "reconstruction_excludes_target": (
                self.reconstruction_protocol == "causal_prefix_excludes_target"
            ),
            "phi_formula_fixed": bool(self.phi_formula),
            "geo_to_qk_ratio_policy_fixed": (
                self.geo_to_qk_ratio_policy == "report_and_match_before_comparison"
            ),
            "controls_built_at_w_level": (
                self.control_generation_level == "W_level_before_distance_normalization"
            ),
            "future_inputs_forbidden": "future_tokens" in self.forbidden_metric_inputs,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["forbidden_metric_inputs"] = list(self.forbidden_metric_inputs)
        return payload


def contract_from_project_config(project_config: dict[str, Any]) -> MeasurementContract:
    """Build a measurement contract from an existing ERGT project config."""

    graph_config = project_config.get("relational_graph", {})
    distance_config = project_config.get("distance", {})
    attention_config = project_config.get("attention", {})

    graph_diagonal = graph_config.get("diagonal_policy", "keep_for_distance")
    if graph_diagonal == "keep":
        graph_diagonal = "keep_for_distance"

    return MeasurementContract(
        graph_diagonal_policy=graph_diagonal,
        distance_diagonal_policy=distance_config.get("diagonal_policy", "zero"),
        causal_policy="lower_triangular_no_future_edges"
        if attention_config.get("causal_runtime_distance", False)
        else "attention_mask_only_no_runtime_distance",
        normalization=distance_config.get("normalization", "offdiag_zscore_clamp"),
        clip_value=float(distance_config.get("clip_value", 5.0)),
    )


def valid_edge_mask(
    *,
    sequence_length: int,
    attention_mask: torch.Tensor | None = None,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return `[batch, 1, sequence, sequence]` valid-edge mask.

    The row index is the current token `i`, and the column index is the context
    token `j`. A valid edge means `j <= i`, `i != j`, and both positions are
    non-padding.
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    if attention_mask is not None:
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if attention_mask.size(1) != sequence_length:
            raise ValueError("attention_mask sequence length must match sequence_length")
        batch = attention_mask.size(0)
        device = attention_mask.device
    else:
        batch = int(batch_size or 1)
        if batch <= 0:
            raise ValueError("batch_size must be positive")

    edge_device = torch.device(device) if device is not None else None
    positions = torch.arange(sequence_length, device=edge_device)
    current = positions.view(sequence_length, 1)
    context = positions.view(1, sequence_length)
    causal = context <= current
    non_diagonal = context != current
    valid = (causal & non_diagonal).view(1, 1, sequence_length, sequence_length)
    valid = valid.expand(batch, 1, sequence_length, sequence_length).clone()

    if attention_mask is not None:
        nonpadding = attention_mask.to(dtype=torch.bool, device=valid.device)
        pair_mask = nonpadding[:, None, :, None] & nonpadding[:, None, None, :]
        valid &= pair_mask

    return valid


def build_measurement_contract_report(
    contract: MeasurementContract | None = None,
    *,
    sequence_length: int = 5,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build the machine-readable Phase 0 measurement contract report."""

    contract = contract or MeasurementContract()
    if attention_mask is None:
        attention_mask = torch.ones(1, sequence_length, dtype=torch.long)

    mask = valid_edge_mask(sequence_length=sequence_length, attention_mask=attention_mask)
    rows, cols = torch.meshgrid(
        torch.arange(sequence_length, device=mask.device),
        torch.arange(sequence_length, device=mask.device),
        indexing="ij",
    )
    future = cols > rows
    diagonal = cols == rows

    contract_checks = contract.validate()
    tensor_checks = {
        "valid_edge_mask_shape": list(mask.shape)
        == [attention_mask.size(0), 1, sequence_length, sequence_length],
        "valid_edge_mask_has_edges": int(mask.sum().item()) > 0,
        "valid_edge_mask_excludes_future": not bool(mask[:, :, future].any().item()),
        "valid_edge_mask_excludes_diagonal": not bool(mask[:, :, diagonal].any().item()),
        "valid_edge_mask_excludes_padding_pairs": _padding_pairs_are_excluded(
            mask, attention_mask
        ),
    }
    checks = {**contract_checks, **tensor_checks}
    status: ContractStatus = "pass" if all(checks.values()) else "fail"

    return {
        "phase": "phase0_measurement_contracts",
        "status": status,
        "contract": contract.to_dict(),
        "claim_boundaries": list(CLAIM_BOUNDARIES),
        "checks": checks,
        "valid_edge_summary": {
            "sequence_length": sequence_length,
            "batch_size": int(attention_mask.size(0)),
            "valid_edges": int(mask.sum().item()),
            "future_edges_allowed": False,
            "diagonal_edges_allowed": False,
        },
        "next_required_step": "strict_W_level_controls" if status == "pass" else "fix_contract",
    }


def _padding_pairs_are_excluded(mask: torch.Tensor, attention_mask: torch.Tensor) -> bool:
    padding = ~attention_mask.to(dtype=torch.bool, device=mask.device)
    if not bool(padding.any().item()):
        return True
    invalid_rows = mask & padding[:, None, :, None]
    invalid_cols = mask & padding[:, None, None, :]
    return not bool((invalid_rows | invalid_cols).any().item())
