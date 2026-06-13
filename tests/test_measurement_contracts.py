import json

import torch

from evaluation.measurement_contracts import (
    MeasurementContract,
    build_measurement_contract_report,
    contract_from_project_config,
    valid_edge_mask,
)


def test_valid_edge_mask_is_causal_non_diagonal_and_padding_aware() -> None:
    attention_mask = torch.tensor([[1, 1, 1, 0]])

    mask = valid_edge_mask(sequence_length=4, attention_mask=attention_mask)

    assert mask.shape == (1, 1, 4, 4)
    assert not mask[0, 0, 0, 0]
    assert not mask[0, 0, 0, 1]
    assert mask[0, 0, 1, 0]
    assert mask[0, 0, 2, 0]
    assert mask[0, 0, 2, 1]
    assert not mask[0, 0, 2, 3]
    assert not mask[0, 0, 3, 0]
    assert not mask[0, 0, 0, 3]


def test_default_measurement_contract_report_passes_and_serializes() -> None:
    report = build_measurement_contract_report(sequence_length=5)

    assert report["status"] == "pass"
    assert report["checks"]["valid_edge_mask_excludes_future"]
    assert report["checks"]["valid_edge_mask_excludes_diagonal"]
    assert report["contract"]["reconstruction_protocol"] == "causal_prefix_excludes_target"
    assert report["next_required_step"] == "strict_W_level_controls"
    json.dumps(report)


def test_contract_from_project_config_reads_existing_phase3_policies() -> None:
    config = {
        "attention": {"causal_runtime_distance": True},
        "relational_graph": {"diagonal_policy": "keep"},
        "distance": {
            "diagonal_policy": "zero",
            "normalization": "offdiag_zscore_clamp",
            "clip_value": 5.0,
        },
    }

    contract = contract_from_project_config(config)
    report = build_measurement_contract_report(contract, sequence_length=4)

    assert contract.graph_diagonal_policy == "keep_for_distance"
    assert contract.distance_diagonal_policy == "zero"
    assert contract.causal_policy == "lower_triangular_no_future_edges"
    assert report["status"] == "pass"


def test_contract_report_fails_when_causal_policy_does_not_forbid_future() -> None:
    contract = MeasurementContract(causal_policy="attention_mask_only_no_runtime_distance")

    report = build_measurement_contract_report(contract, sequence_length=4)

    assert report["status"] == "fail"
    assert report["checks"]["causal_policy_forbids_future"] is False
