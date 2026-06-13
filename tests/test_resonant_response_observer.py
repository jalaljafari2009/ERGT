import json

import torch

from evaluation.relational_field_observer import synthetic_structured_hidden_layers
from evaluation.resonant_response_observer import (
    build_resonant_response_observer_report,
    controlled_resonant_perturbation,
    phi_proxy,
)


def test_controlled_resonant_perturbation_does_not_mutate_input() -> None:
    hidden_layers, attention_mask = synthetic_structured_hidden_layers(seed=1)
    before = [layer.clone() for layer in hidden_layers]

    after = controlled_resonant_perturbation(
        hidden_layers,
        attention_mask=attention_mask,
        scale=0.08,
    )

    assert all(
        torch.allclose(original, current)
        for original, current in zip(before, hidden_layers, strict=True)
    )
    assert any(
        not torch.allclose(original, perturbed)
        for original, perturbed in zip(before, after, strict=True)
    )


def test_phi_proxy_is_finite_and_anti_collapse_aware() -> None:
    metrics = {
        "coherence_mean": 0.5,
        "local_relational_entropy_mean": 0.25,
        "saturation_fraction": 0.0,
        "valid_weight_variance": 0.1,
    }
    collapsed = {**metrics, "saturation_fraction": 1.0}

    assert phi_proxy(metrics, stability=0.9) > 0
    assert phi_proxy(collapsed, stability=0.9) == 0


def test_resonant_response_report_passes_on_structured_smoke_field() -> None:
    report = build_resonant_response_observer_report(seed=2027)

    assert report["status"] == "pass"
    assert report["input_source"] == "synthetic_structured_smoke"
    assert report["model_intervention"] == "none"
    assert report["checks"]["probe_reset_to_before_supported"]
    assert report["checks"]["response_not_scale_artifact"]
    assert report["checks"]["all_layers_real_response_beats_or_stabilizes_vs_controls"]
    assert report["next_required_step"] == "information_potential_phi"
    json.dumps(report)


def test_resonant_response_report_accepts_explicit_hidden_layers() -> None:
    hidden_layers, attention_mask = synthetic_structured_hidden_layers(seed=2, layers=2)

    report = build_resonant_response_observer_report(
        hidden_layers=hidden_layers,
        attention_mask=attention_mask,
        seed=2,
    )

    assert report["input_source"] == "provided_hidden_layers"
    assert sorted(report["layers"].keys()) == ["layer_0", "layer_1"]
