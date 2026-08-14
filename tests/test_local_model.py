from __future__ import annotations

import torch

from equivariant_linear_attention import ELA, ELAGraph


def test_local_jet_branch_is_off_by_default() -> None:
    """The default configuration must be exactly the canonical edge-free model."""

    model = ELA("3x0e", "1x0e", width=32, depth=1)
    assert model.config.local_points == 0
    assert model.config.uses_local_jet is False
    assert not hasattr(model.layers[0], "local_geometry")
    assert not hasattr(model.layers[0], "local_closure")
    assert not hasattr(model.layers[0], "local_norm")
    description = model.describe()
    assert description["canonical_edge_free_path"] is True
    assert description["transient_local_support"] is False
    assert description["local_support"] == "none"


def test_enabling_local_points_adds_parameters_only_then() -> None:
    canonical = ELA("3x0e", "1x0e", width=32, depth=1)
    local = ELA("3x0e", "1x0e", width=32, depth=1, local_points=8)
    canonical_keys = set(canonical.state_dict())
    local_keys = set(local.state_dict())
    assert canonical_keys < local_keys
    assert all(".local_" in key for key in local_keys - canonical_keys)
    description = local.describe()
    assert description["canonical_edge_free_path"] is False
    assert description["transient_local_support"] is True
    assert description["local_points"] == 8


def test_local_extractors_receive_gradients_at_initialization() -> None:
    generator = torch.Generator().manual_seed(557)
    model = ELA("3x0e", "1x0e", width=32, depth=1, local_points=8)
    x = torch.randn(10, 3, generator=generator, requires_grad=True)
    position = torch.randn(10, 3, generator=generator, requires_grad=True)
    output = model(ELAGraph(x, position))
    output.x.square().mean().backward()

    cumulant_grad = model.layers[0].local_geometry.cumulants.source_weight.weight.grad
    probe_grad = model.layers[0].local_geometry.jet.probe.weight.grad
    assert cumulant_grad is not None and bool(torch.isfinite(cumulant_grad).all().item())
    assert probe_grad is not None and bool(torch.isfinite(probe_grad).all().item())
    assert float(cumulant_grad.abs().sum()) > 0.0
    assert float(probe_grad.abs().sum()) > 0.0
    assert position.grad is not None and bool(torch.isfinite(position.grad).all().item())
