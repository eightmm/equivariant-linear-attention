from __future__ import annotations

import torch

from equivariant_linear_attention import ELA, ELAGraph


def test_local_extractors_receive_gradients_at_initialization() -> None:
    generator = torch.Generator().manual_seed(557)
    model = ELA("3x0e", "1x0e", width=32, depth=1)
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
