from __future__ import annotations

import torch

from equivariant_linear_attention.advanced import (
    DirectVectorForceHead,
    ScalarEnergyHead,
    conservative_forces,
)


def test_energy_head_and_conservative_forces() -> None:
    generator = torch.Generator().manual_seed(501)
    positions = torch.randn(5, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    scalar = positions.square().sum(dim=-1, keepdim=True).expand(-1, 4)
    head = ScalarEnergyHead(4).double()
    batch = torch.tensor([0, 0, 0, 1, 1])
    energy = head(scalar, batch)
    force = conservative_forces(energy.sum(), positions, create_graph=True)
    assert force.shape == positions.shape
    assert torch.isfinite(force).all()
    second = torch.autograd.grad(force.square().sum(), positions)[0]
    assert torch.isfinite(second).all()


def test_direct_vector_head_is_equivariant() -> None:
    generator = torch.Generator().manual_seed(503)
    head = DirectVectorForceHead(5, 3).double()
    scalar = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    vector = torch.randn(7, 3, 3, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    reference = head(scalar, vector)
    moved = head(scalar, vector @ q.T)
    torch.testing.assert_close(moved, reference @ q.T, atol=2e-12, rtol=2e-12)
