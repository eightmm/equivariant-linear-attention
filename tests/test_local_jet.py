from __future__ import annotations

import torch

from equivariant_linear_attention.nn.geometry import GeometryContext
from equivariant_linear_attention.nn.local_jet import ReproducingLocalJet
from equivariant_linear_attention.nn.local_support import build_local_support
from equivariant_linear_attention.nn.ops import (
    bounded_scalar,
    bounded_st,
    matrix_to_st,
    unit_ball,
)


def _geometry(position: torch.Tensor, index: torch.Tensor) -> GeometryContext:
    return GeometryContext.build(
        position,
        index,
        num_segments=int(index.max().item()) + 1,
        eps=1e-10,
    )


def test_reproducing_jet_recovers_quadratic_field() -> None:
    generator = torch.Generator().manual_seed(523)
    position = 0.8 * torch.randn(40, 3, generator=generator, dtype=torch.float64)
    index = torch.zeros(40, dtype=torch.long)
    support = build_local_support(
        _geometry(position, index), max_points=40, chunk_size=8, eps=1e-10
    )
    linear = torch.tensor([0.030, -0.020, 0.015], dtype=torch.float64)
    hessian = torch.tensor(
        [[0.012, -0.004, 0.003], [-0.004, -0.008, 0.002], [0.003, 0.002, 0.006]],
        dtype=torch.float64,
    )
    value = (
        0.02
        + position @ linear
        + 0.5 * torch.einsum("na,ab,nb->n", position, hessian, position)
    )
    jet = ReproducingLocalJet(
        scalar_width=1, probe_rank=1, num_scales=2, eps=1e-10
    ).double()
    with torch.no_grad():
        jet.probe.weight.fill_(1.0)
        jet.raw_scale.fill_(12.0)
        jet.raw_ridge.fill_(-30.0)
    output = jet(value[:, None], support)

    expected_value = bounded_scalar(value[:, None, None].expand(-1, 2, 1), 1e-10)
    expected_gradient = unit_ball(
        (linear[None, :] + position @ hessian.T)[:, None, None, :].expand(-1, 2, 1, -1),
        1e-10,
    )
    expected_laplacian = bounded_scalar(
        torch.full(
            (40, 2, 1),
            torch.trace(hessian).item() / 3.0,
            dtype=torch.float64,
        ),
        1e-10,
    )
    expected_hessian = bounded_st(
        matrix_to_st(hessian)[None, None, None, :].expand(40, 2, 1, -1),
        1e-10,
    )
    torch.testing.assert_close(output.value, expected_value, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(output.gradient, expected_gradient, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(
        output.laplacian, expected_laplacian, atol=3e-6, rtol=3e-6
    )
    torch.testing.assert_close(output.hessian, expected_hessian, atol=3e-6, rtol=3e-6)
