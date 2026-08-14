from __future__ import annotations

import torch
from conftest import orthogonal

from equivariant_linear_attention.nn.geometry import GeometryContext
from equivariant_linear_attention.nn.local_moments import LocalCumulantBank
from equivariant_linear_attention.nn.local_support import (
    build_local_support,
    wendland_c2,
)


def _geometry(position: torch.Tensor, index: torch.Tensor) -> GeometryContext:
    return GeometryContext.build(
        position,
        index,
        num_segments=int(index.max().item()) + 1,
        eps=1e-10,
    )


def test_wendland_support_is_compact_and_c2_at_boundary() -> None:
    value = torch.tensor([0.0, 0.5, 1.0, 1.2], dtype=torch.float64)
    output = wendland_c2(value)
    torch.testing.assert_close(
        output[[0, 2, 3]],
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
    )
    assert 0.0 < float(output[1]) < 1.0

    point = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    first = torch.autograd.grad(wendland_c2(point), point, create_graph=True)[0]
    second = torch.autograd.grad(first, point)[0]
    torch.testing.assert_close(first, torch.zeros_like(first))
    torch.testing.assert_close(second, torch.zeros_like(second))


def test_transient_support_is_bounded_isolated_and_o3_equivariant() -> None:
    generator = torch.Generator().manual_seed(501)
    position = torch.randn(13, 3, generator=generator, dtype=torch.float64)
    index = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    support = build_local_support(
        _geometry(position, index),
        max_points=4,
        chunk_size=3,
        eps=1e-10,
    )
    assert support.source.numel() <= position.shape[0] * 5
    assert bool((index[support.source] == index[support.receiver]).all().item())

    transform = orthogonal(reflection=True, seed=503)
    moved = build_local_support(
        _geometry(
            position @ transform.T
            + torch.tensor([1.5, -2.0, 0.75], dtype=torch.float64),
            index,
        ),
        max_points=4,
        chunk_size=3,
        eps=1e-10,
    )
    torch.testing.assert_close(moved.source, support.source)
    torch.testing.assert_close(moved.receiver, support.receiver)
    torch.testing.assert_close(moved.distance, support.distance, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(moved.scale, support.scale, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(
        moved.displacement,
        support.displacement @ transform.T,
        atol=2e-10,
        rtol=2e-10,
    )


def test_relative_and_physical_scales_have_distinct_semantics() -> None:
    bank = LocalCumulantBank(scalar_width=4, rank=4, eps=1e-10).double()
    with torch.no_grad():
        bank.raw_physical_mix[:2].fill_(-30.0)
        bank.raw_physical_mix[2:].fill_(30.0)
    support_scale = torch.tensor([1.0, 2.0], dtype=torch.float64)
    original = bank.scales(support_scale, dtype=torch.float64)
    expanded = bank.scales(2.0 * support_scale, dtype=torch.float64)
    torch.testing.assert_close(
        expanded[:, :2], 2.0 * original[:, :2], atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(
        expanded[:, 2:], original[:, 2:], atol=1e-10, rtol=1e-10
    )
