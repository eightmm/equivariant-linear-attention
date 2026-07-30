from __future__ import annotations

import torch

from equivariant_attention import (
    ImplicitGaussianSpatialKernel,
    ImplicitSpatialKernelConfig,
)


def test_chunk_size_does_not_change_transport_or_moments() -> None:
    torch.manual_seed(23)
    common = dict(
        scales=(0.75, 1.5, 3.0),
        order=2,
        exclude_self=True,
        normalization="one_plus_mass",
    )
    small_chunk = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(**common, chunk_size=1)
    ).double()
    large_chunk = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(**common, chunk_size=128)
    ).double()
    values = torch.randn(11, 7, dtype=torch.float64)
    positions = torch.randn(11, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)

    small = small_chunk(values, positions, batch)
    large = large_chunk(values, positions, batch)
    torch.testing.assert_close(small.output, large.output, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(small.mass, large.mass, atol=2e-10, rtol=2e-10)

    small_moments = small_chunk.moments(positions, batch)
    large_moments = large_chunk.moments(positions, batch)
    torch.testing.assert_close(
        small_moments.relative_vector,
        large_moments.relative_vector,
        atol=2e-10,
        rtol=2e-10,
    )
    torch.testing.assert_close(
        small_moments.relative_tensor,
        large_moments.relative_tensor,
        atol=2e-10,
        rtol=2e-10,
    )


def test_many_singleton_graphs_remain_isolated_and_zero() -> None:
    torch.manual_seed(29)
    nodes = 16
    kernel = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(
            scales=(1.0, 2.0),
            exclude_self=True,
            chunk_size=3,
        )
    ).double()
    values = torch.randn(nodes, 5, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    batch = torch.arange(nodes, dtype=torch.long)

    result = kernel(values, positions, batch)
    torch.testing.assert_close(result.output, torch.zeros_like(result.output))
    torch.testing.assert_close(result.mass, torch.zeros_like(result.mass))
