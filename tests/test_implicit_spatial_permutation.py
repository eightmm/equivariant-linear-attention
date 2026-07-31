from __future__ import annotations

import torch

from equivariant_attention import (
    ImplicitGaussianSpatialKernel,
    ImplicitSpatialKernelConfig,
)


def test_implicit_transport_is_node_permutation_equivariant() -> None:
    torch.manual_seed(31)
    kernel = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(
            scales=(0.7, 1.4, 2.8),
            order=2,
            exclude_self=True,
            normalization="one_plus_mass",
            chunk_size=3,
        )
    ).double()
    positions = torch.randn(10, 3, dtype=torch.float64)
    values = torch.randn(10, 6, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    permutation = torch.randperm(10)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(10)

    reference = kernel(values, positions, batch).output
    permuted = kernel(
        values[permutation],
        positions[permutation],
        batch[permutation],
    ).output

    torch.testing.assert_close(
        permuted[inverse],
        reference,
        atol=3e-10,
        rtol=3e-10,
    )
