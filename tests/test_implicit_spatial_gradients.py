from __future__ import annotations

import torch

from equivariant_attention import (
    ImplicitGaussianSpatialKernel,
    ImplicitSpatialKernelConfig,
)


def test_implicit_kernel_value_coordinate_and_scale_gradients_are_finite() -> None:
    torch.manual_seed(17)
    kernel = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(
            scales=(0.8, 1.6, 3.2),
            order=2,
            exclude_self=True,
            normalization="one_plus_mass",
            learnable_scale_weights=True,
        )
    ).double()
    values = torch.randn(9, 6, dtype=torch.float64, requires_grad=True)
    positions = torch.randn(9, 3, dtype=torch.float64, requires_grad=True)
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    transport = kernel(values, positions, batch)
    moments = kernel.moments(positions, batch)
    loss = (
        transport.output.square().mean()
        + 0.1 * transport.mass.square().mean()
        + moments.relative_vector.square().mean()
        + moments.relative_tensor.square().mean()
    )
    loss.backward()

    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert kernel.raw_scale_weights.grad is not None
    assert torch.isfinite(kernel.raw_scale_weights.grad).all()


def test_implicit_kernel_supports_double_backward_in_coordinates() -> None:
    torch.manual_seed(19)
    kernel = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(
            scales=(1.0, 2.0),
            order=2,
            exclude_self=True,
            normalization="one_plus_mass",
        )
    ).double()
    values = torch.randn(6, 3, dtype=torch.float64)
    positions = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    batch = torch.zeros(6, dtype=torch.long)

    output = kernel(values, positions, batch).output
    first = torch.autograd.grad(output.square().sum(), positions, create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), positions)[0]

    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
