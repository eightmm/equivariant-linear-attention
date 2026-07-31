from __future__ import annotations

import torch

from equivariant_attention import (
    ImplicitGaussianSpatialKernel,
    ImplicitSpatialKernelConfig,
    ImplicitSpatialResidual,
    UnifiedSE3State,
)


def _state(nodes: int = 6, scalar_width: int = 8, heads: int = 3) -> UnifiedSE3State:
    return UnifiedSE3State(
        even_scalar=torch.randn(nodes, scalar_width, dtype=torch.float64),
        odd_scalar=torch.randn(nodes, heads, dtype=torch.float64),
        polar_vector=torch.randn(nodes, heads, 3, dtype=torch.float64),
        axial_vector=torch.randn(nodes, heads, 3, dtype=torch.float64),
        even_tensor=torch.randn(nodes, heads, 5, dtype=torch.float64),
        odd_tensor=torch.randn(nodes, heads, 5, dtype=torch.float64),
    )


def test_zero_initialized_implicit_residual_is_exact_identity() -> None:
    torch.manual_seed(37)
    state = _state()
    kernel = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(scales=(1.0, 2.0))
    ).double()
    residual = ImplicitSpatialResidual(
        kernel,
        scalar_width=8,
        num_heads=3,
        residual_scale_init=0.0,
    ).double()
    positions = torch.randn(6, 3, dtype=torch.float64)
    batch = torch.zeros(6, dtype=torch.long)

    output = residual(state, positions, batch)
    for name in (
        "even_scalar",
        "odd_scalar",
        "polar_vector",
        "axial_vector",
        "even_tensor",
        "odd_tensor",
    ):
        torch.testing.assert_close(getattr(output, name), getattr(state, name))


def test_implicit_residual_scales_receive_gradient_from_even_objective() -> None:
    torch.manual_seed(41)
    state = _state()
    kernel = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(scales=(1.0, 2.0))
    ).double()
    residual = ImplicitSpatialResidual(
        kernel,
        scalar_width=8,
        num_heads=3,
        residual_scale_init=0.0,
    ).double()
    positions = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    batch = torch.zeros(6, dtype=torch.long)

    output = residual(state, positions, batch)
    output.even_scalar.square().mean().backward()

    assert residual.even_scale.grad is not None
    assert torch.isfinite(residual.even_scale.grad).all()
    assert torch.count_nonzero(residual.even_scale.grad) > 0
