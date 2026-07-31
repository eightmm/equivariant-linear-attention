from __future__ import annotations

import pytest
import torch

from equivariant_attention import (
    ImplicitGaussianSpatialKernel,
    ImplicitSpatialKernelConfig,
)


def test_implicit_config_rejects_nonpositive_scales() -> None:
    with pytest.raises(ValueError, match="scales"):
        ImplicitSpatialKernelConfig(scales=(1.0, 0.0))


def test_implicit_config_rejects_float32_underflow_scale() -> None:
    with pytest.raises(ValueError, match="representable"):
        ImplicitSpatialKernelConfig(scales=(1e-50,))


def test_implicit_config_rejects_unsupported_taylor_order() -> None:
    with pytest.raises(ValueError, match="order 0 or 2"):
        ImplicitSpatialKernelConfig(order=1)


def test_implicit_config_rejects_zero_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        ImplicitSpatialKernelConfig(chunk_size=0)


def test_implicit_kernel_rejects_nonlong_batch() -> None:
    kernel = ImplicitGaussianSpatialKernel(ImplicitSpatialKernelConfig())
    with pytest.raises(TypeError, match="torch.long"):
        kernel(
            torch.randn(4, 3),
            torch.randn(4, 3),
            torch.zeros(4, dtype=torch.int32),
        )


def test_implicit_kernel_rejects_mismatched_value_count() -> None:
    kernel = ImplicitGaussianSpatialKernel(ImplicitSpatialKernelConfig())
    context = kernel.prepare(
        torch.randn(4, 3),
        torch.zeros(4, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="same node count"):
        kernel.transport_prepared(torch.randn(3, 2), context)


def test_implicit_kernel_fails_closed_on_unstable_finite_coordinates() -> None:
    kernel = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(scales=(1.0,))
    )
    positions = torch.tensor(
        [
            [3.0e38, 3.0e38, 3.0e38],
            [-3.0e38, -3.0e38, -3.0e38],
        ],
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="must be finite"):
        kernel(
            torch.ones(2, 1),
            positions,
            torch.zeros(2, dtype=torch.long),
        )
