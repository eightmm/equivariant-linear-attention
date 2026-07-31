from __future__ import annotations

import pytest
import torch

from equivariant_attention import (
    ImplicitGaussianSpatialKernel,
    ImplicitSpatialKernelConfig,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_implicit_kernel_cuda_bfloat16_forward_backward() -> None:
    torch.manual_seed(43)
    device = torch.device("cuda")
    kernel = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(
            scales=(1.0, 2.0, 4.0),
            order=2,
            exclude_self=True,
            normalization="one_plus_mass",
            learnable_scale_weights=True,
            chunk_size=128,
        )
    ).to(device=device, dtype=torch.bfloat16)
    values = torch.randn(
        512,
        32,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    positions = torch.randn(
        512,
        3,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    batch = torch.arange(8, device=device).repeat_interleave(64).to(torch.long)

    result = kernel(values, positions, batch)
    loss = result.output.float().square().mean() + 0.01 * result.mass.square().mean()
    loss.backward()

    assert result.output.dtype == torch.bfloat16
    assert result.mass.dtype == torch.float32
    assert torch.isfinite(result.output).all()
    assert torch.isfinite(result.mass).all()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert kernel.raw_scale_weights.grad is not None
    assert torch.isfinite(kernel.raw_scale_weights.grad).all()
