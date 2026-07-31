from __future__ import annotations

import torch

from equivariant_attention import (
    ImplicitGaussianSpatialKernel,
    ImplicitSpatialKernelConfig,
)


def _c2_cutoff_matrix(positions: torch.Tensor, cutoff: float) -> torch.Tensor:
    displacement = positions[:, None, :] - positions[None, :, :]
    ratio = displacement.square().sum(dim=-1) / cutoff**2
    inside = ratio < 1.0
    u = ratio.clamp(min=0.0, max=1.0)
    value = 1.0 - 10.0 * u.pow(3) + 15.0 * u.pow(4) - 6.0 * u.pow(5)
    value = torch.where(inside, value, torch.zeros_like(value))
    value.fill_diagonal_(0.0)
    return value


def test_distant_fragment_does_not_change_explicit_original_pair_block() -> None:
    base = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    fragment = torch.tensor([[20.0, 0.0, 0.0]], dtype=torch.float64)
    combined = torch.cat([base, fragment], dim=0)

    reference = _c2_cutoff_matrix(base, cutoff=2.0)
    after = _c2_cutoff_matrix(combined, cutoff=2.0)[:3, :3]
    torch.testing.assert_close(after, reference, atol=0.0, rtol=0.0)


def test_truncated_centered_implicit_kernel_detects_fragment_drift() -> None:
    base = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    fragment = torch.tensor([[20.0, 0.0, 0.0]], dtype=torch.float64)
    combined = torch.cat([base, fragment], dim=0)
    kernel = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(
            scales=(10.0,),
            order=2,
            exclude_self=False,
            normalization="none",
        )
    ).double()

    base_features = kernel.prepare(
        base,
        torch.zeros(3, dtype=torch.long),
    ).features
    combined_features = kernel.prepare(
        combined,
        torch.zeros(4, dtype=torch.long),
    ).features
    reference = base_features @ base_features.T
    after = combined_features[:3] @ combined_features[:3].T

    assert (after - reference).abs().max() > 1e-8


def test_zero_value_fragment_message_drift_is_reportable() -> None:
    base = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    fragment = torch.tensor([[20.0, 0.0, 0.0]], dtype=torch.float64)
    values = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    combined_values = torch.cat([values, torch.zeros(1, 1, dtype=torch.float64)])
    kernel = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(
            scales=(10.0,),
            order=2,
            exclude_self=True,
            normalization="none",
        )
    ).double()

    reference = kernel(
        values,
        base,
        torch.zeros(3, dtype=torch.long),
    ).output
    after = kernel(
        combined_values,
        torch.cat([base, fragment], dim=0),
        torch.zeros(4, dtype=torch.long),
    ).output[:3]

    assert (after - reference).abs().max() > 1e-8
