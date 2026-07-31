from __future__ import annotations

from math import isfinite

import torch
from torch import nn

from .implicit_spatial import (
    ImplicitGaussianSpatialKernel,
    ImplicitSpatialStateTransport,
)
from .layered_se3 import UnifiedSE3State


def _finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return numeric


class ImplicitSpatialResidual(nn.Module):
    """Zero-initializable edge-free residual over the canonical hidden state.

    This module is the integration boundary for replacing or augmenting an
    explicit sparse local residual. It applies one shared invariant kernel to
    every irrep sector, then uses copy-wise LayerScale before the residual add.
    The default scale is zero, so inserting the module does not change the
    incumbent function until training activates it.
    """

    def __init__(
        self,
        kernel: ImplicitGaussianSpatialKernel,
        *,
        scalar_width: int,
        num_heads: int,
        residual_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(kernel, ImplicitGaussianSpatialKernel):
            raise TypeError("kernel must be an ImplicitGaussianSpatialKernel")
        if isinstance(scalar_width, bool) or not isinstance(scalar_width, int):
            raise TypeError("scalar_width must be an integer")
        if isinstance(num_heads, bool) or not isinstance(num_heads, int):
            raise TypeError("num_heads must be an integer")
        if scalar_width <= 0 or num_heads <= 0:
            raise ValueError("scalar_width and num_heads must be positive")
        scale = _finite_nonnegative("residual_scale_init", residual_scale_init)
        self.transport = ImplicitSpatialStateTransport(kernel)
        self.even_scale = nn.Parameter(torch.full((scalar_width,), scale))
        self.odd_scale = nn.Parameter(torch.full((num_heads,), scale))
        self.polar_scale = nn.Parameter(torch.full((num_heads, 1), scale))
        self.axial_scale = nn.Parameter(torch.full((num_heads, 1), scale))
        self.even_tensor_scale = nn.Parameter(torch.full((num_heads, 1), scale))
        self.odd_tensor_scale = nn.Parameter(torch.full((num_heads, 1), scale))

    def delta(
        self,
        state: UnifiedSE3State,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> UnifiedSE3State:
        transported = self.transport(state, positions, batch)
        return UnifiedSE3State(
            even_scalar=transported.even_scalar
            * self.even_scale.to(dtype=transported.even_scalar.dtype),
            odd_scalar=transported.odd_scalar
            * self.odd_scale.to(dtype=transported.odd_scalar.dtype),
            polar_vector=transported.polar_vector
            * self.polar_scale.to(dtype=transported.polar_vector.dtype),
            axial_vector=transported.axial_vector
            * self.axial_scale.to(dtype=transported.axial_vector.dtype),
            even_tensor=transported.even_tensor
            * self.even_tensor_scale.to(dtype=transported.even_tensor.dtype),
            odd_tensor=transported.odd_tensor
            * self.odd_tensor_scale.to(dtype=transported.odd_tensor.dtype),
        )

    def forward(
        self,
        state: UnifiedSE3State,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> UnifiedSE3State:
        delta = self.delta(state, positions, batch)
        return UnifiedSE3State(
            even_scalar=state.even_scalar + delta.even_scalar,
            odd_scalar=state.odd_scalar + delta.odd_scalar,
            polar_vector=state.polar_vector + delta.polar_vector,
            axial_vector=state.axial_vector + delta.axial_vector,
            even_tensor=state.even_tensor + delta.even_tensor,
            odd_tensor=state.odd_tensor + delta.odd_tensor,
        )


__all__ = ["ImplicitSpatialResidual"]
