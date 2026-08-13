from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch

from .ops import bounded_scalar, bounded_st, matrix_to_st, unit_ball


@dataclass(frozen=True)
class LocalFeatureJet:
    value: torch.Tensor
    gradient: torch.Tensor
    laplacian: torch.Tensor
    hessian: torch.Tensor
    wavelet_value: torch.Tensor
    wavelet_gradient: torch.Tensor
    wavelet_laplacian: torch.Tensor
    wavelet_hessian: torch.Tensor
    confidence: torch.Tensor
    scale: torch.Tensor


def decode_local_jet(
    coefficient: torch.Tensor,
    system: torch.Tensor,
    scale: torch.Tensor,
    *,
    eps: float,
) -> LocalFeatureJet:
    value = coefficient[..., 0, :]
    linear = coefficient[..., 1:4, :].movedim(-1, -2)
    quadratic = coefficient[..., 4:10, :].movedim(-1, -2)
    xx, xy, xz, yy, yz, zz = quadratic.unbind(dim=-1)
    root_two = sqrt(2.0)
    hessian = torch.stack(
        (
            torch.stack((2.0 * xx, root_two * xy, root_two * xz), dim=-1),
            torch.stack((root_two * xy, 2.0 * yy, root_two * yz), dim=-1),
            torch.stack((root_two * xz, root_two * yz, 2.0 * zz), dim=-1),
        ),
        dim=-2,
    )
    gradient = linear / scale[..., None, None]
    hessian = hessian / scale[..., None, None, None].square()
    laplacian = hessian.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
    hessian_st = matrix_to_st(hessian)
    normalized_hessian = hessian * scale[..., None, None, None].square()
    normalized_laplacian = laplacian * scale[..., None].square()
    wavelet_value = value[:, :-1] - value[:, 1:]
    wavelet_gradient = linear[:, :-1] - linear[:, 1:]
    wavelet_laplacian = normalized_laplacian[:, :-1] - normalized_laplacian[:, 1:]
    wavelet_hessian = matrix_to_st(
        normalized_hessian[:, :-1] - normalized_hessian[:, 1:]
    )

    trace = system.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    frobenius_square = system.square().sum(dim=(-2, -1))
    dimension = float(system.shape[-1])
    confidence = (
        trace.square() / (dimension * frobenius_square + eps)
    ).clamp(0.0, 1.0)
    return LocalFeatureJet(
        bounded_scalar(value, eps),
        unit_ball(gradient, eps),
        bounded_scalar(laplacian, eps),
        bounded_st(hessian_st, eps),
        bounded_scalar(wavelet_value, eps),
        unit_ball(wavelet_gradient, eps),
        bounded_scalar(wavelet_laplacian, eps),
        bounded_st(wavelet_hessian, eps),
        confidence,
        scale,
    )


__all__ = ["LocalFeatureJet", "decode_local_jet"]
