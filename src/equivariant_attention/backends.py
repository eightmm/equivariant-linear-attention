from __future__ import annotations

from typing import Literal

import torch
from torch import nn

BackendName = Literal["cuequivariance", "e3nn", "cartesian"]


class SphericalHarmonicsBackend(nn.Module):
    """Small runtime adapter around cuEquivariance with explicit fallbacks."""

    def __init__(self, preferred: BackendName = "cuequivariance", lmax: int = 1) -> None:
        super().__init__()
        if preferred not in {"cuequivariance", "e3nn", "cartesian"}:
            msg = f"unknown backend: {preferred}"
            raise ValueError(msg)
        if lmax not in {1, 2}:
            msg = f"lmax must be 1 or 2, got {lmax}"
            raise ValueError(msg)
        self.preferred = preferred
        self.lmax = lmax
        self._active: BackendName = "cartesian"
        self._cu_sh: nn.Module | None = None
        self._e3nn_spherical_harmonics = None
        self._degrees = list(range(lmax + 1))

        if preferred == "cuequivariance":
            try:
                import cuequivariance_torch as cuet
            except (ImportError, ModuleNotFoundError):
                self._load_e3nn()
            else:
                self._cu_sh = cuet.SphericalHarmonics(self._degrees, normalize=True)
                self._active = "cuequivariance"
        elif preferred == "e3nn":
            self._load_e3nn()

    @property
    def active(self) -> BackendName:
        return self._active

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        if vectors.ndim != 2 or vectors.shape[-1] != 3:
            msg = f"vectors must have shape (N, 3), got {tuple(vectors.shape)}"
            raise ValueError(msg)
        if vectors.shape[0] == 0:
            return vectors.new_zeros((0, 1 + 3 + (5 if self.lmax == 2 else 0)))

        if self._active == "cuequivariance" and self._cu_sh is not None:
            try:
                return self._cu_sh(vectors)
            except (RuntimeError, ValueError):
                self._load_e3nn()

        if self._active == "e3nn" and self._e3nn_spherical_harmonics is not None:
            try:
                return self._e3nn_spherical_harmonics(
                    self._degrees,
                    vectors,
                    normalize=True,
                    normalization="component",
                )
            except (RuntimeError, ValueError):
                self._active = "cartesian"

        return self._cartesian_sh(vectors)

    def _load_e3nn(self) -> None:
        try:
            from e3nn.o3 import spherical_harmonics
        except (ImportError, ModuleNotFoundError):
            self._active = "cartesian"
        else:
            self._e3nn_spherical_harmonics = spherical_harmonics
            self._active = "e3nn"

    def _cartesian_sh(self, vectors: torch.Tensor) -> torch.Tensor:
        norm = vectors.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(vectors.dtype).eps)
        unit = vectors / norm
        ones = torch.ones((vectors.shape[0], 1), dtype=vectors.dtype, device=vectors.device)
        if self.lmax == 1:
            return torch.cat([ones, unit], dim=-1)
        x, y, z = unit.unbind(dim=-1)
        l2 = torch.stack(
            [
                x * y,
                y * z,
                2.0 * z * z - x * x - y * y,
                x * z,
                x * x - y * y,
            ],
            dim=-1,
        )
        return torch.cat([ones, unit, l2], dim=-1)
