"""Exact dense directed triangle mixing for the canonical pair stack."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


def _validate_pair_inputs(
    z: torch.Tensor,
    pair_mask: torch.Tensor,
    *,
    pair_width: int,
) -> None:
    if not isinstance(z, torch.Tensor) or not torch.is_floating_point(z):
        raise TypeError("z must be a floating-point tensor")
    if z.ndim != 4 or z.shape[1] != z.shape[2] or z.shape[-1] != pair_width:
        raise ValueError(f"z must have shape (B,N,N,{pair_width})")
    if not isinstance(pair_mask, torch.Tensor):
        raise TypeError("pair_mask must be a tensor")
    if pair_mask.dtype != torch.bool or pair_mask.shape != z.shape[:3]:
        raise ValueError("pair_mask must be boolean with shape (B,N,N)")
    if pair_mask.device != z.device:
        raise ValueError("pair_mask and z must share one device")


class GatedTriangleMultiplication(nn.Module):
    """One exact, gated outgoing or incoming triangle contraction."""

    def __init__(
        self,
        pair_width: int,
        hidden_width: int,
        direction: Literal["outgoing", "incoming"],
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if pair_width <= 0 or hidden_width <= 0:
            raise ValueError("pair_width and hidden_width must be positive")
        if direction not in {"outgoing", "incoming"}:
            raise ValueError("direction must be 'outgoing' or 'incoming'")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.pair_width = int(pair_width)
        self.hidden_width = int(hidden_width)
        self.direction = direction

        self.norm = nn.LayerNorm(pair_width, eps=eps)
        self.linear_a = nn.Linear(pair_width, hidden_width)
        self.gate_a = nn.Linear(pair_width, hidden_width)
        self.linear_b = nn.Linear(pair_width, hidden_width)
        self.gate_b = nn.Linear(pair_width, hidden_width)
        self.center_norm = nn.LayerNorm(hidden_width, eps=eps)
        self.output_projection = nn.Linear(hidden_width, pair_width)
        self.output_gate = nn.Linear(pair_width, pair_width)

        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        nn.init.zeros_(self.output_gate.weight)
        nn.init.ones_(self.output_gate.bias)

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        _validate_pair_inputs(z, pair_mask, pair_width=self.pair_width)
        z_norm = self.norm(z)
        a = self.linear_a(z_norm) * torch.sigmoid(self.gate_a(z_norm))
        b = self.linear_b(z_norm) * torch.sigmoid(self.gate_b(z_norm))
        mask = pair_mask.unsqueeze(-1).to(dtype=z.dtype)
        a = a * mask
        b = b * mask

        if self.direction == "outgoing":
            contracted = torch.einsum("bikc,bjkc->bijc", a, b)
        else:
            contracted = torch.einsum("bkjc,bkic->bijc", a, b)
        contracted = self.center_norm(contracted)
        update = self.output_projection(contracted)
        update = update * torch.sigmoid(self.output_gate(z_norm))
        return update * mask


class PairTransition(nn.Module):
    """Zero-initialized pre-norm SwiGLU transition over pair channels."""

    def __init__(
        self,
        pair_width: int,
        expansion: int = 4,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if pair_width <= 0:
            raise ValueError("pair_width must be positive")
        if isinstance(expansion, bool) or not isinstance(expansion, int):
            raise TypeError("expansion must be an integer")
        if expansion <= 0:
            raise ValueError("expansion must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.pair_width = int(pair_width)
        self.hidden_width = int(pair_width * expansion)
        self.norm = nn.LayerNorm(pair_width, eps=eps)
        self.in_projection = nn.Linear(pair_width, 2 * self.hidden_width)
        self.out_projection = nn.Linear(self.hidden_width, pair_width)
        nn.init.zeros_(self.out_projection.weight)
        nn.init.zeros_(self.out_projection.bias)

    def _project(self, value: torch.Tensor) -> torch.Tensor:
        a, b = self.in_projection(value).chunk(2, dim=-1)
        return self.out_projection(F.silu(a) * b)

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        _validate_pair_inputs(z, pair_mask, pair_width=self.pair_width)
        normalized = self.norm(z)
        if chunk_size is None:
            update = self._project(normalized)
        else:
            if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
                raise TypeError("chunk_size must be an integer or None")
            if chunk_size <= 0:
                raise ValueError("chunk_size must be positive")
            flat = normalized.reshape(-1, self.pair_width)
            chunks = tuple(
                self._project(flat[start : start + chunk_size])
                for start in range(0, flat.shape[0], chunk_size)
            )
            update = (
                torch.cat(chunks, dim=0).reshape_as(z)
                if chunks
                else torch.empty_like(z)
            )
        return update * pair_mask.unsqueeze(-1).to(dtype=z.dtype)


class _RowwiseDropout(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        self.probability = float(probability)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return value
        row_mask = value.new_ones((value.shape[0], value.shape[1], 1, value.shape[-1]))
        return value * F.dropout(
            row_mask,
            p=self.probability,
            training=True,
        )


class TrianglePairBlock(nn.Module):
    """The single canonical dense pair order: out, in, then transition."""

    def __init__(
        self,
        pair_width: int,
        triangle_hidden: int,
        transition_factor: int = 4,
        dropout: float = 0.1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.pair_width = int(pair_width)
        self.outgoing = GatedTriangleMultiplication(
            pair_width,
            triangle_hidden,
            "outgoing",
            eps,
        )
        self.incoming = GatedTriangleMultiplication(
            pair_width,
            triangle_hidden,
            "incoming",
            eps,
        )
        self.transition = PairTransition(pair_width, transition_factor, eps)
        self.triangle_dropout = _RowwiseDropout(dropout)

    def _remask(self, z: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        return z * pair_mask.unsqueeze(-1).to(dtype=z.dtype)

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        _validate_pair_inputs(z, pair_mask, pair_width=self.pair_width)
        z = self._remask(z, pair_mask)
        z = self._remask(
            z + self.triangle_dropout(self.outgoing(z, pair_mask)),
            pair_mask,
        )
        z = self._remask(
            z + self.triangle_dropout(self.incoming(z, pair_mask)),
            pair_mask,
        )
        z = self._remask(z + self.transition(z, pair_mask), pair_mask)
        return z


__all__ = [
    "GatedTriangleMultiplication",
    "PairTransition",
    "TrianglePairBlock",
]
