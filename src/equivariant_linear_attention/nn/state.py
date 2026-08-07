from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn

from ..irreps import IrrepLayout, pack_irreps, split_irreps
from .ops import bounded_scalar, bounded_st, st_square, unit_ball


class ChannelMix(nn.Module):
    """GEMM-backed mixing on multiplicity axes, shared across irrep components."""

    def __init__(
        self, in_channels: int, out_channels: int, *, zero_init: bool = False
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        if zero_init:
            nn.init.zeros_(self.weight)
        else:
            nn.init.normal_(self.weight, std=1.0 / sqrt(max(1, in_channels)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[1] != self.in_channels:
            raise ValueError(f"value multiplicity must be {self.in_channels}")
        # Move multiplicity to the final dimension so F.linear lowers to GEMM
        # for scalars, vectors, and tensors without separate einsum kernels.
        mixed = F.linear(
            value.movedim(1, -1),
            self.weight.to(dtype=value.dtype),
        )
        return mixed.movedim(-1, 1)


class PairedChannelMix(nn.Module):
    """Apply two independent parity-preserving channel maps in one batched GEMM."""

    def __init__(
        self, in_channels: int, out_channels: int, *, zero_init: bool = False
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.weight = nn.Parameter(torch.empty(2, out_channels, in_channels))
        if zero_init:
            nn.init.zeros_(self.weight)
        else:
            nn.init.normal_(self.weight, std=1.0 / sqrt(max(1, in_channels)))

    def forward(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if first.shape != second.shape:
            raise ValueError("paired channel inputs must have identical shapes")
        if first.shape[1] != self.in_channels:
            raise ValueError(f"value multiplicity must be {self.in_channels}")
        stacked = torch.stack((first, second), dim=0)
        moved = stacked.movedim(2, -1)
        flat = moved.reshape(2, -1, self.in_channels)
        output = torch.bmm(
            flat,
            self.weight.to(dtype=first.dtype).transpose(1, 2),
        )
        output = output.reshape(2, *moved.shape[1:-1], self.out_channels)
        output = output.movedim(-1, 2)
        return output[0], output[1]


@dataclass(frozen=True)
class ParityState:
    even_scalar: torch.Tensor
    odd_scalar: torch.Tensor
    polar_vector: torch.Tensor
    axial_vector: torch.Tensor
    even_tensor: torch.Tensor
    odd_tensor: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.even_scalar.shape[0])

    def add(self, other: ParityState) -> ParityState:
        return ParityState(
            self.even_scalar + other.even_scalar,
            self.odd_scalar + other.odd_scalar,
            self.polar_vector + other.polar_vector,
            self.axial_vector + other.axial_vector,
            self.even_tensor + other.even_tensor,
            self.odd_tensor + other.odd_tensor,
        )

    def subtract(self, other: ParityState) -> ParityState:
        return ParityState(
            self.even_scalar - other.even_scalar,
            self.odd_scalar - other.odd_scalar,
            self.polar_vector - other.polar_vector,
            self.axial_vector - other.axial_vector,
            self.even_tensor - other.even_tensor,
            self.odd_tensor - other.odd_tensor,
        )

    def scale(self, scalar: torch.Tensor | float) -> ParityState:
        if isinstance(scalar, torch.Tensor):

            def apply(value: torch.Tensor) -> torch.Tensor:
                suffix = (1,) * (value.ndim - scalar.ndim)
                return value * scalar.reshape(*scalar.shape, *suffix)

        else:

            def apply(value: torch.Tensor) -> torch.Tensor:
                return value * float(scalar)

        return ParityState(*(apply(value) for value in self.as_tuple()))

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return (
            self.even_scalar,
            self.odd_scalar,
            self.polar_vector,
            self.axial_vector,
            self.even_tensor,
            self.odd_tensor,
        )

    def bounded(self, eps: float) -> ParityState:
        return ParityState(
            bounded_scalar(self.even_scalar, eps),
            bounded_scalar(self.odd_scalar, eps),
            unit_ball(self.polar_vector, eps),
            unit_ball(self.axial_vector, eps),
            bounded_st(self.even_tensor, eps),
            bounded_st(self.odd_tensor, eps),
        )


class InputProjection(nn.Module):
    def __init__(
        self, layout: IrrepLayout, *, scalar_width: int, num_heads: int
    ) -> None:
        super().__init__()
        self.layout = layout
        self.scalar_width = scalar_width
        self.num_heads = num_heads
        modules: dict[str, nn.Module] = {}
        for block in layout.blocks:
            name = str(block.irrep)
            width = scalar_width if name == "0e" else num_heads
            if block.irrep.degree == 0:
                modules[name] = nn.Linear(block.multiplicity, width, bias=name == "0e")
            elif block.irrep.degree in {1, 2}:
                modules[name] = ChannelMix(block.multiplicity, width)
            else:
                raise ValueError("persistent input irreps must have l<=2")
        self.projectors = nn.ModuleDict(modules)

    def forward(self, value: torch.Tensor) -> ParityState:
        blocks = split_irreps(self.layout, value)
        nodes = value.shape[0]

        def scalar(name: str, width: int) -> torch.Tensor:
            if name not in self.projectors:
                return value.new_zeros((nodes, width))
            return self.projectors[name](blocks[name].squeeze(-1))

        def geometric(name: str, dim: int) -> torch.Tensor:
            if name not in self.projectors:
                return value.new_zeros((nodes, self.num_heads, dim))
            return self.projectors[name](blocks[name])

        return ParityState(
            scalar("0e", self.scalar_width),
            scalar("0o", self.num_heads),
            geometric("1o", 3),
            geometric("1e", 3),
            geometric("2e", 5),
            geometric("2o", 5),
        )


class OutputProjection(nn.Module):
    def __init__(
        self, layout: IrrepLayout, *, scalar_width: int, num_heads: int
    ) -> None:
        super().__init__()
        self.layout = layout
        modules: dict[str, nn.Module] = {}
        for block in layout.blocks:
            name = str(block.irrep)
            in_channels = scalar_width if name == "0e" else num_heads
            if block.irrep.degree == 0:
                modules[name] = nn.Linear(
                    in_channels, block.multiplicity, bias=name == "0e"
                )
            elif block.irrep.degree in {1, 2}:
                modules[name] = ChannelMix(in_channels, block.multiplicity)
            else:
                raise ValueError("persistent output irreps must have l<=2")
        self.projectors = nn.ModuleDict(modules)

    def forward(self, state: ParityState) -> torch.Tensor:
        source = {
            "0e": state.even_scalar.unsqueeze(-1),
            "0o": state.odd_scalar.unsqueeze(-1),
            "1o": state.polar_vector,
            "1e": state.axial_vector,
            "2e": state.even_tensor,
            "2o": state.odd_tensor,
        }
        output: dict[str, torch.Tensor] = {}
        for block in self.layout.blocks:
            name = str(block.irrep)
            if block.irrep.degree == 0:
                output[name] = self.projectors[name](
                    source[name].squeeze(-1)
                ).unsqueeze(-1)
            else:
                output[name] = self.projectors[name](source[name])
        return pack_irreps(self.layout, output)


class EquivariantRMSNorm(nn.Module):
    def __init__(self, *, scalar_width: int, num_heads: int, eps: float) -> None:
        super().__init__()
        self.eps = float(eps)
        self.even_gain = nn.Parameter(torch.ones(scalar_width))
        self.odd_gain = nn.Parameter(torch.ones(num_heads))
        self.polar_gain = nn.Parameter(torch.ones(num_heads))
        self.axial_gain = nn.Parameter(torch.ones(num_heads))
        self.even_tensor_gain = nn.Parameter(torch.ones(num_heads))
        self.odd_tensor_gain = nn.Parameter(torch.ones(num_heads))

    def forward(self, state: ParityState) -> ParityState:
        dtype = (
            torch.float64 if state.even_scalar.dtype == torch.float64 else torch.float32
        )

        def scalar(value: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
            work = value.to(dtype=dtype)
            rms = torch.sqrt(work.square().mean(dim=-1, keepdim=True) + self.eps)
            return (work / rms * gain.to(dtype=dtype)).to(dtype=value.dtype)

        def vector(value: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
            work = value.to(dtype=dtype)
            rms = torch.sqrt(work.square().mean(dim=-1, keepdim=True) + self.eps)
            return (work / rms * gain.to(dtype=dtype)[None, :, None]).to(
                dtype=value.dtype
            )

        def tensor(value: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
            work = value.to(dtype=dtype)
            rms = torch.sqrt(st_square(work).unsqueeze(-1) / 5.0 + self.eps)
            return (work / rms * gain.to(dtype=dtype)[None, :, None]).to(
                dtype=value.dtype
            )

        return ParityState(
            scalar(state.even_scalar, self.even_gain),
            scalar(state.odd_scalar, self.odd_gain),
            vector(state.polar_vector, self.polar_gain),
            vector(state.axial_vector, self.axial_gain),
            tensor(state.even_tensor, self.even_tensor_gain),
            tensor(state.odd_tensor, self.odd_tensor_gain),
        )


def state_invariants(state: ParityState, eps: float) -> torch.Tensor:
    return torch.cat(
        (
            state.even_scalar,
            torch.sqrt(state.odd_scalar.square() + eps),
            torch.sqrt(state.polar_vector.square().mean(dim=-1) + eps),
            torch.sqrt(state.axial_vector.square().mean(dim=-1) + eps),
            torch.sqrt(st_square(state.even_tensor) / 5.0 + eps),
            torch.sqrt(st_square(state.odd_tensor) / 5.0 + eps),
        ),
        dim=-1,
    )


__all__ = [
    "ChannelMix",
    "EquivariantRMSNorm",
    "InputProjection",
    "OutputProjection",
    "PairedChannelMix",
    "ParityState",
    "state_invariants",
]
