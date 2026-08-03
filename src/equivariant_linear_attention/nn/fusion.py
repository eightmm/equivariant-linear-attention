from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn

from .parity import _ParityState, _st_square


MessageTuple = tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class BranchFusionDiagnostics:
    """Invariant routing information for one global/local fusion call."""

    weights: torch.Tensor
    global_rms: torch.Tensor
    local_rms: torch.Tensor
    balance_strength: torch.Tensor


class RMSAwareBranchFusion(nn.Module):
    """Identity-initialized global/local fusion for equivariant messages.

    Routing weights depend only on the even scalar state and invariant message
    RMS values. The same scalar weight is broadcast over all components of one
    irrep sector, so the operation commutes with the complete tracked O(3)
    action. A learnable balance path can interpolate from the incumbent raw sum
    to a variance-balanced mixture without changing the initial function.
    """

    sector_names = (
        "0e",
        "0o",
        "1o",
        "1e",
        "2e",
        "2o",
    )

    def __init__(
        self,
        *,
        scalar_width: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if isinstance(scalar_width, bool) or not isinstance(scalar_width, int):
            raise TypeError("scalar_width must be an integer")
        if scalar_width <= 0:
            raise ValueError("scalar_width must be positive")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            raise TypeError("eps must be a real number")
        numeric_eps = float(eps)
        if not isfinite(numeric_eps) or numeric_eps <= 0.0:
            raise ValueError("eps must be finite and positive")

        self.scalar_width = scalar_width
        self.eps = max(numeric_eps, 1e-8)
        hidden = max(8, min(32, scalar_width))
        self.router = nn.Sequential(
            nn.Linear(scalar_width + 2 * len(self.sector_names), hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * len(self.sector_names)),
        )
        self.balance_strength = nn.Parameter(
            torch.zeros(len(self.sector_names))
        )
        self.reset_identity()

    def reset_identity(self) -> None:
        """Restore exact raw ``global + local`` routing."""

        output = self.router[-1]
        if not isinstance(output, nn.Linear):
            raise RuntimeError("unexpected branch-router output module")
        with torch.no_grad():
            output.weight.zero_()
            output.bias.zero_()
            self.balance_strength.zero_()

    @staticmethod
    def _work_dtype(value: torch.Tensor) -> torch.dtype:
        return torch.float64 if value.dtype == torch.float64 else torch.float32

    def _rms(self, value: torch.Tensor, sector_index: int) -> torch.Tensor:
        if value.ndim < 2:
            raise ValueError("message tensors must have a node and feature axis")
        dtype = self._work_dtype(value)
        work = value.to(dtype=dtype)
        if sector_index in {4, 5}:
            if work.ndim != 3 or work.shape[-1] != 5:
                raise ValueError(
                    f"{self.sector_names[sector_index]} messages must have "
                    "shape (nodes, channels, 5)"
                )
            # The five stored ST coordinates are not an orthonormal basis:
            # zz=-xx-yy and off-diagonal entries occur twice in the full
            # Frobenius norm. Use the actual invariant norm before reducing
            # channels so a learned router still commutes with generic O(3).
            square = _st_square(work).mean(dim=1, keepdim=True) / 5.0
            return torch.sqrt(square.unsqueeze(-1) + self.eps)
        dimensions = tuple(range(1, work.ndim))
        return torch.sqrt(
            work.square().mean(dim=dimensions, keepdim=True) + self.eps
        )

    @staticmethod
    def _summary(rms: torch.Tensor) -> torch.Tensor:
        return rms.reshape(rms.shape[0], -1)

    @staticmethod
    def _broadcast(weight: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return weight.reshape(
            weight.shape[0],
            *((1,) * (value.ndim - 1)),
        ).to(dtype=value.dtype)

    def routing_weights(
        self,
        state: _ParityState,
        global_message: Sequence[torch.Tensor],
        local_message: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        if len(global_message) != len(self.sector_names):
            raise ValueError("global_message must contain six irrep sectors")
        if len(local_message) != len(self.sector_names) + 1:
            raise ValueError(
                "local_message must contain six sectors plus one chiral scalar"
            )
        global_rms = tuple(
            self._rms(value, index)
            for index, value in enumerate(global_message)
        )
        local_rms = tuple(
            self._rms(value, index)
            for index, value in enumerate(local_message[:6])
        )
        descriptor = torch.cat(
            [
                state.even_scalar,
                *(
                    torch.log(self._summary(value).clamp_min(self.eps)).to(
                        dtype=state.even_scalar.dtype
                    )
                    for value in global_rms
                ),
                *(
                    torch.log(self._summary(value).clamp_min(self.eps)).to(
                        dtype=state.even_scalar.dtype
                    )
                    for value in local_rms
                ),
            ],
            dim=-1,
        )
        logits = self.router(descriptor).reshape(
            descriptor.shape[0],
            len(self.sector_names),
            2,
        )
        # A two-class softmax is exactly a sigmoid of the logit difference.
        # Form the complementary local weight directly to avoid launching a
        # generic softmax kernel for six independent binary decisions.
        global_weight = 2.0 * torch.sigmoid(logits[..., 0] - logits[..., 1])
        weights = torch.stack((global_weight, 2.0 - global_weight), dim=-1)
        return weights, global_rms, local_rms

    def diagnostics(
        self,
        state: _ParityState,
        global_message: Sequence[torch.Tensor],
        local_message: Sequence[torch.Tensor],
    ) -> BranchFusionDiagnostics:
        weights, global_rms, local_rms = self.routing_weights(
            state,
            global_message,
            local_message,
        )
        return BranchFusionDiagnostics(
            weights=weights,
            global_rms=torch.cat(
                [self._summary(value) for value in global_rms],
                dim=-1,
            ),
            local_rms=torch.cat(
                [self._summary(value) for value in local_rms],
                dim=-1,
            ),
            balance_strength=torch.tanh(self.balance_strength),
        )

    def forward(
        self,
        state: _ParityState,
        global_message: Sequence[torch.Tensor],
        local_message: Sequence[torch.Tensor],
    ) -> tuple[MessageTuple, MessageTuple]:
        weights, global_rms, local_rms = self.routing_weights(
            state,
            global_message,
            local_message,
        )
        fused: list[torch.Tensor] = []
        for index, (global_value, local_value) in enumerate(
            zip(global_message, local_message[:6], strict=True)
        ):
            if global_value.shape != local_value.shape:
                raise ValueError(
                    f"global/local {self.sector_names[index]} shapes must match"
                )

            # Autocast may independently select BF16 or FP32 for the global
            # and local operators. Match ordinary ``global + local`` promotion
            # at their boundary so zero-initialized routing still reproduces
            # the admitted path without rejecting a valid mixed-precision run.
            native_dtype = torch.promote_types(
                global_value.dtype,
                local_value.dtype,
            )
            global_native = global_value.to(dtype=native_dtype)
            local_native = local_value.to(dtype=native_dtype)
            global_weight_native = self._broadcast(
                weights[:, index, 0],
                global_native,
            )
            local_weight_native = self._broadcast(
                weights[:, index, 1],
                local_native,
            )

            work_dtype = self._work_dtype(global_native)
            global_weight = global_weight_native.to(dtype=work_dtype)
            local_weight = local_weight_native.to(dtype=work_dtype)
            global_scale = global_rms[index].to(dtype=work_dtype)
            local_scale = local_rms[index].to(dtype=work_dtype)
            reference_scale = torch.sqrt(
                0.5 * (global_scale.square() + local_scale.square())
            )
            weight_norm = torch.sqrt(
                0.5 * (global_weight.square() + local_weight.square())
                + self.eps
            )
            # Interpolate the small invariant coefficients before touching the
            # full irrep tensors. This is algebraically the same as materializing
            # both ``weighted`` and ``balanced`` messages and blending them, but
            # it performs only one pair of message multiplications and avoids a
            # second full-size autograd branch.
            balanced_global_weight = (
                reference_scale * global_weight / global_scale / weight_norm
            )
            balanced_local_weight = (
                reference_scale * local_weight / local_scale / weight_norm
            )
            strength = torch.tanh(self.balance_strength[index]).to(
                dtype=work_dtype
            )
            effective_global_weight = global_weight + strength * (
                balanced_global_weight - global_weight
            )
            effective_local_weight = local_weight + strength * (
                balanced_local_weight - local_weight
            )
            fused.append(
                effective_global_weight.to(dtype=native_dtype) * global_native
                + effective_local_weight.to(dtype=native_dtype) * local_native
            )

        chiral = local_message[6]
        chiral_weight = self._broadcast(weights[:, 1, 1], chiral)
        routed_global: MessageTuple = tuple(fused)
        routed_local: MessageTuple = (
            *(torch.zeros_like(value) for value in local_message[:6]),
            chiral_weight * chiral,
        )
        return routed_global, routed_local


__all__ = [
    "BranchFusionDiagnostics",
    "RMSAwareBranchFusion",
]
