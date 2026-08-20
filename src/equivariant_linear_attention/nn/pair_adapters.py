"""Invariant adapters between ordered pair memory and equivariant node state."""

from __future__ import annotations

import torch
from torch import nn

from .pair_state import DensePairState
from .state import ParityState


class PairToNodeSummary(nn.Module):
    """Normalized gated outgoing/incoming summaries in packed node order."""

    def __init__(self, *, pair_width: int, context_width: int, eps: float) -> None:
        super().__init__()
        self.eps = float(eps)
        self.out_gate = nn.Linear(pair_width, context_width)
        self.out_value = nn.Linear(pair_width, context_width)
        self.in_gate = nn.Linear(pair_width, context_width)
        self.in_value = nn.Linear(pair_width, context_width)
        self.output = nn.Linear(2 * context_width, context_width)
        # This produces a context representation, not a residual update.  The
        # downstream injection projections are zero-initialized; zeroing this
        # map as well would put two zero Jacobians in series and make the whole
        # pair-to-node route gradient-dead.
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _summary(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        gate_projection: nn.Linear,
        value_projection: nn.Linear,
    ) -> torch.Tensor:
        gate = torch.sigmoid(gate_projection(z))
        gate = gate * mask[..., None].to(dtype=gate.dtype)
        value = value_projection(z)
        work_dtype = (
            torch.float32 if z.dtype in (torch.float16, torch.bfloat16) else z.dtype
        )
        numerator = (gate.to(dtype=work_dtype) * value.to(dtype=work_dtype)).sum(dim=2)
        denominator = gate.to(dtype=work_dtype).sum(dim=2).clamp_min(self.eps)
        return (numerator / denominator).to(dtype=z.dtype)

    def forward(self, pair: DensePairState) -> torch.Tensor:
        outgoing = self._summary(
            pair.z,
            pair.pair_mask,
            self.out_gate,
            self.out_value,
        )
        incoming = self._summary(
            pair.z.transpose(1, 2),
            pair.pair_mask.transpose(1, 2),
            self.in_gate,
            self.in_value,
        )
        dense = self.output(torch.cat((outgoing, incoming), dim=-1))
        dense = dense * pair.node_mask[..., None].to(dtype=dense.dtype)
        return pair.gather_nodes(dense)


class PairContextInjection(nn.Module):
    """Inject invariant context without mixing any geometric component axis."""

    def __init__(
        self,
        *,
        context_width: int,
        scalar_width: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.scalar_width = int(scalar_width)
        self.num_heads = int(num_heads)
        self.even_residual = nn.Linear(context_width, scalar_width)
        self.gates = nn.Linear(context_width, scalar_width + 5 * num_heads)
        nn.init.zeros_(self.even_residual.weight)
        nn.init.zeros_(self.even_residual.bias)
        nn.init.zeros_(self.gates.weight)
        nn.init.zeros_(self.gates.bias)

    def forward(self, state: ParityState, context: torch.Tensor) -> ParityState:
        if context.shape != (state.num_nodes, self.gates.in_features):
            raise ValueError("context must have shape (N,context_width)")
        raw = torch.tanh(self.gates(context))
        even_gate, other = torch.split(
            raw,
            (self.scalar_width, 5 * self.num_heads),
            dim=-1,
        )
        gates = other.reshape(state.num_nodes, 5, self.num_heads)
        return ParityState(
            (1.0 + even_gate) * state.even_scalar + self.even_residual(context),
            (1.0 + gates[:, 0]) * state.odd_scalar,
            (1.0 + gates[:, 1, :, None]) * state.polar_vector,
            (1.0 + gates[:, 2, :, None]) * state.axial_vector,
            (1.0 + gates[:, 3, :, None]) * state.even_tensor,
            (1.0 + gates[:, 4, :, None]) * state.odd_tensor,
        )


__all__ = ["PairContextInjection", "PairToNodeSummary"]
