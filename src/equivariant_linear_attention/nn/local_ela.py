"""Pair-conditioned bounded local equivariant transport for TriELA."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .geometry import GeometryContext
from .local_closure import LocalEquivariantClosure
from .local_geometry import PointwiseLocalGeometry
from .local_support import LocalSupport, wendland_c2
from .ops import segment_sum
from .pair_state import DensePairState
from .state import EquivariantRMSNorm, ParityState


@dataclass(frozen=True)
class LocalELAOutput:
    state: ParityState
    context: torch.Tensor


class LocalELABlock(nn.Module):
    """Apply local tensor geometry and gate its delta with ordered pair memory."""

    def __init__(
        self,
        *,
        scalar_width: int,
        pair_width: int,
        num_heads: int,
        moment_rank: int,
        probe_rank: int,
        num_scales: int,
        max_points: int,
        chunk_size: int,
        residual_scale: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.scalar_width = int(scalar_width)
        self.pair_width = int(pair_width)
        self.num_heads = int(num_heads)
        self.eps = float(eps)
        self.norm = EquivariantRMSNorm(
            scalar_width=scalar_width,
            num_heads=num_heads,
            eps=eps,
        )
        self.geometry = PointwiseLocalGeometry(
            scalar_width=scalar_width,
            moment_rank=moment_rank,
            probe_rank=probe_rank,
            num_scales=num_scales,
            max_points=max_points,
            chunk_size=chunk_size,
            eps=eps,
        )
        self.closure = LocalEquivariantClosure(
            scalar_width=scalar_width,
            num_heads=num_heads,
            moment_rank=moment_rank,
            probe_rank=probe_rank,
            num_scales=num_scales,
            eps=eps,
        )
        self.edge_gate = nn.Linear(pair_width, 1)
        self.edge_value = nn.Linear(pair_width, pair_width)
        self.context_projection = nn.Linear(
            pair_width,
            scalar_width + 5 * num_heads,
        )
        # 2*sigmoid(0)=1: pair conditioning starts as an exact identity gate.
        nn.init.zeros_(self.context_projection.weight)
        nn.init.zeros_(self.context_projection.bias)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def _pair_context(
        self,
        pair: DensePairState,
        support: LocalSupport,
        *,
        num_nodes: int,
    ) -> torch.Tensor:
        source = support.source
        receiver = support.receiver
        graph = pair.packed_batch[receiver]
        receiver_slot = pair.packed_slot[receiver]
        source_slot = pair.packed_slot[source]
        z_edge = pair.z[graph, receiver_slot, source_slot]
        valid = pair.pair_mask[graph, receiver_slot, source_slot]
        gate = torch.sigmoid(self.edge_gate(z_edge)).squeeze(-1)
        scale = support.scale[support.receiver].to(dtype=support.distance.dtype)
        outer_window = wendland_c2(support.distance / scale.clamp_min(self.eps)).to(
            dtype=gate.dtype
        )
        gate = gate * valid.to(dtype=gate.dtype) * outer_window
        value = self.edge_value(z_edge)
        work_dtype = (
            torch.float32
            if z_edge.dtype in (torch.float16, torch.bfloat16)
            else z_edge.dtype
        )
        numerator = segment_sum(
            gate[:, None].to(dtype=work_dtype) * value.to(dtype=work_dtype),
            receiver,
            num_nodes,
        )
        denominator = segment_sum(
            gate.to(dtype=work_dtype),
            receiver,
            num_nodes,
        )
        work = numerator / denominator.clamp_min(self.eps)[:, None]
        return work.to(dtype=z_edge.dtype)

    def _gate_delta(
        self,
        delta: ParityState,
        context: torch.Tensor,
    ) -> ParityState:
        raw = self.context_projection(context)
        even, other = torch.split(
            raw,
            (self.scalar_width, 5 * self.num_heads),
            dim=-1,
        )
        geometric = other.reshape(delta.num_nodes, 5, self.num_heads)
        even_gate = 2.0 * torch.sigmoid(even)
        gates = 2.0 * torch.sigmoid(geometric)
        return ParityState(
            even_gate * delta.even_scalar,
            gates[:, 0] * delta.odd_scalar,
            gates[:, 1, :, None] * delta.polar_vector,
            gates[:, 2, :, None] * delta.axial_vector,
            gates[:, 3, :, None] * delta.even_tensor,
            gates[:, 4, :, None] * delta.odd_tensor,
        )

    def forward(
        self,
        state: ParityState,
        geometry: GeometryContext,
        support: LocalSupport,
        pair: DensePairState,
    ) -> LocalELAOutput:
        normalized = self.norm(state)
        local = self.geometry(normalized.even_scalar, geometry, support)
        delta = self.closure(normalized, local)
        context = self._pair_context(pair, support, num_nodes=state.num_nodes)
        delta = self._gate_delta(delta, context)
        return LocalELAOutput(
            state=state.add(delta.scale(self.residual_scale)),
            context=context,
        )


__all__ = ["LocalELABlock", "LocalELAOutput"]
