from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .irreps import CartesianIrreps
from .model import _scatter_mean, _symmetric_traceless, _symmetric_traceless_cross

RichAttentionMode = Literal["linear", "local"]


@dataclass(frozen=True)
class RichEquivariantAttentionConfig:
    node_dim: int
    hidden_irreps: str | CartesianIrreps = "32x0e + 8x1o + 4x2e"
    output_irreps: str | CartesianIrreps = "1x0e + 1x1o + 1x2e"
    num_layers: int = 3
    num_heads: int = 4
    attention_mode: RichAttentionMode = "local"
    vector_edge_bias: bool = False
    vector_edge_bias_scale: float = 0.1
    residual_scale_init: float = 0.1
    eps: float = 1e-12


class RichEquivariantAttention(nn.Module):
    """Persistent scalar/vector/tensor equivariant attention."""

    def __init__(self, config: RichEquivariantAttentionConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.hidden_irreps = CartesianIrreps.parse(config.hidden_irreps)
        self.output_irreps = CartesianIrreps.parse(config.output_irreps)

        self.scalar_in = nn.Linear(config.node_dim + 2, self.hidden_irreps.scalars)
        self.vector_in = _ScalarChannelLinear(config.node_dim + 1, self.hidden_irreps.vectors)
        self.tensor_in = _ScalarChannelLinear(config.node_dim + 1, self.hidden_irreps.tensors)
        layer_residual_scale = config.residual_scale_init / sqrt(config.num_layers)
        self.layers = nn.ModuleList(
            [
                _RichEquivariantLayer(
                    irreps=self.hidden_irreps,
                    num_heads=config.num_heads,
                    attention_mode=config.attention_mode,
                    vector_edge_bias=config.vector_edge_bias,
                    vector_edge_bias_scale=config.vector_edge_bias_scale,
                    residual_scale_init=layer_residual_scale,
                    eps=config.eps,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.scalar_out_norm = nn.LayerNorm(self.hidden_irreps.scalars)
        self.scalar_out = nn.Linear(self.hidden_irreps.scalars, self.output_irreps.scalars)
        self.vector_out = _ChannelMix(self.hidden_irreps.vectors, self.output_irreps.vectors)
        self.tensor_out = _ChannelMix(self.hidden_irreps.tensors, self.output_irreps.tensors)

    def forward(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None = None,
        neighbor_index: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | str]:
        single_graph = batch is None
        node_feats, pos, batch, neighbor_index, neighbor_mask = self._check_inputs(
            node_feats,
            pos,
            batch,
            neighbor_index,
            neighbor_mask,
        )
        center = pos.mean(dim=0, keepdim=True) if single_graph else _scatter_mean(pos, batch)
        centered = pos - center[batch]
        radius = centered.norm(dim=-1, keepdim=True)
        bounded_centered = _bounded_geometry(centered, self.config.eps)
        scalar_base = torch.cat([node_feats, torch.log1p(radius), torch.log1p(radius.square())], dim=-1)
        gate_base = torch.cat([node_feats, torch.log1p(radius)], dim=-1)

        scalars = self.scalar_in(scalar_base)
        vector_gate = torch.tanh(self.vector_in(gate_base))
        tensor_gate = torch.tanh(self.tensor_in(gate_base))
        vectors = vector_gate.unsqueeze(-1) * bounded_centered.unsqueeze(1)
        tensors = tensor_gate[..., None, None] * _symmetric_traceless(bounded_centered).unsqueeze(1)

        for layer in self.layers:
            scalars, vectors, tensors = layer(
                scalars,
                vectors,
                tensors,
                centered,
                None if single_graph else batch,
                neighbor_index,
                neighbor_mask,
            )

        node_scalars = self.scalar_out(self.scalar_out_norm(scalars))
        node_vectors = self.vector_out(vectors)
        node_tensors = self.tensor_out(tensors)
        if single_graph:
            graph_scalars = node_scalars.mean(dim=0, keepdim=True)
            graph_vectors = node_vectors.mean(dim=0, keepdim=True)
            graph_tensors = node_tensors.mean(dim=0, keepdim=True)
        else:
            graph_scalars = _scatter_mean(node_scalars, batch)
            graph_vectors = _scatter_mean(node_vectors, batch)
            graph_tensors = _scatter_mean(node_tensors, batch)

        return {
            "node_scalars": node_scalars,
            "node_vectors": node_vectors,
            "node_tensors": node_tensors,
            "graph_scalars": graph_scalars,
            "graph_vectors": graph_vectors,
            "graph_tensors": graph_tensors,
            "hidden_irreps": str(self.hidden_irreps),
            "output_irreps": str(self.output_irreps),
            "attention_mode": self.config.attention_mode,
        }

    def _check_inputs(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None,
        neighbor_index: torch.Tensor | None,
        neighbor_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if node_feats.ndim != 2 or node_feats.shape[1] != self.config.node_dim:
            msg = f"node_feats must have shape (N, {self.config.node_dim})"
            raise ValueError(msg)
        if pos.shape != (node_feats.shape[0], 3):
            msg = f"pos must have shape (N, 3), got {tuple(pos.shape)}"
            raise ValueError(msg)
        if not torch.is_floating_point(node_feats) or not torch.is_floating_point(pos):
            raise TypeError("node_feats and pos must be floating point tensors")
        if node_feats.device != pos.device:
            raise ValueError("node_feats and pos must be on the same device")
        if pos.dtype != node_feats.dtype:
            pos = pos.to(dtype=node_feats.dtype)
        if not torch.isfinite(node_feats).all() or not torch.isfinite(pos).all():
            raise ValueError("node_feats and pos must be finite")

        if batch is None:
            batch = torch.zeros(node_feats.shape[0], dtype=torch.long, device=node_feats.device)
        else:
            if batch.shape != (node_feats.shape[0],):
                msg = f"batch must have shape (N,), got {tuple(batch.shape)}"
                raise ValueError(msg)
            batch = batch.to(device=node_feats.device, dtype=torch.long)

        if neighbor_index is None:
            if neighbor_mask is not None:
                raise ValueError("neighbor_mask requires neighbor_index")
            return node_feats, pos, batch, None, None
        if self.config.attention_mode != "local":
            raise ValueError("neighbor_index is only supported in local mode")
        if neighbor_index.ndim != 2 or neighbor_index.shape[0] != node_feats.shape[0]:
            msg = f"neighbor_index must have shape (N, K), got {tuple(neighbor_index.shape)}"
            raise ValueError(msg)
        neighbor_index = neighbor_index.to(device=node_feats.device, dtype=torch.long)
        if (neighbor_index < 0).any() or (neighbor_index >= node_feats.shape[0]).any():
            raise ValueError("neighbor_index values are out of range")
        if neighbor_mask is None:
            neighbor_mask = torch.ones(neighbor_index.shape, dtype=torch.bool, device=node_feats.device)
        elif neighbor_mask.shape != neighbor_index.shape:
            msg = f"neighbor_mask must match neighbor_index shape {tuple(neighbor_index.shape)}"
            raise ValueError(msg)
        else:
            neighbor_mask = neighbor_mask.to(device=node_feats.device, dtype=torch.bool)
        return node_feats, pos, batch, neighbor_index, neighbor_mask


class _RichEquivariantLayer(nn.Module):
    def __init__(
        self,
        irreps: CartesianIrreps,
        num_heads: int,
        attention_mode: RichAttentionMode,
        vector_edge_bias: bool,
        vector_edge_bias_scale: float,
        residual_scale_init: float,
        eps: float,
    ) -> None:
        super().__init__()
        if irreps.scalars % num_heads != 0:
            raise ValueError("scalar hidden channels must be divisible by num_heads")
        self.irreps = irreps
        self.num_heads = num_heads
        self.head_dim = irreps.scalars // num_heads
        self.attention_mode = attention_mode
        self.vector_edge_bias = vector_edge_bias
        self.vector_edge_bias_scale = vector_edge_bias_scale
        self.eps = eps

        self.norm = nn.LayerNorm(irreps.scalars)
        self.query = nn.Linear(irreps.scalars, irreps.scalars)
        self.key = nn.Linear(irreps.scalars, irreps.scalars)
        self.scalar_value = nn.Linear(irreps.scalars, irreps.scalars)
        self.scalar_update_norm = nn.LayerNorm(irreps.scalars + irreps.vectors + irreps.tensors)
        self.scalar_update = nn.Linear(irreps.scalars + irreps.vectors + irreps.tensors, irreps.scalars)
        self.vector_mix = _ChannelMix(irreps.vectors, irreps.vectors)
        self.vector_from_tensor = _ChannelMix(irreps.tensors, irreps.vectors)
        self.vector_rel_gate = _ScalarChannelLinear(irreps.scalars, irreps.vectors)
        self.tensor_mix = _ChannelMix(irreps.tensors, irreps.tensors)
        self.tensor_from_vector = _ChannelMix(irreps.vectors, irreps.tensors)
        self.tensor_rel_gate = _ScalarChannelLinear(irreps.scalars, irreps.tensors)
        if vector_edge_bias:
            self.edge_query_vector = _HeadVectorMix(irreps.vectors, num_heads)
            self.edge_key_vector = _HeadVectorMix(irreps.vectors, num_heads)
            self.edge_vector_bias_weight = nn.Parameter(torch.zeros(num_heads, 4))
            self.edge_vector_bias_offset = nn.Parameter(torch.zeros(num_heads))
        self.scalar_residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.vector_residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.tensor_residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
        tensors: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None,
        neighbor_index: torch.Tensor | None,
        neighbor_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_norm = self.norm(scalars)
        q = self.query(s_norm).reshape(scalars.shape[0], self.num_heads, self.head_dim)
        k = self.key(s_norm).reshape(scalars.shape[0], self.num_heads, self.head_dim)
        scalar_value = self.scalar_value(s_norm).reshape(scalars.shape[0], self.num_heads, self.head_dim)
        vector_norm = _bounded_irrep(vectors, self.eps)
        tensor_norm = _bounded_irrep(tensors, self.eps)
        vector_value = self.vector_mix(vector_norm)
        tensor_value = self.tensor_mix(tensor_norm)
        vector_gate = torch.tanh(self.vector_rel_gate(s_norm))
        tensor_gate = torch.tanh(self.tensor_rel_gate(s_norm))
        vector_to_tensor = self.tensor_from_vector(vector_norm)
        tensor_to_vector = self.vector_from_tensor(tensor_norm)

        if self.attention_mode == "local":
            s_msg, v_msg, t_msg = self._local_messages(
                q,
                k,
                scalar_value,
                vector_value,
                tensor_value,
                vector_gate,
                tensor_gate,
                vector_to_tensor,
                tensor_to_vector,
                pos,
                neighbor_index,
                neighbor_mask,
            )
        else:
            s_msg, v_msg, t_msg = self._linear_messages(
                F.elu(q) + 1.0,
                F.elu(k) + 1.0,
                scalar_value,
                vector_value,
                tensor_value,
                vector_gate,
                tensor_gate,
                vector_to_tensor,
                tensor_to_vector,
                pos,
                batch,
            )

        scalar_invariants = torch.cat(
            [
                s_msg,
                _bounded_irrep(v_msg, self.eps).square().sum(dim=-1),
                _bounded_irrep(t_msg, self.eps).square().sum(dim=(-1, -2)),
            ],
            dim=-1,
        )
        scalar_delta = self.scalar_update(self.scalar_update_norm(scalar_invariants))
        scalars = scalars + self.scalar_residual_scale * scalar_delta
        vectors = vectors + self.vector_residual_scale * _bounded_irrep(v_msg, self.eps)
        tensors = tensors + self.tensor_residual_scale * _bounded_irrep(t_msg, self.eps)
        return scalars, vectors, tensors

    def _linear_messages(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        scalar_value: torch.Tensor,
        vector_value: torch.Tensor,
        tensor_value: torch.Tensor,
        vector_gate: torch.Tensor,
        tensor_gate: torch.Tensor,
        vector_to_tensor: torch.Tensor,
        tensor_to_vector: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch is None:
            return self._linear_graph(
                q,
                k,
                scalar_value,
                vector_value,
                tensor_value,
                vector_gate,
                tensor_gate,
                vector_to_tensor,
                tensor_to_vector,
                pos,
            )
        return self._linear_batched(
            q,
            k,
            scalar_value,
            vector_value,
            tensor_value,
            vector_gate,
            tensor_gate,
            vector_to_tensor,
            tensor_to_vector,
            pos,
            batch,
        )

    def _linear_batched(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        scalar_value: torch.Tensor,
        vector_value: torch.Tensor,
        tensor_value: torch.Tensor,
        vector_gate: torch.Tensor,
        tensor_gate: torch.Tensor,
        vector_to_tensor: torch.Tensor,
        tensor_to_vector: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = _bounded_geometry(pos, self.eps)
        num_graphs = int(batch.max().item()) + 1

        k_sum = _segment_sum(k, batch, num_graphs)
        denom = (q * k_sum[batch]).sum(dim=-1).clamp_min(self.eps)
        scalar_kv = _segment_sum(torch.einsum("nhd,nhe->nhde", k, scalar_value), batch, num_graphs)
        scalar_msg_h = torch.einsum("nhd,nhde->nhe", q, scalar_kv[batch]) / denom.unsqueeze(-1)
        scalar_msg = scalar_msg_h.reshape(q.shape[0], self.irreps.scalars)

        vector_base = _linear_value_attention_batched(q, k, vector_value, denom, batch, num_graphs)
        vector_rel = _linear_relative_vector_batched(q, k, vector_gate, pos, denom, batch, num_graphs)
        tensor_vec = torch.einsum("ncab,nb->nca", tensor_to_vector, pos)
        vector_tensor = _linear_value_attention_batched(q, k, tensor_vec, denom, batch, num_graphs)
        vector_msg = vector_base + vector_rel + vector_tensor

        tensor_base = _linear_value_attention_batched(q, k, tensor_value, denom, batch, num_graphs)
        tensor_rel = _linear_relative_tensor_batched(q, k, tensor_gate, pos, denom, batch, num_graphs)
        tensor_cross = _linear_cross_tensor_batched(q, k, vector_to_tensor, pos, denom, batch, num_graphs)
        tensor_msg = tensor_base + tensor_rel + tensor_cross
        return scalar_msg, vector_msg, tensor_msg

    def _linear_graph(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        scalar_value: torch.Tensor,
        vector_value: torch.Tensor,
        tensor_value: torch.Tensor,
        vector_gate: torch.Tensor,
        tensor_gate: torch.Tensor,
        vector_to_tensor: torch.Tensor,
        tensor_to_vector: torch.Tensor,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = _bounded_geometry(pos, self.eps)
        denom = (q * k.sum(dim=0).unsqueeze(0)).sum(dim=-1).clamp_min(self.eps)
        scalar_kv = torch.einsum("nhd,nhe->hde", k, scalar_value)
        scalar_msg_h = torch.einsum("nhd,hde->nhe", q, scalar_kv) / denom.unsqueeze(-1)
        scalar_msg = scalar_msg_h.reshape(q.shape[0], self.irreps.scalars)

        vector_base = _linear_value_attention(q, k, vector_value, denom)
        vector_rel = _linear_relative_vector(q, k, vector_gate, pos, denom)
        tensor_vec = torch.einsum("ncab,nb->nca", tensor_to_vector, pos)
        vector_tensor = _linear_value_attention(q, k, tensor_vec, denom)
        vector_msg = vector_base + vector_rel + vector_tensor

        tensor_base = _linear_value_attention(q, k, tensor_value, denom)
        tensor_rel = _linear_relative_tensor(q, k, tensor_gate, pos, denom)
        tensor_cross = _linear_cross_tensor(q, k, vector_to_tensor, pos, denom)
        tensor_msg = tensor_base + tensor_rel + tensor_cross
        return scalar_msg, vector_msg, tensor_msg

    def _local_messages(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        scalar_value: torch.Tensor,
        vector_value: torch.Tensor,
        tensor_value: torch.Tensor,
        vector_gate: torch.Tensor,
        tensor_gate: torch.Tensor,
        vector_to_tensor: torch.Tensor,
        tensor_to_vector: torch.Tensor,
        pos: torch.Tensor,
        neighbor_index: torch.Tensor | None,
        neighbor_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if neighbor_index is None or neighbor_mask is None:
            raise ValueError("rich local attention requires neighbor_index and neighbor_mask")
        k_n = k[neighbor_index]
        rel = pos[neighbor_index] - pos.unsqueeze(1)
        dist = rel.float().norm(dim=-1).to(dtype=pos.dtype).clamp_min(self.eps)
        rel_msg = rel / torch.sqrt(1.0 + dist.square()).unsqueeze(-1)
        logits = torch.einsum("nhd,nkhd->nkh", q, k_n) / sqrt(self.head_dim)
        if self.vector_edge_bias:
            logits = logits + self._vector_edge_attention_bias(vector_value, rel, dist, neighbor_index)
        logits = logits.masked_fill(~neighbor_mask.unsqueeze(-1), torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=1)

        scalar_n = scalar_value[neighbor_index]
        scalar_msg = torch.einsum("nkh,nkhd->nhd", attn, scalar_n).reshape(pos.shape[0], self.irreps.scalars)

        vector_msg = torch.einsum("nkh,nkca->nca", attn, vector_value[neighbor_index])
        vector_msg = vector_msg + torch.einsum("nkh,nkc,nka->nca", attn, vector_gate[neighbor_index], rel_msg)
        tensor_vec = torch.einsum("nkcab,nkb->nkca", tensor_to_vector[neighbor_index], rel_msg)
        vector_msg = vector_msg + torch.einsum("nkh,nkca->nca", attn, tensor_vec)

        tensor_msg = torch.einsum("nkh,nkcab->ncab", attn, tensor_value[neighbor_index])
        tensor_msg = tensor_msg + torch.einsum(
            "nkh,nkc,nkab->ncab",
            attn,
            tensor_gate[neighbor_index],
            _symmetric_traceless(rel_msg),
        )
        cross = _symmetric_traceless_cross(vector_to_tensor[neighbor_index], rel_msg.unsqueeze(2))
        tensor_msg = tensor_msg + torch.einsum("nkh,nkcab->ncab", attn, cross)
        return scalar_msg, vector_msg, tensor_msg

    def _vector_edge_attention_bias(
        self,
        vectors: torch.Tensor,
        rel: torch.Tensor,
        dist: torch.Tensor,
        neighbor_index: torch.Tensor,
    ) -> torch.Tensor:
        direction = rel / dist.unsqueeze(-1)
        query_vector = self.edge_query_vector(vectors)
        key_vector = self.edge_key_vector(vectors)[neighbor_index]
        query_edge = torch.einsum("nha,nka->nkh", query_vector, direction)
        key_edge = torch.einsum("nkha,nka->nkh", key_vector, direction)
        vector_dot = torch.einsum("nha,nkha->nkh", query_vector, key_vector)
        radial = torch.log1p(dist).unsqueeze(-1).expand_as(query_edge)
        features = torch.stack([radial, query_edge, key_edge, vector_dot], dim=-1)
        raw = torch.einsum("nkhf,hf->nkh", features, self.edge_vector_bias_weight)
        raw = raw + self.edge_vector_bias_offset
        return self.vector_edge_bias_scale * torch.tanh(raw)


class _ChannelMix(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        if out_channels and in_channels:
            nn.init.normal_(self.weight, std=1.0 / sqrt(in_channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.out_channels == 0:
            return value.new_zeros((value.shape[0], 0, *value.shape[2:]))
        if self.in_channels == 0:
            return value.new_zeros((value.shape[0], self.out_channels, *value.shape[2:]))
        return torch.einsum("oc,nc...->no...", self.weight, value)


class _ScalarChannelLinear(nn.Module):
    def __init__(self, in_features: int, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.linear = nn.Linear(in_features, out_channels) if out_channels else None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.linear is None:
            return value.new_zeros((value.shape[0], 0))
        return self.linear(value)


class _HeadVectorMix(nn.Module):
    def __init__(self, in_channels: int, num_heads: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, in_channels))
        nn.init.normal_(self.weight, std=1.0 / sqrt(in_channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.einsum("hc,nca->nha", self.weight, value)


def _linear_value_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    denom: torch.Tensor,
) -> torch.Tensor:
    kv = torch.einsum("nhd,nc...->hdc...", k, value)
    return torch.einsum("nhd,hdc...->nhc...", q, kv).mean(dim=1) / denom.mean(dim=-1)[:, None, *([None] * (value.ndim - 2))]


def _linear_value_attention_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    denom: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    kv = _segment_sum(torch.einsum("nhd,nc...->nhdc...", k, value), batch, num_graphs)
    return torch.einsum("nhd,nhdc...->nhc...", q, kv[batch]).mean(dim=1) / denom.mean(dim=-1)[
        :, None, *([None] * (value.ndim - 2))
    ]


def _linear_relative_vector(
    q: torch.Tensor,
    k: torch.Tensor,
    gate: torch.Tensor,
    pos: torch.Tensor,
    denom: torch.Tensor,
) -> torch.Tensor:
    gate_sum = torch.einsum("nhd,nc->hdc", k, gate)
    pos_sum = torch.einsum("nhd,nc,na->hdca", k, gate, pos)
    weighted_gate = torch.einsum("nhd,hdc->nhc", q, gate_sum) / denom.unsqueeze(-1)
    weighted_pos = torch.einsum("nhd,hdca->nhca", q, pos_sum) / denom.unsqueeze(-1).unsqueeze(-1)
    return (weighted_pos - pos[:, None, None, :] * weighted_gate.unsqueeze(-1)).mean(dim=1)


def _linear_relative_vector_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    gate: torch.Tensor,
    pos: torch.Tensor,
    denom: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    gate_sum = _segment_sum(torch.einsum("nhd,nc->nhdc", k, gate), batch, num_graphs)
    pos_sum = _segment_sum(torch.einsum("nhd,nc,na->nhdca", k, gate, pos), batch, num_graphs)
    weighted_gate = torch.einsum("nhd,nhdc->nhc", q, gate_sum[batch]) / denom.unsqueeze(-1)
    weighted_pos = torch.einsum("nhd,nhdca->nhca", q, pos_sum[batch]) / denom.unsqueeze(-1).unsqueeze(-1)
    return (weighted_pos - pos[:, None, None, :] * weighted_gate.unsqueeze(-1)).mean(dim=1)


def _linear_relative_tensor(
    q: torch.Tensor,
    k: torch.Tensor,
    gate: torch.Tensor,
    pos: torch.Tensor,
    denom: torch.Tensor,
) -> torch.Tensor:
    st_pos = _symmetric_traceless(pos)
    gate_sum = torch.einsum("nhd,nc->hdc", k, gate)
    pos_sum = torch.einsum("nhd,nc,na->hdca", k, gate, pos)
    st_sum = torch.einsum("nhd,nc,nab->hdcab", k, gate, st_pos)
    weighted_gate = torch.einsum("nhd,hdc->nhc", q, gate_sum) / denom.unsqueeze(-1)
    weighted_pos = torch.einsum("nhd,hdca->nhca", q, pos_sum) / denom.unsqueeze(-1).unsqueeze(-1)
    weighted_st = torch.einsum("nhd,hdcab->nhcab", q, st_sum) / denom[..., None, None, None]
    cross = _symmetric_traceless_cross(pos[:, None, None, :], weighted_pos)
    return (weighted_st + st_pos[:, None, None] * weighted_gate[..., None, None] - cross).mean(dim=1)


def _linear_relative_tensor_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    gate: torch.Tensor,
    pos: torch.Tensor,
    denom: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    st_pos = _symmetric_traceless(pos)
    gate_sum = _segment_sum(torch.einsum("nhd,nc->nhdc", k, gate), batch, num_graphs)
    pos_sum = _segment_sum(torch.einsum("nhd,nc,na->nhdca", k, gate, pos), batch, num_graphs)
    st_sum = _segment_sum(torch.einsum("nhd,nc,nab->nhdcab", k, gate, st_pos), batch, num_graphs)
    weighted_gate = torch.einsum("nhd,nhdc->nhc", q, gate_sum[batch]) / denom.unsqueeze(-1)
    weighted_pos = torch.einsum("nhd,nhdca->nhca", q, pos_sum[batch]) / denom.unsqueeze(-1).unsqueeze(-1)
    weighted_st = torch.einsum("nhd,nhdcab->nhcab", q, st_sum[batch]) / denom[..., None, None, None]
    cross = _symmetric_traceless_cross(pos[:, None, None, :], weighted_pos)
    return (weighted_st + st_pos[:, None, None] * weighted_gate[..., None, None] - cross).mean(dim=1)


def _linear_cross_tensor(
    q: torch.Tensor,
    k: torch.Tensor,
    vector_value: torch.Tensor,
    pos: torch.Tensor,
    denom: torch.Tensor,
) -> torch.Tensor:
    vector_sum = torch.einsum("nhd,nca->hdca", k, vector_value)
    vector_pos_sum = torch.einsum("nhd,nca,nb->hdcab", k, vector_value, pos)
    weighted_vector = torch.einsum("nhd,hdca->nhca", q, vector_sum) / denom.unsqueeze(-1).unsqueeze(-1)
    weighted_vector_pos = torch.einsum("nhd,hdcab->nhcab", q, vector_pos_sum) / denom[..., None, None, None]
    cross = _symmetric_traceless_cross(weighted_vector, pos[:, None, None, :])
    return (weighted_vector_pos - cross).mean(dim=1)


def _linear_cross_tensor_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    vector_value: torch.Tensor,
    pos: torch.Tensor,
    denom: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    vector_sum = _segment_sum(torch.einsum("nhd,nca->nhdca", k, vector_value), batch, num_graphs)
    vector_pos_sum = _segment_sum(torch.einsum("nhd,nca,nb->nhdcab", k, vector_value, pos), batch, num_graphs)
    weighted_vector = torch.einsum("nhd,nhdca->nhca", q, vector_sum[batch]) / denom.unsqueeze(-1).unsqueeze(-1)
    weighted_vector_pos = torch.einsum("nhd,nhdcab->nhcab", q, vector_pos_sum[batch]) / denom[..., None, None, None]
    cross = _symmetric_traceless_cross(weighted_vector, pos[:, None, None, :])
    return (weighted_vector_pos - cross).mean(dim=1)


def _segment_sum(value: torch.Tensor, batch: torch.Tensor, num_segments: int) -> torch.Tensor:
    out = value.new_zeros((num_segments, *value.shape[1:]))
    return out.index_add(0, batch, value)


def _bounded_geometry(value: torch.Tensor, eps: float) -> torch.Tensor:
    scale = torch.sqrt(1.0 + value.square().sum(dim=-1, keepdim=True).clamp_min(eps))
    return value / scale


def _bounded_irrep(value: torch.Tensor, eps: float) -> torch.Tensor:
    if value.shape[1] == 0:
        return value
    reduce_dims = tuple(range(2, value.ndim))
    scale = torch.sqrt(1.0 + value.square().mean(dim=reduce_dims, keepdim=True).clamp_min(eps))
    return value / scale


def _validate_config(config: RichEquivariantAttentionConfig) -> None:
    if config.node_dim <= 0:
        raise ValueError("node_dim must be positive")
    if config.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if config.num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if config.attention_mode not in {"linear", "local"}:
        raise ValueError("attention_mode must be 'linear' or 'local'")
    if config.vector_edge_bias and config.attention_mode != "local":
        raise ValueError("vector_edge_bias requires local attention")
    if config.vector_edge_bias_scale < 0:
        raise ValueError("vector_edge_bias_scale must be nonnegative")
    if config.residual_scale_init < 0:
        raise ValueError("residual_scale_init must be nonnegative")
    hidden = CartesianIrreps.parse(config.hidden_irreps)
    output = CartesianIrreps.parse(config.output_irreps)
    if hidden.scalars <= 0:
        raise ValueError("hidden_irreps must include scalar channels")
    if hidden.scalars % config.num_heads != 0:
        raise ValueError("hidden scalar channels must be divisible by num_heads")
    if config.vector_edge_bias and hidden.vectors <= 0:
        raise ValueError("vector_edge_bias requires vector channels")
    if output.scalars + output.vectors + output.tensors <= 0:
        raise ValueError("output_irreps must include at least one term")
