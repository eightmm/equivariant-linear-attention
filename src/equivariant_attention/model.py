from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .backends import BackendName, SphericalHarmonicsBackend

AttentionMode = Literal["linear", "linear_sh", "local", "dense"]


@dataclass(frozen=True)
class EquivariantAttentionConfig:
    node_dim: int
    edge_dim: int = 0
    hidden_dim: int = 64
    num_layers: int = 3
    num_heads: int = 4
    preferred_backend: BackendName = "cuequivariance"
    attention_mode: AttentionMode = "linear_sh"
    sh_lmax: int = 2
    local_radius: float | None = None
    max_neighbors: int = 32
    eps: float = 1e-12


class EquivariantAttention(nn.Module):
    """Global SE(3)-equivariant attention for dense 3D structural graphs."""

    def __init__(self, config: EquivariantAttentionConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.geometry = SphericalHarmonicsBackend(config.preferred_backend, lmax=config.sh_lmax)
        self.input = nn.Linear(config.node_dim, config.hidden_dim)
        self.geometry_input = _mlp(3, config.hidden_dim, config.hidden_dim)
        self.layers = nn.ModuleList(
            [
                _EquivariantAttentionLayer(
                    hidden_dim=config.hidden_dim,
                    edge_dim=config.edge_dim,
                    num_heads=config.num_heads,
                    attention_mode=config.attention_mode,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.node_scalar = nn.Linear(config.hidden_dim, 1)
        self.graph_scalar = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        edge_feats: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        neighbor_index: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | str]:
        single_graph = batch is None
        node_feats, pos, edge_feats, batch, neighbor_index, neighbor_mask = self._check_inputs(
            node_feats,
            pos,
            edge_feats,
            batch,
            neighbor_index,
            neighbor_mask,
        )
        if single_graph:
            centers = pos.mean(dim=0, keepdim=True)
        else:
            centers = _scatter_mean(pos, batch)
        centered_pos = pos - centers[batch]
        radius = centered_pos.norm(dim=-1, keepdim=True)
        sh = self.geometry(centered_pos)
        l2_norm = sh[:, 4:].norm(dim=-1, keepdim=True) if self.config.sh_lmax == 2 else radius.new_zeros(radius.shape)
        geometry_features = torch.cat([radius, radius.square(), l2_norm], dim=-1)

        h = self.input(node_feats) + self.geometry_input(geometry_features)
        node_vector = pos.new_zeros((pos.shape[0], 3))
        node_tensor = pos.new_zeros((pos.shape[0], 3, 3))
        if self.config.attention_mode == "dense":
            rel, dist, same_graph, sh_l0 = self._pair_geometry(centered_pos, batch)
            for layer in self.layers:
                h, node_vector, node_tensor = layer.forward_dense(h, centered_pos, rel, dist, same_graph, sh_l0, edge_feats)
        elif self.config.attention_mode == "local":
            for layer in self.layers:
                h, node_vector, node_tensor = layer.forward_local(
                    h,
                    centered_pos,
                    None if single_graph else batch,
                    self.config.local_radius,
                    self.config.max_neighbors,
                    neighbor_index,
                    neighbor_mask,
                )
        else:
            for layer in self.layers:
                h, node_vector, node_tensor = layer.forward_linear(
                    h,
                    centered_pos,
                    None if single_graph else batch,
                    self.config.eps,
                )

        if single_graph:
            graph_hidden = h.mean(dim=0, keepdim=True)
            graph_vector = node_vector.mean(dim=0, keepdim=True)
            graph_tensor = node_tensor.mean(dim=0, keepdim=True)
        else:
            graph_hidden = _scatter_mean(h, batch)
            graph_vector = _scatter_mean(node_vector, batch)
            graph_tensor = _scatter_mean(node_tensor, batch)
        return {
            "node_scalar": self.node_scalar(h),
            "node_vector": node_vector,
            "node_tensor": node_tensor,
            "graph_scalar": self.graph_scalar(graph_hidden),
            "graph_vector": graph_vector,
            "graph_tensor": graph_tensor,
            "backend": self.geometry.active,
            "attention_mode": self.config.attention_mode,
        }

    def _pair_geometry(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist = rel.norm(dim=-1, keepdim=True)
        same_graph = batch.unsqueeze(0).eq(batch.unsqueeze(1))
        flat_sh = self.geometry(rel.reshape(-1, 3))
        sh_l0 = flat_sh[:, :1].reshape(pos.shape[0], pos.shape[0], 1)
        return rel, dist, same_graph, sh_l0

    def _check_inputs(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        edge_feats: torch.Tensor | None,
        batch: torch.Tensor | None,
        neighbor_index: torch.Tensor | None,
        neighbor_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if node_feats.ndim != 2:
            msg = f"node_feats must have shape (N, F), got {tuple(node_feats.shape)}"
            raise ValueError(msg)
        if node_feats.shape[1] != self.config.node_dim:
            msg = f"node_feats width must be {self.config.node_dim}, got {node_feats.shape[1]}"
            raise ValueError(msg)
        if pos.shape != (node_feats.shape[0], 3):
            msg = f"pos must have shape (N, 3), got {tuple(pos.shape)}"
            raise ValueError(msg)
        if node_feats.shape[0] == 0:
            raise ValueError("at least one node is required")
        if not torch.is_floating_point(node_feats) or not torch.is_floating_point(pos):
            raise TypeError("node_feats and pos must be floating point tensors")
        if node_feats.device != pos.device:
            raise ValueError("node_feats and pos must be on the same device")
        if node_feats.dtype != pos.dtype:
            pos = pos.to(dtype=node_feats.dtype)
        if not torch.isfinite(node_feats).all() or not torch.isfinite(pos).all():
            raise ValueError("node_feats and pos must be finite")

        if batch is None:
            batch = torch.zeros(node_feats.shape[0], dtype=torch.long, device=node_feats.device)
        elif batch.shape != (node_feats.shape[0],):
            msg = f"batch must have shape (N,), got {tuple(batch.shape)}"
            raise ValueError(msg)
        else:
            batch = batch.to(device=node_feats.device, dtype=torch.long)
        if (batch < 0).any():
            raise ValueError("batch indices must be nonnegative")

        edge_feats = self._check_edge_feats(edge_feats, node_feats)
        neighbor_index, neighbor_mask = self._check_neighbors(neighbor_index, neighbor_mask, node_feats)
        return node_feats, pos, edge_feats, batch, neighbor_index, neighbor_mask

    def _check_edge_feats(
        self,
        edge_feats: torch.Tensor | None,
        node_feats: torch.Tensor,
    ) -> torch.Tensor | None:
        n_nodes = node_feats.shape[0]
        if self.config.edge_dim == 0:
            if edge_feats is not None:
                raise ValueError("edge_feats require config.edge_dim > 0")
            return None
        if edge_feats is None:
            return node_feats.new_zeros((n_nodes, n_nodes, self.config.edge_dim))
        if edge_feats.ndim == 2:
            edge_feats = edge_feats.unsqueeze(-1)
        expected = (n_nodes, n_nodes, self.config.edge_dim)
        if edge_feats.shape != expected:
            msg = f"edge_feats must have shape {expected}, got {tuple(edge_feats.shape)}"
            raise ValueError(msg)
        edge_feats = edge_feats.to(device=node_feats.device, dtype=node_feats.dtype)
        if not torch.isfinite(edge_feats).all():
            raise ValueError("edge_feats must be finite")
        return edge_feats

    def _check_neighbors(
        self,
        neighbor_index: torch.Tensor | None,
        neighbor_mask: torch.Tensor | None,
        node_feats: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if neighbor_index is None:
            if neighbor_mask is not None:
                raise ValueError("neighbor_mask requires neighbor_index")
            return None, None
        if self.config.attention_mode != "local":
            raise ValueError("neighbor_index is only supported with local attention")
        if neighbor_index.ndim != 2:
            msg = f"neighbor_index must have shape (N, K), got {tuple(neighbor_index.shape)}"
            raise ValueError(msg)
        if neighbor_index.shape[0] != node_feats.shape[0]:
            msg = f"neighbor_index first dimension must be N={node_feats.shape[0]}"
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
        if not neighbor_mask.any(dim=1).all():
            raise ValueError("each node needs at least one valid neighbor")
        return neighbor_index, neighbor_mask


class _EquivariantAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, num_heads: int, attention_mode: AttentionMode) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attention_mode = attention_mode

        self.norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.radial_bias = _mlp(2, hidden_dim, num_heads)
        self.edge_bias = _mlp(edge_dim, hidden_dim, num_heads) if edge_dim > 0 else None
        self.scalar_out = nn.Linear(hidden_dim, hidden_dim)
        self.tensor_scalar = nn.Linear(num_heads, hidden_dim)
        self.update = _mlp(hidden_dim * 2, hidden_dim, hidden_dim)
        self.vector_gate = nn.Linear(hidden_dim, num_heads)
        self.tensor_gate = nn.Linear(hidden_dim, num_heads)
        self.vector_mix = nn.Parameter(torch.empty(num_heads))
        self.tensor_mix = nn.Parameter(torch.empty(num_heads))
        nn.init.normal_(self.vector_mix, std=1.0 / sqrt(num_heads))
        nn.init.normal_(self.tensor_mix, std=1.0 / sqrt(num_heads))

    def forward_dense(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        rel: torch.Tensor,
        dist: torch.Tensor,
        same_graph: torch.Tensor,
        sh_l0: torch.Tensor,
        edge_feats: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_norm = self.norm(h)
        n_nodes = h.shape[0]
        q = self.query(h_norm).reshape(n_nodes, self.num_heads, self.head_dim)
        k = self.key(h_norm).reshape(n_nodes, self.num_heads, self.head_dim)
        v = self.value(h_norm).reshape(n_nodes, self.num_heads, self.head_dim)

        logits = torch.einsum("ihd,jhd->ijh", q, k) / sqrt(self.head_dim)
        logits = logits + self.radial_bias(torch.cat([dist, sh_l0], dim=-1))
        if self.edge_bias is not None:
            if edge_feats is None:
                raise ValueError("edge_feats are required for this edge_dim")
            logits = logits + self.edge_bias(edge_feats)

        min_value = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~same_graph.unsqueeze(-1), min_value)
        attn = torch.softmax(logits, dim=1)

        scalar_message = torch.einsum("ijh,jhd->ihd", attn, v).reshape(n_nodes, self.hidden_dim)
        tensor_per_head = self._dense_tensor_message(attn, h_norm, rel)
        tensor_scalar = _tensor_invariant_features(tensor_per_head, pos)
        scalar_delta = self.scalar_out(scalar_message) + self.tensor_scalar(tensor_scalar)
        h_next = h + self.update(torch.cat([h, scalar_delta], dim=-1))

        gates = torch.tanh(self.vector_gate(h_norm))
        vector_per_head = torch.einsum("ijh,jh,ijc->ihc", attn, gates, rel)
        node_vector = torch.einsum("ihc,h->ic", vector_per_head, self.vector_mix)
        node_tensor = torch.einsum("ihab,h->iab", tensor_per_head, self.tensor_mix)
        return h_next, node_vector, node_tensor

    def forward_linear(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_norm = self.norm(h)
        n_nodes = h.shape[0]
        q = _positive_feature(self.query(h_norm).reshape(n_nodes, self.num_heads, self.head_dim))
        k = _positive_feature(self.key(h_norm).reshape(n_nodes, self.num_heads, self.head_dim))
        v = self.value(h_norm).reshape(n_nodes, self.num_heads, self.head_dim)
        gates = torch.tanh(self.vector_gate(h_norm))

        if batch is None:
            scalar_message, vector_per_head, tensor_per_head = _linear_attention_graph(
                q,
                k,
                v,
                gates,
                torch.tanh(self.tensor_gate(h_norm)),
                pos,
                eps,
                use_tensor=self.attention_mode == "linear_sh",
            )
        else:
            scalar_message = torch.empty_like(v)
            vector_per_head = pos.new_empty((n_nodes, self.num_heads, 3))
            tensor_per_head = pos.new_empty((n_nodes, self.num_heads, 3, 3))
            for graph_id in torch.unique(batch, sorted=True):
                idx = batch.eq(graph_id)
                scalar_g, vector_g, tensor_g = _linear_attention_graph(
                    q[idx],
                    k[idx],
                    v[idx],
                    gates[idx],
                    torch.tanh(self.tensor_gate(h_norm[idx])),
                    pos[idx],
                    eps,
                    use_tensor=self.attention_mode == "linear_sh",
                )
                scalar_message[idx] = scalar_g
                vector_per_head[idx] = vector_g
                tensor_per_head[idx] = tensor_g

        tensor_scalar = _tensor_invariant_features(tensor_per_head, pos)
        scalar_delta = self.scalar_out(scalar_message.reshape(n_nodes, self.hidden_dim))
        if self.attention_mode == "linear_sh":
            scalar_delta = scalar_delta + self.tensor_scalar(tensor_scalar)
        h_next = h + self.update(torch.cat([h, scalar_delta], dim=-1))
        node_vector = torch.einsum("ihc,h->ic", vector_per_head, self.vector_mix)
        node_tensor = torch.einsum("ihab,h->iab", tensor_per_head, self.tensor_mix)
        return h_next, node_vector, node_tensor

    def forward_local(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None,
        radius: float | None,
        max_neighbors: int,
        neighbor_index: torch.Tensor | None,
        neighbor_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_norm = self.norm(h)
        n_nodes = h.shape[0]
        q = self.query(h_norm).reshape(n_nodes, self.num_heads, self.head_dim)
        k = self.key(h_norm).reshape(n_nodes, self.num_heads, self.head_dim)
        v = self.value(h_norm).reshape(n_nodes, self.num_heads, self.head_dim)
        gates = torch.tanh(self.vector_gate(h_norm))
        tensor_gates = torch.tanh(self.tensor_gate(h_norm))

        scalar_message = torch.empty_like(v)
        vector_per_head = pos.new_empty((n_nodes, self.num_heads, 3))
        tensor_per_head = pos.new_empty((n_nodes, self.num_heads, 3, 3))
        if neighbor_index is not None:
            if neighbor_mask is None:
                raise ValueError("neighbor_mask is required when neighbor_index is provided")
            scalar_message, vector_per_head, tensor_per_head = self._local_attention_indexed(
                q,
                k,
                v,
                gates,
                tensor_gates,
                pos,
                neighbor_index,
                neighbor_mask,
            )
        else:
            graph_ids = [None] if batch is None else torch.unique(batch, sorted=True).unbind(0)
            for graph_id in graph_ids:
                idx = torch.ones(n_nodes, dtype=torch.bool, device=h.device) if graph_id is None else batch.eq(graph_id)
                scalar_g, vector_g, tensor_g = self._local_attention_graph(
                    q[idx],
                    k[idx],
                    v[idx],
                    gates[idx],
                    tensor_gates[idx],
                    pos[idx],
                    radius,
                    max_neighbors,
                )
                scalar_message[idx] = scalar_g
                vector_per_head[idx] = vector_g
                tensor_per_head[idx] = tensor_g

        tensor_scalar = _tensor_invariant_features(tensor_per_head, pos)
        scalar_delta = self.scalar_out(scalar_message.reshape(n_nodes, self.hidden_dim)) + self.tensor_scalar(tensor_scalar)
        h_next = h + self.update(torch.cat([h, scalar_delta], dim=-1))
        node_vector = torch.einsum("ihc,h->ic", vector_per_head, self.vector_mix)
        node_tensor = torch.einsum("ihab,h->iab", tensor_per_head, self.tensor_mix)
        return h_next, node_vector, node_tensor

    def _dense_tensor_message(self, attn: torch.Tensor, h_norm: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        tensor_gates = torch.tanh(self.tensor_gate(h_norm))
        rel_tensor = _symmetric_traceless(rel)
        return torch.einsum("ijh,jh,ijab->ihab", attn, tensor_gates, rel_tensor)

    def _local_attention_graph(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gates: torch.Tensor,
        tensor_gates: torch.Tensor,
        pos: torch.Tensor,
        radius: float | None,
        max_neighbors: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_nodes = pos.shape[0]
        k_neighbors = min(max_neighbors, n_nodes)
        dist = torch.cdist(pos.float(), pos.float()).to(dtype=pos.dtype)
        if radius is not None:
            dist = dist.masked_fill(dist > radius, torch.inf)
        neighbor_dist, neighbor_idx = torch.topk(dist, k=k_neighbors, dim=1, largest=False)
        valid = torch.isfinite(neighbor_dist)
        k_n = k[neighbor_idx]
        v_n = v[neighbor_idx]
        gates_n = gates[neighbor_idx]
        tensor_gates_n = tensor_gates[neighbor_idx]
        rel = pos[neighbor_idx] - pos.unsqueeze(1)

        logits = torch.einsum("ihd,ikhd->ikh", q, k_n) / sqrt(self.head_dim)
        radial = self.radial_bias(torch.stack([neighbor_dist.nan_to_num(posinf=0.0), valid.to(pos.dtype)], dim=-1))
        logits = logits + radial
        logits = logits.masked_fill(~valid.unsqueeze(-1), torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=1)

        scalar_message = torch.einsum("ikh,ikhd->ihd", attn, v_n)
        vector_per_head = torch.einsum("ikh,ikh,ikc->ihc", attn, gates_n, rel)
        tensor_per_head = torch.einsum(
            "ikh,ikh,ikab->ihab",
            attn,
            tensor_gates_n,
            _symmetric_traceless(rel),
        )
        return scalar_message, vector_per_head, tensor_per_head

    def _local_attention_indexed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gates: torch.Tensor,
        tensor_gates: torch.Tensor,
        pos: torch.Tensor,
        neighbor_index: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_n = k[neighbor_index]
        v_n = v[neighbor_index]
        gates_n = gates[neighbor_index]
        tensor_gates_n = tensor_gates[neighbor_index]
        rel = pos[neighbor_index] - pos.unsqueeze(1)
        dist = rel.float().norm(dim=-1).to(dtype=pos.dtype)

        logits = torch.einsum("ihd,ikhd->ikh", q, k_n) / sqrt(self.head_dim)
        radial = self.radial_bias(torch.stack([dist, neighbor_mask.to(pos.dtype)], dim=-1))
        logits = logits + radial
        logits = logits.masked_fill(~neighbor_mask.unsqueeze(-1), torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=1)

        scalar_message = torch.einsum("ikh,ikhd->ihd", attn, v_n)
        vector_per_head = torch.einsum("ikh,ikh,ikc->ihc", attn, gates_n, rel)
        tensor_per_head = torch.einsum(
            "ikh,ikh,ikab->ihab",
            attn,
            tensor_gates_n,
            _symmetric_traceless(rel),
        )
        return scalar_message, vector_per_head, tensor_per_head


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _positive_feature(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


def _linear_attention_graph(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gates: torch.Tensor,
    tensor_gates: torch.Tensor,
    pos: torch.Tensor,
    eps: float,
    use_tensor: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_sum = k.sum(dim=0)
    denom = (q * k_sum.unsqueeze(0)).sum(dim=-1).clamp_min(eps)
    kv_sum = torch.einsum("nhd,nhe->hde", k, v)
    scalar_message = torch.einsum("nhd,hde->nhe", q, kv_sum) / denom.unsqueeze(-1)

    gated_key_sum = torch.einsum("nhd,nh->hd", k, gates)
    gated_pos_sum = torch.einsum("nhd,nh,nc->hdc", k, gates, pos)
    weighted_gate = (q * gated_key_sum.unsqueeze(0)).sum(dim=-1) / denom
    weighted_pos = torch.einsum("nhd,hdc->nhc", q, gated_pos_sum) / denom.unsqueeze(-1)
    vector_per_head = weighted_pos - pos.unsqueeze(1) * weighted_gate.unsqueeze(-1)
    if not use_tensor:
        tensor_per_head = pos.new_zeros((pos.shape[0], q.shape[1], 3, 3))
        return scalar_message, vector_per_head, tensor_per_head

    st_pos = _symmetric_traceless(pos)
    tensor_key_sum = torch.einsum("nhd,nh->hd", k, tensor_gates)
    tensor_pos_sum = torch.einsum("nhd,nh,nc->hdc", k, tensor_gates, pos)
    tensor_st_sum = torch.einsum("nhd,nh,nab->hdab", k, tensor_gates, st_pos)
    weighted_tensor_gate = (q * tensor_key_sum.unsqueeze(0)).sum(dim=-1) / denom
    weighted_pos_t = torch.einsum("nhd,hdc->nhc", q, tensor_pos_sum) / denom.unsqueeze(-1)
    weighted_st = torch.einsum("nhd,hdab->nhab", q, tensor_st_sum) / denom[..., None, None]
    cross = _symmetric_traceless_cross(pos.unsqueeze(1), weighted_pos_t)
    tensor_per_head = weighted_st + st_pos.unsqueeze(1) * weighted_tensor_gate[..., None, None] - cross
    return scalar_message, vector_per_head, tensor_per_head


def _symmetric_traceless(vectors: torch.Tensor) -> torch.Tensor:
    outer = vectors.unsqueeze(-1) * vectors.unsqueeze(-2)
    norm2 = vectors.square().sum(dim=-1)
    eye = torch.eye(3, dtype=vectors.dtype, device=vectors.device)
    return outer - eye * (norm2 / 3.0).unsqueeze(-1).unsqueeze(-1)


def _symmetric_traceless_cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    sym = a.unsqueeze(-1) * b.unsqueeze(-2) + b.unsqueeze(-1) * a.unsqueeze(-2)
    dot = (a * b).sum(dim=-1)
    eye = torch.eye(3, dtype=a.dtype, device=a.device)
    return sym - eye * (2.0 * dot / 3.0).unsqueeze(-1).unsqueeze(-1)


def _tensor_invariant_features(tensor_per_head: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    node_tensor = _symmetric_traceless(pos)
    return torch.einsum("ihab,iab->ih", tensor_per_head, node_tensor)


def _scatter_mean(values: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    n_graphs = int(batch.max().item()) + 1
    out = values.new_zeros((n_graphs, *values.shape[1:]))
    out.index_add_(0, batch, values)
    counts = values.new_zeros((n_graphs, *([1] * (values.ndim - 1))))
    counts.index_add_(0, batch, torch.ones((values.shape[0], *([1] * (values.ndim - 1))), device=values.device, dtype=values.dtype))
    return out / counts.clamp_min(1)


def _validate_config(config: EquivariantAttentionConfig) -> None:
    if config.node_dim <= 0:
        raise ValueError("node_dim must be positive")
    if config.edge_dim < 0:
        raise ValueError("edge_dim must be nonnegative")
    if config.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if config.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if config.num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if config.hidden_dim % config.num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads")
    if config.attention_mode not in {"linear", "linear_sh", "local", "dense"}:
        raise ValueError("attention_mode must be one of: linear, linear_sh, local, dense")
    if config.attention_mode != "dense" and config.edge_dim != 0:
        raise ValueError("edge_dim is only supported with dense attention")
    if config.sh_lmax not in {1, 2}:
        raise ValueError("sh_lmax must be 1 or 2")
    if config.local_radius is not None and config.local_radius <= 0:
        raise ValueError("local_radius must be positive")
    if config.max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive")
