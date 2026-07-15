from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn

from .irreps import CartesianIrreps


@dataclass(frozen=True)
class EquivariantAttentionConfig:
    node_dim: int
    hidden_irreps: str | CartesianIrreps = "64x0e + 4x1o"
    output_irreps: str | CartesianIrreps = "1x0e"
    num_layers: int = 3
    num_heads: int = 4
    vector_kernel_init: float = 0.05
    residual_scale_init: float = 0.1
    eps: float = 1e-12


class EquivariantAttention(nn.Module):
    """O(3)-equivariant global attention with exact factorized moments."""

    attention_kind = "factorized_moment"
    symmetry = "O3"

    def __init__(self, config: EquivariantAttentionConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.hidden_irreps = CartesianIrreps.parse(config.hidden_irreps)
        self.output_irreps = CartesianIrreps.parse(config.output_irreps)

        scalar_input_dim = config.node_dim + 3
        self.scalar_in = nn.Linear(scalar_input_dim, self.hidden_irreps.scalars)
        self.vector_in = nn.Linear(self.hidden_irreps.scalars, self.hidden_irreps.vectors)
        layer_scale = config.residual_scale_init / sqrt(config.num_layers)
        self.layers = nn.ModuleList(
            [
                _EquivariantMomentLayer(
                    scalars=self.hidden_irreps.scalars,
                    vectors=self.hidden_irreps.vectors,
                    num_heads=config.num_heads,
                    vector_kernel_init=config.vector_kernel_init,
                    residual_scale_init=layer_scale,
                    eps=config.eps,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.scalar_out_norm = nn.LayerNorm(self.hidden_irreps.scalars)
        self.scalar_out = nn.Linear(self.hidden_irreps.scalars, self.output_irreps.scalars)
        self.vector_out = _ChannelMix(self.hidden_irreps.vectors, self.output_irreps.vectors)
        self.tensor_out = _ChannelMix(config.num_heads, self.output_irreps.tensors)

    def forward(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        node_feats, pos, batch = self._check_inputs(node_feats, pos, batch)
        geometry_dtype = torch.float64 if pos.dtype == torch.float64 else torch.float32
        geometry_pos = pos.to(dtype=geometry_dtype)
        center = _scatter_mean(geometry_pos, batch)
        centered = geometry_pos - center[batch]
        radius = centered.norm(dim=-1, keepdim=True)
        graph_scale = torch.sqrt(
            _scatter_mean(centered.square().sum(dim=-1, keepdim=True), batch).clamp_min(self.config.eps)
        )
        normalized_pos = centered / graph_scale[batch]
        model_pos = normalized_pos.to(dtype=node_feats.dtype)
        scalar_input = torch.cat(
            [
                node_feats,
                torch.log1p(radius).to(dtype=node_feats.dtype),
                torch.log1p(graph_scale[batch]).to(dtype=node_feats.dtype),
                torch.log1p(normalized_pos.square().sum(dim=-1, keepdim=True)).to(
                    dtype=node_feats.dtype
                ),
            ],
            dim=-1,
        )

        scalars = self.scalar_in(scalar_input)
        vectors = torch.tanh(self.vector_in(scalars)).unsqueeze(-1) * model_pos.unsqueeze(1)
        transient_tensor = normalized_pos.new_zeros((normalized_pos.shape[0], self.config.num_heads, 5))
        for layer in self.layers:
            scalars, vectors, transient_tensor = layer(scalars, vectors, normalized_pos, batch)

        node_scalars = self.scalar_out(self.scalar_out_norm(scalars))
        node_vectors = self.vector_out(vectors)
        node_tensors = _st_features_to_matrix(self.tensor_out(transient_tensor))
        return {
            "node_scalars": node_scalars,
            "node_vectors": node_vectors,
            "node_tensors": node_tensors,
            "graph_scalars": _scatter_mean(node_scalars, batch),
            "graph_vectors": _scatter_mean(node_vectors, batch),
            "graph_tensors": _scatter_mean(node_tensors, batch),
        }

    def _check_inputs(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if node_feats.ndim != 2 or node_feats.shape[1] != self.config.node_dim:
            raise ValueError(f"node_feats must have shape (N, {self.config.node_dim})")
        if node_feats.shape[0] == 0:
            raise ValueError("at least one node is required")
        if pos.shape != (node_feats.shape[0], 3):
            raise ValueError(f"pos must have shape (N, 3), got {tuple(pos.shape)}")
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
            raise ValueError(f"batch must have shape (N,), got {tuple(batch.shape)}")
        else:
            if batch.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise TypeError("batch must use an integer dtype")
            if batch.device != node_feats.device:
                raise ValueError("batch, node_feats, and pos must be on the same device")
            batch = batch.to(dtype=torch.long)
        if (batch < 0).any():
            raise ValueError("batch indices must be nonnegative")
        graph_ids = torch.unique(batch, sorted=True)
        expected = torch.arange(graph_ids.numel(), device=batch.device)
        if not torch.equal(graph_ids, expected):
            raise ValueError("batch indices must be contiguous and start at zero")
        return node_feats, pos, batch


class _EquivariantMomentLayer(nn.Module):
    def __init__(
        self,
        scalars: int,
        vectors: int,
        num_heads: int,
        vector_kernel_init: float,
        residual_scale_init: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.scalars = scalars
        self.vectors = vectors
        self.num_heads = num_heads
        self.head_dim = scalars // num_heads
        self.eps = eps

        self.norm = nn.LayerNorm(scalars)
        self.query_scalar = nn.Linear(scalars, scalars)
        self.key_scalar = nn.Linear(scalars, scalars)
        self.value_scalar = nn.Linear(scalars, scalars)
        self.query_vector = _ChannelMix(vectors, num_heads)
        self.key_vector = _ChannelMix(vectors, num_heads)
        self.value_vector = _ChannelMix(vectors, num_heads)
        self.query_vector_gate = nn.Linear(scalars, num_heads)
        self.key_vector_gate = nn.Linear(scalars, num_heads)
        self.relative_gate = nn.Linear(scalars, num_heads)
        self.tensor_gate = nn.Linear(scalars, num_heads)
        self.raw_vector_kernel = nn.Parameter(
            torch.full((num_heads,), _inverse_softplus(vector_kernel_init))
        )
        self.relative_mix = nn.Parameter(torch.full((num_heads,), 0.1))
        self.tensor_mix = nn.Parameter(torch.full((num_heads,), 0.1))
        self.vector_update = _ChannelMix(num_heads, vectors)
        invariant_dim = scalars + 5 * num_heads
        self.scalar_update_norm = nn.LayerNorm(invariant_dim)
        self.scalar_update = nn.Sequential(
            nn.Linear(invariant_dim, scalars),
            nn.SiLU(),
            nn.Linear(scalars, scalars),
        )
        self.scalar_residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.vector_residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        ffn_hidden = 2 * scalars
        self.ffn_norm = nn.LayerNorm(scalars)
        self.ffn_in = nn.Linear(scalars + vectors, 2 * ffn_hidden)
        self.ffn_out = nn.Linear(ffn_hidden, scalars)
        self.ffn_vector_gate = nn.Linear(scalars, vectors)
        self.ffn_vector_mix = _ChannelMix(vectors, vectors)
        self.ffn_scalar_residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.ffn_vector_residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_norm = self.norm(scalars)
        bounded_vectors = _bounded_irrep(vectors, self.eps)
        n_nodes = scalars.shape[0]

        q0 = F.elu(self.query_scalar(s_norm).reshape(n_nodes, self.num_heads, self.head_dim)) + 1.0
        k0 = F.elu(self.key_scalar(s_norm).reshape(n_nodes, self.num_heads, self.head_dim)) + 1.0
        q1 = self.query_vector(bounded_vectors) * torch.tanh(self.query_vector_gate(s_norm)).unsqueeze(-1)
        k1 = self.key_vector(bounded_vectors) * torch.tanh(self.key_vector_gate(s_norm)).unsqueeze(-1)
        query_outer = _symmetric_outer_features(q1)
        key_outer = _symmetric_outer_features(k1)
        moment_dtype = query_outer.dtype
        kernel_scale = F.softplus(self.raw_vector_kernel).to(dtype=moment_dtype)
        ones = torch.ones(
            (n_nodes, self.num_heads, 1),
            dtype=moment_dtype,
            device=scalars.device,
        )
        outer_scale = kernel_scale.sqrt()[None, :, None]
        query_angular = torch.cat([ones, outer_scale * query_outer], dim=-1)
        key_angular = torch.cat([ones, outer_scale * key_outer], dim=-1)
        query = torch.cat([q0.to(dtype=moment_dtype), query_angular], dim=-1)
        key = torch.cat([k0.to(dtype=moment_dtype), key_angular], dim=-1)

        scalar_value = self.value_scalar(s_norm).reshape(n_nodes, self.num_heads, self.head_dim)
        vector_value = self.value_vector(bounded_vectors)
        relative_gate = torch.tanh(self.relative_gate(s_norm))
        tensor_gate = torch.tanh(self.tensor_gate(s_norm))
        pos_h = pos.unsqueeze(1).expand(-1, self.num_heads, -1)
        st_pos = _symmetric_traceless_features(pos).unsqueeze(1).expand(-1, self.num_heads, -1)
        value = torch.cat(
            [
                scalar_value.to(dtype=moment_dtype),
                vector_value.to(dtype=moment_dtype),
                relative_gate.to(dtype=moment_dtype).unsqueeze(-1),
                relative_gate.to(dtype=moment_dtype).unsqueeze(-1) * pos_h,
                tensor_gate.to(dtype=moment_dtype).unsqueeze(-1),
                tensor_gate.to(dtype=moment_dtype).unsqueeze(-1) * pos_h,
                tensor_gate.to(dtype=moment_dtype).unsqueeze(-1) * st_pos,
            ],
            dim=-1,
        )
        transported = _factorized_attention(
            query,
            key,
            value,
            batch,
            balanced=True,
            eps=self.eps,
        )

        offset = self.head_dim
        scalar_message = transported[..., :offset].reshape(n_nodes, self.scalars)
        vector_base = transported[..., offset : offset + 3]
        relative_mass = transported[..., offset + 3 : offset + 4]
        relative_position = transported[..., offset + 4 : offset + 7]
        relative = relative_position - pos_h * relative_mass
        tensor_mass = transported[..., offset + 7 : offset + 8]
        tensor_position = transported[..., offset + 8 : offset + 11]
        tensor_second = transported[..., offset + 11 : offset + 16]
        tensor = tensor_second + st_pos * tensor_mass - 2.0 * _symmetric_traceless_cross_features(
            tensor_position, pos_h
        )
        moment_q1 = q1.to(dtype=moment_dtype)
        tensor_vector = _st_matrix_vector(tensor, moment_q1)
        query_base_dot = (moment_q1 * vector_base).sum(dim=-1)
        query_relative_dot = (moment_q1 * relative).sum(dim=-1)
        relative_square = relative.square().sum(dim=-1)
        tensor_square = _st_frobenius_square(tensor)
        query_tensor_dot = (moment_q1 * tensor_vector).sum(dim=-1)

        vector_per_head = (
            vector_base
            + self.relative_mix.to(dtype=moment_dtype)[None, :, None] * relative
            + self.tensor_mix.to(dtype=moment_dtype)[None, :, None] * tensor_vector
        )

        scalar_invariants = torch.cat(
            [
                scalar_message,
                query_base_dot,
                query_relative_dot,
                relative_square,
                tensor_square,
                query_tensor_dot,
            ],
            dim=-1,
        )
        normalized_invariants = _stable_layer_norm(self.scalar_update_norm, scalar_invariants)
        scalar_delta = self.scalar_update(normalized_invariants.to(dtype=scalars.dtype))
        vector_delta = self.vector_update(vector_per_head)
        scalars = scalars + self.scalar_residual_scale * scalar_delta
        bounded_delta = _bounded_irrep(vector_delta, self.eps).to(dtype=vectors.dtype)
        vectors = vectors + self.vector_residual_scale * bounded_delta
        ffn_scalars = self.ffn_norm(scalars)
        ffn_vectors = _bounded_irrep(vectors, self.eps)
        ffn_invariants = torch.cat([ffn_scalars, ffn_vectors.square().sum(dim=-1)], dim=-1)
        ffn_content, ffn_gate = self.ffn_in(ffn_invariants).chunk(2, dim=-1)
        scalar_ffn = self.ffn_out(F.silu(ffn_content) * ffn_gate)
        vector_ffn = self.ffn_vector_mix(ffn_vectors) * torch.tanh(
            self.ffn_vector_gate(ffn_scalars)
        ).unsqueeze(-1)
        scalars = scalars + self.ffn_scalar_residual_scale * scalar_ffn
        vectors = vectors + self.ffn_vector_residual_scale * _bounded_irrep(vector_ffn, self.eps)
        return scalars, vectors, tensor


class _ChannelMix(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        if in_channels and out_channels:
            nn.init.normal_(self.weight, std=1.0 / sqrt(in_channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.out_channels == 0:
            return value.new_zeros((value.shape[0], 0, *value.shape[2:]))
        return torch.einsum("oc,nc...->no...", self.weight.to(dtype=value.dtype), value)


def _factorized_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    batch: torch.Tensor,
    *,
    balanced: bool,
    balance_exponent: torch.Tensor | None = None,
    eps: float,
    sinkhorn_iterations: int = 1,
) -> torch.Tensor:
    if not isinstance(sinkhorn_iterations, int) or sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn_iterations must be positive")
    output_dtype = value.dtype
    reduction_dtype = torch.float64 if torch.float64 in {query.dtype, key.dtype, value.dtype} else torch.float32
    query = query.to(dtype=reduction_dtype)
    key = key.to(dtype=reduction_dtype)
    value = value.to(dtype=reduction_dtype)
    if balance_exponent is not None:
        balance_exponent = balance_exponent.to(dtype=reduction_dtype)
    num_graphs = int(batch.max().item()) + 1
    if balanced:
        row_scale = query.new_ones(query.shape[:2])
        for _ in range(sinkhorn_iterations):
            query_sum = _segment_sum(query * row_scale.unsqueeze(-1), batch, num_graphs)
            key_mass = (key * query_sum[batch]).sum(dim=-1).clamp_min(eps)
            if balance_exponent is None:
                key_scale = key_mass.reciprocal()
            else:
                key_scale = torch.exp(-balance_exponent[None, :] * torch.log(key_mass))
            weighted_key = key * key_scale.unsqueeze(-1)
            key_sum = _segment_sum(weighted_key, batch, num_graphs)
            denominator = (query * key_sum[batch]).sum(dim=-1).clamp_min(eps)
            row_scale = denominator.reciprocal()
    else:
        key_scale = key.new_ones(key.shape[:2])
        weighted_key = key
        key_sum = _segment_sum(weighted_key, batch, num_graphs)
        denominator = (query * key_sum[batch]).sum(dim=-1).clamp_min(eps)
    summary = _segment_sum(weighted_key.unsqueeze(-1) * value.unsqueeze(-2), batch, num_graphs)
    numerator = torch.einsum("nhd,nhdf->nhf", query, summary[batch])
    return (numerator / denominator.unsqueeze(-1)).to(dtype=output_dtype)


def _symmetric_outer_features(value: torch.Tensor) -> torch.Tensor:
    value = value.to(dtype=_moment_dtype(value))
    x, y, z = value.unbind(dim=-1)
    sqrt_two = sqrt(2.0)
    return torch.stack([x * x, y * y, z * z, sqrt_two * x * y, sqrt_two * x * z, sqrt_two * y * z], dim=-1)


def _symmetric_traceless_features(value: torch.Tensor) -> torch.Tensor:
    value = value.to(dtype=_moment_dtype(value))
    x, y, z = value.unbind(dim=-1)
    trace_third = (x.square() + y.square() + z.square()) / 3.0
    return torch.stack([x.square() - trace_third, y.square() - trace_third, x * y, x * z, y * z], dim=-1)


def _symmetric_traceless_cross_features(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    dtype = _moment_dtype(left, right)
    left = left.to(dtype=dtype)
    right = right.to(dtype=dtype)
    lx, ly, lz = left.unbind(dim=-1)
    rx, ry, rz = right.unbind(dim=-1)
    trace_third = (lx * rx + ly * ry + lz * rz) / 3.0
    return torch.stack(
        [
            lx * rx - trace_third,
            ly * ry - trace_third,
            0.5 * (lx * ry + ly * rx),
            0.5 * (lx * rz + lz * rx),
            0.5 * (ly * rz + lz * ry),
        ],
        dim=-1,
    )


def _st_features_to_matrix(value: torch.Tensor) -> torch.Tensor:
    xx, yy, xy, xz, yz = value.unbind(dim=-1)
    zz = -xx - yy
    return torch.stack(
        [
            torch.stack([xx, xy, xz], dim=-1),
            torch.stack([xy, yy, yz], dim=-1),
            torch.stack([xz, yz, zz], dim=-1),
        ],
        dim=-2,
    )


def _st_matrix_vector(tensor: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    dtype = _moment_dtype(tensor, vector)
    return torch.einsum(
        "...ab,...b->...a",
        _st_features_to_matrix(tensor.to(dtype=dtype)),
        vector.to(dtype=dtype),
    )


def _st_frobenius_square(value: torch.Tensor) -> torch.Tensor:
    value = value.to(dtype=_moment_dtype(value))
    xx, yy, xy, xz, yz = value.unbind(dim=-1)
    zz = -xx - yy
    return xx.square() + yy.square() + zz.square() + 2.0 * (xy.square() + xz.square() + yz.square())


def _segment_sum(value: torch.Tensor, batch: torch.Tensor, num_segments: int) -> torch.Tensor:
    out = value.new_zeros((num_segments, *value.shape[1:]))
    return out.index_add(0, batch, value)


def _scatter_mean(value: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    output_dtype = value.dtype
    reduction_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
    reduced = value.to(dtype=reduction_dtype)
    num_graphs = int(batch.max().item()) + 1
    summed = _segment_sum(reduced, batch, num_graphs)
    count = torch.bincount(batch, minlength=num_graphs).to(device=value.device, dtype=reduction_dtype)
    count = count.reshape(num_graphs, *((1,) * (value.ndim - 1)))
    return (summed / count).to(dtype=output_dtype)


def _bounded_irrep(value: torch.Tensor, eps: float) -> torch.Tensor:
    output_dtype = value.dtype
    reduction_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
    reduced = value.to(dtype=reduction_dtype)
    scale = torch.sqrt(1.0 + reduced.square().mean(dim=-1, keepdim=True).clamp_min(eps))
    return (reduced / scale).to(dtype=output_dtype)


def _stable_layer_norm(layer: nn.LayerNorm, value: torch.Tensor) -> torch.Tensor:
    dtype = _moment_dtype(value)
    weight = None if layer.weight is None else layer.weight.to(dtype=dtype)
    bias = None if layer.bias is None else layer.bias.to(dtype=dtype)
    return F.layer_norm(
        value.to(dtype=dtype),
        layer.normalized_shape,
        weight,
        bias,
        layer.eps,
    )


def _moment_dtype(*values: torch.Tensor) -> torch.dtype:
    return torch.float64 if any(value.dtype == torch.float64 for value in values) else torch.float32


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("vector_kernel_init must be positive")
    return float(torch.log(torch.expm1(torch.tensor(value))).item())


def _validate_config(config: EquivariantAttentionConfig) -> None:
    if config.node_dim <= 0:
        raise ValueError("node_dim must be positive")
    if config.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if config.num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if config.eps <= 0:
        raise ValueError("eps must be positive")
    if config.residual_scale_init < 0:
        raise ValueError("residual_scale_init must be nonnegative")
    if config.vector_kernel_init <= 0:
        raise ValueError("vector_kernel_init must be positive")
    hidden = CartesianIrreps.parse(config.hidden_irreps)
    output = CartesianIrreps.parse(config.output_irreps)
    if hidden.scalars <= 0 or hidden.vectors <= 0 or hidden.tensors:
        raise ValueError("hidden_irreps supports only scalar and vector channels, both with positive multiplicity")
    if hidden.scalars % config.num_heads:
        raise ValueError("hidden scalar channels must be divisible by num_heads")
    if output.scalars + output.vectors + output.tensors <= 0:
        raise ValueError("output_irreps must include at least one term")
