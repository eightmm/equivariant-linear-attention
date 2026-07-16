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
    linear_kernel_init: float = 0.05
    linear_kernel_max: float = 1.0
    vector_kernel_init: float = 0.05
    vector_kernel_max: float = 1.0
    kernel_floor: float = 1.0
    use_linear_kernel: bool = True
    use_key_balancing: bool = True
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
                    linear_kernel_init=config.linear_kernel_init,
                    linear_kernel_max=config.linear_kernel_max,
                    vector_kernel_init=config.vector_kernel_init,
                    vector_kernel_max=config.vector_kernel_max,
                    kernel_floor=config.kernel_floor,
                    use_linear_kernel=config.use_linear_kernel,
                    use_key_balancing=config.use_key_balancing,
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
        node_feats, pos, batch, num_graphs, graph_counts = self._check_inputs(
            node_feats,
            pos,
            batch,
        )
        geometry_dtype = torch.float64 if pos.dtype == torch.float64 else torch.float32
        geometry_pos = pos.to(dtype=geometry_dtype)
        center = _scatter_mean(geometry_pos, batch, num_graphs, graph_counts)
        centered = geometry_pos - center[batch]
        radius = centered.norm(dim=-1, keepdim=True)
        graph_scale = torch.sqrt(
            _scatter_mean(
                centered.square().sum(dim=-1, keepdim=True),
                batch,
                num_graphs,
                graph_counts,
            ).clamp_min(self.config.eps)
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
            scalars, vectors, transient_tensor = layer(
                scalars,
                vectors,
                normalized_pos,
                batch,
                num_graphs,
            )

        node_scalars = self.scalar_out(self.scalar_out_norm(scalars))
        node_vectors = self.vector_out(vectors)
        node_tensors = _st_features_to_matrix(self.tensor_out(transient_tensor))
        return {
            "node_scalars": node_scalars,
            "node_vectors": node_vectors,
            "node_tensors": node_tensors,
            "graph_scalars": _scatter_mean(node_scalars, batch, num_graphs, graph_counts),
            "graph_vectors": _scatter_mean(node_vectors, batch, num_graphs, graph_counts),
            "graph_tensors": _scatter_mean(node_tensors, batch, num_graphs, graph_counts),
        }

    def _check_inputs(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
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
        num_graphs, graph_counts = _graph_metadata(batch)
        return node_feats, pos, batch, num_graphs, graph_counts


class _EquivariantMomentLayer(nn.Module):
    def __init__(
        self,
        scalars: int,
        vectors: int,
        num_heads: int,
        linear_kernel_init: float,
        linear_kernel_max: float,
        vector_kernel_init: float,
        vector_kernel_max: float,
        kernel_floor: float,
        use_linear_kernel: bool,
        use_key_balancing: bool,
        residual_scale_init: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.scalars = scalars
        self.vectors = vectors
        self.num_heads = num_heads
        self.head_dim = scalars // num_heads
        self.eps = eps
        self.linear_kernel_max = linear_kernel_max
        self.vector_kernel_max = vector_kernel_max
        self.kernel_floor = kernel_floor
        self.use_linear_kernel = use_linear_kernel
        self.use_key_balancing = use_key_balancing

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
        self.raw_linear_kernel = nn.Parameter(
            torch.full(
                (num_heads,),
                _inverse_sigmoid(linear_kernel_init / linear_kernel_max),
            )
        )
        self.raw_vector_kernel = nn.Parameter(
            torch.full(
                (num_heads,),
                _inverse_sigmoid(vector_kernel_init / vector_kernel_max),
            )
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
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_norm = self.norm(scalars)
        bounded_vectors = _bounded_irrep(vectors, self.eps)
        n_nodes = scalars.shape[0]

        q0 = _normalize_positive_features(
            F.elu(self.query_scalar(s_norm).reshape(n_nodes, self.num_heads, self.head_dim)) + 1.0,
            self.eps,
        )
        k0 = _normalize_positive_features(
            F.elu(self.key_scalar(s_norm).reshape(n_nodes, self.num_heads, self.head_dim)) + 1.0,
            self.eps,
        )
        q1 = _unit_ball(
            self.query_vector(bounded_vectors)
            * torch.tanh(self.query_vector_gate(s_norm)).unsqueeze(-1),
            self.eps,
        )
        k1 = _unit_ball(
            self.key_vector(bounded_vectors)
            * torch.tanh(self.key_vector_gate(s_norm)).unsqueeze(-1),
            self.eps,
        )
        moment_dtype = _moment_dtype(q0, k0, q1, k1, pos)
        linear_scale = _bounded_kernel_scale(
            self.raw_linear_kernel,
            self.linear_kernel_max,
        ).to(dtype=moment_dtype)
        if not self.use_linear_kernel:
            linear_scale = torch.zeros_like(linear_scale)
        kernel_scale = _bounded_kernel_scale(
            self.raw_vector_kernel,
            self.vector_kernel_max,
        ).to(dtype=moment_dtype)

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
        transported = _factorized_moment_attention(
            q0,
            k0,
            q1,
            k1,
            kernel_scale,
            value,
            batch,
            num_graphs=num_graphs,
            balanced=self.use_key_balancing,
            linear_scale=linear_scale,
            kernel_floor=self.kernel_floor,
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


def _factorized_moment_attention(
    query_scalar: torch.Tensor,
    key_scalar: torch.Tensor,
    query_vector: torch.Tensor,
    key_vector: torch.Tensor,
    kernel_scale: torch.Tensor,
    value: torch.Tensor,
    batch: torch.Tensor,
    *,
    num_graphs: int,
    balanced: bool,
    linear_scale: torch.Tensor | None = None,
    kernel_floor: float = 1.0,
    balance_exponent: torch.Tensor | None = None,
    sinkhorn_iterations: int = 1,
) -> torch.Tensor:
    if not isinstance(sinkhorn_iterations, int) or sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn_iterations must be positive")
    output_dtype = value.dtype
    reduction_dtype = _moment_dtype(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        value,
    )
    query_scalar = query_scalar.to(dtype=reduction_dtype)
    key_scalar = key_scalar.to(dtype=reduction_dtype)
    query_vector = query_vector.to(dtype=reduction_dtype)
    key_vector = key_vector.to(dtype=reduction_dtype)
    kernel_scale = kernel_scale.to(dtype=reduction_dtype)
    if linear_scale is None:
        linear_scale = torch.zeros_like(kernel_scale)
    else:
        linear_scale = linear_scale.to(dtype=reduction_dtype)
    value = value.to(dtype=reduction_dtype)
    if balance_exponent is not None:
        balance_exponent = balance_exponent.to(dtype=reduction_dtype)
    if balanced:
        row_scale = query_scalar.new_ones(query_scalar.shape[:2])
        for _ in range(sinkhorn_iterations):
            key_mass = _structured_key_mass(
                query_scalar,
                key_scalar,
                query_vector,
                key_vector,
                kernel_scale,
                row_scale,
                batch,
                num_graphs,
                linear_scale=linear_scale,
                kernel_floor=kernel_floor,
            )
            if balance_exponent is None:
                key_scale = key_mass.reciprocal()
            else:
                key_scale = torch.exp(-balance_exponent[None, :] * torch.log(key_mass))
            denominator = _structured_row_denominator(
                query_scalar,
                key_scalar,
                query_vector,
                key_vector,
                kernel_scale,
                key_scale,
                batch,
                num_graphs,
                linear_scale=linear_scale,
                kernel_floor=kernel_floor,
            )
            row_scale = denominator.reciprocal()
    else:
        key_scale = key_scalar.new_ones(key_scalar.shape[:2])
        denominator = _structured_row_denominator(
            query_scalar,
            key_scalar,
            query_vector,
            key_vector,
            kernel_scale,
            key_scale,
            batch,
            num_graphs,
            linear_scale=linear_scale,
            kernel_floor=kernel_floor,
        )
    numerator = _structured_numerator(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        kernel_scale,
        key_scale,
        value,
        batch,
        num_graphs,
        linear_scale=linear_scale,
        kernel_floor=kernel_floor,
    )
    return (numerator / denominator.unsqueeze(-1)).to(dtype=output_dtype)


def _structured_key_mass(
    query_scalar: torch.Tensor,
    key_scalar: torch.Tensor,
    query_vector: torch.Tensor,
    key_vector: torch.Tensor,
    kernel_scale: torch.Tensor,
    row_scale: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    *,
    linear_scale: torch.Tensor | None = None,
    kernel_floor: float = 1.0,
) -> torch.Tensor:
    if linear_scale is None:
        linear_scale = torch.zeros_like(kernel_scale)
    scalar_sum = _segment_sum(
        query_scalar * row_scale.unsqueeze(-1),
        batch,
        num_graphs,
    )
    constant_sum = _segment_sum(row_scale, batch, num_graphs)
    linear_sum = _segment_sum(
        query_vector * row_scale.unsqueeze(-1),
        batch,
        num_graphs,
    )
    vector_outer_sum = _segment_sum(
        _vector_outer(query_vector) * row_scale[..., None, None],
        batch,
        num_graphs,
    )
    content = (key_scalar * scalar_sum[batch]).sum(dim=-1)
    linear = (key_vector * linear_sum[batch]).sum(dim=-1)
    quadratic = _positive_quadratic_form(key_vector, vector_outer_sum[batch])
    return (
        content
        + (kernel_floor + linear_scale)[None, :] * constant_sum[batch]
        + linear_scale[None, :] * linear
        + kernel_scale[None, :] * quadratic
    )


def _structured_row_denominator(
    query_scalar: torch.Tensor,
    key_scalar: torch.Tensor,
    query_vector: torch.Tensor,
    key_vector: torch.Tensor,
    kernel_scale: torch.Tensor,
    key_scale: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    *,
    linear_scale: torch.Tensor | None = None,
    kernel_floor: float = 1.0,
) -> torch.Tensor:
    if linear_scale is None:
        linear_scale = torch.zeros_like(kernel_scale)
    scalar_sum = _segment_sum(
        key_scalar * key_scale.unsqueeze(-1),
        batch,
        num_graphs,
    )
    constant_sum = _segment_sum(key_scale, batch, num_graphs)
    linear_sum = _segment_sum(
        key_vector * key_scale.unsqueeze(-1),
        batch,
        num_graphs,
    )
    vector_outer_sum = _segment_sum(
        _vector_outer(key_vector) * key_scale[..., None, None],
        batch,
        num_graphs,
    )
    content = (query_scalar * scalar_sum[batch]).sum(dim=-1)
    linear = (query_vector * linear_sum[batch]).sum(dim=-1)
    quadratic = _positive_quadratic_form(query_vector, vector_outer_sum[batch])
    return (
        content
        + (kernel_floor + linear_scale)[None, :] * constant_sum[batch]
        + linear_scale[None, :] * linear
        + kernel_scale[None, :] * quadratic
    )


def _structured_numerator(
    query_scalar: torch.Tensor,
    key_scalar: torch.Tensor,
    query_vector: torch.Tensor,
    key_vector: torch.Tensor,
    kernel_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    *,
    linear_scale: torch.Tensor | None = None,
    kernel_floor: float = 1.0,
) -> torch.Tensor:
    if linear_scale is None:
        linear_scale = torch.zeros_like(kernel_scale)
    weighted_value = key_scale.unsqueeze(-1) * value
    scalar_summary = _segment_sum(
        key_scalar.unsqueeze(-1) * weighted_value.unsqueeze(-2),
        batch,
        num_graphs,
    )
    constant_summary = _segment_sum(weighted_value, batch, num_graphs)
    linear_summary = _segment_sum(
        key_vector.unsqueeze(-1) * weighted_value.unsqueeze(-2),
        batch,
        num_graphs,
    )
    quadratic_summary = _segment_sum(
        _vector_outer(key_vector).unsqueeze(-1) * weighted_value[..., None, None, :],
        batch,
        num_graphs,
    )
    content = torch.einsum("nhd,nhdf->nhf", query_scalar, scalar_summary[batch])
    linear = torch.einsum("nha,nhaf->nhf", query_vector, linear_summary[batch])
    quadratic = torch.einsum(
        "nha,nhabf,nhb->nhf",
        query_vector,
        quadratic_summary[batch],
        query_vector,
    )
    return (
        content
        + (kernel_floor + linear_scale)[None, :, None] * constant_summary[batch]
        + linear_scale[None, :, None] * linear
        + kernel_scale[None, :, None] * quadratic
    )


def _vector_outer(value: torch.Tensor) -> torch.Tensor:
    return value.unsqueeze(-1) * value.unsqueeze(-2)


def _positive_quadratic_form(vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nha,nhab,nhb->nh", vector, matrix, vector).clamp_min(0.0)


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


def _scatter_mean(
    value: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    graph_counts: torch.Tensor,
) -> torch.Tensor:
    output_dtype = value.dtype
    reduction_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
    reduced = value.to(dtype=reduction_dtype)
    summed = _segment_sum(reduced, batch, num_graphs)
    count = graph_counts.to(device=value.device, dtype=reduction_dtype)
    count = count.reshape(num_graphs, *((1,) * (value.ndim - 1)))
    return (summed / count).to(dtype=output_dtype)


def _graph_metadata(batch: torch.Tensor) -> tuple[int, torch.Tensor]:
    if (batch < 0).any():
        raise ValueError("batch indices must be nonnegative")
    graph_ids = torch.unique(batch, sorted=True)
    expected = torch.arange(graph_ids.numel(), device=batch.device)
    if not torch.equal(graph_ids, expected):
        raise ValueError("batch indices must be contiguous and start at zero")
    num_graphs = graph_ids.numel()
    return num_graphs, torch.bincount(batch, minlength=num_graphs)


def _bounded_irrep(value: torch.Tensor, eps: float) -> torch.Tensor:
    output_dtype = value.dtype
    reduction_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
    reduced = value.to(dtype=reduction_dtype)
    rms = _stable_vector_norm(reduced) / sqrt(reduced.shape[-1])
    scale = torch.hypot(rms, torch.ones_like(rms))
    return (reduced / scale).to(dtype=output_dtype)


def _unit_ball(value: torch.Tensor, eps: float) -> torch.Tensor:
    reduced = value.to(dtype=_moment_dtype(value))
    norm = _stable_vector_norm(reduced)
    normalized = reduced / torch.hypot(norm, torch.ones_like(norm))
    return normalized * _inward_unit_margin(normalized)


def _normalize_positive_features(value: torch.Tensor, eps: float) -> torch.Tensor:
    reduced = value.to(dtype=_moment_dtype(value))
    norm = _stable_vector_norm(reduced)
    floor = torch.full_like(norm, sqrt(eps))
    normalized = reduced / torch.hypot(norm, floor)
    return normalized * _inward_unit_margin(normalized)


def _inward_unit_margin(value: torch.Tensor) -> float:
    dimension = value.shape[-1]
    rounding_margin = 4.0 * dimension * torch.finfo(value.dtype).eps
    return max(0.5, 1.0 - rounding_margin)


def _stable_vector_norm(value: torch.Tensor) -> torch.Tensor:
    magnitude = value.abs().amax(dim=-1, keepdim=True)
    safe_magnitude = magnitude.clamp_min(torch.finfo(value.dtype).tiny)
    scaled = value / safe_magnitude
    scaled_norm_square = scaled.square().sum(dim=-1, keepdim=True)
    scaled_norm = torch.sqrt(scaled_norm_square.clamp_min(torch.finfo(value.dtype).eps))
    return magnitude * scaled_norm


def _bounded_kernel_scale(raw: torch.Tensor, maximum: float) -> torch.Tensor:
    return maximum * torch.sigmoid(raw)


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


def _inverse_sigmoid(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    return float(torch.logit(torch.tensor(probability)).item())


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
    if config.vector_kernel_max <= 0:
        raise ValueError("vector_kernel_max must be positive")
    if config.vector_kernel_init >= config.vector_kernel_max:
        raise ValueError("vector_kernel_init must be smaller than vector_kernel_max")
    if config.linear_kernel_init <= 0:
        raise ValueError("linear_kernel_init must be positive")
    if config.linear_kernel_max <= 0:
        raise ValueError("linear_kernel_max must be positive")
    if config.linear_kernel_init >= config.linear_kernel_max:
        raise ValueError("linear_kernel_init must be smaller than linear_kernel_max")
    if config.kernel_floor <= 0:
        raise ValueError("kernel_floor must be positive")
    hidden = CartesianIrreps.parse(config.hidden_irreps)
    output = CartesianIrreps.parse(config.output_irreps)
    if hidden.scalars <= 0 or hidden.vectors <= 0 or hidden.tensors:
        raise ValueError("hidden_irreps supports only scalar and vector channels, both with positive multiplicity")
    if hidden.scalars % config.num_heads:
        raise ValueError("hidden scalar channels must be divisible by num_heads")
    if output.scalars + output.vectors + output.tensors <= 0:
        raise ValueError("output_irreps must include at least one term")
