from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn

from .irreps import CartesianIrreps
from .model import _scatter_mean


@dataclass(frozen=True)
class EquivariantMomentAttentionConfig:
    node_dim: int
    hidden_irreps: str | CartesianIrreps = "64x0e + 4x1o"
    output_irreps: str | CartesianIrreps = "1x0e"
    num_layers: int = 3
    num_heads: int = 4
    balance_attention: bool = True
    sinkhorn_iterations: int = 1
    radial_trace: bool = False
    full_gram_invariants: bool = False
    shifted_angular_kernel: bool = False
    radial_distance_kernel: bool = False
    dynamic_moment_routing: bool = False
    learnable_balance_exponent: bool = False
    equivariant_ffn: bool = True
    ffn_hidden_ratio: float = 2.0
    vector_kernel_init: float = 0.05
    angular_shift_init: float = 1.1
    radial_distance_shift_init: float = 1.1
    routing_hidden_dim: int = 16
    routing_delta_scale: float = 0.25
    balance_exponent_init: float = 0.9
    residual_scale_init: float = 0.1
    eps: float = 1e-12


class EquivariantMomentAttention(nn.Module):
    """Global linear attention with persistent scalar/vector and transient l=2 states."""

    def __init__(self, config: EquivariantMomentAttentionConfig) -> None:
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
                    balance_attention=config.balance_attention,
                    sinkhorn_iterations=config.sinkhorn_iterations,
                    radial_trace=config.radial_trace,
                    full_gram_invariants=config.full_gram_invariants,
                    shifted_angular_kernel=config.shifted_angular_kernel,
                    radial_distance_kernel=config.radial_distance_kernel,
                    dynamic_moment_routing=config.dynamic_moment_routing,
                    learnable_balance_exponent=config.learnable_balance_exponent,
                    equivariant_ffn=config.equivariant_ffn,
                    ffn_hidden_ratio=config.ffn_hidden_ratio,
                    vector_kernel_init=config.vector_kernel_init,
                    angular_shift_init=config.angular_shift_init,
                    radial_distance_shift_init=config.radial_distance_shift_init,
                    routing_hidden_dim=config.routing_hidden_dim,
                    routing_delta_scale=config.routing_delta_scale,
                    balance_exponent_init=config.balance_exponent_init,
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
    ) -> dict[str, torch.Tensor | str]:
        node_feats, pos, batch = self._check_inputs(node_feats, pos, batch)
        center = _scatter_mean(pos, batch)
        centered = pos - center[batch]
        radius = centered.norm(dim=-1, keepdim=True)
        graph_scale = torch.sqrt(
            _scatter_mean(centered.square().sum(dim=-1, keepdim=True), batch).clamp_min(self.config.eps)
        )
        normalized_pos = centered / graph_scale[batch]
        scalar_input = torch.cat(
            [
                node_feats,
                torch.log1p(radius),
                torch.log1p(graph_scale[batch]),
                normalized_pos.square().sum(dim=-1, keepdim=True),
            ],
            dim=-1,
        )

        scalars = self.scalar_in(scalar_input)
        vectors = torch.tanh(self.vector_in(scalars)).unsqueeze(-1) * normalized_pos.unsqueeze(1)
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
            "hidden_irreps": str(self.hidden_irreps),
            "output_irreps": str(self.output_irreps),
            "attention_mode": "moment_linear",
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
            batch = batch.to(device=node_feats.device, dtype=torch.long)
        if (batch < 0).any():
            raise ValueError("batch indices must be nonnegative")
        return node_feats, pos, batch


class _EquivariantMomentLayer(nn.Module):
    def __init__(
        self,
        scalars: int,
        vectors: int,
        num_heads: int,
        balance_attention: bool,
        sinkhorn_iterations: int,
        radial_trace: bool,
        full_gram_invariants: bool,
        shifted_angular_kernel: bool,
        radial_distance_kernel: bool,
        dynamic_moment_routing: bool,
        learnable_balance_exponent: bool,
        equivariant_ffn: bool,
        ffn_hidden_ratio: float,
        vector_kernel_init: float,
        angular_shift_init: float,
        radial_distance_shift_init: float,
        routing_hidden_dim: int,
        routing_delta_scale: float,
        balance_exponent_init: float,
        residual_scale_init: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.scalars = scalars
        self.vectors = vectors
        self.num_heads = num_heads
        self.head_dim = scalars // num_heads
        self.balance_attention = balance_attention
        self.sinkhorn_iterations = sinkhorn_iterations
        self.radial_trace = radial_trace
        self.full_gram_invariants = full_gram_invariants
        self.shifted_angular_kernel = shifted_angular_kernel
        self.radial_distance_kernel = radial_distance_kernel
        self.dynamic_moment_routing = dynamic_moment_routing
        self.routing_delta_scale = routing_delta_scale
        self.equivariant_ffn = equivariant_ffn
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
        if shifted_angular_kernel:
            self.raw_angular_shift = nn.Parameter(
                torch.full((num_heads,), _inverse_softplus(angular_shift_init - 1.0))
            )
        else:
            self.register_parameter("raw_angular_shift", None)
        if radial_distance_kernel:
            self.raw_radial_distance_shift = nn.Parameter(
                torch.full((num_heads,), _inverse_softplus(radial_distance_shift_init - 1.0))
            )
        else:
            self.register_parameter("raw_radial_distance_shift", None)
        if learnable_balance_exponent:
            self.raw_balance_exponent = nn.Parameter(
                torch.full((num_heads,), _inverse_sigmoid(balance_exponent_init))
            )
        else:
            self.register_parameter("raw_balance_exponent", None)
        self.relative_mix = nn.Parameter(torch.full((num_heads,), 0.1))
        self.tensor_mix = nn.Parameter(torch.full((num_heads,), 0.1))
        self.vector_update = _ChannelMix(num_heads, vectors)
        invariant_dim = scalars + 5 * num_heads
        if radial_trace:
            invariant_dim += num_heads
        if full_gram_invariants:
            invariant_dim += vectors * (vectors + 1) // 2 + vectors * num_heads + 2 * num_heads
        self.scalar_update_norm = nn.LayerNorm(invariant_dim)
        self.scalar_update = nn.Sequential(
            nn.Linear(invariant_dim, scalars),
            nn.SiLU(),
            nn.Linear(scalars, scalars),
        )
        self.scalar_residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.vector_residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        if equivariant_ffn:
            ffn_hidden = max(1, round(scalars * ffn_hidden_ratio))
            self.ffn_norm = nn.LayerNorm(scalars)
            self.ffn_in = nn.Linear(scalars + vectors, 2 * ffn_hidden)
            self.ffn_out = nn.Linear(ffn_hidden, scalars)
            self.ffn_vector_gate = nn.Linear(scalars, vectors)
            self.ffn_vector_mix = _ChannelMix(vectors, vectors)
            self.ffn_scalar_residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
            self.ffn_vector_residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        else:
            self.ffn_norm = None
            self.ffn_in = None
            self.ffn_out = None
            self.ffn_vector_gate = None
            self.ffn_vector_mix = None
            self.register_parameter("ffn_scalar_residual_scale", None)
            self.register_parameter("ffn_vector_residual_scale", None)
        if dynamic_moment_routing:
            with torch.random.fork_rng(devices=[]):
                self.routing_norm = nn.LayerNorm(7)
                self.routing_mlp = nn.Sequential(
                    nn.Linear(7, routing_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(routing_hidden_dim, 4),
                )
                self.routing_context = nn.Linear(scalars, num_heads * 4)
            nn.init.zeros_(self.routing_mlp[-1].weight)
            nn.init.zeros_(self.routing_mlp[-1].bias)
            nn.init.zeros_(self.routing_context.weight)
            nn.init.zeros_(self.routing_context.bias)
        else:
            self.routing_norm = None
            self.routing_mlp = None
            self.routing_context = None

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
        kernel_scale = F.softplus(self.raw_vector_kernel)
        if self.shifted_angular_kernel:
            shift = 1.0 + F.softplus(self.raw_angular_shift)
            query_angular = _shifted_angular_features(_smooth_unit_ball(q1), kernel_scale, shift)
            key_angular = _shifted_angular_features(_smooth_unit_ball(k1), kernel_scale, shift)
        else:
            ones = q0.new_ones((n_nodes, self.num_heads, 1))
            outer_scale = kernel_scale.sqrt()[None, :, None]
            query_angular = torch.cat([ones, outer_scale * _symmetric_outer_features(q1)], dim=-1)
            key_angular = torch.cat([ones, outer_scale * _symmetric_outer_features(k1)], dim=-1)
        query = torch.cat([q0, query_angular], dim=-1)
        key = torch.cat([k0, key_angular], dim=-1)
        if self.radial_distance_kernel:
            radial_shift = 1.0 + F.softplus(self.raw_radial_distance_shift)
            radial_query, radial_key = _radial_distance_features(pos, batch, radial_shift, self.eps)
            query = torch.einsum("nhd,nhr->nhdr", query, radial_query).flatten(start_dim=2)
            key = torch.einsum("nhd,nhr->nhdr", key, radial_key).flatten(start_dim=2)

        scalar_value = self.value_scalar(s_norm).reshape(n_nodes, self.num_heads, self.head_dim)
        vector_value = self.value_vector(bounded_vectors)
        relative_gate = torch.tanh(self.relative_gate(s_norm))
        tensor_gate = torch.tanh(self.tensor_gate(s_norm))
        pos_h = pos.unsqueeze(1).expand(-1, self.num_heads, -1)
        st_pos = _symmetric_traceless_features(pos).unsqueeze(1).expand(-1, self.num_heads, -1)
        value_parts = [
            scalar_value,
            vector_value,
            relative_gate.unsqueeze(-1),
            relative_gate.unsqueeze(-1) * pos_h,
            tensor_gate.unsqueeze(-1),
            tensor_gate.unsqueeze(-1) * pos_h,
            tensor_gate.unsqueeze(-1) * st_pos,
        ]
        if self.radial_trace:
            value_parts.append(tensor_gate.unsqueeze(-1) * pos.square().sum(dim=-1)[:, None, None])
        value = torch.cat(value_parts, dim=-1)
        balance_exponent = (
            torch.sigmoid(self.raw_balance_exponent) if self.raw_balance_exponent is not None else None
        )
        transported = _factorized_attention(
            query,
            key,
            value,
            batch,
            balanced=self.balance_attention,
            balance_exponent=balance_exponent,
            eps=self.eps,
            sinkhorn_iterations=self.sinkhorn_iterations,
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
        tensor_vector = _st_matrix_vector(tensor, q1)
        query_base_dot = (q1 * vector_base).sum(dim=-1)
        query_relative_dot = (q1 * relative).sum(dim=-1)
        relative_square = relative.square().sum(dim=-1)
        tensor_square = _st_frobenius_square(tensor)
        query_tensor_dot = (q1 * tensor_vector).sum(dim=-1)

        if self.dynamic_moment_routing:
            tensor_square_vector = _normalized_st_square_vector(tensor, q1, self.eps)
            query_tensor_square_dot = (q1 * tensor_square_vector).sum(dim=-1)
            routing_features = torch.stack(
                [
                    _signed_unit_interval(query_base_dot),
                    _signed_unit_interval(query_relative_dot),
                    torch.log1p(relative_square),
                    torch.log1p(tensor_square),
                    _signed_unit_interval(query_tensor_dot),
                    _signed_unit_interval(query_tensor_square_dot),
                    _normalized_st_cubic_trace(tensor, self.eps),
                ],
                dim=-1,
            )
            routing_logits = self.routing_mlp(self.routing_norm(routing_features))
            routing_logits = routing_logits + self.routing_context(s_norm).reshape(n_nodes, self.num_heads, 4)
            routing_delta = self.routing_delta_scale * torch.tanh(routing_logits)
            vector_per_head = (
                (1.0 + routing_delta[..., 0, None]) * vector_base
                + (self.relative_mix[None, :, None] + routing_delta[..., 1, None]) * relative
                + (self.tensor_mix[None, :, None] + routing_delta[..., 2, None]) * tensor_vector
                + routing_delta[..., 3, None] * tensor_square_vector
            )
        else:
            vector_per_head = (
                vector_base
                + self.relative_mix[None, :, None] * relative
                + self.tensor_mix[None, :, None] * tensor_vector
            )

        invariant_parts = [
            scalar_message,
            query_base_dot,
            query_relative_dot,
            relative_square,
            tensor_square,
            query_tensor_dot,
        ]
        if self.radial_trace:
            radial_second = transported[..., offset + 16]
            invariant_parts.append(
                _relative_radial_trace(tensor_mass.squeeze(-1), tensor_position, radial_second, pos_h)
            )
        if self.full_gram_invariants:
            invariant_parts.append(_gram_invariants(bounded_vectors, vector_per_head, vector_base, relative))
        scalar_invariants = torch.cat(invariant_parts, dim=-1)
        scalar_delta = self.scalar_update(self.scalar_update_norm(scalar_invariants))
        vector_delta = self.vector_update(vector_per_head)
        scalars = scalars + self.scalar_residual_scale * scalar_delta
        vectors = vectors + self.vector_residual_scale * _bounded_irrep(vector_delta, self.eps)
        if self.equivariant_ffn:
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
        return torch.einsum("oc,nc...->no...", self.weight, value)


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
    return numerator / denominator.unsqueeze(-1)


def _symmetric_outer_features(value: torch.Tensor) -> torch.Tensor:
    x, y, z = value.unbind(dim=-1)
    sqrt_two = sqrt(2.0)
    return torch.stack([x * x, y * y, z * z, sqrt_two * x * y, sqrt_two * x * z, sqrt_two * y * z], dim=-1)


def _shifted_angular_features(
    value: torch.Tensor,
    kernel_scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    scale = kernel_scale[None, :, None]
    offset = shift[None, :, None]
    return torch.cat(
        [
            scale.sqrt() * offset.expand(value.shape[0], -1, -1),
            (2.0 * scale * offset).sqrt() * value,
            scale.sqrt() * _symmetric_outer_features(value),
        ],
        dim=-1,
    )


def _radial_distance_features(
    position: torch.Tensor,
    batch: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_graphs = int(batch.max().item()) + 1
    max_radius = position.new_zeros(num_graphs).scatter_reduce_(
        0,
        batch,
        position.norm(dim=-1),
        reduce="amax",
        include_self=True,
    )
    scaled = position / (2.0 * max_radius.clamp_min(eps))[batch, None]
    squared_radius = scaled.square().sum(dim=-1, keepdim=True)
    ones = squared_radius.new_ones(squared_radius.shape)
    spatial = sqrt(2.0) * scaled
    query = torch.cat([ones, squared_radius, spatial], dim=-1)
    key = torch.cat(
        [
            shift[None, :, None] - squared_radius[:, None, :],
            -ones[:, None, :].expand(-1, shift.numel(), -1),
            spatial[:, None, :].expand(-1, shift.numel(), -1),
        ],
        dim=-1,
    )
    return query[:, None, :].expand(-1, shift.numel(), -1), key


def _relative_radial_trace(
    mass: torch.Tensor,
    first_moment: torch.Tensor,
    second_moment: torch.Tensor,
    position: torch.Tensor,
) -> torch.Tensor:
    return (
        second_moment
        + position.square().sum(dim=-1) * mass
        - 2.0 * (position * first_moment).sum(dim=-1)
    )


def _gram_invariants(
    state_vectors: torch.Tensor,
    message_vectors: torch.Tensor,
    vector_base: torch.Tensor,
    relative: torch.Tensor,
) -> torch.Tensor:
    channels = state_vectors.shape[1]
    row, col = torch.triu_indices(channels, channels, device=state_vectors.device)
    state_gram = torch.einsum("nca,nda->ncd", state_vectors, state_vectors)[:, row, col]
    cross_gram = torch.einsum("nca,nha->nch", state_vectors, message_vectors).flatten(start_dim=1)
    message_invariants = torch.cat(
        [vector_base.square().sum(dim=-1), (vector_base * relative).sum(dim=-1)], dim=-1
    )
    return torch.cat([state_gram, cross_gram, message_invariants], dim=-1)


def _symmetric_traceless_features(value: torch.Tensor) -> torch.Tensor:
    x, y, z = value.unbind(dim=-1)
    trace_third = (x.square() + y.square() + z.square()) / 3.0
    return torch.stack([x.square() - trace_third, y.square() - trace_third, x * y, x * z, y * z], dim=-1)


def _symmetric_traceless_cross_features(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
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
    return torch.einsum("...ab,...b->...a", _st_features_to_matrix(tensor), vector)


def _normalized_st_square_vector(tensor: torch.Tensor, vector: torch.Tensor, eps: float) -> torch.Tensor:
    matrix = _st_features_to_matrix(tensor)
    square = torch.matmul(matrix, matrix)
    trace = square.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    identity = torch.eye(3, dtype=tensor.dtype, device=tensor.device)
    traceless_square = square - (trace / 3.0)[..., None, None] * identity
    return torch.einsum("...ab,...b->...a", traceless_square, vector) / trace.clamp_min(eps)[..., None]


def _normalized_st_cubic_trace(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    matrix = _st_features_to_matrix(tensor)
    square = torch.matmul(matrix, matrix)
    quadratic_trace = square.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cubic_trace = torch.einsum("...ab,...ba->...", square, matrix)
    return cubic_trace / quadratic_trace.clamp_min(eps).pow(1.5)


def _st_frobenius_square(value: torch.Tensor) -> torch.Tensor:
    xx, yy, xy, xz, yz = value.unbind(dim=-1)
    zz = -xx - yy
    return xx.square() + yy.square() + zz.square() + 2.0 * (xy.square() + xz.square() + yz.square())


def _signed_unit_interval(value: torch.Tensor) -> torch.Tensor:
    return value / (1.0 + value.abs())


def _segment_sum(value: torch.Tensor, batch: torch.Tensor, num_segments: int) -> torch.Tensor:
    out = value.new_zeros((num_segments, *value.shape[1:]))
    return out.index_add(0, batch, value)


def _bounded_irrep(value: torch.Tensor, eps: float) -> torch.Tensor:
    scale = torch.sqrt(1.0 + value.square().mean(dim=-1, keepdim=True).clamp_min(eps))
    return value / scale


def _smooth_unit_ball(value: torch.Tensor) -> torch.Tensor:
    return value / torch.sqrt(1.0 + value.square().sum(dim=-1, keepdim=True))


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("vector_kernel_init must be positive")
    return float(torch.log(torch.expm1(torch.tensor(value))).item())


def _inverse_sigmoid(value: float) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError("balance_exponent_init must be between zero and one")
    return float(torch.logit(torch.tensor(value)).item())


def _validate_config(config: EquivariantMomentAttentionConfig) -> None:
    if config.node_dim <= 0:
        raise ValueError("node_dim must be positive")
    if config.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if config.num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if not isinstance(config.sinkhorn_iterations, int) or config.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn_iterations must be positive")
    if config.ffn_hidden_ratio <= 0:
        raise ValueError("ffn_hidden_ratio must be positive")
    if config.routing_hidden_dim <= 0:
        raise ValueError("routing_hidden_dim must be positive")
    if config.routing_delta_scale <= 0:
        raise ValueError("routing_delta_scale must be positive")
    if config.eps <= 0:
        raise ValueError("eps must be positive")
    if config.residual_scale_init < 0:
        raise ValueError("residual_scale_init must be nonnegative")
    if config.vector_kernel_init <= 0:
        raise ValueError("vector_kernel_init must be positive")
    if config.angular_shift_init <= 1:
        raise ValueError("angular_shift_init must be greater than one")
    if config.radial_distance_shift_init <= 1:
        raise ValueError("radial_distance_shift_init must be greater than one")
    if not 0 < config.balance_exponent_init < 1:
        raise ValueError("balance_exponent_init must be between zero and one")
    hidden = CartesianIrreps.parse(config.hidden_irreps)
    output = CartesianIrreps.parse(config.output_irreps)
    if hidden.scalars <= 0 or hidden.vectors <= 0 or hidden.tensors:
        raise ValueError("hidden_irreps supports only scalar and vector channels, both with positive multiplicity")
    if hidden.scalars % config.num_heads:
        raise ValueError("hidden scalar channels must be divisible by num_heads")
    if output.scalars + output.vectors + output.tensors <= 0:
        raise ValueError("output_irreps must include at least one term")
