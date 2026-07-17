from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import torch
import torch.nn.functional as F
from torch import nn

from .irreps import CartesianIrreps


_MEMORY_ROUTER_DIM = 8
_MEMORY_ROUTER_LOGIT_SCALE = 4.0


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
    kernel_floor_mode: str = "fixed"
    use_alignment_linear_term: bool = True
    use_key_balancing: bool = True
    local_head_counts: tuple[int, ...] | None = None
    local_cutoff: float = 2.5
    num_rbf: int = 16
    learn_local_radial_gate: bool = False
    global_memory_count: int = 1
    use_memory_interaction: bool = False
    memory_assignment_temperature: float = 1.0
    memory_assignment_scale: float = 2.5
    memory_interaction_cutoff: float = 2.5
    use_radial_trace: bool = False
    residual_scale_init: float = 0.1
    eps: float = 1e-12


class EquivariantAttention(nn.Module):
    """O(3)-equivariant local/global attention with exact factorized moments."""

    attention_kind = "factorized_moment"
    symmetry = "O3"

    def __init__(self, config: EquivariantAttentionConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.hidden_irreps = CartesianIrreps.parse(config.hidden_irreps)
        self.output_irreps = CartesianIrreps.parse(config.output_irreps)

        self.scalar_in = nn.Linear(config.node_dim, self.hidden_irreps.scalars)
        self.global_scalar_in = nn.Linear(3, self.hidden_irreps.scalars, bias=False)
        self.vector_in = nn.Linear(
            self.hidden_irreps.scalars, self.hidden_irreps.vectors
        )
        local_head_counts = config.local_head_counts or (0,) * config.num_layers
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
                    kernel_floor_mode=config.kernel_floor_mode,
                    use_alignment_linear_term=config.use_alignment_linear_term,
                    use_key_balancing=config.use_key_balancing,
                    local_head_count=local_head_counts[layer_index],
                    local_cutoff=config.local_cutoff,
                    num_rbf=config.num_rbf,
                    learn_local_radial_gate=config.learn_local_radial_gate,
                    global_memory_count=config.global_memory_count,
                    use_memory_interaction=config.use_memory_interaction,
                    memory_assignment_temperature=config.memory_assignment_temperature,
                    memory_assignment_scale=config.memory_assignment_scale,
                    memory_interaction_cutoff=config.memory_interaction_cutoff,
                    use_radial_trace=config.use_radial_trace,
                    residual_scale_init=layer_scale,
                    eps=config.eps,
                )
                for layer_index in range(config.num_layers)
            ]
        )
        self.scalar_out_norm = nn.LayerNorm(self.hidden_irreps.scalars)
        self.scalar_out = nn.Linear(
            self.hidden_irreps.scalars, self.output_irreps.scalars
        )
        self.vector_out = _ChannelMix(
            self.hidden_irreps.vectors, self.output_irreps.vectors
        )
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
        normalized_pos, log_radius, log_graph_scale, log_normalized_square = (
            _scale_first_geometry(
                pos,
                batch,
                num_graphs=num_graphs,
                graph_counts=graph_counts,
            )
        )
        global_scalar_input = torch.cat(
            [
                log_radius.to(dtype=node_feats.dtype),
                log_graph_scale[batch].to(dtype=node_feats.dtype),
                log_normalized_square.to(dtype=node_feats.dtype),
            ],
            dim=-1,
        )

        scalars = self.scalar_in(node_feats)
        vectors = scalars.new_zeros((scalars.shape[0], self.hidden_irreps.vectors, 3))
        transient_tensor = normalized_pos.new_zeros(
            (normalized_pos.shape[0], self.config.num_heads, 5)
        )
        local_geometry = None
        if any(layer.local_head_count for layer in self.layers):
            local_geometry = _local_geometry(
                pos,
                batch,
                num_graphs=num_graphs,
                cutoff=self.config.local_cutoff,
                num_rbf=self.config.num_rbf,
                graph_counts=graph_counts,
            )
        global_geometry_injected = False
        for layer in self.layers:
            if layer.global_head_count and not global_geometry_injected:
                scalars = scalars + self.global_scalar_in(global_scalar_input)
                vector_gate = torch.tanh(self.vector_in(scalars)).unsqueeze(-1)
                geometry_vector = vector_gate.to(
                    dtype=normalized_pos.dtype
                ) * normalized_pos.unsqueeze(1)
                vectors = vectors + geometry_vector.to(dtype=vectors.dtype)
                global_geometry_injected = True
            scalars, vectors, transient_tensor = layer(
                scalars,
                vectors,
                normalized_pos,
                pos,
                batch,
                num_graphs,
                graph_counts,
                local_geometry,
            )

        node_scalars = self.scalar_out(self.scalar_out_norm(scalars))
        node_vectors = self.vector_out(vectors)
        node_tensors = _st_features_to_matrix(self.tensor_out(transient_tensor))
        return {
            "node_scalars": node_scalars,
            "node_vectors": node_vectors,
            "node_tensors": node_tensors,
            "graph_scalars": _scatter_mean(
                node_scalars, batch, num_graphs, graph_counts
            ),
            "graph_vectors": _scatter_mean(
                node_vectors, batch, num_graphs, graph_counts
            ),
            "graph_tensors": _scatter_mean(
                node_tensors, batch, num_graphs, graph_counts
            ),
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
        if not torch.is_floating_point(node_feats):
            raise TypeError("node_feats must be a floating point tensor")
        if pos.dtype not in {torch.float32, torch.float64}:
            raise TypeError("pos must use float32 or float64 coordinates")
        if node_feats.device != pos.device:
            raise ValueError("node_feats and pos must be on the same device")
        if not torch.isfinite(node_feats).all() or not torch.isfinite(pos).all():
            raise ValueError("node_feats and pos must be finite")
        if batch is None:
            batch = torch.zeros(
                node_feats.shape[0], dtype=torch.long, device=node_feats.device
            )
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
                raise ValueError(
                    "batch, node_feats, and pos must be on the same device"
                )
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
        kernel_floor_mode: str,
        use_alignment_linear_term: bool,
        use_key_balancing: bool,
        local_head_count: int,
        local_cutoff: float,
        num_rbf: int,
        learn_local_radial_gate: bool,
        global_memory_count: int,
        use_memory_interaction: bool,
        memory_assignment_temperature: float,
        memory_assignment_scale: float,
        memory_interaction_cutoff: float,
        use_radial_trace: bool,
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
        self.kernel_floor_mode = kernel_floor_mode
        self.use_alignment_linear_term = use_alignment_linear_term
        self.use_key_balancing = use_key_balancing
        self.local_head_count = local_head_count
        self.global_head_count = num_heads - local_head_count
        self.local_cutoff = local_cutoff
        self.num_rbf = num_rbf
        self.global_memory_count = global_memory_count
        self.use_memory_interaction = use_memory_interaction
        self.memory_assignment_temperature = memory_assignment_temperature
        self.memory_assignment_scale = memory_assignment_scale
        self.memory_interaction_cutoff = memory_interaction_cutoff
        self.use_radial_trace = use_radial_trace

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
        self.radial_trace_gate = nn.Linear(scalars, num_heads)
        self.local_radial_weight = nn.Parameter(
            torch.zeros(num_heads, num_rbf),
            requires_grad=learn_local_radial_gate,
        )
        self.local_radial_bias = nn.Parameter(
            torch.zeros(num_heads),
            requires_grad=learn_local_radial_gate,
        )
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
        invariant_dim = scalars + 6 * num_heads
        self.scalar_update_norm = nn.LayerNorm(invariant_dim)
        self.scalar_update = nn.Sequential(
            nn.Linear(invariant_dim, scalars),
            nn.SiLU(),
            nn.Linear(scalars, scalars),
        )
        self.scalar_residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )
        self.vector_residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )
        ffn_hidden = 2 * scalars
        self.ffn_norm = nn.LayerNorm(scalars)
        self.ffn_in = nn.Linear(scalars + vectors, 2 * ffn_hidden)
        self.ffn_out = nn.Linear(ffn_hidden, scalars)
        self.ffn_vector_gate = nn.Linear(scalars, vectors)
        self.ffn_vector_mix = _ChannelMix(vectors, vectors)
        self.ffn_scalar_residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )
        self.ffn_vector_residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )
        # Allocated for every route and M so comparisons retain one state schema.
        self.memory_router_in = nn.Linear(self.head_dim, _MEMORY_ROUTER_DIM)
        self.memory_router_out = nn.Linear(_MEMORY_ROUTER_DIM, _MEMORY_ROUTER_DIM)

    def forward(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
        global_pos: torch.Tensor,
        raw_pos: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        graph_counts: torch.Tensor,
        local_geometry: tuple[torch.Tensor, ...] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_norm = self.norm(scalars)
        bounded_vectors = _bounded_irrep(vectors, self.eps)
        n_nodes = scalars.shape[0]

        q0 = _normalize_positive_features(
            F.elu(
                self.query_scalar(s_norm).reshape(
                    n_nodes, self.num_heads, self.head_dim
                )
            )
            + 1.0,
            self.eps,
        )
        k0 = _normalize_positive_features(
            F.elu(
                self.key_scalar(s_norm).reshape(n_nodes, self.num_heads, self.head_dim)
            )
            + 1.0,
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
        moment_dtype = _moment_dtype(q0, k0, q1, k1, global_pos, raw_pos)
        alignment_scale = _bounded_kernel_scale(
            self.raw_linear_kernel,
            self.linear_kernel_max,
        ).to(dtype=moment_dtype)
        alignment_dot_scale = (
            alignment_scale
            if self.use_alignment_linear_term
            else torch.zeros_like(alignment_scale)
        )
        kernel_scale = _bounded_kernel_scale(
            self.raw_vector_kernel,
            self.vector_kernel_max,
        ).to(dtype=moment_dtype)

        scalar_value = self.value_scalar(s_norm).reshape(
            n_nodes, self.num_heads, self.head_dim
        )
        vector_value = self.value_vector(bounded_vectors)
        relative_gate = torch.tanh(self.relative_gate(s_norm))
        tensor_gate = torch.tanh(self.tensor_gate(s_norm))
        radial_trace_gate = torch.tanh(self.radial_trace_gate(s_norm))

        message_groups: list[tuple[torch.Tensor, ...]] = []
        if self.local_head_count:
            local = slice(0, self.local_head_count)
            receiver, sender, weights, displacement, squared_distance = (
                _local_attention_weights(
                    q0[:, local],
                    k0[:, local],
                    q1[:, local],
                    k1[:, local],
                    kernel_scale[local],
                    raw_pos,
                    batch,
                    num_graphs=num_graphs,
                    balanced=self.use_key_balancing,
                    alignment_scale=alignment_scale[local],
                    alignment_dot_scale=alignment_dot_scale[local],
                    kernel_floor=self.kernel_floor,
                    cutoff=self.local_cutoff,
                    num_rbf=self.num_rbf,
                    radial_weight=self.local_radial_weight[local],
                    radial_bias=self.local_radial_bias[local],
                    local_geometry=local_geometry,
                )
            )
            message_groups.append(
                _local_moment_messages(
                    receiver,
                    sender,
                    weights,
                    displacement,
                    squared_distance,
                    scalar_value[:, local],
                    vector_value[:, local],
                    relative_gate[:, local],
                    tensor_gate[:, local],
                    radial_trace_gate[:, local],
                    use_radial_trace=self.use_radial_trace,
                    num_nodes=n_nodes,
                )
            )
        if self.global_head_count:
            global_heads = slice(self.local_head_count, self.num_heads)
            memory_router_latent = None
            if self.use_memory_interaction and self.global_memory_count > 1:
                memory_router_latent = torch.tanh(
                    self.memory_router_out(
                        F.silu(self.memory_router_in(k0[:, global_heads]))
                    )
                )
                router_norm = _stable_vector_norm(memory_router_latent)
                memory_router_latent = memory_router_latent / router_norm.clamp_min(
                    torch.finfo(memory_router_latent.dtype).tiny
                )
            message_groups.append(
                _global_moment_messages(
                    q0[:, global_heads],
                    k0[:, global_heads],
                    q1[:, global_heads],
                    k1[:, global_heads],
                    kernel_scale[global_heads],
                    scalar_value[:, global_heads],
                    vector_value[:, global_heads],
                    relative_gate[:, global_heads],
                    tensor_gate[:, global_heads],
                    radial_trace_gate[:, global_heads],
                    global_pos,
                    batch,
                    num_graphs=num_graphs,
                    graph_counts=graph_counts,
                    balanced=self.use_key_balancing,
                    alignment_scale=alignment_scale[global_heads],
                    alignment_dot_scale=alignment_dot_scale[global_heads],
                    kernel_floor=self.kernel_floor,
                    kernel_floor_mode=self.kernel_floor_mode,
                    memory_count=self.global_memory_count,
                    memory_temperature=self.memory_assignment_temperature,
                    memory_assignment_scale=self.memory_assignment_scale,
                    memory_interaction_cutoff=self.memory_interaction_cutoff,
                    memory_router_latent=memory_router_latent,
                    use_memory_interaction=self.use_memory_interaction,
                    use_radial_trace=self.use_radial_trace,
                )
            )

        scalar_message, vector_base, relative, tensor, radial_trace = (
            torch.cat(values, dim=1) for values in zip(*message_groups, strict=True)
        )
        scalar_message = scalar_message.reshape(n_nodes, self.scalars)
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
                radial_trace,
            ],
            dim=-1,
        )
        normalized_invariants = _stable_layer_norm(
            self.scalar_update_norm, scalar_invariants
        )
        scalar_delta = self.scalar_update(normalized_invariants.to(dtype=scalars.dtype))
        vector_delta = self.vector_update(vector_per_head)
        scalars = scalars + self.scalar_residual_scale * scalar_delta
        bounded_delta = _bounded_irrep(vector_delta, self.eps).to(dtype=vectors.dtype)
        vectors = vectors + self.vector_residual_scale * bounded_delta
        ffn_scalars = self.ffn_norm(scalars)
        ffn_vectors = _bounded_irrep(vectors, self.eps)
        ffn_invariants = torch.cat(
            [ffn_scalars, ffn_vectors.square().sum(dim=-1)], dim=-1
        )
        ffn_content, ffn_gate = self.ffn_in(ffn_invariants).chunk(2, dim=-1)
        scalar_ffn = self.ffn_out(F.silu(ffn_content) * ffn_gate)
        vector_ffn = self.ffn_vector_mix(ffn_vectors) * torch.tanh(
            self.ffn_vector_gate(ffn_scalars)
        ).unsqueeze(-1)
        scalars = scalars + self.ffn_scalar_residual_scale * scalar_ffn
        vectors = vectors + self.ffn_vector_residual_scale * _bounded_irrep(
            vector_ffn, self.eps
        )
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


def _global_moment_messages(
    query_scalar: torch.Tensor,
    key_scalar: torch.Tensor,
    query_vector: torch.Tensor,
    key_vector: torch.Tensor,
    kernel_scale: torch.Tensor,
    scalar_value: torch.Tensor,
    vector_value: torch.Tensor,
    relative_gate: torch.Tensor,
    tensor_gate: torch.Tensor,
    radial_trace_gate: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    *,
    num_graphs: int,
    graph_counts: torch.Tensor,
    balanced: bool,
    alignment_scale: torch.Tensor,
    alignment_dot_scale: torch.Tensor,
    kernel_floor: float,
    kernel_floor_mode: str,
    memory_count: int,
    memory_temperature: float,
    memory_assignment_scale: float,
    memory_interaction_cutoff: float,
    memory_router_latent: torch.Tensor | None = None,
    use_memory_interaction: bool,
    use_radial_trace: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    moment_dtype = _moment_dtype(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        scalar_value,
        vector_value,
        pos,
    )
    num_heads = query_scalar.shape[1]
    pos = pos.to(dtype=moment_dtype)
    pos_h = pos.unsqueeze(1).expand(-1, num_heads, -1)
    st_pos = _symmetric_traceless_features(pos).unsqueeze(1).expand(-1, num_heads, -1)
    relative_gate = relative_gate.to(dtype=moment_dtype)
    tensor_gate = tensor_gate.to(dtype=moment_dtype)
    value_parts = [
        scalar_value.to(dtype=moment_dtype),
        vector_value.to(dtype=moment_dtype),
        relative_gate.unsqueeze(-1),
        relative_gate.unsqueeze(-1) * pos_h,
        tensor_gate.unsqueeze(-1),
        tensor_gate.unsqueeze(-1) * pos_h,
        tensor_gate.unsqueeze(-1) * st_pos,
    ]
    if use_radial_trace:
        radial_trace_gate = radial_trace_gate.to(dtype=moment_dtype)
        value_parts.extend(
            [
                radial_trace_gate.unsqueeze(-1),
                radial_trace_gate.unsqueeze(-1) * pos_h,
                radial_trace_gate.unsqueeze(-1)
                * pos_h.square().sum(dim=-1, keepdim=True),
            ]
        )
    value = torch.cat(value_parts, dim=-1)
    if use_memory_interaction and memory_count > 1:
        assignment, coupling, _ = _memory_assignments_and_coupling(
            key_scalar,
            pos,
            batch,
            num_graphs=num_graphs,
            memory_count=memory_count,
            temperature=memory_temperature,
            assignment_scale=memory_assignment_scale,
            interaction_cutoff=memory_interaction_cutoff,
            router_latent=memory_router_latent,
            interact=True,
        )
        transported = _memory_factorized_attention(
            query_scalar,
            key_scalar,
            query_vector,
            key_vector,
            kernel_scale,
            value,
            assignment,
            coupling,
            batch,
            num_graphs=num_graphs,
            balanced=balanced,
            alignment_scale=alignment_scale,
            alignment_dot_scale=alignment_dot_scale,
            kernel_floor=kernel_floor,
            kernel_floor_mode=kernel_floor_mode,
            graph_counts=graph_counts,
        )
    else:
        transported = _factorized_moment_attention(
            query_scalar,
            key_scalar,
            query_vector,
            key_vector,
            kernel_scale,
            value,
            batch,
            num_graphs=num_graphs,
            balanced=balanced,
            alignment_scale=alignment_scale,
            alignment_dot_scale=alignment_dot_scale,
            kernel_floor=kernel_floor,
            kernel_floor_mode=kernel_floor_mode,
            graph_counts=graph_counts,
        )

    offset = scalar_value.shape[-1]
    scalar_message = transported[..., :offset]
    vector_base = transported[..., offset : offset + 3]
    relative_mass = transported[..., offset + 3]
    relative_position = transported[..., offset + 4 : offset + 7]
    relative = relative_position - pos_h * relative_mass.unsqueeze(-1)
    tensor_mass = transported[..., offset + 7]
    tensor_position = transported[..., offset + 8 : offset + 11]
    tensor_second = transported[..., offset + 11 : offset + 16]
    tensor = (
        tensor_second
        + st_pos * tensor_mass.unsqueeze(-1)
        - 2.0 * _symmetric_traceless_cross_features(tensor_position, pos_h)
    )
    if use_radial_trace:
        radial_mass = transported[..., offset + 16]
        radial_first = transported[..., offset + 17 : offset + 20]
        radial_second = transported[..., offset + 20]
        radial_trace = _relative_radial_trace(
            radial_second, radial_first, radial_mass, pos_h
        )
    else:
        radial_trace = transported.new_zeros(transported.shape[:2])
    return scalar_message, vector_base, relative, tensor, radial_trace


def _local_moment_messages(
    receiver: torch.Tensor,
    sender: torch.Tensor,
    weights: torch.Tensor,
    displacement: torch.Tensor,
    squared_distance: torch.Tensor,
    scalar_value: torch.Tensor,
    vector_value: torch.Tensor,
    relative_gate: torch.Tensor,
    tensor_gate: torch.Tensor,
    radial_trace_gate: torch.Tensor,
    *,
    use_radial_trace: bool,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = _moment_dtype(
        weights,
        displacement,
        scalar_value,
        vector_value,
        relative_gate,
        tensor_gate,
    )
    weights = weights.to(dtype=dtype)
    displacement = displacement.to(dtype=dtype)
    scalar_message = _edge_sum(
        weights,
        scalar_value[sender].to(dtype=dtype),
        receiver,
        num_nodes,
    )
    vector_base = _edge_sum(
        weights,
        vector_value[sender].to(dtype=dtype),
        receiver,
        num_nodes,
    )
    relative = _edge_sum(
        weights,
        relative_gate[sender].to(dtype=dtype).unsqueeze(-1) * displacement.unsqueeze(1),
        receiver,
        num_nodes,
    )
    tensor = _edge_sum(
        weights,
        tensor_gate[sender].to(dtype=dtype).unsqueeze(-1)
        * _symmetric_traceless_features(displacement).unsqueeze(1),
        receiver,
        num_nodes,
    )
    if use_radial_trace:
        radial_value = radial_trace_gate[sender].to(dtype=dtype) * squared_distance.to(
            dtype=dtype
        ).unsqueeze(-1)
        radial_trace = _edge_sum(
            weights,
            radial_value,
            receiver,
            num_nodes,
        )
    else:
        radial_trace = weights.new_zeros((num_nodes, weights.shape[1]))
    return scalar_message, vector_base, relative, tensor, radial_trace


def _edge_sum(
    weights: torch.Tensor,
    value: torch.Tensor,
    receiver: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    expanded_weights = weights.reshape(
        weights.shape[0],
        weights.shape[1],
        *((1,) * (value.ndim - 2)),
    )
    output = value.new_zeros((num_nodes, *value.shape[1:]))
    return output.index_add(0, receiver, expanded_weights * value)


def _local_attention_weights(
    query_scalar: torch.Tensor,
    key_scalar: torch.Tensor,
    query_vector: torch.Tensor,
    key_vector: torch.Tensor,
    kernel_scale: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    *,
    num_graphs: int,
    balanced: bool,
    alignment_scale: torch.Tensor | None = None,
    alignment_dot_scale: torch.Tensor | None = None,
    kernel_floor: float = 1.0,
    cutoff: float = 2.5,
    num_rbf: int = 16,
    radial_weight: torch.Tensor,
    radial_bias: torch.Tensor,
    local_geometry: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if local_geometry is None:
        local_geometry = _local_geometry(
            pos,
            batch,
            num_graphs=num_graphs,
            cutoff=cutoff,
            num_rbf=num_rbf,
        )
    receiver, sender, displacement, squared_distance, rbf = local_geometry

    dtype = _moment_dtype(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        kernel_scale,
        pos,
        radial_weight,
        radial_bias,
    )
    query_scalar = query_scalar.to(dtype=dtype)
    key_scalar = key_scalar.to(dtype=dtype)
    query_vector = query_vector.to(dtype=dtype)
    key_vector = key_vector.to(dtype=dtype)
    kernel_scale = kernel_scale.to(dtype=dtype)
    alignment_scale, alignment_dot_scale = _resolve_alignment_scales(
        kernel_scale,
        alignment_scale,
        alignment_dot_scale,
    )
    content = (query_scalar[receiver] * key_scalar[sender]).sum(dim=-1)
    angular = (query_vector[receiver] * key_vector[sender]).sum(dim=-1)
    kernel = (
        float(kernel_floor)
        + content
        + alignment_scale.unsqueeze(0)
        + alignment_dot_scale.unsqueeze(0) * angular
        + kernel_scale.unsqueeze(0) * angular.square()
    )
    radial_logits = torch.einsum(
        "ek,hk->eh",
        rbf.to(dtype=dtype),
        radial_weight.to(dtype=dtype),
    ) + radial_bias.to(dtype=dtype).unsqueeze(0)
    radial_floor = 1e-3
    radial_gate = _cosine_of_squared_distance_cutoff(
        squared_distance.to(dtype=dtype)
    ).unsqueeze(-1) * (
        radial_floor + (1.0 - radial_floor) * torch.sigmoid(radial_logits)
    )
    weighted = kernel * radial_gate
    num_nodes = query_scalar.shape[0]
    if balanced:
        key_mass = weighted.new_zeros((num_nodes, weighted.shape[1])).index_add(
            0,
            sender,
            weighted,
        )
        weighted = weighted / key_mass[sender]
    denominator = weighted.new_zeros((num_nodes, weighted.shape[1])).index_add(
        0,
        receiver,
        weighted,
    )
    weights = weighted / denominator[receiver]
    return (
        receiver,
        sender,
        weights,
        displacement.to(dtype=dtype),
        squared_distance.to(dtype=dtype),
    )


def _local_geometry(
    pos: torch.Tensor,
    batch: torch.Tensor,
    *,
    num_graphs: int,
    cutoff: float,
    num_rbf: int,
    graph_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if graph_counts is None:
        graph_counts = torch.bincount(batch, minlength=num_graphs)
    receiver, sender = _batched_complete_graph_edges(batch, graph_counts)
    cutoff_tensor = pos.new_full((), float(cutoff))
    candidate_distance = _stable_vector_norm(pos[sender] - pos[receiver]).squeeze(-1)
    inside = candidate_distance < cutoff_tensor
    receiver = receiver[inside]
    sender = sender[inside]
    displacement = (pos[sender] - pos[receiver]) / cutoff_tensor
    squared_distance = displacement.square().sum(dim=-1)
    rbf_centers = torch.linspace(
        0.0,
        1.0,
        num_rbf,
        dtype=pos.dtype,
        device=pos.device,
    )
    rbf_width = 1.0 / max(1, num_rbf - 1)
    rbf = torch.exp(
        -0.5 * ((squared_distance.unsqueeze(-1) - rbf_centers) / rbf_width).square()
    )
    return receiver, sender, displacement, squared_distance, rbf


def _batched_complete_graph_edges(
    batch: torch.Tensor,
    graph_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build graph-local Cartesian products with one vectorized index expansion."""
    sorted_nodes = torch.argsort(batch, stable=True)
    counts_per_receiver = graph_counts[batch[sorted_nodes]]
    total_edges = int(graph_counts.square().sum().item())
    receiver_positions = torch.arange(
        batch.numel(), device=batch.device
    ).repeat_interleave(counts_per_receiver, output_size=total_edges)
    receiver = sorted_nodes[receiver_positions]

    receiver_starts = torch.cumsum(counts_per_receiver, dim=0) - counts_per_receiver
    sender_local = torch.arange(total_edges, device=batch.device) - receiver_starts[
        receiver_positions
    ]
    graph_starts = torch.cumsum(graph_counts, dim=0) - graph_counts
    sender_positions = graph_starts[batch[receiver]] + sender_local
    sender = sorted_nodes[sender_positions]
    return receiver, sender


def _cosine_of_squared_distance_cutoff(
    squared_scaled_distance: torch.Tensor,
) -> torch.Tensor:
    inside = (squared_scaled_distance >= 0.0) & (squared_scaled_distance < 1.0)
    bounded_distance = squared_scaled_distance.clamp(min=0.0, max=1.0)
    smooth = 0.5 * (torch.cos(torch.pi * bounded_distance) + 1.0)
    return torch.where(inside, smooth, torch.zeros_like(smooth))


def _relative_radial_trace(
    second: torch.Tensor,
    first: torch.Tensor,
    mass: torch.Tensor,
    pos_h: torch.Tensor,
) -> torch.Tensor:
    dtype = _moment_dtype(second, first, mass, pos_h)
    second = second.to(dtype=dtype)
    first = first.to(dtype=dtype)
    mass = mass.to(dtype=dtype)
    pos_h = pos_h.to(dtype=dtype)
    return (
        second - 2.0 * (first * pos_h).sum(dim=-1) + mass * pos_h.square().sum(dim=-1)
    )


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
    alignment_scale: torch.Tensor | None = None,
    alignment_dot_scale: torch.Tensor | None = None,
    kernel_floor: float = 1.0,
    kernel_floor_mode: str = "fixed",
    graph_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    if balanced and kernel_floor_mode == "inverse_graph_size":
        raise ValueError(
            "inverse_graph_size kernel floor is not registered with key balancing"
        )
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
    if alignment_scale is None:
        alignment_scale = torch.zeros_like(kernel_scale)
    alignment_scale = alignment_scale.to(dtype=reduction_dtype)
    if alignment_dot_scale is None:
        alignment_dot_scale = alignment_scale
    alignment_dot_scale = alignment_dot_scale.to(dtype=reduction_dtype)
    value = value.to(dtype=reduction_dtype)
    if balanced:
        row_scale = query_scalar.new_ones(query_scalar.shape[:2])
        key_mass = _structured_key_mass(
            query_scalar,
            key_scalar,
            query_vector,
            key_vector,
            kernel_scale,
            row_scale,
            batch,
            num_graphs,
            alignment_scale=alignment_scale,
            alignment_dot_scale=alignment_dot_scale,
            kernel_floor=kernel_floor,
            kernel_floor_mode=kernel_floor_mode,
            graph_counts=graph_counts,
        )
        key_scale = key_mass.reciprocal()
        denominator = _structured_row_denominator(
            query_scalar,
            key_scalar,
            query_vector,
            key_vector,
            kernel_scale,
            key_scale,
            batch,
            num_graphs,
            alignment_scale=alignment_scale,
            alignment_dot_scale=alignment_dot_scale,
            kernel_floor=kernel_floor,
            kernel_floor_mode=kernel_floor_mode,
            graph_counts=graph_counts,
        )
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
            alignment_scale=alignment_scale,
            alignment_dot_scale=alignment_dot_scale,
            kernel_floor=kernel_floor,
            kernel_floor_mode=kernel_floor_mode,
            graph_counts=graph_counts,
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
        alignment_scale=alignment_scale,
        alignment_dot_scale=alignment_dot_scale,
        kernel_floor=kernel_floor,
        kernel_floor_mode=kernel_floor_mode,
        graph_counts=graph_counts,
    )
    return (numerator / denominator.unsqueeze(-1)).to(dtype=output_dtype)


def _memory_factorized_attention(
    query_scalar: torch.Tensor,
    key_scalar: torch.Tensor,
    query_vector: torch.Tensor,
    key_vector: torch.Tensor,
    kernel_scale: torch.Tensor,
    value: torch.Tensor,
    assignment: torch.Tensor,
    coupling: torch.Tensor,
    batch: torch.Tensor,
    *,
    num_graphs: int,
    balanced: bool,
    alignment_scale: torch.Tensor | None = None,
    alignment_dot_scale: torch.Tensor | None = None,
    kernel_floor: float = 1.0,
    kernel_floor_mode: str = "fixed",
    graph_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    if assignment.shape[-1] == 1:
        return _factorized_moment_attention(
            query_scalar,
            key_scalar,
            query_vector,
            key_vector,
            kernel_scale,
            value,
            batch,
            num_graphs=num_graphs,
            balanced=balanced,
            alignment_scale=alignment_scale,
            alignment_dot_scale=alignment_dot_scale,
            kernel_floor=kernel_floor,
            kernel_floor_mode=kernel_floor_mode,
            graph_counts=graph_counts,
        )
    if balanced and kernel_floor_mode == "inverse_graph_size":
        raise ValueError(
            "inverse_graph_size kernel floor is not registered with key balancing"
        )
    output_dtype = value.dtype
    dtype = _moment_dtype(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        kernel_scale,
        value,
        assignment,
        coupling,
    )
    query_scalar = query_scalar.to(dtype=dtype)
    key_scalar = key_scalar.to(dtype=dtype)
    query_vector = query_vector.to(dtype=dtype)
    key_vector = key_vector.to(dtype=dtype)
    kernel_scale = kernel_scale.to(dtype=dtype)
    value = value.to(dtype=dtype)
    assignment = assignment.to(dtype=dtype)
    coupling = coupling.to(dtype=dtype)
    alignment_scale, alignment_dot_scale = _resolve_alignment_scales(
        kernel_scale,
        alignment_scale,
        alignment_dot_scale,
    )
    if balanced:
        row_scale = query_scalar.new_ones(query_scalar.shape[:2])
        key_mass = _memory_key_mass(
            query_scalar,
            key_scalar,
            query_vector,
            key_vector,
            kernel_scale,
            row_scale,
            assignment,
            coupling,
            batch,
            num_graphs,
            alignment_scale=alignment_scale,
            alignment_dot_scale=alignment_dot_scale,
            kernel_floor=kernel_floor,
            kernel_floor_mode=kernel_floor_mode,
            graph_counts=graph_counts,
        )
        key_scale = key_mass.reciprocal()
    else:
        key_scale = key_scalar.new_ones(key_scalar.shape[:2])
    denominator = _memory_row_denominator(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        kernel_scale,
        key_scale,
        assignment,
        coupling,
        batch,
        num_graphs,
        alignment_scale=alignment_scale,
        alignment_dot_scale=alignment_dot_scale,
        kernel_floor=kernel_floor,
        kernel_floor_mode=kernel_floor_mode,
        graph_counts=graph_counts,
    )
    numerator = _memory_numerator(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        kernel_scale,
        key_scale,
        value,
        assignment,
        coupling,
        batch,
        num_graphs,
        alignment_scale=alignment_scale,
        alignment_dot_scale=alignment_dot_scale,
        kernel_floor=kernel_floor,
        kernel_floor_mode=kernel_floor_mode,
        graph_counts=graph_counts,
    )
    return (numerator / denominator.unsqueeze(-1)).to(dtype=output_dtype)


def _memory_key_mass(
    query_scalar: torch.Tensor,
    key_scalar: torch.Tensor,
    query_vector: torch.Tensor,
    key_vector: torch.Tensor,
    kernel_scale: torch.Tensor,
    row_scale: torch.Tensor,
    assignment: torch.Tensor,
    coupling: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    *,
    alignment_scale: torch.Tensor,
    alignment_dot_scale: torch.Tensor,
    kernel_floor: float,
    kernel_floor_mode: str,
    graph_counts: torch.Tensor | None,
) -> torch.Tensor:
    weighted_assignment = assignment * row_scale.unsqueeze(-1)
    scalar_summary = _segment_sum(
        weighted_assignment.unsqueeze(-1) * query_scalar.unsqueeze(-2),
        batch,
        num_graphs,
    )
    constant_summary = _segment_sum(weighted_assignment, batch, num_graphs)
    linear_summary = _segment_sum(
        weighted_assignment.unsqueeze(-1) * query_vector.unsqueeze(-2),
        batch,
        num_graphs,
    )
    quadratic_summary = _segment_sum(
        weighted_assignment[..., None, None]
        * _vector_outer(query_vector).unsqueeze(-3),
        batch,
        num_graphs,
    )
    mixed_scalar = torch.einsum("ghmn,ghmd->ghnd", coupling, scalar_summary)
    mixed_constant = torch.einsum("ghmn,ghm->ghn", coupling, constant_summary)
    mixed_linear = torch.einsum("ghmn,ghma->ghna", coupling, linear_summary)
    mixed_quadratic = torch.einsum("ghmn,ghmab->ghnab", coupling, quadratic_summary)
    content = torch.einsum(
        "nhm,nhd,nhmd->nh",
        assignment,
        key_scalar,
        mixed_scalar[batch],
    )
    constant = torch.einsum("nhm,nhm->nh", assignment, mixed_constant[batch])
    linear = torch.einsum(
        "nhm,nha,nhma->nh",
        assignment,
        key_vector,
        mixed_linear[batch],
    )
    quadratic = torch.einsum(
        "nhm,nha,nhmab,nhb->nh",
        assignment,
        key_vector,
        mixed_quadratic[batch],
        key_vector,
    )
    pair_floor = _pair_floor(
        key_scalar,
        batch,
        kernel_floor,
        kernel_floor_mode,
        graph_counts,
    )
    pair_alignment_scale, pair_alignment_dot_scale = _pair_alignment_scales(
        key_scalar,
        batch,
        alignment_scale,
        alignment_dot_scale,
        kernel_floor_mode,
        graph_counts,
    )
    return (
        content
        + (pair_floor + pair_alignment_scale) * constant
        + pair_alignment_dot_scale * linear
        + kernel_scale.unsqueeze(0) * quadratic
    )


def _memory_row_denominator(
    query_scalar: torch.Tensor,
    key_scalar: torch.Tensor,
    query_vector: torch.Tensor,
    key_vector: torch.Tensor,
    kernel_scale: torch.Tensor,
    key_scale: torch.Tensor,
    assignment: torch.Tensor,
    coupling: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    *,
    alignment_scale: torch.Tensor,
    alignment_dot_scale: torch.Tensor,
    kernel_floor: float,
    kernel_floor_mode: str,
    graph_counts: torch.Tensor | None,
) -> torch.Tensor:
    weighted_assignment = assignment * key_scale.unsqueeze(-1)
    scalar_summary = _segment_sum(
        weighted_assignment.unsqueeze(-1) * key_scalar.unsqueeze(-2),
        batch,
        num_graphs,
    )
    constant_summary = _segment_sum(weighted_assignment, batch, num_graphs)
    linear_summary = _segment_sum(
        weighted_assignment.unsqueeze(-1) * key_vector.unsqueeze(-2),
        batch,
        num_graphs,
    )
    quadratic_summary = _segment_sum(
        weighted_assignment[..., None, None] * _vector_outer(key_vector).unsqueeze(-3),
        batch,
        num_graphs,
    )
    mixed_scalar = torch.einsum("ghmn,ghnd->ghmd", coupling, scalar_summary)
    mixed_constant = torch.einsum("ghmn,ghn->ghm", coupling, constant_summary)
    mixed_linear = torch.einsum("ghmn,ghna->ghma", coupling, linear_summary)
    mixed_quadratic = torch.einsum("ghmn,ghnab->ghmab", coupling, quadratic_summary)
    content = torch.einsum(
        "nhm,nhd,nhmd->nh",
        assignment,
        query_scalar,
        mixed_scalar[batch],
    )
    constant = torch.einsum("nhm,nhm->nh", assignment, mixed_constant[batch])
    linear = torch.einsum(
        "nhm,nha,nhma->nh",
        assignment,
        query_vector,
        mixed_linear[batch],
    )
    quadratic = torch.einsum(
        "nhm,nha,nhmab,nhb->nh",
        assignment,
        query_vector,
        mixed_quadratic[batch],
        query_vector,
    )
    pair_floor = _pair_floor(
        query_scalar,
        batch,
        kernel_floor,
        kernel_floor_mode,
        graph_counts,
    )
    pair_alignment_scale, pair_alignment_dot_scale = _pair_alignment_scales(
        query_scalar,
        batch,
        alignment_scale,
        alignment_dot_scale,
        kernel_floor_mode,
        graph_counts,
    )
    return (
        content
        + (pair_floor + pair_alignment_scale) * constant
        + pair_alignment_dot_scale * linear
        + kernel_scale.unsqueeze(0) * quadratic
    )


def _memory_numerator(
    query_scalar: torch.Tensor,
    key_scalar: torch.Tensor,
    query_vector: torch.Tensor,
    key_vector: torch.Tensor,
    kernel_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value: torch.Tensor,
    assignment: torch.Tensor,
    coupling: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    *,
    alignment_scale: torch.Tensor,
    alignment_dot_scale: torch.Tensor,
    kernel_floor: float,
    kernel_floor_mode: str,
    graph_counts: torch.Tensor | None,
) -> torch.Tensor:
    weighted_value = key_scale.unsqueeze(-1) * value
    memory_value = assignment.unsqueeze(-1) * weighted_value.unsqueeze(-2)
    scalar_summary = _segment_sum(
        key_scalar.unsqueeze(-2).unsqueeze(-1) * memory_value.unsqueeze(-2),
        batch,
        num_graphs,
    )
    constant_summary = _segment_sum(memory_value, batch, num_graphs)
    linear_summary = _segment_sum(
        key_vector.unsqueeze(-2).unsqueeze(-1) * memory_value.unsqueeze(-2),
        batch,
        num_graphs,
    )
    quadratic_summary = _segment_sum(
        _vector_outer(key_vector).unsqueeze(-3).unsqueeze(-1)
        * memory_value[..., None, None, :],
        batch,
        num_graphs,
    )
    mixed_scalar = torch.einsum("ghmn,ghndf->ghmdf", coupling, scalar_summary)
    mixed_constant = torch.einsum("ghmn,ghnf->ghmf", coupling, constant_summary)
    mixed_linear = torch.einsum("ghmn,ghnaf->ghmaf", coupling, linear_summary)
    mixed_quadratic = torch.einsum("ghmn,ghnabf->ghmabf", coupling, quadratic_summary)
    content = torch.einsum(
        "nhm,nhd,nhmdf->nhf",
        assignment,
        query_scalar,
        mixed_scalar[batch],
    )
    constant = torch.einsum("nhm,nhmf->nhf", assignment, mixed_constant[batch])
    linear = torch.einsum(
        "nhm,nha,nhmaf->nhf",
        assignment,
        query_vector,
        mixed_linear[batch],
    )
    quadratic = torch.einsum(
        "nhm,nha,nhmabf,nhb->nhf",
        assignment,
        query_vector,
        mixed_quadratic[batch],
        query_vector,
    )
    pair_floor = _pair_floor(
        query_scalar,
        batch,
        kernel_floor,
        kernel_floor_mode,
        graph_counts,
    )
    pair_alignment_scale, pair_alignment_dot_scale = _pair_alignment_scales(
        query_scalar,
        batch,
        alignment_scale,
        alignment_dot_scale,
        kernel_floor_mode,
        graph_counts,
    )
    return (
        content
        + (pair_floor + pair_alignment_scale).unsqueeze(-1) * constant
        + pair_alignment_dot_scale.unsqueeze(-1) * linear
        + kernel_scale[None, :, None] * quadratic
    )


def _memory_assignments_and_coupling(
    key_scalar: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    *,
    num_graphs: int,
    memory_count: int,
    temperature: float,
    assignment_scale: float,
    interaction_cutoff: float,
    interact: bool,
    router_latent: torch.Tensor | None = None,
    identity_mix: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = _moment_dtype(key_scalar, pos)
    key_scalar = key_scalar.to(dtype=dtype)
    pos = pos.to(dtype=dtype)
    memory_index = torch.arange(
        memory_count,
        dtype=dtype,
        device=key_scalar.device,
    )
    if router_latent is None:
        feature_index = torch.arange(
            key_scalar.shape[-1],
            dtype=dtype,
            device=key_scalar.device,
        )
        feature_code = torch.cos(
            torch.pi * (feature_index + 0.5) / (2.0 * max(1, key_scalar.shape[-1]))
        )
        feature_code = feature_code / _stable_vector_norm(feature_code)
        scalar_coordinate = torch.tanh(
            torch.einsum("nhd,d->nh", key_scalar, feature_code)
        )
        basis_index = torch.arange(
            memory_count,
            dtype=dtype,
            device=key_scalar.device,
        )
        invariant_basis = scalar_coordinate.unsqueeze(-1).pow(basis_index)
        logit_scale = 1.0
    else:
        if router_latent.ndim != 3 or router_latent.shape[:2] != key_scalar.shape[:2]:
            raise ValueError(
                "router_latent must have shape (nodes, heads, router_dimension)"
            )
        if router_latent.shape[-1] <= 0 or router_latent.is_complex():
            raise ValueError("router_latent must be a nonempty real-valued tensor")
        invariant_basis = router_latent.to(device=key_scalar.device, dtype=dtype)
        if not bool(torch.isfinite(invariant_basis).all().item()):
            raise ValueError("router_latent must contain only finite values")
        basis_index = torch.arange(
            invariant_basis.shape[-1],
            dtype=dtype,
            device=key_scalar.device,
        )
        logit_scale = _MEMORY_ROUTER_LOGIT_SCALE
    slot_codes = torch.cos(
        torch.pi
        * (memory_index[:, None] + 0.5)
        * (basis_index[None, :] + 0.5)
        / memory_count
    )
    slot_codes = slot_codes / _stable_vector_norm(slot_codes)
    projected_logits = logit_scale * torch.einsum(
        "nhd,md->nhm", invariant_basis, slot_codes
    )
    temperature_tensor = projected_logits.new_full((), float(temperature))
    bounded_denominator = torch.maximum(
        temperature_tensor,
        projected_logits.abs() / 32.0,
    )
    logits = 8.0 * torch.tanh(projected_logits / bounded_denominator / 8.0)
    preliminary_assignment = torch.softmax(logits, dim=-1)
    preliminary_occupancy = _segment_sum(
        preliminary_assignment,
        batch,
        num_graphs,
    )
    preliminary_centers = _segment_sum(
        preliminary_assignment.unsqueeze(-1) * pos[:, None, None, :],
        batch,
        num_graphs,
    ) / preliminary_occupancy.unsqueeze(-1)
    assignment_distance = _stable_vector_norm(
        pos[:, None, None, :] - preliminary_centers[batch]
    ).squeeze(-1)
    bounded_penalty = 4.0 * _bounded_square_fraction(
        assignment_distance,
        assignment_scale,
    )
    assignment = torch.softmax(logits - bounded_penalty, dim=-1)
    occupancy = _segment_sum(assignment, batch, num_graphs)
    centers = _segment_sum(
        assignment.unsqueeze(-1) * pos[:, None, None, :],
        batch,
        num_graphs,
    ) / occupancy.unsqueeze(-1)
    if interact:
        center_distance = _stable_vector_norm(
            centers.unsqueeze(-2) - centers.unsqueeze(-3)
        ).squeeze(-1)
        cutoff_tensor = center_distance.new_full((), float(interaction_cutoff))
        distance_normalizer = torch.maximum(center_distance, cutoff_tensor)
        center_square = (center_distance / distance_normalizer).square()
        radial_coupling = _cosine_of_squared_distance_cutoff(center_square)
        coupling = _mix_memory_coupling(radial_coupling, identity_mix)
    else:
        coupling = key_scalar.new_ones(
            (num_graphs, key_scalar.shape[1], memory_count, memory_count)
        )
    return assignment, coupling, centers


def _mix_memory_coupling(
    radial_coupling: torch.Tensor,
    identity_mix: float,
) -> torch.Tensor:
    """Mix radial coupling with identity while retaining an exact unit diagonal."""

    if isinstance(identity_mix, bool) or not isinstance(identity_mix, (int, float)):
        raise TypeError("identity_mix must be a real number")
    value = float(identity_mix)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("identity_mix must be finite and lie in [0, 1]")
    if radial_coupling.ndim < 2 or radial_coupling.shape[-1] != radial_coupling.shape[-2]:
        raise ValueError("radial_coupling must contain square memory matrices")
    if value == 0.0:
        return radial_coupling
    memories = radial_coupling.shape[-1]
    diagonal = torch.eye(
        memories,
        dtype=torch.bool,
        device=radial_coupling.device,
    )
    return torch.where(
        diagonal,
        torch.ones_like(radial_coupling),
        radial_coupling * (1.0 - value),
    )


def _bounded_square_fraction(distance: torch.Tensor, scale: float) -> torch.Tensor:
    """Evaluate d² / (s² + d²) without first forming either square."""

    scale_tensor = distance.new_full((), float(scale))
    normalizer = torch.maximum(distance, scale_tensor)
    distance_ratio = distance / normalizer
    scale_ratio = scale_tensor / normalizer
    distance_square = distance_ratio.square()
    return distance_square / (distance_square + scale_ratio.square())


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
    alignment_scale: torch.Tensor | None = None,
    alignment_dot_scale: torch.Tensor | None = None,
    kernel_floor: float = 1.0,
    kernel_floor_mode: str = "fixed",
    graph_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    alignment_scale, alignment_dot_scale = _resolve_alignment_scales(
        kernel_scale,
        alignment_scale,
        alignment_dot_scale,
    )
    pair_floor = _pair_floor(
        query_scalar,
        batch,
        kernel_floor,
        kernel_floor_mode,
        graph_counts,
    )
    pair_alignment_scale, pair_alignment_dot_scale = _pair_alignment_scales(
        query_scalar,
        batch,
        alignment_scale,
        alignment_dot_scale,
        kernel_floor_mode,
        graph_counts,
    )
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
        + (pair_floor + pair_alignment_scale) * constant_sum[batch]
        + pair_alignment_dot_scale * linear
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
    alignment_scale: torch.Tensor | None = None,
    alignment_dot_scale: torch.Tensor | None = None,
    kernel_floor: float = 1.0,
    kernel_floor_mode: str = "fixed",
    graph_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    alignment_scale, alignment_dot_scale = _resolve_alignment_scales(
        kernel_scale,
        alignment_scale,
        alignment_dot_scale,
    )
    pair_floor = _pair_floor(
        query_scalar,
        batch,
        kernel_floor,
        kernel_floor_mode,
        graph_counts,
    )
    pair_alignment_scale, pair_alignment_dot_scale = _pair_alignment_scales(
        query_scalar,
        batch,
        alignment_scale,
        alignment_dot_scale,
        kernel_floor_mode,
        graph_counts,
    )
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
        + (pair_floor + pair_alignment_scale) * constant_sum[batch]
        + pair_alignment_dot_scale * linear
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
    alignment_scale: torch.Tensor | None = None,
    alignment_dot_scale: torch.Tensor | None = None,
    kernel_floor: float = 1.0,
    kernel_floor_mode: str = "fixed",
    graph_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    alignment_scale, alignment_dot_scale = _resolve_alignment_scales(
        kernel_scale,
        alignment_scale,
        alignment_dot_scale,
    )
    pair_floor = _pair_floor(
        query_scalar,
        batch,
        kernel_floor,
        kernel_floor_mode,
        graph_counts,
    )
    pair_alignment_scale, pair_alignment_dot_scale = _pair_alignment_scales(
        query_scalar,
        batch,
        alignment_scale,
        alignment_dot_scale,
        kernel_floor_mode,
        graph_counts,
    )
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
        + (pair_floor + pair_alignment_scale).unsqueeze(-1) * constant_summary[batch]
        + pair_alignment_dot_scale.unsqueeze(-1) * linear
        + kernel_scale[None, :, None] * quadratic
    )


def _resolve_alignment_scales(
    kernel_scale: torch.Tensor,
    alignment_scale: torch.Tensor | None,
    alignment_dot_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if alignment_scale is None:
        alignment_scale = torch.zeros_like(kernel_scale)
    else:
        alignment_scale = alignment_scale.to(
            device=kernel_scale.device, dtype=kernel_scale.dtype
        )
    if alignment_dot_scale is None:
        alignment_dot_scale = alignment_scale
    else:
        alignment_dot_scale = alignment_dot_scale.to(
            device=kernel_scale.device,
            dtype=kernel_scale.dtype,
        )
    return alignment_scale, alignment_dot_scale


def _pair_floor(
    reference: torch.Tensor,
    batch: torch.Tensor,
    kernel_floor: float,
    kernel_floor_mode: str,
    graph_counts: torch.Tensor | None,
) -> torch.Tensor:
    floor = reference.new_full((reference.shape[0], 1), float(kernel_floor))
    if kernel_floor_mode == "fixed":
        return floor
    if kernel_floor_mode != "inverse_graph_size":
        raise ValueError(f"unknown kernel_floor_mode: {kernel_floor_mode}")
    if graph_counts is None:
        raise ValueError(
            "graph_counts are required for inverse_graph_size kernel baseline"
        )
    counts = graph_counts.to(device=reference.device, dtype=reference.dtype)
    return floor / counts[batch].unsqueeze(-1)


def _pair_alignment_scales(
    reference: torch.Tensor,
    batch: torch.Tensor,
    alignment_scale: torch.Tensor,
    alignment_dot_scale: torch.Tensor,
    kernel_floor_mode: str,
    graph_counts: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    graph_scale = _pair_graph_scale(
        reference,
        batch,
        kernel_floor_mode,
        graph_counts,
    )
    return (
        graph_scale * alignment_scale.unsqueeze(0),
        graph_scale * alignment_dot_scale.unsqueeze(0),
    )


def _pair_graph_scale(
    reference: torch.Tensor,
    batch: torch.Tensor,
    kernel_floor_mode: str,
    graph_counts: torch.Tensor | None,
) -> torch.Tensor:
    scale = reference.new_ones((reference.shape[0], 1))
    if kernel_floor_mode == "fixed":
        return scale
    if kernel_floor_mode != "inverse_graph_size":
        raise ValueError(f"unknown kernel_floor_mode: {kernel_floor_mode}")
    if graph_counts is None:
        raise ValueError(
            "graph_counts are required for inverse_graph_size kernel baseline"
        )
    counts = graph_counts.to(device=reference.device, dtype=reference.dtype)
    return scale / counts[batch].unsqueeze(-1)


def _vector_outer(value: torch.Tensor) -> torch.Tensor:
    return value.unsqueeze(-1) * value.unsqueeze(-2)


def _positive_quadratic_form(
    vector: torch.Tensor, matrix: torch.Tensor
) -> torch.Tensor:
    return torch.einsum("nha,nhab,nhb->nh", vector, matrix, vector).clamp_min(0.0)


def _symmetric_traceless_features(value: torch.Tensor) -> torch.Tensor:
    value = value.to(dtype=_moment_dtype(value))
    x, y, z = value.unbind(dim=-1)
    trace_third = (x.square() + y.square() + z.square()) / 3.0
    return torch.stack(
        [x.square() - trace_third, y.square() - trace_third, x * y, x * z, y * z],
        dim=-1,
    )


def _symmetric_traceless_cross_features(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
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
    return (
        xx.square()
        + yy.square()
        + zz.square()
        + 2.0 * (xy.square() + xz.square() + yz.square())
    )


def _segment_sum(
    value: torch.Tensor, batch: torch.Tensor, num_segments: int
) -> torch.Tensor:
    out = value.new_zeros((num_segments, *value.shape[1:]))
    return out.index_add(0, batch, value)


def _segment_amax(
    value: torch.Tensor, batch: torch.Tensor, num_segments: int
) -> torch.Tensor:
    out = value.new_zeros((num_segments, *value.shape[1:]))
    index = batch.reshape(-1, *((1,) * (value.ndim - 1))).expand_as(value)
    return out.scatter_reduce(0, index, value, reduce="amax", include_self=True)


def _scale_first_geometry(
    pos: torch.Tensor,
    batch: torch.Tensor,
    *,
    num_graphs: int,
    graph_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    graph_magnitude = _segment_amax(
        pos.abs().amax(dim=-1, keepdim=True),
        batch,
        num_graphs,
    )
    safe_magnitude = torch.where(
        graph_magnitude > 0.0,
        graph_magnitude,
        torch.ones_like(graph_magnitude),
    )
    scaled = pos / safe_magnitude[batch]
    center = _scatter_mean(scaled, batch, num_graphs, graph_counts)
    centered = scaled - center[batch]
    scaled_radius = _stable_vector_norm(centered)
    scaled_mean_square = _scatter_mean(
        centered.square().sum(dim=-1, keepdim=True),
        batch,
        num_graphs,
        graph_counts,
    )
    has_extent = scaled_mean_square > 0.0
    sqrt_input = torch.where(
        has_extent,
        scaled_mean_square,
        torch.ones_like(scaled_mean_square),
    )
    scaled_rms = torch.where(
        has_extent,
        torch.sqrt(sqrt_input),
        torch.zeros_like(scaled_mean_square),
    )
    safe_rms = torch.where(scaled_rms > 0.0, scaled_rms, torch.ones_like(scaled_rms))
    normalized = centered / safe_rms[batch]
    log_radius = _stable_log1p_product(safe_magnitude[batch], scaled_radius)
    log_graph_scale = _stable_log1p_product(safe_magnitude, scaled_rms)
    log_normalized_square = torch.log1p(normalized.square().sum(dim=-1, keepdim=True))
    return normalized, log_radius, log_graph_scale, log_normalized_square


def _stable_log1p_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(left.dtype).tiny
    zero = (left == 0.0) | (right == 0.0)
    log_product = torch.log(left.clamp_min(tiny)) + torch.log(right.clamp_min(tiny))
    return torch.where(zero, torch.zeros_like(log_product), F.softplus(log_product))


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
    return (
        torch.float64
        if any(value.dtype == torch.float64 for value in values)
        else torch.float32
    )


def _inverse_sigmoid(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    return float(torch.logit(torch.tensor(probability)).item())


def _validate_config(config: EquivariantAttentionConfig) -> None:
    for name in (
        "node_dim",
        "num_layers",
        "num_heads",
        "num_rbf",
        "global_memory_count",
    ):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for name in (
        "use_alignment_linear_term",
        "use_key_balancing",
        "learn_local_radial_gate",
        "use_memory_interaction",
        "use_radial_trace",
    ):
        if not isinstance(getattr(config, name), bool):
            raise TypeError(f"{name} must be a bool")
    if config.local_head_counts is None:
        local_head_counts = (0,) * config.num_layers
    else:
        if not isinstance(config.local_head_counts, tuple):
            raise TypeError("local_head_counts must be a tuple of integers or None")
        if len(config.local_head_counts) != config.num_layers:
            raise ValueError("local_head_counts length must equal num_layers")
        for local_heads in config.local_head_counts:
            if not isinstance(local_heads, int) or isinstance(local_heads, bool):
                raise TypeError("local_head_counts must contain only integers")
            if not 0 <= local_heads <= config.num_heads:
                raise ValueError(
                    "each local_head_counts value must lie between zero and num_heads"
                )
        local_head_counts = config.local_head_counts
    if config.use_memory_interaction:
        registered_lgl = (
            config.num_heads,
            0,
            config.num_heads,
        )
        if config.num_layers != 3 or local_head_counts != registered_lgl:
            raise ValueError(
                "memory interaction is registered only for the middle global stage of a three-layer lgl route"
            )
    if config.kernel_floor_mode not in {"fixed", "inverse_graph_size"}:
        raise ValueError("kernel_floor_mode must be 'fixed' or 'inverse_graph_size'")
    if config.kernel_floor_mode == "inverse_graph_size" and config.use_key_balancing:
        raise ValueError(
            "inverse_graph_size kernel floor is not registered with key balancing"
        )

    eps = _float32_control("eps", config.eps, positive=True)
    residual_scale = _float32_control(
        "residual_scale_init",
        config.residual_scale_init,
        nonnegative=True,
    )
    vector_init = _normal_float32_control(
        "vector_kernel_init", config.vector_kernel_init, positive=True
    )
    vector_max = _normal_float32_control(
        "vector_kernel_max", config.vector_kernel_max, positive=True
    )
    linear_init = _normal_float32_control(
        "linear_kernel_init", config.linear_kernel_init, positive=True
    )
    linear_max = _normal_float32_control(
        "linear_kernel_max", config.linear_kernel_max, positive=True
    )
    kernel_floor = _normal_float32_control(
        "kernel_floor", config.kernel_floor, positive=True
    )
    _float32_control("local_cutoff", config.local_cutoff, positive=True)
    _float32_control(
        "memory_assignment_temperature",
        config.memory_assignment_temperature,
        positive=True,
    )
    _float32_control(
        "memory_assignment_scale", config.memory_assignment_scale, positive=True
    )
    _float32_control(
        "memory_interaction_cutoff",
        config.memory_interaction_cutoff,
        positive=True,
    )
    del eps, residual_scale
    if vector_init >= vector_max:
        raise ValueError(
            "vector_kernel_init must be smaller than vector_kernel_max in float32"
        )
    if linear_init >= linear_max:
        raise ValueError(
            "linear_kernel_init must be smaller than linear_kernel_max in float32"
        )
    upper_bound = torch.tensor(kernel_floor, dtype=torch.float32)
    upper_bound = upper_bound + 1.0 + 2.0 * linear_max + vector_max
    if not torch.isfinite(upper_bound):
        raise ValueError("kernel upper bound must be finite in float32")
    _normal_float32_ratio(
        "vector_kernel_init/vector_kernel_max",
        vector_init,
        vector_max,
    )
    _normal_float32_ratio(
        "linear_kernel_init/linear_kernel_max",
        linear_init,
        linear_max,
    )
    hidden = CartesianIrreps.parse(config.hidden_irreps)
    output = CartesianIrreps.parse(config.output_irreps)
    if hidden.scalars <= 0 or hidden.vectors <= 0 or hidden.tensors:
        raise ValueError(
            "hidden_irreps supports only scalar and vector channels, both with positive multiplicity"
        )
    if hidden.scalars % config.num_heads:
        raise ValueError("hidden scalar channels must be divisible by num_heads")
    if output.scalars + output.vectors + output.tensors <= 0:
        raise ValueError("output_irreps must include at least one term")


def _float32_control(
    name: str,
    value: object,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    converted = float(torch.tensor(numeric, dtype=torch.float32).item())
    if not isfinite(converted):
        raise ValueError(f"{name} must be finite in float32")
    if positive and converted <= 0.0:
        raise ValueError(f"{name} must be positive in float32")
    if nonnegative and converted < 0.0:
        raise ValueError(f"{name} must be nonnegative in float32")
    return converted


def _normal_float32_control(
    name: str,
    value: object,
    *,
    positive: bool = False,
) -> float:
    converted = _float32_control(name, value, positive=positive)
    if positive and converted < torch.finfo(torch.float32).tiny:
        raise ValueError(f"{name} must be a normal float32 value")
    return converted


def _normal_float32_ratio(name: str, numerator: float, denominator: float) -> None:
    ratio = torch.tensor(numerator, dtype=torch.float32) / torch.tensor(
        denominator,
        dtype=torch.float32,
    )
    if float(ratio) < torch.finfo(torch.float32).tiny:
        raise ValueError(f"{name} ratio must be a normal float32 value")
