from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import factorial, isfinite, prod, sqrt
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .irreps import CartesianIrreps


_MEMORY_ROUTER_DIM = 8
_MEMORY_ROUTER_LOGIT_SCALE = 4.0
_GLOBAL_TRANSPORT_MODES = frozenset({"learned", "uniform", "none"})
_SCALAR_CONTENT_MODES = frozenset({"bounded", "unit"})
_COORDINATE_NEIGHBOR_POLICIES = frozenset({"error", "fixed", "rebuild"})
_LOCAL_RBF_SPACINGS = frozenset({"squared", "distance"})
_READOUT_MODES = frozenset({"interaction", "mean", "sum"})


@dataclass(frozen=True)
class _LocalGeometry:
    receiver: torch.Tensor
    sender: torch.Tensor
    displacement: torch.Tensor
    squared_distance: torch.Tensor
    rbf: torch.Tensor
    nonself_receiver: torch.Tensor
    nonself_sender: torch.Tensor
    nonself_displacement: torch.Tensor
    nonself_squared_distance: torch.Tensor
    nonself_rbf: torch.Tensor
    nonself_cutoff: torch.Tensor
    nonself_tensor_features: torch.Tensor

    def _base(self) -> tuple[torch.Tensor, ...]:
        return (
            self.receiver,
            self.sender,
            self.displacement,
            self.squared_distance,
            self.rbf,
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self._base())

    def __len__(self) -> int:
        return 5

    def __getitem__(
        self,
        index: int | slice,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return self._base()[index]


_LocalGeometryInput = _LocalGeometry | tuple[torch.Tensor, ...]


def _nonself_local_geometry(
    geometry: _LocalGeometryInput,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(geometry, _LocalGeometry):
        return (
            geometry.nonself_receiver,
            geometry.nonself_sender,
            geometry.nonself_displacement,
            geometry.nonself_squared_distance,
            geometry.nonself_rbf,
        )
    receiver, sender, displacement, squared_distance, rbf = geometry
    nonself = receiver != sender
    return (
        receiver[nonself],
        sender[nonself],
        displacement[nonself],
        squared_distance[nonself],
        rbf[nonself],
    )


def _nonself_cutoff(
    geometry: _LocalGeometryInput,
    squared_distance: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if (
        isinstance(geometry, _LocalGeometry)
        and geometry.nonself_cutoff.dtype == dtype
    ):
        return geometry.nonself_cutoff
    return _cosine_of_squared_distance_cutoff(squared_distance.to(dtype=dtype))


def _nonself_tensor_features(
    geometry: _LocalGeometryInput,
    displacement: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if (
        isinstance(geometry, _LocalGeometry)
        and geometry.nonself_tensor_features.dtype == dtype
    ):
        return geometry.nonself_tensor_features
    return _symmetric_traceless_features(displacement.to(dtype=dtype))


def routing_head_counts(
    routing: str,
    *,
    num_layers: int,
    num_heads: int,
) -> tuple[int, ...]:
    """Resolve the registered all-local/all-global routing presets."""
    if not isinstance(routing, str):
        raise TypeError("routing must be a string")
    for name, value in (("num_layers", num_layers), ("num_heads", num_heads)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if routing == "ggg":
        return (0,) * num_layers
    if num_layers != 3:
        raise ValueError(
            "lgg, ggl, lgl, and lll routing presets require exactly three layers"
        )
    routes = {
        "lgg": (num_heads, 0, 0),
        "ggl": (0, 0, num_heads),
        "lgl": (num_heads, 0, num_heads),
        "lll": (num_heads, num_heads, num_heads),
    }
    try:
        return routes[routing]
    except KeyError as exc:
        raise ValueError(f"unknown routing preset: {routing}") from exc


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
    global_transport_mode: str = "learned"
    coordinate_updates: bool = False
    coordinate_neighbor_policy: str = "error"
    use_multiscale_spatial_kernel: bool = False
    use_pairwise_local_content: bool = False
    pairwise_residual_scale_init: float = 0.1
    use_edge_conditioned_local_transport: bool = False
    normalize_edge_conditioned_local_by_sqrt_degree: bool = False
    use_gated_local_transport: bool = False
    use_grouped_invariant_normalization: bool = False
    readout_mode: str = "mean"
    scalar_content_mode: str = "unit"
    use_tensor_product_kernel: bool = False
    tensor_kernel_init: float = 0.05
    tensor_kernel_max: float = 1.0
    input_vector_dim: int = 0
    input_tensor_dim: int = 0
    use_irrep_rms_normalization: bool = False
    angular_feature_rank: int = 1
    use_quartic_kernel: bool = False
    quartic_kernel_init: float = 0.01
    quartic_kernel_max: float = 1.0
    checkpoint_gated_local_mlp: bool = False
    # Appended so existing positional construction keeps its meaning.
    local_rbf_spacing: str = "squared"


class EquivariantAttention(nn.Module):
    """O(3)-equivariant local/global attention with exact factorized moments."""

    attention_kind = "factorized_moment"
    symmetry = "O3"

    def __init__(self, config: EquivariantAttentionConfig) -> None:
        super().__init__()
        _validate_config(config)
        if config.use_memory_interaction and config.global_memory_count > 1:
            warnings.warn(
                "Interacting HEMM remains experimental and Stage-0 blocked; "
                "do not interpret this arm as an admitted performance candidate.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.config = config
        self.hidden_irreps = CartesianIrreps.parse(config.hidden_irreps)
        self.output_irreps = CartesianIrreps.parse(config.output_irreps)
        local_head_counts = config.local_head_counts or (0,) * config.num_layers
        if config.use_edge_conditioned_local_transport:
            if self.hidden_irreps.vectors != config.num_heads:
                raise ValueError(
                    "edge-conditioned local transport requires hidden vector "
                    "channels to equal num_heads"
                )
            if any(
                local_heads not in {0, config.num_heads}
                for local_heads in local_head_counts
            ):
                raise ValueError(
                    "edge-conditioned local transport requires each local stage "
                    "to use all heads"
                )
        if config.use_gated_local_transport:
            if self.hidden_irreps.vectors != config.num_heads:
                raise ValueError(
                    "gated local transport requires hidden vector channels "
                    "to equal num_heads"
                )
            if any(
                local_heads not in {0, config.num_heads}
                for local_heads in local_head_counts
            ):
                raise ValueError(
                    "gated local transport requires each local stage to use all heads"
                )

        self.scalar_in = nn.Linear(config.node_dim, self.hidden_irreps.scalars)
        self.global_scalar_in = nn.Linear(3, self.hidden_irreps.scalars, bias=False)
        self.vector_in = nn.Linear(
            self.hidden_irreps.scalars, self.hidden_irreps.vectors
        )
        self.external_vector_in = (
            _ChannelMix(config.input_vector_dim, self.hidden_irreps.vectors)
            if config.input_vector_dim
            else None
        )
        self.external_tensor_in = (
            _ChannelMix(config.input_tensor_dim, self.hidden_irreps.tensors)
            if config.input_tensor_dim
            else None
        )
        layer_scale = config.residual_scale_init / sqrt(config.num_layers)
        self.layers = nn.ModuleList(
            [
                _EquivariantMomentLayer(
                    scalars=self.hidden_irreps.scalars,
                    vectors=self.hidden_irreps.vectors,
                    tensors=self.hidden_irreps.tensors,
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
                    global_transport_mode=config.global_transport_mode,
                    use_multiscale_spatial_kernel=(
                        config.use_multiscale_spatial_kernel
                    ),
                    local_cutoff=config.local_cutoff,
                    num_rbf=config.num_rbf,
                    local_rbf_spacing=config.local_rbf_spacing,
                    learn_local_radial_gate=config.learn_local_radial_gate,
                    use_edge_conditioned_local_transport=(
                        config.use_edge_conditioned_local_transport
                    ),
                    normalize_edge_conditioned_local_by_sqrt_degree=(
                        config.normalize_edge_conditioned_local_by_sqrt_degree
                    ),
                    use_gated_local_transport=config.use_gated_local_transport,
                    use_grouped_invariant_normalization=(
                        config.use_grouped_invariant_normalization
                    ),
                    use_irrep_rms_normalization=(
                        config.use_irrep_rms_normalization
                    ),
                    angular_feature_rank=config.angular_feature_rank,
                    use_quartic_kernel=config.use_quartic_kernel,
                    quartic_kernel_init=config.quartic_kernel_init,
                    quartic_kernel_max=config.quartic_kernel_max,
                    checkpoint_gated_local_mlp=config.checkpoint_gated_local_mlp,
                    scalar_content_mode=config.scalar_content_mode,
                    use_tensor_product_kernel=config.use_tensor_product_kernel,
                    tensor_kernel_init=config.tensor_kernel_init,
                    tensor_kernel_max=config.tensor_kernel_max,
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
        self.coordinate_updaters = nn.ModuleList(
            [
                _CoordinateUpdater(
                    scalars=self.hidden_irreps.scalars,
                    vectors=self.hidden_irreps.vectors,
                    eps=config.eps,
                )
                for _ in range(config.num_layers - 1)
            ]
            if config.coordinate_updates
            else []
        )
        self.local_pairwise_content = (
            _LocalPairwiseContent(
                head_dim=self.hidden_irreps.scalars // config.num_heads,
                num_rbf=config.num_rbf,
                residual_scale_init=config.pairwise_residual_scale_init,
                eps=config.eps,
            )
            if config.use_pairwise_local_content
            else None
        )
        self.scalar_out_norm = nn.LayerNorm(self.hidden_irreps.scalars)
        self.scalar_out = nn.Linear(
            self.hidden_irreps.scalars, self.output_irreps.scalars
        )
        self.vector_out = _ChannelMix(
            self.hidden_irreps.vectors, self.output_irreps.vectors
        )
        tensor_out_channels = self.hidden_irreps.tensors or config.num_heads
        self.tensor_out = _ChannelMix(
            tensor_out_channels,
            self.output_irreps.tensors,
        )
        self.interaction_readout = (
            _InteractionReadout(
                scalars=self.hidden_irreps.scalars,
                output_scalars=self.output_irreps.scalars,
                num_rbf=config.num_rbf,
                cutoff=config.local_cutoff,
                rbf_spacing=config.local_rbf_spacing,
                eps=config.eps,
            )
            if config.readout_mode == "interaction"
            else None
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None = None,
        *,
        edge_index: torch.Tensor | None = None,
        edge_index_is_validated: bool = False,
        readout_mask: torch.Tensor | None = None,
        node_vectors: torch.Tensor | None = None,
        node_tensors: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(edge_index_is_validated, bool):
            raise TypeError("edge_index_is_validated must be boolean")
        if edge_index is None and edge_index_is_validated:
            raise ValueError("validated edge mode requires edge_index")
        node_feats, pos, batch, num_graphs, graph_counts = self._check_inputs(
            node_feats,
            pos,
            batch,
        )
        node_vectors, node_tensors = self._check_equivariant_inputs(
            node_feats,
            node_vectors,
            node_tensors,
        )
        pool_mask, pool_batch, pool_counts = _readout_metadata(
            readout_mask,
            batch,
            num_graphs=num_graphs,
            graph_counts=graph_counts,
        )
        if self.interaction_readout is not None:
            _validate_interaction_roles(
                readout_mask,
                batch,
                num_graphs=num_graphs,
            )
        scalars = self.scalar_in(node_feats)
        vectors = scalars.new_zeros((scalars.shape[0], self.hidden_irreps.vectors, 3))
        if self.external_vector_in is not None:
            if node_vectors is None:
                raise RuntimeError("validated external vector input is missing")
            vectors = vectors + self.external_vector_in(
                node_vectors.to(dtype=vectors.dtype)
            )
        persistent_tensor = (
            pos.new_zeros((pos.shape[0], self.hidden_irreps.tensors, 5))
            if self.hidden_irreps.tensors
            else None
        )
        if self.external_tensor_in is not None:
            if node_tensors is None or persistent_tensor is None:
                raise RuntimeError("validated external tensor input is missing")
            tensor_features = _st_matrix_to_features(
                node_tensors.to(dtype=persistent_tensor.dtype)
            )
            persistent_tensor = persistent_tensor + self.external_tensor_in(
                tensor_features
            )
        transient_tensor = pos.new_zeros((pos.shape[0], self.config.num_heads, 5))
        local_geometry = None
        has_local_heads = any(layer.local_head_count for layer in self.layers)
        if edge_index is not None and not (
            has_local_heads or self.interaction_readout is not None
        ):
            raise ValueError("edge_index requires local heads or interaction readout")
        resolved_edge_index = edge_index
        resolved_edge_index_is_validated = edge_index_is_validated
        if self.config.coordinate_updates and edge_index is not None:
            if self.config.coordinate_neighbor_policy == "error":
                raise ValueError(
                    "coordinate_updates with external sparse candidates requires "
                    "coordinate_neighbor_policy='fixed' or 'rebuild'"
                )
            if self.config.coordinate_neighbor_policy == "rebuild":
                resolved_edge_index = None
                resolved_edge_index_is_validated = False
        if has_local_heads and not self.config.coordinate_updates:
            local_geometry = _local_geometry(
                pos,
                batch,
                num_graphs=num_graphs,
                cutoff=self.config.local_cutoff,
                num_rbf=self.config.num_rbf,
                rbf_spacing=self.config.local_rbf_spacing,
                graph_counts=graph_counts,
                edge_index=resolved_edge_index,
                edge_index_is_validated=resolved_edge_index_is_validated,
            )
        normalized_pos: torch.Tensor | None = None
        global_scalar_input: torch.Tensor | None = None
        global_geometry_injected = False
        for layer_index, layer in enumerate(self.layers):
            if self.config.coordinate_updates:
                local_geometry = (
                    _local_geometry(
                        pos,
                        batch,
                        num_graphs=num_graphs,
                        cutoff=self.config.local_cutoff,
                        num_rbf=self.config.num_rbf,
                        rbf_spacing=self.config.local_rbf_spacing,
                        graph_counts=graph_counts,
                        edge_index=resolved_edge_index,
                        edge_index_is_validated=(resolved_edge_index_is_validated),
                    )
                    if layer.local_head_count
                    else None
                )
                normalized_pos = None
                global_scalar_input = None
            if layer.has_active_global_transport and (
                self.config.coordinate_updates or normalized_pos is None
            ):
                (
                    normalized_pos,
                    log_radius,
                    log_graph_scale,
                    log_normalized_square,
                ) = _scale_first_geometry(
                    pos,
                    batch,
                    num_graphs=num_graphs,
                    graph_counts=graph_counts,
                )
                global_scalar_input = torch.cat(
                    [
                        log_radius.to(dtype=node_feats.dtype),
                        log_graph_scale[batch].to(dtype=node_feats.dtype),
                        log_normalized_square.to(dtype=node_feats.dtype),
                    ],
                    dim=-1,
                )
            if layer.has_active_global_transport and not global_geometry_injected:
                if normalized_pos is None or global_scalar_input is None:
                    raise RuntimeError(
                        "active global transport requires global geometry"
                    )
                scalars = scalars + self.global_scalar_in(global_scalar_input)
                vector_gate = torch.tanh(self.vector_in(scalars)).unsqueeze(-1)
                geometry_vector = vector_gate.to(
                    dtype=normalized_pos.dtype
                ) * normalized_pos.unsqueeze(1)
                vectors = vectors + geometry_vector.to(dtype=vectors.dtype)
                global_geometry_injected = True
            layer_args = (
                scalars,
                vectors,
                normalized_pos,
                pos,
                batch,
                num_graphs,
                graph_counts,
                local_geometry,
                self.local_pairwise_content,
            )
            if self.hidden_irreps.tensors:
                if persistent_tensor is None:
                    raise RuntimeError("persistent tensor state is missing")
                (
                    scalars,
                    vectors,
                    persistent_tensor,
                    transient_tensor,
                ) = layer(
                    *layer_args,
                    persistent_tensor=persistent_tensor,
                )
            else:
                scalars, vectors, transient_tensor = layer(*layer_args)
            if layer_index < len(self.coordinate_updaters):
                pos = pos + self.coordinate_updaters[layer_index](
                    scalars,
                    vectors,
                    pos,
                    batch,
                    num_graphs,
                    graph_counts,
                )

        node_scalars = self.scalar_out(self.scalar_out_norm(scalars))
        node_vectors = self.vector_out(vectors)
        tensor_state = (
            persistent_tensor if self.hidden_irreps.tensors else transient_tensor
        )
        if tensor_state is None:
            raise RuntimeError("tensor output state is missing")
        node_tensors = _st_features_to_matrix(self.tensor_out(tensor_state))
        pooled_scalars = node_scalars if pool_mask is None else node_scalars[pool_mask]
        pooled_vectors = node_vectors if pool_mask is None else node_vectors[pool_mask]
        pooled_tensors = node_tensors if pool_mask is None else node_tensors[pool_mask]
        pool = _scatter_sum if self.config.readout_mode == "sum" else _scatter_mean
        graph_scalars = pool(
            pooled_scalars,
            pool_batch,
            num_graphs,
            pool_counts,
        )
        if self.interaction_readout is not None:
            if readout_mask is None:
                raise RuntimeError("interaction readout role validation was skipped")
            graph_scalars = graph_scalars + self.interaction_readout(
                scalars,
                pos,
                batch,
                readout_mask,
                num_graphs=num_graphs,
                graph_counts=graph_counts,
                edge_index=resolved_edge_index,
                edge_index_is_validated=resolved_edge_index_is_validated,
            ).to(dtype=graph_scalars.dtype)
        output = {
            "node_scalars": node_scalars,
            "node_vectors": node_vectors,
            "node_tensors": node_tensors,
            "graph_scalars": graph_scalars,
            "graph_vectors": pool(pooled_vectors, pool_batch, num_graphs, pool_counts),
            "graph_tensors": pool(pooled_tensors, pool_batch, num_graphs, pool_counts),
        }
        if self.config.coordinate_updates:
            output["node_positions"] = pos
        return output

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

    def _check_equivariant_inputs(
        self,
        node_feats: torch.Tensor,
        node_vectors: torch.Tensor | None,
        node_tensors: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        num_nodes = node_feats.shape[0]
        vector_dim = self.config.input_vector_dim
        tensor_dim = self.config.input_tensor_dim
        if vector_dim:
            if node_vectors is None:
                raise ValueError(
                    "node_vectors is required when input_vector_dim is positive"
                )
            expected = (num_nodes, vector_dim, 3)
            if node_vectors.shape != expected:
                raise ValueError(
                    f"node_vectors must have shape {expected}, "
                    f"got {tuple(node_vectors.shape)}"
                )
            _validate_equivariant_input_tensor(
                "node_vectors",
                node_vectors,
                reference=node_feats,
            )
        elif node_vectors is not None:
            raise ValueError("node_vectors requires positive input_vector_dim")

        if tensor_dim:
            if node_tensors is None:
                raise ValueError(
                    "node_tensors is required when input_tensor_dim is positive"
                )
            expected = (num_nodes, tensor_dim, 3, 3)
            if node_tensors.shape != expected:
                raise ValueError(
                    f"node_tensors must have shape {expected}, "
                    f"got {tuple(node_tensors.shape)}"
                )
            _validate_equivariant_input_tensor(
                "node_tensors",
                node_tensors,
                reference=node_feats,
            )
            tolerance = 1e-9 if node_tensors.dtype == torch.float64 else 1e-5
            if not torch.allclose(
                node_tensors,
                node_tensors.transpose(-1, -2),
                atol=tolerance,
                rtol=tolerance,
            ):
                raise ValueError("node_tensors must be symmetric")
            trace = node_tensors.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            if not torch.allclose(
                trace,
                torch.zeros_like(trace),
                atol=tolerance,
                rtol=tolerance,
            ):
                raise ValueError("node_tensors must be traceless")
        elif node_tensors is not None:
            raise ValueError("node_tensors requires positive input_tensor_dim")
        return node_vectors, node_tensors


class _QuarticFeatureMap(nn.Module):
    """Symmetric degree-four map with dot(phi(x), phi(y)) = (x . y)^4."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise TypeError("quartic feature dimension must be an integer")
        if dimension <= 0:
            raise ValueError("quartic feature dimension must be positive")
        terms = list(combinations_with_replacement(range(dimension), 4))
        indices = torch.tensor(terms, dtype=torch.long)
        coefficients = []
        for term in terms:
            counts = [term.count(index) for index in range(dimension)]
            multinomial = factorial(4) / prod(factorial(count) for count in counts)
            coefficients.append(sqrt(multinomial))
        self.dimension = dimension
        self.output_dim = len(terms)
        self.register_buffer("indices", indices, persistent=False)
        self.register_buffer(
            "coefficients",
            torch.tensor(coefficients, dtype=torch.float64),
            persistent=False,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.dimension:
            raise ValueError(
                f"quartic input requires final dimension {self.dimension}"
            )
        indices = self.indices
        features = (
            value.index_select(-1, indices[:, 0])
            * value.index_select(-1, indices[:, 1])
            * value.index_select(-1, indices[:, 2])
            * value.index_select(-1, indices[:, 3])
        )
        return features * self.coefficients.to(dtype=value.dtype)


class _SeparableIrrepRMSNorm(nn.Module):
    """Invariant RMS pre-normalization shared by all non-scalar irreps."""

    def __init__(
        self,
        vectors: int,
        tensors: int,
        *,
        eps: float,
    ) -> None:
        super().__init__()
        self.vectors = vectors
        self.tensors = tensors
        self.eps = eps
        self.vector_weight = nn.Parameter(torch.ones(vectors))
        self.tensor_weight = (
            nn.Parameter(torch.ones(tensors)) if tensors else None
        )

    def forward(
        self,
        vectors: torch.Tensor,
        tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if vectors.ndim != 3 or vectors.shape[1:] != (self.vectors, 3):
            raise ValueError("vector state has an invalid shape")
        if tensors.shape != (vectors.shape[0], self.tensors, 5):
            raise ValueError("tensor state has an invalid shape")
        dtype = _moment_dtype(vectors, tensors)
        vector_energy = vectors.to(dtype=dtype).square().sum(dim=(-2, -1))
        tensor_energy = (
            _st_frobenius_square(tensors.to(dtype=dtype)).sum(dim=-1)
            if self.tensors
            else torch.zeros_like(vector_energy)
        )
        degrees_of_freedom = 3 * self.vectors + 5 * self.tensors
        inverse_rms = (
            (vector_energy + tensor_energy) / degrees_of_freedom + self.eps
        ).rsqrt()
        normalized_vectors = (
            vectors.to(dtype=dtype)
            * inverse_rms[:, None, None]
            * self.vector_weight.to(dtype=dtype)[None, :, None]
        ).to(dtype=vectors.dtype)
        if self.tensors:
            if self.tensor_weight is None:
                raise RuntimeError("tensor normalization weight is missing")
            normalized_tensors = (
                tensors.to(dtype=dtype)
                * inverse_rms[:, None, None]
                * self.tensor_weight.to(dtype=dtype)[None, :, None]
            ).to(dtype=tensors.dtype)
        else:
            normalized_tensors = tensors
        return normalized_vectors, normalized_tensors


class _EquivariantMomentLayer(nn.Module):
    def __init__(
        self,
        scalars: int,
        vectors: int,
        tensors: int,
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
        global_transport_mode: str,
        use_multiscale_spatial_kernel: bool,
        local_cutoff: float,
        num_rbf: int,
        local_rbf_spacing: str,
        learn_local_radial_gate: bool,
        use_edge_conditioned_local_transport: bool,
        normalize_edge_conditioned_local_by_sqrt_degree: bool,
        use_gated_local_transport: bool,
        use_grouped_invariant_normalization: bool,
        use_irrep_rms_normalization: bool,
        angular_feature_rank: int,
        use_quartic_kernel: bool,
        quartic_kernel_init: float,
        quartic_kernel_max: float,
        checkpoint_gated_local_mlp: bool,
        scalar_content_mode: str,
        use_tensor_product_kernel: bool,
        tensor_kernel_init: float,
        tensor_kernel_max: float,
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
        self.tensors = tensors
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
        self.global_transport_mode = global_transport_mode
        self.has_active_global_transport = (
            self.global_head_count > 0 and global_transport_mode != "none"
        )
        spatial_scales = (
            torch.logspace(-3.0, 0.0, num_heads, base=2.0)
            if use_multiscale_spatial_kernel
            else None
        )
        self.register_buffer(
            "_spatial_scales",
            spatial_scales,
            persistent=False,
        )
        self.local_cutoff = local_cutoff
        self.num_rbf = num_rbf
        self.local_rbf_spacing = local_rbf_spacing
        self.global_memory_count = global_memory_count
        self.use_memory_interaction = use_memory_interaction
        self.memory_assignment_temperature = memory_assignment_temperature
        self.memory_assignment_scale = memory_assignment_scale
        self.memory_interaction_cutoff = memory_interaction_cutoff
        self.use_radial_trace = use_radial_trace
        self.scalar_content_mode = scalar_content_mode
        self.tensor_kernel_max = tensor_kernel_max
        self.angular_feature_rank = angular_feature_rank
        self.quartic_kernel_max = quartic_kernel_max
        self.irrep_rms_norm = (
            _SeparableIrrepRMSNorm(vectors, tensors, eps=eps)
            if use_irrep_rms_normalization
            else None
        )
        self.edge_conditioned_local = (
            _EdgeConditionedLocalTransport(
                scalars=scalars,
                vectors=vectors,
                num_heads=local_head_count,
                num_rbf=num_rbf,
                eps=eps,
                normalize_by_sqrt_degree=(
                    normalize_edge_conditioned_local_by_sqrt_degree
                ),
            )
            if use_edge_conditioned_local_transport and local_head_count
            else None
        )
        self.gated_local = None
        if use_gated_local_transport and local_head_count:
            # Candidate-only modules must not shift any incumbent initialization.
            with torch.random.fork_rng(devices=[]):
                self.gated_local = _GatedEquivariantLocalTransport(
                    scalars=scalars,
                    vectors=vectors,
                    num_heads=local_head_count,
                    num_rbf=num_rbf,
                    eps=eps,
                    checkpoint_mlp=checkpoint_gated_local_mlp,
                )

        self.norm = nn.LayerNorm(scalars)
        self.query_scalar = nn.Linear(scalars, scalars)
        self.key_scalar = nn.Linear(scalars, scalars)
        self.value_scalar = nn.Linear(scalars, scalars)
        self.query_vector = _ChannelMix(vectors, num_heads)
        self.key_vector = _ChannelMix(vectors, num_heads)
        self.value_vector = _ChannelMix(vectors, num_heads)
        self.query_vector_gate = nn.Linear(scalars, num_heads)
        self.key_vector_gate = nn.Linear(scalars, num_heads)
        self.query_vector_extra = None
        self.key_vector_extra = None
        self.query_vector_extra_gate = None
        self.key_vector_extra_gate = None
        if angular_feature_rank > 1:
            extra_channels = num_heads * (angular_feature_rank - 1)
            with torch.random.fork_rng(devices=[]):
                self.query_vector_extra = _ChannelMix(vectors, extra_channels)
                self.key_vector_extra = _ChannelMix(vectors, extra_channels)
                self.query_vector_extra_gate = nn.Linear(scalars, extra_channels)
                self.key_vector_extra_gate = nn.Linear(scalars, extra_channels)
        self.quartic_feature_map = (
            _QuarticFeatureMap(3)
            if use_quartic_kernel
            else None
        )
        self.raw_quartic_kernel = (
            nn.Parameter(
                torch.full(
                    (num_heads,),
                    _inverse_sigmoid(
                        quartic_kernel_init / quartic_kernel_max
                    ),
                )
            )
            if use_quartic_kernel
            else None
        )
        self.tensor_kernel_query = None
        self.tensor_kernel_key = None
        self.raw_tensor_kernel = None
        if use_tensor_product_kernel:
            # Candidate-only projections must not shift any incumbent
            # initialization under a matched model seed.
            with torch.random.fork_rng(devices=[]):
                self.tensor_kernel_query = _ChannelMix(tensors, num_heads)
                self.tensor_kernel_key = _ChannelMix(tensors, num_heads)
                self.raw_tensor_kernel = nn.Parameter(
                    torch.full(
                        (num_heads,),
                        _inverse_sigmoid(tensor_kernel_init / tensor_kernel_max),
                    )
                )
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
        self.persistent_tensor_to_head = (
            _ChannelMix(tensors, num_heads) if tensors else None
        )
        self.persistent_tensor_from_head = (
            _ChannelMix(num_heads, tensors) if tensors else None
        )
        self.persistent_tensor_gate = nn.Linear(scalars, tensors) if tensors else None
        self.persistent_tensor_residual_scale = (
            nn.Parameter(torch.tensor(float(residual_scale_init))) if tensors else None
        )
        invariant_dim = scalars + 6 * num_heads + tensors
        self.use_grouped_invariant_normalization = use_grouped_invariant_normalization
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
        self.ffn_in = nn.Linear(scalars + vectors + tensors, 2 * ffn_hidden)
        self.ffn_out = nn.Linear(ffn_hidden, scalars)
        self.ffn_vector_gate = nn.Linear(scalars, vectors)
        self.ffn_vector_mix = _ChannelMix(vectors, vectors)
        self.persistent_tensor_ffn_gate = (
            nn.Linear(scalars, tensors) if tensors else None
        )
        self.persistent_tensor_ffn_mix = (
            _ChannelMix(tensors, tensors) if tensors else None
        )
        self.ffn_scalar_residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )
        self.ffn_vector_residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )
        self.persistent_tensor_ffn_residual_scale = (
            nn.Parameter(torch.tensor(float(residual_scale_init))) if tensors else None
        )
        # Allocated for every route and M so comparisons retain one state schema.
        self.memory_router_in = nn.Linear(self.head_dim, _MEMORY_ROUTER_DIM)
        self.memory_router_out = nn.Linear(_MEMORY_ROUTER_DIM, _MEMORY_ROUTER_DIM)

    def forward(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
        global_pos: torch.Tensor | None,
        raw_pos: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        graph_counts: torch.Tensor,
        local_geometry: _LocalGeometryInput | None,
        local_pairwise_content: _LocalPairwiseContent | None = None,
        *,
        persistent_tensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if self.tensors:
            if persistent_tensor is None:
                raise ValueError(
                    "persistent_tensor is required when hidden 2e channels are enabled"
                )
            if persistent_tensor.shape != (scalars.shape[0], self.tensors, 5):
                raise ValueError(
                    "persistent_tensor must have shape "
                    f"({scalars.shape[0]}, {self.tensors}, 5)"
                )
        elif persistent_tensor is not None:
            raise ValueError("persistent_tensor requires positive hidden 2e channels")
        else:
            persistent_tensor = raw_pos.new_zeros((scalars.shape[0], 0, 5))
        if self.local_head_count == 0 and self.global_transport_mode == "none":
            tensor = raw_pos.new_zeros((scalars.shape[0], self.num_heads, 5))
            scalars, vectors, persistent_tensor = self._apply_ffn(
                scalars,
                vectors,
                persistent_tensor,
            )
            if self.tensors:
                return scalars, vectors, persistent_tensor, tensor
            return scalars, vectors, tensor

        s_norm = self.norm(scalars)
        normalized_vectors, normalized_persistent_tensor = (
            self._normalize_non_scalars(vectors, persistent_tensor)
        )
        bounded_vectors = _bounded_irrep(normalized_vectors, self.eps)
        bounded_persistent_tensor = _bounded_st_tensor(
            normalized_persistent_tensor
        )
        n_nodes = scalars.shape[0]

        q1 = _unit_ball(
            self.query_vector(bounded_vectors)
            * torch.tanh(self.query_vector_gate(s_norm)).unsqueeze(-1),
            self.eps,
        )
        q1_kernel = self._angular_kernel_vector(
            q1,
            bounded_vectors,
            s_norm,
            query=True,
        )
        all_local_gated = (
            self.gated_local is not None
            and self.global_head_count == 0
            and local_pairwise_content is None
        )
        message_groups: list[tuple[torch.Tensor, ...]] = []
        if all_local_gated:
            if local_geometry is None or self.gated_local is None:
                raise RuntimeError("gated local transport requires local geometry")
            message_groups.append(
                self.gated_local(
                    s_norm,
                    bounded_vectors,
                    local_geometry,
                    num_nodes=n_nodes,
                )
            )
            moment_dtype = _moment_dtype(q1_kernel, raw_pos)
        else:
            raw_query_scalar = self.query_scalar(s_norm).reshape(
                n_nodes, self.num_heads, self.head_dim
            )
            raw_key_scalar = self.key_scalar(s_norm).reshape(
                n_nodes, self.num_heads, self.head_dim
            )
            query_content = _positive_scalar_features(
                raw_query_scalar,
                self.eps,
                mode=self.scalar_content_mode,
            )
            key_content = _positive_scalar_features(
                raw_key_scalar,
                self.eps,
                mode=self.scalar_content_mode,
            )
            q0 = query_content
            k0 = key_content
            if self.tensor_kernel_query is not None:
                if self.tensor_kernel_key is None or self.raw_tensor_kernel is None:
                    raise RuntimeError("tensor-product kernel modules are incomplete")
                tensor_kernel_scale = _bounded_kernel_scale(
                    self.raw_tensor_kernel,
                    self.tensor_kernel_max,
                )
                tensor_query_features, tensor_key_features = _tensor_product_features(
                    self.tensor_kernel_query(persistent_tensor),
                    self.tensor_kernel_key(persistent_tensor),
                    tensor_kernel_scale,
                    eps=self.eps,
                )
                q0 = torch.cat([q0, tensor_query_features], dim=-1)
                k0 = torch.cat([k0, tensor_key_features], dim=-1)
            k1 = _unit_ball(
                self.key_vector(bounded_vectors)
                * torch.tanh(self.key_vector_gate(s_norm)).unsqueeze(-1),
                self.eps,
            )
            k1_kernel = self._angular_kernel_vector(
                k1,
                bounded_vectors,
                s_norm,
                query=False,
            )
            if self.quartic_feature_map is not None:
                if self.raw_quartic_kernel is None:
                    raise RuntimeError("quartic kernel scale is missing")
                quartic_scale = _bounded_kernel_scale(
                    self.raw_quartic_kernel,
                    self.quartic_kernel_max,
                )
                quartic_root = quartic_scale.sqrt()[None, :, None]
                q0 = torch.cat(
                    [
                        q0,
                        self.quartic_feature_map(q1) * quartic_root,
                    ],
                    dim=-1,
                )
                k0 = torch.cat(
                    [
                        k0,
                        self.quartic_feature_map(k1) * quartic_root,
                    ],
                    dim=-1,
                )
            moment_dtype = _moment_dtype(
                q0,
                k0,
                q1_kernel,
                k1_kernel,
                raw_pos,
            )
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

            if self.local_head_count:
                local = slice(0, self.local_head_count)
                if self.gated_local is not None:
                    if local_geometry is None:
                        raise RuntimeError(
                            "gated local transport requires local geometry"
                        )
                    local_messages = self.gated_local(
                        s_norm,
                        bounded_vectors,
                        local_geometry,
                        num_nodes=n_nodes,
                    )
                elif self.edge_conditioned_local is not None:
                    if local_geometry is None:
                        raise RuntimeError(
                            "edge-conditioned local transport requires local geometry"
                        )
                    local_messages = self.edge_conditioned_local(
                        s_norm,
                        bounded_vectors,
                        local_geometry,
                        num_nodes=n_nodes,
                    )
                else:
                    receiver, sender, weights, displacement, squared_distance = (
                        _local_attention_weights(
                            q0[:, local],
                            k0[:, local],
                            q1_kernel[:, local],
                            k1_kernel[:, local],
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
                            rbf_spacing=self.local_rbf_spacing,
                            radial_weight=self.local_radial_weight[local],
                            radial_bias=self.local_radial_bias[local],
                            local_geometry=local_geometry,
                        )
                    )
                    local_messages = _local_moment_messages(
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
                if local_pairwise_content is not None:
                    if local_geometry is None:
                        raise RuntimeError(
                            "pairwise local content requires local geometry"
                        )
                    pairwise_scalar = local_pairwise_content(
                        raw_query_scalar[:, local],
                        raw_key_scalar[:, local],
                        local_geometry,
                        num_nodes=n_nodes,
                    )
                    local_messages = (
                        local_messages[0] + pairwise_scalar,
                        *local_messages[1:],
                    )
                message_groups.append(local_messages)
            if self.global_head_count:
                global_heads = slice(self.local_head_count, self.num_heads)
                if self.global_transport_mode == "none":
                    message_groups.append(
                        _zero_moment_messages(
                            n_nodes,
                            self.global_head_count,
                            self.head_dim,
                            dtype=moment_dtype,
                            device=scalars.device,
                        )
                    )
                else:
                    if global_pos is None:
                        raise RuntimeError("active global transport requires global_pos")
                    spatial_features = (
                        _quadratic_gaussian_spatial_features(
                            global_pos,
                            self._spatial_scales,
                        )
                        if self._spatial_scales is not None
                        else None
                    )
                    memory_router_latent = None
                    if self.use_memory_interaction and self.global_memory_count > 1:
                        memory_router_latent = torch.tanh(
                            self.memory_router_out(
                                F.silu(
                                    self.memory_router_in(
                                        key_content[:, global_heads]
                                    )
                                )
                            )
                        )
                        router_norm = _stable_vector_norm(memory_router_latent)
                        memory_router_latent = (
                            memory_router_latent
                            / router_norm.clamp_min(
                                torch.finfo(memory_router_latent.dtype).tiny
                            )
                        )
                    message_groups.append(
                        _global_moment_messages(
                            q0[:, global_heads],
                            k0[:, global_heads],
                            q1_kernel[:, global_heads],
                            k1_kernel[:, global_heads],
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
                            global_transport_mode=self.global_transport_mode,
                            spatial_features=spatial_features,
                        )
                    )

        if len(message_groups) == 1:
            scalar_message, vector_base, relative, tensor, radial_trace = (
                message_groups[0]
            )
        else:
            scalar_message, vector_base, relative, tensor, radial_trace = (
                torch.cat(values, dim=1)
                for values in zip(*message_groups, strict=True)
            )
        scalar_message = scalar_message.reshape(n_nodes, self.scalars)
        moment_q1 = q1.to(dtype=moment_dtype)
        tensor_context = tensor
        if self.persistent_tensor_to_head is not None:
            tensor_context = tensor_context + self.persistent_tensor_to_head(
                bounded_persistent_tensor
            ).to(dtype=moment_dtype)
        tensor_vector = _st_matrix_vector(tensor_context, moment_q1)
        query_base_dot = (moment_q1 * vector_base).sum(dim=-1)
        query_relative_dot = (moment_q1 * relative).sum(dim=-1)
        relative_square = relative.square().sum(dim=-1)
        tensor_square = _st_frobenius_square(tensor_context)
        query_tensor_dot = (moment_q1 * tensor_vector).sum(dim=-1)

        vector_per_head = (
            vector_base
            + self.relative_mix.to(dtype=moment_dtype)[None, :, None] * relative
            + self.tensor_mix.to(dtype=moment_dtype)[None, :, None] * tensor_vector
        )

        angular_invariants = [
            query_base_dot,
            query_relative_dot,
            relative_square,
            tensor_square,
            query_tensor_dot,
            radial_trace,
        ]
        persistent_invariants = (
            _st_frobenius_square(bounded_persistent_tensor) if self.tensors else None
        )
        if self.use_grouped_invariant_normalization:
            scalar_message = _stable_group_norm(scalar_message)
            angular = _stable_group_norm(torch.cat(angular_invariants, dim=-1))
            invariant_parts = [scalar_message, angular]
            if persistent_invariants is not None:
                invariant_parts.append(_stable_group_norm(persistent_invariants))
        else:
            invariant_parts = [scalar_message, *angular_invariants]
            if persistent_invariants is not None:
                invariant_parts.append(persistent_invariants)
        scalar_invariants = torch.cat(invariant_parts, dim=-1)
        normalized_invariants = _stable_layer_norm(
            self.scalar_update_norm, scalar_invariants
        )
        scalar_delta = self.scalar_update(normalized_invariants.to(dtype=scalars.dtype))
        vector_delta = self.vector_update(vector_per_head)
        scalars = scalars + self.scalar_residual_scale * scalar_delta
        bounded_delta = _bounded_irrep(vector_delta, self.eps).to(dtype=vectors.dtype)
        vectors = vectors + self.vector_residual_scale * bounded_delta
        if self.persistent_tensor_from_head is not None:
            if (
                self.persistent_tensor_gate is None
                or self.persistent_tensor_residual_scale is None
            ):
                raise RuntimeError("persistent tensor update is incomplete")
            tensor_delta = self.persistent_tensor_from_head(tensor)
            tensor_delta = tensor_delta * torch.tanh(
                self.persistent_tensor_gate(s_norm)
            ).to(dtype=tensor_delta.dtype).unsqueeze(-1)
            persistent_tensor = persistent_tensor + (
                self.persistent_tensor_residual_scale * _bounded_st_tensor(tensor_delta)
            )
        scalars, vectors, persistent_tensor = self._apply_ffn(
            scalars,
            vectors,
            persistent_tensor,
        )
        if self.tensors:
            return scalars, vectors, persistent_tensor, tensor
        return scalars, vectors, tensor

    def _normalize_non_scalars(
        self,
        vectors: torch.Tensor,
        persistent_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.irrep_rms_norm is None:
            return vectors, persistent_tensor
        return self.irrep_rms_norm(vectors, persistent_tensor)

    def _angular_kernel_vector(
        self,
        base: torch.Tensor,
        bounded_vectors: torch.Tensor,
        scalars: torch.Tensor,
        *,
        query: bool,
    ) -> torch.Tensor:
        if self.angular_feature_rank == 1:
            return base
        mix = self.query_vector_extra if query else self.key_vector_extra
        gate = (
            self.query_vector_extra_gate
            if query
            else self.key_vector_extra_gate
        )
        if mix is None or gate is None:
            raise RuntimeError("ranked angular feature modules are incomplete")
        num_nodes = base.shape[0]
        extra = mix(bounded_vectors).reshape(
            num_nodes,
            self.num_heads,
            self.angular_feature_rank - 1,
            3,
        )
        extra_gate = torch.tanh(gate(scalars)).reshape(
            num_nodes,
            self.num_heads,
            self.angular_feature_rank - 1,
            1,
        )
        direct_sum = torch.cat(
            [base.unsqueeze(-2), extra * extra_gate],
            dim=-2,
        )
        return _unit_ball(direct_sum.flatten(start_dim=-2), self.eps)

    def _apply_ffn(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
        persistent_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ffn_scalars = self.ffn_norm(scalars)
        normalized_vectors, normalized_tensor = self._normalize_non_scalars(
            vectors,
            persistent_tensor,
        )
        ffn_vectors = _bounded_irrep(normalized_vectors, self.eps)
        if self.persistent_tensor_ffn_mix is not None:
            if (
                self.persistent_tensor_ffn_gate is None
                or self.persistent_tensor_ffn_residual_scale is None
            ):
                raise RuntimeError("persistent tensor FFN is incomplete")
            tensor_ffn = self.persistent_tensor_ffn_mix(
                _bounded_st_tensor(normalized_tensor)
            )
            tensor_ffn = tensor_ffn * torch.tanh(
                self.persistent_tensor_ffn_gate(ffn_scalars)
            ).to(dtype=tensor_ffn.dtype).unsqueeze(-1)
            persistent_tensor = persistent_tensor + (
                self.persistent_tensor_ffn_residual_scale
                * _bounded_st_tensor(tensor_ffn)
            )
            _, normalized_tensor = self._normalize_non_scalars(
                vectors,
                persistent_tensor,
            )
        ffn_parts = [ffn_scalars, ffn_vectors.square().sum(dim=-1)]
        if self.tensors:
            ffn_parts.append(
                _st_frobenius_square(_bounded_st_tensor(normalized_tensor)).to(
                    dtype=scalars.dtype
                )
            )
        ffn_invariants = torch.cat(ffn_parts, dim=-1)
        ffn_content, ffn_gate = self.ffn_in(ffn_invariants).chunk(2, dim=-1)
        scalar_ffn = self.ffn_out(F.silu(ffn_content) * ffn_gate)
        vector_ffn = self.ffn_vector_mix(ffn_vectors) * torch.tanh(
            self.ffn_vector_gate(ffn_scalars)
        ).unsqueeze(-1)
        scalars = scalars + self.ffn_scalar_residual_scale * scalar_ffn
        vectors = vectors + self.ffn_vector_residual_scale * _bounded_irrep(
            vector_ffn, self.eps
        )
        return scalars, vectors, persistent_tensor


class _EdgeConditionedLocalTransport(nn.Module):
    """Invariant edge filters driving equivariant sparse local sums."""

    def __init__(
        self,
        *,
        scalars: int,
        vectors: int,
        num_heads: int,
        num_rbf: int,
        eps: float = 1e-12,
        normalize_by_sqrt_degree: bool = False,
    ) -> None:
        super().__init__()
        if scalars % num_heads:
            raise ValueError("scalars must be divisible by num_heads")
        if vectors != num_heads:
            raise ValueError(
                "edge-conditioned local transport requires vectors == num_heads"
            )
        self.scalars = scalars
        self.num_heads = num_heads
        self.head_dim = scalars // num_heads
        self.eps = eps
        self.normalize_by_sqrt_degree = normalize_by_sqrt_degree
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * scalars + num_rbf, 12),
            nn.SiLU(),
            nn.Linear(12, scalars + 3 * num_heads),
        )

    def forward(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
        local_geometry: _LocalGeometryInput,
        *,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        receiver, sender, displacement, squared_distance, rbf = (
            _nonself_local_geometry(local_geometry)
        )

        edge_features = torch.cat(
            [
                scalars[receiver],
                scalars[sender],
                rbf.to(dtype=scalars.dtype),
            ],
            dim=-1,
        )
        edge_output = self.edge_mlp(edge_features)
        scalar_edge, sender_gate, relative_gate, tensor_gate = torch.split(
            edge_output,
            [self.scalars, self.num_heads, self.num_heads, self.num_heads],
            dim=-1,
        )
        dtype = _moment_dtype(scalars, vectors, displacement)
        cutoff = _nonself_cutoff(
            local_geometry,
            squared_distance,
            dtype=dtype,
        )
        edge_weight = cutoff[:, None, None]
        scalar_edge_message = edge_weight * scalar_edge.to(dtype=dtype).reshape(
            -1,
            self.num_heads,
            self.head_dim,
        )
        vector_edge_message = edge_weight * (
            torch.tanh(sender_gate).to(dtype=dtype).unsqueeze(-1)
            * vectors[sender].to(dtype=dtype)
        )
        relative_edge_message = edge_weight * (
            torch.tanh(relative_gate).to(dtype=dtype).unsqueeze(-1)
            * displacement.to(dtype=dtype).unsqueeze(1)
        )
        tensor_edge_message = edge_weight * (
            torch.tanh(tensor_gate).to(dtype=dtype).unsqueeze(-1)
            * _nonself_tensor_features(
                local_geometry,
                displacement,
                dtype=dtype,
            ).unsqueeze(1)
        )
        if self.normalize_by_sqrt_degree:
            (
                scalar_message,
                vector_base,
                relative,
                tensor,
                cutoff_mass,
            ) = _fused_index_sum(
                receiver,
                num_nodes,
                scalar_edge_message,
                vector_edge_message,
                relative_edge_message,
                tensor_edge_message,
                cutoff.unsqueeze(-1),
            )
            inverse_sqrt_mass = (1.0 + cutoff_mass.squeeze(-1)).rsqrt()
            scalar_message = scalar_message * inverse_sqrt_mass[:, None, None]
            vector_base = vector_base * inverse_sqrt_mass[:, None, None]
            relative = relative * inverse_sqrt_mass[:, None, None]
            tensor = tensor * inverse_sqrt_mass[:, None, None]
        else:
            scalar_message, vector_base, relative, tensor = _fused_index_sum(
                receiver,
                num_nodes,
                scalar_edge_message,
                vector_edge_message,
                relative_edge_message,
                tensor_edge_message,
            )
        radial_trace = scalar_message.new_zeros((num_nodes, self.num_heads))
        return scalar_message, vector_base, relative, tensor, radial_trace


class _GatedEquivariantLocalTransport(nn.Module):
    """Same-feature nonlinear local transport with bounded-degree aggregation."""

    def __init__(
        self,
        *,
        scalars: int,
        vectors: int,
        num_heads: int,
        num_rbf: int,
        eps: float = 1e-12,
        checkpoint_mlp: bool = False,
    ) -> None:
        super().__init__()
        if scalars % num_heads:
            raise ValueError("scalars must be divisible by num_heads")
        if vectors != num_heads:
            raise ValueError("gated local transport requires vectors == num_heads")
        self.scalars = scalars
        self.num_heads = num_heads
        self.head_dim = scalars // num_heads
        self.num_rbf = num_rbf
        self.eps = eps
        self.checkpoint_mlp = checkpoint_mlp
        hidden_dim = max(32, 2 * self.head_dim)
        edge_input_dim = 2 * self.head_dim + num_rbf + 5
        edge_output_dim = self.head_dim + 5
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_output_dim),
        )
        self.scalar_message_norm = nn.LayerNorm(self.head_dim)
        self.mass_projection = nn.Linear(2, scalars, bias=False)

    def forward(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
        local_geometry: _LocalGeometryInput,
        *,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        receiver, sender, displacement, squared_distance, rbf = (
            _nonself_local_geometry(local_geometry)
        )

        scalar_heads = scalars.reshape(num_nodes, self.num_heads, self.head_dim)
        dtype = _moment_dtype(scalars, vectors, displacement)
        receiver_vector = vectors[receiver].to(dtype=dtype)
        sender_vector = vectors[sender].to(dtype=dtype)
        edge_direction = displacement.to(dtype=dtype).unsqueeze(1)
        vector_invariants = torch.stack(
            [
                (receiver_vector * sender_vector).sum(dim=-1),
                receiver_vector.square().sum(dim=-1),
                sender_vector.square().sum(dim=-1),
                (receiver_vector * edge_direction).sum(dim=-1),
                (sender_vector * edge_direction).sum(dim=-1),
            ],
            dim=-1,
        )
        edge_output = self._factorized_edge_mlp(
            scalar_heads,
            receiver,
            sender,
            rbf.to(dtype=scalars.dtype),
            vector_invariants.to(dtype=scalars.dtype),
        )
        (
            scalar_edge,
            scalar_gate,
            receiver_gate,
            sender_gate,
            relative_gate,
            tensor_gate,
        ) = torch.split(
            edge_output,
            [self.head_dim, 1, 1, 1, 1, 1],
            dim=-1,
        )

        cutoff = _nonself_cutoff(
            local_geometry,
            squared_distance,
            dtype=dtype,
        )
        edge_weight = cutoff[:, None, None]
        (
            scalar_message,
            vector_base,
            relative,
            tensor,
            cutoff_mass,
            effective_degree,
        ) = _fused_index_sum(
            receiver,
            num_nodes,
            edge_weight * (scalar_edge * torch.sigmoid(scalar_gate)).to(dtype=dtype),
            edge_weight
            * (
                torch.tanh(receiver_gate).to(dtype=dtype) * receiver_vector
                + torch.tanh(sender_gate).to(dtype=dtype) * sender_vector
            ),
            edge_weight * torch.tanh(relative_gate).to(dtype=dtype) * edge_direction,
            edge_weight
            * torch.tanh(tensor_gate).to(dtype=dtype)
            * _nonself_tensor_features(
                local_geometry,
                displacement,
                dtype=dtype,
            ).unsqueeze(1),
            cutoff.unsqueeze(-1),
            cutoff.square().unsqueeze(-1),
        )
        cutoff_mass = cutoff_mass.squeeze(-1)
        effective_degree = effective_degree.squeeze(-1)
        inverse_sqrt_mass = (1.0 + cutoff_mass).rsqrt()
        scalar_message = scalar_message * inverse_sqrt_mass[:, None, None]
        vector_base = vector_base * inverse_sqrt_mass[:, None, None]
        relative = relative * inverse_sqrt_mass[:, None, None]
        tensor = tensor * inverse_sqrt_mass[:, None, None]

        scalar_message = _stable_layer_norm(
            self.scalar_message_norm,
            scalar_message,
        )
        mass_features = torch.stack(
            [torch.log1p(cutoff_mass), torch.log1p(effective_degree)],
            dim=-1,
        )
        mass_message = self.mass_projection(
            mass_features.to(dtype=scalars.dtype)
        ).reshape(num_nodes, self.num_heads, self.head_dim)
        scalar_message = scalar_message + mass_message.to(dtype=dtype)
        radial_trace = scalar_message.new_zeros((num_nodes, self.num_heads))
        return scalar_message, vector_base, relative, tensor, radial_trace

    def _factorized_edge_mlp(
        self,
        scalar_heads: torch.Tensor,
        receiver: torch.Tensor,
        sender: torch.Tensor,
        rbf: torch.Tensor,
        vector_invariants: torch.Tensor,
    ) -> torch.Tensor:
        first_linear = self.edge_mlp[0]
        second_linear = self.edge_mlp[2]
        output_linear = self.edge_mlp[4]
        if not all(
            isinstance(module, nn.Linear)
            for module in (first_linear, second_linear, output_linear)
        ):
            raise RuntimeError("gated edge MLP layout is invalid")
        receiver_end = self.head_dim
        sender_end = 2 * self.head_dim
        radial_end = sender_end + self.num_rbf
        weight = first_linear.weight
        receiver_projection = F.linear(
            scalar_heads,
            weight[:, :receiver_end],
        )
        sender_projection = F.linear(
            scalar_heads,
            weight[:, receiver_end:sender_end],
        )
        hidden = (
            receiver_projection[receiver]
            + sender_projection[sender]
            + F.linear(rbf, weight[:, sender_end:radial_end]).unsqueeze(1)
            + F.linear(vector_invariants, weight[:, radial_end:])
        )
        if first_linear.bias is not None:
            hidden = hidden + first_linear.bias

        def finish_mlp(value: torch.Tensor) -> torch.Tensor:
            value = F.silu(value)
            value = F.silu(second_linear(value))
            return output_linear(value)

        if self.checkpoint_mlp and self.training and torch.is_grad_enabled():
            return activation_checkpoint(
                finish_mlp,
                hidden,
                use_reentrant=False,
            )
        return finish_mlp(hidden)


class _LocalPairwiseContent(nn.Module):
    """Invariant receiver/sender/RBF content with explicit neighborhood mass."""

    def __init__(
        self,
        head_dim: int,
        num_rbf: int,
        residual_scale_init: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * head_dim + num_rbf, head_dim),
            nn.SiLU(),
            nn.Linear(head_dim, head_dim),
        )
        self.mass_projection = nn.Linear(2, head_dim)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(
        self,
        query_scalar: torch.Tensor,
        key_scalar: torch.Tensor,
        local_geometry: _LocalGeometryInput,
        *,
        num_nodes: int,
    ) -> torch.Tensor:
        receiver, sender, _displacement, squared_distance, rbf = (
            _nonself_local_geometry(local_geometry)
        )

        head_count = query_scalar.shape[1]
        rbf_features = (
            rbf.to(dtype=query_scalar.dtype).unsqueeze(1).expand(-1, head_count, -1)
        )
        edge_features = torch.cat(
            [query_scalar[receiver], key_scalar[sender], rbf_features], dim=-1
        )
        edge_content = self.edge_mlp(edge_features)
        reduction_dtype = _moment_dtype(edge_content, squared_distance)
        cutoff_weight = _nonself_cutoff(
            local_geometry,
            squared_distance,
            dtype=reduction_dtype,
        )
        weighted_content = (
            edge_content.to(dtype=reduction_dtype) * cutoff_weight[:, None, None]
        )

        aggregated, cutoff_mass, effective_degree = _fused_index_sum(
            receiver,
            num_nodes,
            weighted_content,
            cutoff_weight.unsqueeze(-1),
            cutoff_weight.square().unsqueeze(-1),
        )
        cutoff_mass = cutoff_mass.squeeze(-1)
        effective_degree = effective_degree.squeeze(-1)
        normalized = aggregated / (1.0 + cutoff_mass).sqrt()[:, None, None]
        mass_features = torch.stack(
            [torch.log1p(cutoff_mass), torch.log1p(effective_degree)],
            dim=-1,
        )
        mass_content = self.mass_projection(
            mass_features.to(dtype=query_scalar.dtype)
        ).to(dtype=reduction_dtype)
        message = normalized + mass_content[:, None, :]
        return self.residual_scale.to(dtype=reduction_dtype) * message


class _InteractionReadout(nn.Module):
    """Invariant ligand-pocket pooling with an O(E) parity-aware interface path."""

    def __init__(
        self,
        *,
        scalars: int,
        output_scalars: int,
        num_rbf: int,
        cutoff: float,
        eps: float,
        rbf_spacing: str = "squared",
    ) -> None:
        super().__init__()
        width = max(8, min(32, scalars // 4))
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.rbf_spacing = rbf_spacing
        self.eps = eps
        self.node_norm = nn.LayerNorm(scalars)
        self.entity_projection = nn.Linear(scalars, width, bias=False)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * scalars + num_rbf, width),
            nn.SiLU(),
            nn.Linear(width, width + 6),
        )
        self.context_norm = nn.LayerNorm(3 * width + 3)
        self.context = nn.Sequential(
            nn.Linear(3 * width + 3, width),
            nn.SiLU(),
        )
        self.output = nn.Linear(width, output_scalars, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(
        self,
        scalars: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        ligand_mask: torch.Tensor,
        *,
        num_graphs: int,
        graph_counts: torch.Tensor,
        edge_index: torch.Tensor | None,
        edge_index_is_validated: bool,
    ) -> torch.Tensor:
        ligand_counts, pocket_counts = _interaction_role_counts(
            ligand_mask,
            batch,
            num_graphs=num_graphs,
        )
        node_state = _stable_layer_norm(self.node_norm, scalars).to(
            dtype=scalars.dtype
        )
        entity_state = self.entity_projection(node_state)
        ligand_pool = _scatter_mean(
            entity_state[ligand_mask],
            batch[ligand_mask],
            num_graphs,
            ligand_counts,
        )
        pocket_mask = ~ligand_mask
        pocket_pool = _scatter_mean(
            entity_state[pocket_mask],
            batch[pocket_mask],
            num_graphs,
            pocket_counts,
        )
        receiver, sender, displacement, squared_distance, rbf = _local_geometry(
            pos,
            batch,
            num_graphs=num_graphs,
            cutoff=self.cutoff,
            num_rbf=self.num_rbf,
            rbf_spacing=self.rbf_spacing,
            graph_counts=graph_counts,
            edge_index=edge_index,
            edge_index_is_validated=edge_index_is_validated,
        )
        cross = ligand_mask[receiver] & ~ligand_mask[sender] & (receiver != sender)
        receiver = receiver[cross]
        sender = sender[cross]
        displacement = displacement[cross]
        squared_distance = squared_distance[cross]
        rbf = rbf[cross]

        width = entity_state.shape[-1]
        if receiver.numel() == 0:
            cross_pool = entity_state.new_zeros((num_graphs, width))
            polar_moments = pos.new_zeros((num_graphs, 6, 3))
        else:
            edge_output = self.edge_mlp(
                torch.cat(
                    [
                        node_state[receiver],
                        node_state[sender],
                        rbf.to(dtype=node_state.dtype),
                    ],
                    dim=-1,
                )
            )
            cross_content, polar_gate = torch.split(
                edge_output,
                [width, 6],
                dim=-1,
            )
            reduction_dtype = _moment_dtype(
                edge_output,
                displacement,
                squared_distance,
            )
            cutoff = _cosine_of_squared_distance_cutoff(
                squared_distance.to(dtype=reduction_dtype)
            )
            edge_direction = displacement.to(dtype=reduction_dtype)
            cross_batch = batch[receiver]
            cross_pool, polar_moments, cutoff_mass = _fused_index_sum(
                cross_batch,
                num_graphs,
                cutoff[:, None] * cross_content.to(dtype=reduction_dtype),
                cutoff[:, None, None]
                * torch.tanh(polar_gate).to(dtype=reduction_dtype).unsqueeze(-1)
                * edge_direction.unsqueeze(1),
                cutoff.unsqueeze(-1),
            )
            inverse_mass = (1.0 + cutoff_mass.squeeze(-1)).rsqrt()
            cross_pool = cross_pool * inverse_mass[:, None]
            polar_moments = polar_moments * inverse_mass[:, None, None]

        parity_features = _parity_even_triple_features(polar_moments)
        context = torch.cat(
            [
                ligand_pool,
                pocket_pool,
                cross_pool.to(dtype=scalars.dtype),
                parity_features.to(dtype=scalars.dtype),
            ],
            dim=-1,
        )
        context_state = _stable_layer_norm(self.context_norm, context).to(
            dtype=scalars.dtype
        )
        hidden = self.context(context_state)
        return self.output(hidden)


class _CoordinateUpdater(nn.Module):
    def __init__(self, scalars: int, vectors: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.scalar_norm = nn.LayerNorm(scalars)
        self.scalar_gate = nn.Linear(scalars, 1)
        self.vector_mix = _ChannelMix(vectors, 1)
        nn.init.zeros_(self.scalar_gate.weight)
        nn.init.constant_(self.scalar_gate.bias, 0.1)

    def forward(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        graph_counts: torch.Tensor,
    ) -> torch.Tensor:
        normalized_scalars = _stable_layer_norm(self.scalar_norm, scalars)
        gate = torch.tanh(
            self.scalar_gate(normalized_scalars.to(dtype=scalars.dtype))
        ).to(dtype=pos.dtype)
        direction = self.vector_mix(_bounded_irrep(vectors, self.eps)).squeeze(1)
        raw_displacement = direction.to(dtype=pos.dtype) * gate
        return _bounded_centered_displacement(
            raw_displacement,
            batch,
            num_graphs=num_graphs,
            graph_counts=graph_counts,
            maximum=0.25,
        )


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
    global_transport_mode: str = "learned",
    spatial_features: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if global_transport_mode not in {"learned", "uniform"}:
        raise ValueError(
            "_global_moment_messages supports only learned or uniform transport"
        )
    if spatial_features is not None and global_transport_mode != "learned":
        raise ValueError("spatial features require learned global transport")
    if spatial_features is not None and use_memory_interaction:
        raise ValueError("spatial features cannot use memory interaction")
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
    if global_transport_mode == "uniform":
        if use_memory_interaction:
            raise ValueError("uniform global transport cannot use memory interaction")
        transported = _uniform_global_attention(
            value,
            batch,
            num_graphs=num_graphs,
            graph_counts=graph_counts,
        )
    elif use_memory_interaction and memory_count > 1:
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
            spatial_features=spatial_features,
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


def _uniform_global_attention(
    value: torch.Tensor,
    batch: torch.Tensor,
    *,
    num_graphs: int,
    graph_counts: torch.Tensor,
) -> torch.Tensor:
    """Broadcast the exact graph-wise mean without materializing pair weights."""
    return _scatter_mean(value, batch, num_graphs, graph_counts)[batch]


def _zero_moment_messages(
    num_nodes: int,
    num_heads: int,
    head_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(num_nodes, num_heads, head_dim, dtype=dtype, device=device),
        torch.zeros(num_nodes, num_heads, 3, dtype=dtype, device=device),
        torch.zeros(num_nodes, num_heads, 3, dtype=dtype, device=device),
        torch.zeros(num_nodes, num_heads, 5, dtype=dtype, device=device),
        torch.zeros(num_nodes, num_heads, dtype=dtype, device=device),
    )


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


def _fused_index_sum(
    index: torch.Tensor,
    num_segments: int,
    *values: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if not values:
        raise ValueError("at least one value is required")
    edge_count = index.shape[0]
    if any(value.shape[0] != edge_count for value in values):
        raise ValueError("all fused values must share the index length")
    shapes = [value.shape[1:] for value in values]
    widths = [prod(shape) for shape in shapes]
    groups: list[list[int]] = [[], []]
    group_widths = [0, 0]
    for value_index in sorted(
        range(len(values)),
        key=widths.__getitem__,
        reverse=True,
    ):
        group_index = 0 if group_widths[0] <= group_widths[1] else 1
        groups[group_index].append(value_index)
        group_widths[group_index] += widths[value_index]

    outputs: list[torch.Tensor | None] = [None] * len(values)
    for group in groups:
        if not group:
            continue
        if len(group) == 1:
            value_index = group[0]
            value = values[value_index]
            outputs[value_index] = value.new_zeros(
                (num_segments, *shapes[value_index])
            ).index_add(0, index, value)
            continue
        packed = torch.cat(
            [
                values[value_index].reshape(edge_count, widths[value_index])
                for value_index in group
            ],
            dim=-1,
        )
        reduced = packed.new_zeros((num_segments, sum(widths[i] for i in group)))
        reduced = reduced.index_add(0, index, packed)
        parts = torch.split(
            reduced,
            [widths[value_index] for value_index in group],
            dim=-1,
        )
        for value_index, part in zip(group, parts, strict=True):
            outputs[value_index] = part.reshape(
                num_segments,
                *shapes[value_index],
            )
    if any(output is None for output in outputs):
        raise RuntimeError("fused reduction failed to populate every output")
    return tuple(output for output in outputs if output is not None)


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
    rbf_spacing: str = "squared",
    radial_weight: torch.Tensor,
    radial_bias: torch.Tensor,
    local_geometry: _LocalGeometryInput | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if local_geometry is None:
        local_geometry = _local_geometry(
            pos,
            batch,
            num_graphs=num_graphs,
            cutoff=cutoff,
            num_rbf=num_rbf,
            rbf_spacing=rbf_spacing,
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
    rbf_spacing: str = "squared",
    graph_counts: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    edge_index_is_validated: bool = False,
) -> _LocalGeometry:
    if edge_index is None:
        if graph_counts is None:
            graph_counts = torch.bincount(batch, minlength=num_graphs)
        receiver, sender = _batched_complete_graph_edges(batch, graph_counts)
    else:
        if edge_index_is_validated:
            receiver, sender = _local_edge_index_components(
                edge_index,
                device=pos.device,
            )
        else:
            receiver, sender = _validated_local_edge_index(
                edge_index,
                batch,
                num_nodes=pos.shape[0],
                device=pos.device,
            )
    cutoff_tensor = pos.new_full((), float(cutoff))
    candidate_displacement = pos[sender] - pos[receiver]
    candidate_distance = _stable_vector_norm(candidate_displacement).squeeze(-1)
    inside = candidate_distance < cutoff_tensor
    receiver = receiver[inside]
    sender = sender[inside]
    displacement = candidate_displacement[inside] / cutoff_tensor
    squared_distance = displacement.square().sum(dim=-1)
    rbf = _radial_basis(squared_distance, num_rbf=num_rbf, spacing=rbf_spacing)
    nonself = receiver != sender
    nonself_displacement = displacement[nonself]
    nonself_squared_distance = squared_distance[nonself]
    return _LocalGeometry(
        receiver=receiver,
        sender=sender,
        displacement=displacement,
        squared_distance=squared_distance,
        rbf=rbf,
        nonself_receiver=receiver[nonself],
        nonself_sender=sender[nonself],
        nonself_displacement=nonself_displacement,
        nonself_squared_distance=nonself_squared_distance,
        nonself_rbf=rbf[nonself],
        nonself_cutoff=_cosine_of_squared_distance_cutoff(
            nonself_squared_distance
        ),
        nonself_tensor_features=_symmetric_traceless_features(
            nonself_displacement
        ),
    )


def _radial_basis(
    squared_distance: torch.Tensor,
    *,
    num_rbf: int,
    spacing: str,
) -> torch.Tensor:
    """Gaussian radial basis on the normalized squared distance.

    ``squared`` places centers uniformly in ``u = ||d||^2 / R_c^2``; its radial
    resolution therefore depends on the cutoff and is coarsest at short range.
    ``distance`` places centers uniformly in ``r / R_c`` and gives each basis
    function the squared-distance gap to its successor as width, distributing
    relative radial resolution uniformly over the normalized cutoff interval.
    Both evaluate polynomials of the coordinates without a square root, keeping
    the coincident-node derivative finite.
    """

    if spacing not in _LOCAL_RBF_SPACINGS:
        choices = ", ".join(sorted(_LOCAL_RBF_SPACINGS))
        raise ValueError(f"local_rbf_spacing must be one of: {choices}")
    knots = torch.linspace(
        0.0,
        1.0,
        num_rbf,
        dtype=squared_distance.dtype,
        device=squared_distance.device,
    )
    step = 1.0 / max(1, num_rbf - 1)
    if spacing == "squared":
        centers = knots
        widths: torch.Tensor | float = step
    else:
        centers = knots.square()
        widths = (knots + step).square() - centers
    return torch.exp(
        -0.5 * ((squared_distance.unsqueeze(-1) - centers) / widths).square()
    )


def _validated_local_edge_index(
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    *,
    num_nodes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    receiver, sender = _local_edge_index_components(edge_index, device=device)
    if receiver.numel() == 0:
        raise ValueError("edge_index must contain a self edge for every node")
    if bool((receiver < 0).any().item()) or bool((sender < 0).any().item()):
        raise ValueError("edge_index values must be nonnegative")
    if bool((receiver >= num_nodes).any().item()) or bool(
        (sender >= num_nodes).any().item()
    ):
        raise ValueError("edge_index values are out of range")
    if bool((batch[receiver] != batch[sender]).any().item()):
        raise ValueError("edge_index endpoints must belong to the same graph")

    pair_codes = receiver * num_nodes + sender
    if torch.unique(pair_codes).numel() != pair_codes.numel():
        raise ValueError("edge_index must not contain duplicate directed edges")
    self_nodes = receiver[receiver == sender]
    has_self = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    has_self[self_nodes] = True
    if not bool(has_self.all().item()):
        raise ValueError("edge_index must contain a self edge for every node")
    return receiver, sender


def _local_edge_index_components(
    edge_index: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Check tensor metadata only; callers own validated edge contents."""
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError("edge_index must be a tensor")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if edge_index.device != device:
        raise ValueError("edge_index and model inputs must use the same device")
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if edge_index.dtype not in integer_dtypes:
        raise TypeError("edge_index must use an integer dtype")
    return edge_index.to(dtype=torch.long).unbind(dim=0)


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
    sender_local = (
        torch.arange(total_edges, device=batch.device)
        - receiver_starts[receiver_positions]
    )
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


def _quadratic_gaussian_spatial_features(
    pos: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """O(3)-compatible positive degree-two Gaussian-Taylor features."""
    if pos.ndim != 2 or pos.shape[-1] != 3:
        raise ValueError("pos must have shape (nodes, 3)")
    if scales.ndim != 1 or scales.numel() == 0:
        raise ValueError("scales must be a nonempty one-dimensional tensor")
    scales = scales.to(device=pos.device, dtype=pos.dtype)

    scaled = pos[:, None, :] * torch.sqrt(2.0 * scales)[None, :, None]
    x, y, z = scaled.unbind(dim=-1)
    inverse_sqrt_two = pos.new_tensor(2.0).rsqrt()
    polynomial = torch.stack(
        [
            torch.ones_like(x),
            x,
            y,
            z,
            inverse_sqrt_two * x.square(),
            inverse_sqrt_two * y.square(),
            inverse_sqrt_two * z.square(),
            x * y,
            x * z,
            y * z,
        ],
        dim=-1,
    )
    gaussian = torch.exp(-pos.square().sum(dim=-1, keepdim=True) * scales[None, :])
    return gaussian.unsqueeze(-1) * polynomial


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
    spatial_features: torch.Tensor | None = None,
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
        *(() if spatial_features is None else (spatial_features,)),
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
    if spatial_features is not None:
        if spatial_features.ndim != 3 or spatial_features.shape[:2] != (
            query_scalar.shape[0],
            query_scalar.shape[1],
        ):
            raise ValueError(
                "spatial_features must have shape (nodes, heads, features)"
            )
        spatial_features = spatial_features.to(dtype=reduction_dtype)
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
        if spatial_features is not None:
            key_mass = key_mass + _spatial_key_mass(
                spatial_features,
                row_scale,
                batch,
                num_graphs,
            )
        key_scale = key_mass.reciprocal()
    else:
        key_scale = key_scalar.new_ones(key_scalar.shape[:2])
    augmented_value = torch.cat(
        [value, value.new_ones((*value.shape[:-1], 1))],
        dim=-1,
    )
    transported = _structured_numerator(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        kernel_scale,
        key_scale,
        augmented_value,
        batch,
        num_graphs,
        alignment_scale=alignment_scale,
        alignment_dot_scale=alignment_dot_scale,
        kernel_floor=kernel_floor,
        kernel_floor_mode=kernel_floor_mode,
        graph_counts=graph_counts,
    )
    if spatial_features is not None:
        transported = transported + _spatial_transport(
            spatial_features,
            key_scale,
            augmented_value,
            batch,
            num_graphs,
        )
    numerator = transported[..., :-1]
    denominator = transported[..., -1]
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
    if (
        radial_coupling.ndim < 2
        or radial_coupling.shape[-1] != radial_coupling.shape[-2]
    ):
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


def _spatial_key_mass(
    features: torch.Tensor,
    row_scale: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    weighted_features = features * row_scale.unsqueeze(-1)
    if num_graphs == 1:
        summary = weighted_features.sum(dim=0)
        return torch.einsum("nhf,hf->nh", features, summary)
    summary = _segment_sum(
        weighted_features,
        batch,
        num_graphs,
    )
    return torch.einsum("nhf,nhf->nh", features, summary[batch])


def _spatial_transport(
    features: torch.Tensor,
    key_scale: torch.Tensor,
    value: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    weighted_value = key_scale.unsqueeze(-1) * value
    if num_graphs == 1:
        summary = torch.einsum(
            "nhf,nhv->hfv",
            features,
            weighted_value,
        )
        return torch.einsum("nhf,hfv->nhv", features, summary)
    summary = _segment_sum(
        features.unsqueeze(-1) * weighted_value.unsqueeze(-2),
        batch,
        num_graphs,
    )
    return torch.einsum("nhf,nhfv->nhv", features, summary[batch])


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
    query_quadratic = _symmetric_quadratic_features(
        query_vector,
        left_factor=True,
    )
    quadratic_sum = _segment_sum(
        query_quadratic * row_scale.unsqueeze(-1),
        batch,
        num_graphs,
    )
    node_scalar_sum = _graph_summary_for_nodes(scalar_sum, batch, num_graphs)
    node_linear_sum = _graph_summary_for_nodes(linear_sum, batch, num_graphs)
    node_quadratic_sum = _graph_summary_for_nodes(
        quadratic_sum,
        batch,
        num_graphs,
    )
    node_constant_sum = _graph_summary_for_nodes(constant_sum, batch, num_graphs)
    content = (key_scalar * node_scalar_sum).sum(dim=-1)
    linear = (key_vector * node_linear_sum).sum(dim=-1)
    quadratic = (
        _symmetric_quadratic_features(key_vector, left_factor=False)
        * node_quadratic_sum
    ).sum(dim=-1).clamp_min(0.0)
    return (
        content
        + (pair_floor + pair_alignment_scale) * node_constant_sum
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
    key_quadratic = _symmetric_quadratic_features(
        key_vector,
        left_factor=False,
    )
    quadratic_sum = _segment_sum(
        key_quadratic * key_scale.unsqueeze(-1),
        batch,
        num_graphs,
    )
    node_scalar_sum = _graph_summary_for_nodes(scalar_sum, batch, num_graphs)
    node_linear_sum = _graph_summary_for_nodes(linear_sum, batch, num_graphs)
    node_quadratic_sum = _graph_summary_for_nodes(
        quadratic_sum,
        batch,
        num_graphs,
    )
    node_constant_sum = _graph_summary_for_nodes(constant_sum, batch, num_graphs)
    content = (query_scalar * node_scalar_sum).sum(dim=-1)
    linear = (query_vector * node_linear_sum).sum(dim=-1)
    quadratic = (
        _symmetric_quadratic_features(query_vector, left_factor=True)
        * node_quadratic_sum
    ).sum(dim=-1).clamp_min(0.0)
    return (
        content
        + (pair_floor + pair_alignment_scale) * node_constant_sum
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
    key_quadratic = _symmetric_quadratic_features(
        key_vector,
        left_factor=False,
    )
    quadratic_summary = _segment_sum(
        key_quadratic.unsqueeze(-1) * weighted_value.unsqueeze(-2),
        batch,
        num_graphs,
    )
    node_scalar_summary = _graph_summary_for_nodes(
        scalar_summary,
        batch,
        num_graphs,
    )
    node_linear_summary = _graph_summary_for_nodes(
        linear_summary,
        batch,
        num_graphs,
    )
    node_quadratic_summary = _graph_summary_for_nodes(
        quadratic_summary,
        batch,
        num_graphs,
    )
    node_constant_summary = _graph_summary_for_nodes(
        constant_summary,
        batch,
        num_graphs,
    )
    content = torch.einsum("nhd,nhdf->nhf", query_scalar, node_scalar_summary)
    linear = torch.einsum("nha,nhaf->nhf", query_vector, node_linear_summary)
    quadratic = torch.einsum(
        "nhd,nhdf->nhf",
        _symmetric_quadratic_features(query_vector, left_factor=True),
        node_quadratic_summary,
    )
    return (
        content
        + (pair_floor + pair_alignment_scale).unsqueeze(-1) * node_constant_summary
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


def _symmetric_quadratic_features(
    value: torch.Tensor,
    *,
    left_factor: bool,
) -> torch.Tensor:
    """Return one side of a compressed factorization of ``(x.y)^2``."""
    dimension = value.shape[-1]
    upper = torch.triu_indices(
        dimension,
        dimension,
        offset=1,
        device=value.device,
    )
    off_diagonal = (
        value.index_select(-1, upper[0]) * value.index_select(-1, upper[1])
    )
    if left_factor:
        off_diagonal = 2.0 * off_diagonal
    return torch.cat((value.square(), off_diagonal), dim=-1)


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


def _st_matrix_to_features(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-2:] != (3, 3):
        raise ValueError("symmetric-traceless matrices require final shape (3, 3)")
    return torch.stack(
        [
            value[..., 0, 0],
            value[..., 1, 1],
            value[..., 0, 1],
            value[..., 0, 2],
            value[..., 1, 2],
        ],
        dim=-1,
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


def _bounded_st_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] != 5:
        raise ValueError("symmetric-traceless features must have final dimension 5")
    output_dtype = value.dtype
    reduced = value.to(dtype=_moment_dtype(value))
    scale = torch.sqrt(1.0 + _st_frobenius_square(reduced) / 5.0)
    return (reduced / scale.unsqueeze(-1)).to(dtype=output_dtype)


def _segment_sum(
    value: torch.Tensor, batch: torch.Tensor, num_segments: int
) -> torch.Tensor:
    out = value.new_zeros((num_segments, *value.shape[1:]))
    return out.index_add(0, batch, value)


def _graph_summary_for_nodes(
    summary: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """Broadcast graph summaries without duplicate-index backward for one graph."""
    if summary.shape[0] != num_graphs:
        raise ValueError("summary graph dimension must equal num_graphs")
    if num_graphs == 1:
        return summary[0].unsqueeze(0).expand(batch.shape[0], *summary.shape[1:])
    return summary[batch]


def _segment_amax(
    value: torch.Tensor, batch: torch.Tensor, num_segments: int
) -> torch.Tensor:
    out = value.new_zeros((num_segments, *value.shape[1:]))
    index = batch.reshape(-1, *((1,) * (value.ndim - 1))).expand_as(value)
    return out.scatter_reduce(0, index, value, reduce="amax", include_self=True)


def _bounded_centered_displacement(
    displacement: torch.Tensor,
    batch: torch.Tensor,
    *,
    num_graphs: int,
    graph_counts: torch.Tensor,
    maximum: float,
) -> torch.Tensor:
    centered = (
        displacement
        - _scatter_mean(
            displacement,
            batch,
            num_graphs,
            graph_counts,
        )[batch]
    )
    norm = _stable_vector_norm(centered)
    graph_maximum = _segment_amax(norm, batch, num_graphs)
    limit = centered.new_full((), maximum)
    graph_scale = limit / graph_maximum.clamp_min(limit)
    return centered * graph_scale[batch]


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


def _scatter_sum(
    value: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    _graph_counts: torch.Tensor,
) -> torch.Tensor:
    output_dtype = value.dtype
    reduction_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
    return _segment_sum(
        value.to(dtype=reduction_dtype),
        batch,
        num_graphs,
    ).to(dtype=output_dtype)


def _graph_metadata(batch: torch.Tensor) -> tuple[int, torch.Tensor]:
    if (batch < 0).any():
        raise ValueError("batch indices must be nonnegative")
    num_graphs = int(batch.max().item()) + 1
    graph_counts = torch.bincount(batch, minlength=num_graphs)
    if bool((graph_counts == 0).any().item()):
        raise ValueError("batch indices must be contiguous and start at zero")
    return num_graphs, graph_counts


def _readout_metadata(
    readout_mask: torch.Tensor | None,
    batch: torch.Tensor,
    *,
    num_graphs: int,
    graph_counts: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    if readout_mask is None:
        return None, batch, graph_counts
    if readout_mask.dtype != torch.bool:
        raise TypeError("readout_mask must use boolean dtype")
    if readout_mask.shape != batch.shape:
        raise ValueError(
            f"readout_mask must have shape {tuple(batch.shape)}, "
            f"got {tuple(readout_mask.shape)}"
        )
    if readout_mask.device != batch.device:
        raise ValueError("readout_mask and batch must be on the same device")
    selected_batch = batch[readout_mask]
    selected_counts = torch.bincount(selected_batch, minlength=num_graphs)
    if bool((selected_counts == 0).any().item()):
        raise ValueError("readout_mask must select at least one node from every graph")
    if bool(readout_mask.all().item()):
        return None, batch, graph_counts
    return readout_mask, selected_batch, selected_counts


def _validate_interaction_roles(
    ligand_mask: torch.Tensor | None,
    batch: torch.Tensor,
    *,
    num_graphs: int,
) -> None:
    if ligand_mask is None:
        raise ValueError("interaction readout requires a ligand readout_mask")
    _interaction_role_counts(ligand_mask, batch, num_graphs=num_graphs)


def _interaction_role_counts(
    ligand_mask: torch.Tensor,
    batch: torch.Tensor,
    *,
    num_graphs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if ligand_mask.dtype != torch.bool or ligand_mask.shape != batch.shape:
        raise ValueError("interaction readout_mask must be boolean with shape (N,)")
    if ligand_mask.device != batch.device:
        raise ValueError("interaction readout_mask and batch must share a device")
    ligand_counts = torch.bincount(batch[ligand_mask], minlength=num_graphs)
    pocket_counts = torch.bincount(batch[~ligand_mask], minlength=num_graphs)
    if bool((ligand_counts == 0).any().item()):
        raise ValueError("interaction readout requires ligand nodes in every graph")
    if bool((pocket_counts == 0).any().item()):
        raise ValueError("interaction readout requires pocket nodes in every graph")
    return ligand_counts, pocket_counts


def _parity_even_triple_features(polar_moments: torch.Tensor) -> torch.Tensor:
    if polar_moments.shape[-2:] != (6, 3):
        raise ValueError("polar_moments must end with shape (6, 3)")
    first = torch.linalg.cross(
        polar_moments[..., 0, :],
        polar_moments[..., 1, :],
    )
    second = torch.linalg.cross(
        polar_moments[..., 3, :],
        polar_moments[..., 4, :],
    )
    first_triple = (first * polar_moments[..., 2, :]).sum(dim=-1)
    second_triple = (second * polar_moments[..., 5, :]).sum(dim=-1)
    first_bounded = first_triple / torch.hypot(
        first_triple,
        torch.ones_like(first_triple),
    )
    second_bounded = second_triple / torch.hypot(
        second_triple,
        torch.ones_like(second_triple),
    )
    return torch.stack(
        [
            first_bounded.square(),
            second_bounded.square(),
            first_bounded * second_bounded,
        ],
        dim=-1,
    )


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


def _positive_scalar_features(
    value: torch.Tensor,
    eps: float,
    *,
    mode: str,
) -> torch.Tensor:
    positive = F.elu(value) + 1.0
    unit = _normalize_positive_features(positive, eps)
    if mode == "unit":
        return unit
    if mode != "bounded":
        raise ValueError(f"unknown scalar content mode: {mode}")
    dimension = positive.shape[-1]
    rms = _stable_vector_norm(positive.to(dtype=_moment_dtype(positive))) / sqrt(
        dimension
    )
    amplitude = 2.0 * (rms / (1.0 + rms))
    return unit * amplitude.to(dtype=unit.dtype)


def _normalize_positive_features(value: torch.Tensor, eps: float) -> torch.Tensor:
    reduced = value.to(dtype=_moment_dtype(value))
    norm = _stable_vector_norm(reduced)
    floor = torch.full_like(norm, sqrt(eps))
    normalized = reduced / torch.hypot(norm, floor)
    return normalized * _inward_unit_margin(normalized)


def _tensor_product_features(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.shape != key.shape or query.ndim != 3 or query.shape[-1] != 5:
        raise ValueError("tensor query/key must share shape (nodes, heads, 5)")
    if scale.shape != (query.shape[1],):
        raise ValueError("tensor kernel scale must have one value per head")
    dtype = _moment_dtype(query, key, scale)
    query = _unit_frobenius_st(query.to(dtype=dtype), eps)
    key = _unit_frobenius_st(key.to(dtype=dtype), eps)
    query_matrix = _st_features_to_matrix(query).flatten(start_dim=-2)
    key_matrix = _st_features_to_matrix(key).flatten(start_dim=-2)
    root_scale = torch.sqrt(scale.to(dtype=dtype))[None, :, None]
    query_features = root_scale * torch.cat(
        [torch.ones_like(query_matrix[..., :1]), query_matrix],
        dim=-1,
    )
    key_features = root_scale * torch.cat(
        [torch.ones_like(key_matrix[..., :1]), key_matrix],
        dim=-1,
    )
    return query_features, key_features


def _unit_frobenius_st(value: torch.Tensor, eps: float) -> torch.Tensor:
    del eps
    reduced = value.to(dtype=_moment_dtype(value))
    denominator = torch.sqrt(1.0 + _st_frobenius_square(reduced)).unsqueeze(-1)
    normalized = reduced / denominator
    matrix_features = _st_features_to_matrix(normalized).flatten(start_dim=-2)
    return normalized * _inward_unit_margin(matrix_features)


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


def _stable_group_norm(value: torch.Tensor) -> torch.Tensor:
    dtype = _moment_dtype(value)
    return F.layer_norm(
        value.to(dtype=dtype),
        (value.shape[-1],),
        None,
        None,
        1e-5,
    )


def _moment_dtype(*values: torch.Tensor) -> torch.dtype:
    return (
        torch.float64
        if any(value.dtype == torch.float64 for value in values)
        else torch.float32
    )


def _validate_equivariant_input_tensor(
    name: str,
    value: torch.Tensor,
    *,
    reference: torch.Tensor,
) -> None:
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating point tensor")
    if value.device != reference.device:
        raise ValueError(f"{name} and node_feats must be on the same device")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


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
    for name in ("input_vector_dim", "input_tensor_dim"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if (
        not isinstance(config.angular_feature_rank, int)
        or isinstance(config.angular_feature_rank, bool)
    ):
        raise TypeError("angular_feature_rank must be an integer")
    if not 1 <= config.angular_feature_rank <= 2:
        raise ValueError("angular_feature_rank must lie in [1, 2]")
    for name in (
        "use_alignment_linear_term",
        "use_key_balancing",
        "learn_local_radial_gate",
        "use_pairwise_local_content",
        "use_edge_conditioned_local_transport",
        "normalize_edge_conditioned_local_by_sqrt_degree",
        "use_gated_local_transport",
        "use_grouped_invariant_normalization",
        "use_irrep_rms_normalization",
        "use_quartic_kernel",
        "checkpoint_gated_local_mlp",
        "use_tensor_product_kernel",
        "use_memory_interaction",
        "use_radial_trace",
        "coordinate_updates",
        "use_multiscale_spatial_kernel",
    ):
        if not isinstance(getattr(config, name), bool):
            raise TypeError(f"{name} must be a bool")
    if not isinstance(config.global_transport_mode, str):
        raise TypeError("global_transport_mode must be a string")
    if config.global_transport_mode not in _GLOBAL_TRANSPORT_MODES:
        choices = ", ".join(sorted(_GLOBAL_TRANSPORT_MODES))
        raise ValueError(f"global_transport_mode must be one of: {choices}")
    if not isinstance(config.coordinate_neighbor_policy, str):
        raise TypeError("coordinate_neighbor_policy must be a string")
    if config.coordinate_neighbor_policy not in _COORDINATE_NEIGHBOR_POLICIES:
        choices = ", ".join(sorted(_COORDINATE_NEIGHBOR_POLICIES))
        raise ValueError(f"coordinate_neighbor_policy must be one of: {choices}")
    if not isinstance(config.local_rbf_spacing, str):
        raise TypeError("local_rbf_spacing must be a string")
    if config.local_rbf_spacing not in _LOCAL_RBF_SPACINGS:
        choices = ", ".join(sorted(_LOCAL_RBF_SPACINGS))
        raise ValueError(f"local_rbf_spacing must be one of: {choices}")
    if not isinstance(config.readout_mode, str):
        raise TypeError("readout_mode must be a string")
    if config.readout_mode not in _READOUT_MODES:
        choices = ", ".join(sorted(_READOUT_MODES))
        raise ValueError(f"readout_mode must be one of: {choices}")
    if not isinstance(config.scalar_content_mode, str):
        raise TypeError("scalar_content_mode must be a string")
    if config.scalar_content_mode not in _SCALAR_CONTENT_MODES:
        choices = ", ".join(sorted(_SCALAR_CONTENT_MODES))
        raise ValueError(f"scalar_content_mode must be one of: {choices}")
    if config.coordinate_updates and config.num_layers < 2:
        raise ValueError("coordinate_updates requires at least two layers")
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
    if config.use_multiscale_spatial_kernel:
        if any(local_head_counts):
            raise ValueError(
                "use_multiscale_spatial_kernel requires an all-global route"
            )
        if config.global_transport_mode != "learned":
            raise ValueError(
                "use_multiscale_spatial_kernel requires learned global transport"
            )
        if config.use_memory_interaction:
            raise ValueError(
                "use_multiscale_spatial_kernel cannot use memory interaction"
            )
    if config.use_pairwise_local_content and not any(local_head_counts):
        raise ValueError("use_pairwise_local_content requires at least one local head")
    if config.use_gated_local_transport:
        if not any(local_head_counts):
            raise ValueError("gated local transport requires at least one local head")
        if any(
            local_heads not in {0, config.num_heads}
            for local_heads in local_head_counts
        ):
            raise ValueError(
                "gated local transport requires all-local or all-global stages"
            )
        if config.use_edge_conditioned_local_transport:
            raise ValueError(
                "gated local transport cannot be combined with "
                "edge-conditioned local transport"
            )
        if config.use_pairwise_local_content:
            raise ValueError(
                "gated local transport cannot be combined with pairwise local content"
            )
        if config.learn_local_radial_gate:
            raise ValueError(
                "gated local transport cannot be combined with the "
                "legacy learned local radial gate"
            )
    elif config.checkpoint_gated_local_mlp:
        raise ValueError(
            "checkpoint_gated_local_mlp requires gated local transport"
        )
    if config.use_edge_conditioned_local_transport:
        if not any(local_head_counts):
            raise ValueError(
                "edge-conditioned local transport requires at least one local head"
            )
        if config.use_pairwise_local_content:
            raise ValueError(
                "edge-conditioned local transport cannot be combined with "
                "pairwise local content"
            )
        if config.learn_local_radial_gate:
            raise ValueError(
                "edge-conditioned local transport cannot be combined with the "
                "legacy learned local radial gate"
            )
    elif config.normalize_edge_conditioned_local_by_sqrt_degree:
        raise ValueError(
            "edge-conditioned local degree normalization requires "
            "edge-conditioned local transport"
        )
    _float32_control(
        "pairwise_residual_scale_init",
        config.pairwise_residual_scale_init,
        nonnegative=True,
    )
    if config.use_memory_interaction:
        if config.global_transport_mode != "learned":
            raise ValueError("memory interaction requires learned global transport")
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
    tensor_init = _normal_float32_control(
        "tensor_kernel_init", config.tensor_kernel_init, positive=True
    )
    tensor_max = _normal_float32_control(
        "tensor_kernel_max", config.tensor_kernel_max, positive=True
    )
    quartic_init = _normal_float32_control(
        "quartic_kernel_init", config.quartic_kernel_init, positive=True
    )
    quartic_max = _normal_float32_control(
        "quartic_kernel_max", config.quartic_kernel_max, positive=True
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
    if tensor_init >= tensor_max:
        raise ValueError(
            "tensor_kernel_init must be smaller than tensor_kernel_max in float32"
        )
    if quartic_init >= quartic_max:
        raise ValueError(
            "quartic_kernel_init must be smaller than quartic_kernel_max in float32"
        )
    upper_bound = torch.tensor(kernel_floor, dtype=torch.float32)
    content_upper_bound = 4.0 if config.scalar_content_mode == "bounded" else 1.0
    upper_bound = upper_bound + content_upper_bound + 2.0 * linear_max + vector_max
    if config.use_tensor_product_kernel:
        upper_bound = upper_bound + 2.0 * tensor_max
    if config.use_quartic_kernel:
        upper_bound = upper_bound + quartic_max
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
    _normal_float32_ratio(
        "tensor_kernel_init/tensor_kernel_max",
        tensor_init,
        tensor_max,
    )
    _normal_float32_ratio(
        "quartic_kernel_init/quartic_kernel_max",
        quartic_init,
        quartic_max,
    )
    hidden = CartesianIrreps.parse(config.hidden_irreps)
    output = CartesianIrreps.parse(config.output_irreps)
    if hidden.scalars <= 0 or hidden.vectors <= 0:
        raise ValueError("hidden_irreps requires positive scalar and vector channels")
    if hidden.scalars % config.num_heads:
        raise ValueError("hidden scalar channels must be divisible by num_heads")
    if config.use_tensor_product_kernel and hidden.tensors <= 0:
        raise ValueError("tensor-product kernel requires persistent 2e hidden channels")
    if config.input_tensor_dim and hidden.tensors <= 0:
        raise ValueError("input_tensor_dim requires persistent hidden 2e channels")
    if (
        config.use_tensor_product_kernel
        and config.use_memory_interaction
        and config.global_memory_count > 1
    ):
        raise ValueError(
            "tensor-product kernel is not registered with interacting "
            "multi-memory transport"
        )
    if output.scalars + output.vectors + output.tensors <= 0:
        raise ValueError("output_irreps must include at least one term")
    if config.readout_mode == "interaction" and output.scalars <= 0:
        raise ValueError("interaction readout requires scalar output channels")


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
