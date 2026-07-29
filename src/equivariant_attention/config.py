"""Versioned structured configuration for the equivariant-attention model.

The numerical model still consumes :class:`EquivariantAttentionConfig`.  This
module is a strict, serializable public boundary around that flat compatibility
schema.  Capabilities that the flat Cartesian executor cannot represent are
reported by :attr:`ArchitectureConfig.deferred_features` and are never silently
discarded by :meth:`ArchitectureConfig.to_legacy`.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields
import json
from math import isfinite
from typing import Any, ClassVar

from .irreps import CartesianIrreps, Irrep, IrrepLayout


_PROFILES = frozenset({"minimal", "standard", "chiral", "high_order", "expert"})
_SYMMETRY_GROUPS = frozenset({"O3", "SE3"})
_GLOBAL_BACKENDS = frozenset({"outer_scatter", "feature_gemm", "auto"})
_LOCAL_BACKENDS = frozenset(
    {
        "materialized",
        "segment_csr",
        "auto",
        "streamed_csr",
        "ell",
        "custom",
    }
)
_LOCAL_REDUCTION_BACKENDS = frozenset({"index_add", "segment_csr"})
_CACHE_MODES = frozenset({"full", "compact", "recompute", "auto"})
_SCHEMA = "equivariant_attention.architecture"
_SCHEMA_VERSION = 1


def _require_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")


def _require_integer(
    name: str,
    value: object,
    *,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        relation = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be {relation}")
    return value


def _require_real(
    name: str,
    value: object,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if positive and numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and numeric < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return numeric


def _require_choice(
    name: str,
    value: object,
    choices: frozenset[str],
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value not in choices:
        options = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {options}")
    return value


def _require_optional_integer_tuple(
    name: str,
    value: object,
    *,
    minimum: int = 0,
) -> None:
    if value is None:
        return
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple or None")
    for item in value:
        _require_integer(name, item, minimum=minimum)


def _parse_layout(value: str | CartesianIrreps, *, name: str) -> IrrepLayout:
    if not isinstance(value, (str, CartesianIrreps)):
        raise TypeError(f"{name} must be an irreps string or CartesianIrreps")
    return IrrepLayout.parse(str(value))


def _has_irrep(layout: IrrepLayout, degree: int, parity: str) -> bool:
    return any(
        block.irrep == Irrep(degree, parity) and block.multiplicity > 0
        for block in layout.blocks
    )


def _is_cartesian_layout(layout: IrrepLayout) -> bool:
    supported = {Irrep(0, "e"), Irrep(1, "o"), Irrep(2, "e")}
    return all(block.irrep in supported for block in layout.blocks)


@dataclass(frozen=True, slots=True)
class RepresentationConfig:
    """Persistent carrier, kernel-order, and transient-workspace controls."""

    hidden_irreps: str | CartesianIrreps = "64x0e + 4x1o"
    output_irreps: str | CartesianIrreps = "1x0e"
    input_vector_dim: int = 0
    input_tensor_dim: int = 0
    num_node_roles: int = 0
    use_irrep_rms_normalization: bool = False
    angular_bandwidth: int = 1
    use_tensor_product_kernel: bool = False
    tensor_kernel_init: float = 0.05
    tensor_kernel_max: float = 1.0
    use_quartic_kernel: bool = False
    quartic_kernel_init: float = 0.01
    quartic_kernel_max: float = 1.0
    use_static_tensor_carrier: bool = False
    transient_max_degree: int = 2
    transient_workspace_channels: int = 1
    transient_workspace_layers: tuple[int, ...] | None = None
    transient_residual_scale_init: float = 0.05
    tensor_product_instructions: tuple[str, ...] = ()
    # Explicit public input contract for the optimized Cartesian executor.
    # This is validated against ArchitectureConfig.node_dim and the vector /
    # tensor input channel counts; it does not imply arbitrary-l execution.
    input_irreps: str | CartesianIrreps | None = None

    def __post_init__(self) -> None:
        hidden = _parse_layout(self.hidden_irreps, name="hidden_irreps")
        output = _parse_layout(self.output_irreps, name="output_irreps")
        if not hidden.blocks:
            raise ValueError("hidden_irreps must not be empty")
        if not output.blocks:
            raise ValueError("output_irreps must not be empty")
        if self.input_irreps is not None:
            input_layout = _parse_layout(
                self.input_irreps,
                name="input_irreps",
            )
            if not input_layout.blocks:
                raise ValueError("input_irreps must not be empty")
            if not _is_cartesian_layout(input_layout):
                raise ValueError(
                    "input_irreps must use the optimized Cartesian 0e/1o/2e "
                    "input layout"
                )
        _require_integer("input_vector_dim", self.input_vector_dim, minimum=0)
        _require_integer("input_tensor_dim", self.input_tensor_dim, minimum=0)
        _require_integer("num_node_roles", self.num_node_roles, minimum=0)
        _require_bool(
            "use_irrep_rms_normalization",
            self.use_irrep_rms_normalization,
        )
        _require_integer("angular_bandwidth", self.angular_bandwidth, minimum=1)
        _require_integer(
            "transient_max_degree",
            self.transient_max_degree,
            minimum=0,
        )
        _require_integer(
            "transient_workspace_channels",
            self.transient_workspace_channels,
            minimum=1,
        )
        _require_optional_integer_tuple(
            "transient_workspace_layers",
            self.transient_workspace_layers,
        )
        _require_real(
            "transient_residual_scale_init",
            self.transient_residual_scale_init,
            nonnegative=True,
        )
        if self.angular_bandwidth > self.transient_max_degree:
            raise ValueError("angular_bandwidth cannot exceed transient_max_degree")
        if (
            self.transient_workspace_layers is not None
            and self.transient_max_degree < 3
        ):
            raise ValueError(
                "transient_workspace_layers requires transient_max_degree >= 3"
            )
        for name in (
            "use_tensor_product_kernel",
            "use_quartic_kernel",
            "use_static_tensor_carrier",
        ):
            _require_bool(name, getattr(self, name))
        tensor_init = _require_real(
            "tensor_kernel_init",
            self.tensor_kernel_init,
            positive=True,
        )
        tensor_max = _require_real(
            "tensor_kernel_max",
            self.tensor_kernel_max,
            positive=True,
        )
        if tensor_init >= tensor_max:
            raise ValueError(
                "tensor_kernel_init must be smaller than tensor_kernel_max"
            )
        quartic_init = _require_real(
            "quartic_kernel_init",
            self.quartic_kernel_init,
            positive=True,
        )
        quartic_max = _require_real(
            "quartic_kernel_max",
            self.quartic_kernel_max,
            positive=True,
        )
        if quartic_init >= quartic_max:
            raise ValueError(
                "quartic_kernel_init must be smaller than quartic_kernel_max"
            )
        if not isinstance(self.tensor_product_instructions, tuple):
            raise TypeError("tensor_product_instructions must be a tuple")
        if any(
            not isinstance(instruction, str) or not instruction.strip()
            for instruction in self.tensor_product_instructions
        ):
            raise ValueError(
                "tensor_product_instructions must contain nonempty strings"
            )

    @property
    def persistent_max_degree(self) -> int:
        return _parse_layout(
            self.hidden_irreps,
            name="hidden_irreps",
        ).max_degree

    @property
    def deferred_features(self) -> tuple[str, ...]:
        hidden = _parse_layout(self.hidden_irreps, name="hidden_irreps")
        output = _parse_layout(self.output_irreps, name="output_irreps")
        deferred: list[str] = []
        if not _is_cartesian_layout(hidden):
            deferred.append("generic_persistent_irreps")
        if not _is_cartesian_layout(output):
            deferred.append("generic_output_irreps")
        if self.transient_max_degree > 3:
            deferred.append(f"transient_l{self.transient_max_degree}_workspace")
        if self.tensor_product_instructions:
            deferred.append("expert_tensor_product_instructions")
        return tuple(deferred)

    def _legacy_kwargs(self) -> dict[str, object]:
        return {
            "hidden_irreps": self.hidden_irreps,
            "output_irreps": self.output_irreps,
            "input_vector_dim": self.input_vector_dim,
            "input_tensor_dim": self.input_tensor_dim,
            "num_node_roles": self.num_node_roles,
            "use_irrep_rms_normalization": self.use_irrep_rms_normalization,
            "angular_feature_rank": self.angular_bandwidth,
            "use_tensor_product_kernel": self.use_tensor_product_kernel,
            "tensor_kernel_init": self.tensor_kernel_init,
            "tensor_kernel_max": self.tensor_kernel_max,
            "use_quartic_kernel": self.use_quartic_kernel,
            "quartic_kernel_init": self.quartic_kernel_init,
            "quartic_kernel_max": self.quartic_kernel_max,
            "use_static_tensor_carrier": self.use_static_tensor_carrier,
            "use_transient_l3_workspace": self.transient_max_degree == 3,
            "transient_l3_channels": self.transient_workspace_channels,
            "transient_l3_layers": self.transient_workspace_layers,
            "transient_l3_residual_scale_init": (
                self.transient_residual_scale_init
            ),
        }


@dataclass(frozen=True, slots=True)
class GlobalTransportConfig:
    """Exact finite-feature global kernel and reduction controls."""

    linear_kernel_init: float = 0.05
    linear_kernel_max: float = 1.0
    vector_kernel_init: float = 0.05
    vector_kernel_max: float = 1.0
    kernel_floor: float = 1.0
    kernel_floor_mode: str = "fixed"
    use_alignment_linear_term: bool = True
    use_key_balancing: bool = True
    global_memory_count: int = 1
    use_memory_interaction: bool = False
    memory_assignment_temperature: float = 1.0
    memory_assignment_scale: float = 2.5
    memory_interaction_cutoff: float = 2.5
    use_radial_trace: bool = False
    transport_mode: str = "learned"
    use_multiscale_spatial_kernel: bool = False
    use_adaptive_multiscale_spatial_kernel: bool = False
    use_whitened_global_read: bool = False
    whitened_global_ridge: float = 0.1
    whitened_global_rank_gate: bool = False
    use_global_tensor_value_transport: bool = False
    use_global_key_balancing: bool | None = None
    reduction_backend: str = "outer_scatter"

    def __post_init__(self) -> None:
        for stem in ("linear", "vector"):
            initial = _require_real(
                f"{stem}_kernel_init",
                getattr(self, f"{stem}_kernel_init"),
                positive=True,
            )
            maximum = _require_real(
                f"{stem}_kernel_max",
                getattr(self, f"{stem}_kernel_max"),
                positive=True,
            )
            if initial >= maximum:
                raise ValueError(
                    f"{stem}_kernel_init must be smaller than {stem}_kernel_max"
                )
        _require_real("kernel_floor", self.kernel_floor, positive=True)
        _require_choice(
            "kernel_floor_mode",
            self.kernel_floor_mode,
            frozenset({"fixed", "inverse_graph_size"}),
        )
        for name in (
            "use_alignment_linear_term",
            "use_key_balancing",
            "use_memory_interaction",
            "use_radial_trace",
            "use_multiscale_spatial_kernel",
            "use_adaptive_multiscale_spatial_kernel",
            "use_whitened_global_read",
            "whitened_global_rank_gate",
            "use_global_tensor_value_transport",
        ):
            _require_bool(name, getattr(self, name))
        _require_integer(
            "global_memory_count",
            self.global_memory_count,
            minimum=1,
        )
        for name in (
            "memory_assignment_temperature",
            "memory_assignment_scale",
            "memory_interaction_cutoff",
            "whitened_global_ridge",
        ):
            _require_real(name, getattr(self, name), positive=True)
        _require_choice(
            "transport_mode",
            self.transport_mode,
            frozenset({"learned", "uniform", "none"}),
        )
        if self.use_global_key_balancing is not None:
            _require_bool(
                "use_global_key_balancing",
                self.use_global_key_balancing,
            )
        _require_choice(
            "reduction_backend",
            self.reduction_backend,
            _GLOBAL_BACKENDS,
        )

    def _legacy_kwargs(self) -> dict[str, object]:
        return {
            "linear_kernel_init": self.linear_kernel_init,
            "linear_kernel_max": self.linear_kernel_max,
            "vector_kernel_init": self.vector_kernel_init,
            "vector_kernel_max": self.vector_kernel_max,
            "kernel_floor": self.kernel_floor,
            "kernel_floor_mode": self.kernel_floor_mode,
            "use_alignment_linear_term": self.use_alignment_linear_term,
            "use_key_balancing": self.use_key_balancing,
            "global_memory_count": self.global_memory_count,
            "use_memory_interaction": self.use_memory_interaction,
            "memory_assignment_temperature": self.memory_assignment_temperature,
            "memory_assignment_scale": self.memory_assignment_scale,
            "memory_interaction_cutoff": self.memory_interaction_cutoff,
            "use_radial_trace": self.use_radial_trace,
            "global_transport_mode": self.transport_mode,
            "use_multiscale_spatial_kernel": self.use_multiscale_spatial_kernel,
            "use_adaptive_multiscale_spatial_kernel": (
                self.use_adaptive_multiscale_spatial_kernel
            ),
            "use_whitened_global_read": self.use_whitened_global_read,
            "whitened_global_ridge": self.whitened_global_ridge,
            "whitened_global_rank_gate": self.whitened_global_rank_gate,
            "use_global_tensor_value_transport": (
                self.use_global_tensor_value_transport
            ),
            "use_global_key_balancing": self.use_global_key_balancing,
            "global_reduction_backend": self.reduction_backend,
        }


@dataclass(frozen=True, slots=True)
class LocalResidualConfig:
    """Legacy local route plus homogeneous sparse-residual controls."""

    local_head_counts: tuple[int, ...] | None = None
    local_cutoff: float = 2.5
    num_rbf: int = 16
    learn_local_radial_gate: bool = False
    use_pairwise_local_content: bool = False
    pairwise_residual_scale_init: float = 0.1
    use_edge_conditioned_local_transport: bool = False
    normalize_edge_conditioned_local_by_sqrt_degree: bool = False
    use_gated_local_transport: bool = False
    use_grouped_invariant_normalization: bool = False
    checkpoint_gated_local_mlp: bool = False
    local_rbf_spacing: str = "squared"
    use_cartesian_tensor_product_local_transport: bool = False
    cartesian_tensor_product_local_layers: tuple[int, ...] | None = None
    use_geometry_aware_local_attention: bool = False
    use_se3_axial_tensor_product: bool = False
    geometry_aware_local_layers: tuple[int, ...] | None = None
    use_local_key_balancing: bool | None = None
    reduction_backend: str = "index_add"
    requested_backend: str = "materialized"
    use_sparse_low_rank_local_residual: bool = False
    local_residual_rank: int = 4
    residual_stride: int = 1
    residual_layers: tuple[int, ...] | None = None
    sparse_residual_normalization: str = "positive"
    sparse_residual_score_limit: float = 3.0
    sparse_residual_balancing: str = "receiver"
    distance_bands: tuple[float, ...] = ()
    sparse_residual_stream_chunk_size: int = 64

    def __post_init__(self) -> None:
        _require_optional_integer_tuple(
            "local_head_counts",
            self.local_head_counts,
        )
        _require_real("local_cutoff", self.local_cutoff, positive=True)
        _require_integer("num_rbf", self.num_rbf, minimum=1)
        for name in (
            "learn_local_radial_gate",
            "use_pairwise_local_content",
            "use_edge_conditioned_local_transport",
            "normalize_edge_conditioned_local_by_sqrt_degree",
            "use_gated_local_transport",
            "use_grouped_invariant_normalization",
            "checkpoint_gated_local_mlp",
            "use_cartesian_tensor_product_local_transport",
            "use_geometry_aware_local_attention",
            "use_se3_axial_tensor_product",
            "use_sparse_low_rank_local_residual",
        ):
            _require_bool(name, getattr(self, name))
        _require_real(
            "pairwise_residual_scale_init",
            self.pairwise_residual_scale_init,
            nonnegative=True,
        )
        _require_choice(
            "local_rbf_spacing",
            self.local_rbf_spacing,
            frozenset({"squared", "distance"}),
        )
        for name in (
            "cartesian_tensor_product_local_layers",
            "geometry_aware_local_layers",
            "residual_layers",
        ):
            _require_optional_integer_tuple(name, getattr(self, name))
        if self.use_local_key_balancing is not None:
            _require_bool(
                "use_local_key_balancing",
                self.use_local_key_balancing,
            )
        _require_choice(
            "reduction_backend",
            self.reduction_backend,
            _LOCAL_REDUCTION_BACKENDS,
        )
        _require_choice(
            "requested_backend",
            self.requested_backend,
            _LOCAL_BACKENDS,
        )
        _require_integer(
            "local_residual_rank",
            self.local_residual_rank,
            minimum=1,
        )
        _require_integer("residual_stride", self.residual_stride, minimum=1)
        if self.residual_layers is not None and self.residual_stride != 1:
            raise ValueError(
                "residual_layers and residual_stride cannot both specify a schedule"
            )
        _require_choice(
            "sparse_residual_normalization",
            self.sparse_residual_normalization,
            frozenset({"positive", "softmax"}),
        )
        score_limit = _require_real(
            "sparse_residual_score_limit",
            self.sparse_residual_score_limit,
            positive=True,
        )
        if not 0.5 <= score_limit <= 4.0:
            raise ValueError("sparse_residual_score_limit must lie in [0.5, 4.0]")
        _require_choice(
            "sparse_residual_balancing",
            self.sparse_residual_balancing,
            frozenset({"receiver"}),
        )
        if not isinstance(self.distance_bands, tuple):
            raise TypeError("distance_bands must be a tuple")
        previous = 0.0
        for cutoff in self.distance_bands:
            numeric = _require_real("distance_bands", cutoff, positive=True)
            if numeric <= previous:
                raise ValueError(
                    "distance_bands must be strictly increasing positive cutoffs"
                )
            previous = numeric
        _require_integer(
            "sparse_residual_stream_chunk_size",
            self.sparse_residual_stream_chunk_size,
            minimum=1,
        )

    @property
    def deferred_features(self) -> tuple[str, ...]:
        return ()

    def resolved_residual_layers(self, num_layers: int) -> tuple[int, ...] | None:
        if self.residual_layers is not None:
            return self.residual_layers
        if self.use_sparse_low_rank_local_residual and self.residual_stride > 1:
            return tuple(range(0, num_layers, self.residual_stride))
        return None

    def _legacy_kwargs(
        self,
        *,
        num_layers: int,
        for_validation: bool,
    ) -> dict[str, object]:
        del for_validation
        return {
            "local_head_counts": self.local_head_counts,
            "local_cutoff": self.local_cutoff,
            "num_rbf": self.num_rbf,
            "learn_local_radial_gate": self.learn_local_radial_gate,
            "use_pairwise_local_content": self.use_pairwise_local_content,
            "pairwise_residual_scale_init": self.pairwise_residual_scale_init,
            "use_edge_conditioned_local_transport": (
                self.use_edge_conditioned_local_transport
            ),
            "normalize_edge_conditioned_local_by_sqrt_degree": (
                self.normalize_edge_conditioned_local_by_sqrt_degree
            ),
            "use_gated_local_transport": self.use_gated_local_transport,
            "use_grouped_invariant_normalization": (
                self.use_grouped_invariant_normalization
            ),
            "checkpoint_gated_local_mlp": self.checkpoint_gated_local_mlp,
            "local_rbf_spacing": self.local_rbf_spacing,
            "use_cartesian_tensor_product_local_transport": (
                self.use_cartesian_tensor_product_local_transport
            ),
            "cartesian_tensor_product_local_layers": (
                self.cartesian_tensor_product_local_layers
            ),
            "use_geometry_aware_local_attention": (
                self.use_geometry_aware_local_attention
            ),
            "use_se3_axial_tensor_product": self.use_se3_axial_tensor_product,
            "geometry_aware_local_layers": self.geometry_aware_local_layers,
            "use_local_key_balancing": self.use_local_key_balancing,
            "local_reduction_backend": self.reduction_backend,
            "use_sparse_low_rank_local_residual": (
                self.use_sparse_low_rank_local_residual
            ),
            "local_residual_rank": self.local_residual_rank,
            "local_residual_layers": self.resolved_residual_layers(num_layers),
            "sparse_residual_normalization": (self.sparse_residual_normalization),
            "sparse_residual_score_limit": self.sparse_residual_score_limit,
            "sparse_residual_balancing": self.sparse_residual_balancing,
            "distance_band_cutoffs": self.distance_bands,
            "sparse_residual_backend": self.requested_backend,
            "sparse_residual_stream_chunk_size": (
                self.sparse_residual_stream_chunk_size
            ),
        }


@dataclass(frozen=True, slots=True)
class NeighborConfig:
    """Neighbor-policy, geometry-cache, and immutable relation metadata."""

    coordinate_neighbor_policy: str = "error"
    sparse_residual_neighbor_policy: str = "require"
    sparse_residual_complete_fallback_max_nodes: int = 256
    geometry_cache_mode: str = "full"
    provider_kind: str = "runtime"
    num_edge_relations: int = 0
    relation_cutoffs: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        _require_choice(
            "coordinate_neighbor_policy",
            self.coordinate_neighbor_policy,
            frozenset({"error", "fixed", "rebuild"}),
        )
        _require_choice(
            "sparse_residual_neighbor_policy",
            self.sparse_residual_neighbor_policy,
            frozenset({"require", "complete_fallback"}),
        )
        _require_integer(
            "sparse_residual_complete_fallback_max_nodes",
            self.sparse_residual_complete_fallback_max_nodes,
            minimum=1,
        )
        _require_choice(
            "geometry_cache_mode",
            self.geometry_cache_mode,
            _CACHE_MODES,
        )
        _require_choice(
            "provider_kind",
            self.provider_kind,
            frozenset(
                {
                    "runtime",
                    "precomputed",
                    "reference_radius",
                    "external",
                    "verlet",
                }
            ),
        )
        _require_integer(
            "num_edge_relations",
            self.num_edge_relations,
            minimum=0,
        )
        if self.relation_cutoffs is not None:
            if not isinstance(self.relation_cutoffs, tuple):
                raise TypeError("relation_cutoffs must be a tuple or None")
            if len(self.relation_cutoffs) != self.num_edge_relations:
                raise ValueError(
                    "relation_cutoffs length must equal num_edge_relations"
                )
            for cutoff in self.relation_cutoffs:
                _require_real("relation_cutoffs", cutoff, positive=True)

    def _legacy_kwargs(self) -> dict[str, object]:
        return {
            "coordinate_neighbor_policy": self.coordinate_neighbor_policy,
            "sparse_residual_neighbor_policy": (self.sparse_residual_neighbor_policy),
            "sparse_residual_complete_fallback_max_nodes": (
                self.sparse_residual_complete_fallback_max_nodes
            ),
            "geometry_cache_mode": self.geometry_cache_mode,
            "num_edge_relations": self.num_edge_relations,
            "relation_cutoffs": self.relation_cutoffs,
        }


@dataclass(frozen=True, slots=True)
class ArchitectureConfig:
    """Versioned structured architecture configuration.

    ``profile="expert"`` is the exact compatibility mode used by
    :meth:`from_legacy`.  Named profiles are frozen capability presets and are
    constructed through :meth:`for_profile`.
    """

    SCHEMA: ClassVar[str] = _SCHEMA
    SCHEMA_VERSION: ClassVar[int] = _SCHEMA_VERSION

    node_dim: int
    num_layers: int = 3
    num_heads: int = 4
    profile: str = "expert"
    symmetry_group: str = "O3"
    representation: RepresentationConfig = RepresentationConfig()
    global_transport: GlobalTransportConfig = GlobalTransportConfig()
    local: LocalResidualConfig = LocalResidualConfig()
    neighbor: NeighborConfig = NeighborConfig()
    residual_scale_init: float = 0.1
    eps: float = 1e-12
    coordinate_updates: bool = False
    readout_mode: str = "mean"
    scalar_content_mode: str = "unit"

    def __post_init__(self) -> None:
        _require_integer("node_dim", self.node_dim, minimum=1)
        _require_integer("num_layers", self.num_layers, minimum=1)
        _require_integer("num_heads", self.num_heads, minimum=1)
        _require_choice("profile", self.profile, _PROFILES)
        _require_choice("symmetry_group", self.symmetry_group, _SYMMETRY_GROUPS)
        for name, expected in (
            ("representation", RepresentationConfig),
            ("global_transport", GlobalTransportConfig),
            ("local", LocalResidualConfig),
            ("neighbor", NeighborConfig),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be a {expected.__name__}")
        _require_real(
            "residual_scale_init",
            self.residual_scale_init,
            nonnegative=True,
        )
        _require_real("eps", self.eps, positive=True)
        _require_bool("coordinate_updates", self.coordinate_updates)
        _require_choice(
            "readout_mode",
            self.readout_mode,
            frozenset({"bipartite", "interaction", "mean", "sum"}),
        )
        _require_choice(
            "scalar_content_mode",
            self.scalar_content_mode,
            frozenset({"bounded", "unit"}),
        )
        self._validate_profile()
        self._validate_input_irreps()
        self._validate_schedules()
        self._validate_legacy_compatible_subset()

    @classmethod
    def for_profile(
        cls,
        profile: str,
        *,
        node_dim: int,
        width: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
    ) -> ArchitectureConfig:
        _require_choice("profile", profile, _PROFILES)
        _require_integer("width", width, minimum=1)
        _require_integer("num_heads", num_heads, minimum=1)
        if width % num_heads:
            raise ValueError("width must be divisible by num_heads")
        if profile in {"standard", "chiral", "high_order"}:
            hidden_irreps = f"{width}x0e + {num_heads}x1o + {num_heads}x2e"
        else:
            hidden_irreps = f"{width}x0e + {num_heads}x1o"
        representation = RepresentationConfig(
            hidden_irreps=hidden_irreps,
            input_irreps=f"{node_dim}x0e",
            transient_max_degree=3 if profile == "high_order" else 2,
        )
        return cls(
            node_dim=node_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            profile=profile,
            symmetry_group="SE3" if profile == "chiral" else "O3",
            representation=representation,
        )

    @classmethod
    def from_legacy(cls, config: object) -> ArchitectureConfig:
        from .moment import EquivariantAttentionConfig

        if not isinstance(config, EquivariantAttentionConfig):
            raise TypeError("config must be an EquivariantAttentionConfig")
        values = {
            field.name: getattr(config, field.name)
            for field in fields(EquivariantAttentionConfig)
        }
        _assert_legacy_coverage(values)
        representation = RepresentationConfig(
            hidden_irreps=values["hidden_irreps"],
            output_irreps=values["output_irreps"],
            input_irreps=_input_irrep_spec(
                node_dim=values["node_dim"],
                input_vector_dim=values["input_vector_dim"],
                input_tensor_dim=values["input_tensor_dim"],
            ),
            input_vector_dim=values["input_vector_dim"],
            input_tensor_dim=values["input_tensor_dim"],
            num_node_roles=values["num_node_roles"],
            use_irrep_rms_normalization=values["use_irrep_rms_normalization"],
            angular_bandwidth=values["angular_feature_rank"],
            use_tensor_product_kernel=values["use_tensor_product_kernel"],
            tensor_kernel_init=values["tensor_kernel_init"],
            tensor_kernel_max=values["tensor_kernel_max"],
            use_quartic_kernel=values["use_quartic_kernel"],
            quartic_kernel_init=values["quartic_kernel_init"],
            quartic_kernel_max=values["quartic_kernel_max"],
            use_static_tensor_carrier=values["use_static_tensor_carrier"],
            transient_max_degree=(
                3 if values["use_transient_l3_workspace"] else 2
            ),
            transient_workspace_channels=values["transient_l3_channels"],
            transient_workspace_layers=values["transient_l3_layers"],
            transient_residual_scale_init=values[
                "transient_l3_residual_scale_init"
            ],
        )
        global_transport = GlobalTransportConfig(
            **_select_fields(values, _GLOBAL_LEGACY_FIELDS),
            transport_mode=values["global_transport_mode"],
            reduction_backend=values["global_reduction_backend"],
        )
        local_values = _select_fields(values, _LOCAL_LEGACY_FIELDS)
        local_values["reduction_backend"] = values["local_reduction_backend"]
        local_values["requested_backend"] = values["sparse_residual_backend"]
        local_values["residual_layers"] = values["local_residual_layers"]
        local_values["distance_bands"] = values["distance_band_cutoffs"]
        local = LocalResidualConfig(**local_values)
        neighbor = NeighborConfig(**_select_fields(values, _NEIGHBOR_LEGACY_FIELDS))
        return cls(
            node_dim=values["node_dim"],
            num_layers=values["num_layers"],
            num_heads=values["num_heads"],
            profile="expert",
            symmetry_group=values["symmetry_group"],
            representation=representation,
            global_transport=global_transport,
            local=local,
            neighbor=neighbor,
            residual_scale_init=values["residual_scale_init"],
            eps=values["eps"],
            coordinate_updates=values["coordinate_updates"],
            readout_mode=values["readout_mode"],
            scalar_content_mode=values["scalar_content_mode"],
        )

    @property
    def deferred_features(self) -> tuple[str, ...]:
        return (
            *self.representation.deferred_features,
            *self.local.deferred_features,
        )

    def to_legacy(self) -> object:
        """Return the exact flat compatibility config or reject unsupported data."""
        if self.deferred_features:
            features = ", ".join(self.deferred_features)
            raise NotImplementedError(
                "the flat Cartesian executor cannot represent deferred features: "
                f"{features}"
            )
        from .moment import EquivariantAttentionConfig

        values = self._legacy_kwargs(for_validation=False)
        _assert_legacy_coverage(values)
        return EquivariantAttentionConfig(**values)

    def to_dict(self) -> dict[str, object]:
        config = {
            "node_dim": self.node_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "profile": self.profile,
            "symmetry_group": self.symmetry_group,
            "representation": _dataclass_payload(self.representation),
            "global_transport": _dataclass_payload(self.global_transport),
            "local": _dataclass_payload(self.local),
            "neighbor": _dataclass_payload(self.neighbor),
            "residual_scale_init": self.residual_scale_init,
            "eps": self.eps,
            "coordinate_updates": self.coordinate_updates,
            "readout_mode": self.readout_mode,
            "scalar_content_mode": self.scalar_content_mode,
        }
        return {
            "schema": self.SCHEMA,
            "schema_version": self.SCHEMA_VERSION,
            "config": config,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> ArchitectureConfig:
        value = _strict_json_loads(payload)
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls, payload: object) -> ArchitectureConfig:
        envelope = _strict_mapping(
            payload,
            expected={"schema", "schema_version", "config"},
            name="architecture envelope",
            required={"schema", "schema_version", "config"},
        )
        if envelope["schema"] != cls.SCHEMA:
            raise ValueError("unsupported architecture schema")
        if envelope["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        config = _strict_mapping(
            envelope["config"],
            expected={
                "node_dim",
                "num_layers",
                "num_heads",
                "profile",
                "symmetry_group",
                "representation",
                "global_transport",
                "local",
                "neighbor",
                "residual_scale_init",
                "eps",
                "coordinate_updates",
                "readout_mode",
                "scalar_content_mode",
            },
            name="architecture config",
            required={"node_dim"},
        )
        for key, nested_type, tuple_fields in (
            (
                "representation",
                RepresentationConfig,
                {
                    "tensor_product_instructions",
                    "transient_workspace_layers",
                },
            ),
            ("global_transport", GlobalTransportConfig, set()),
            (
                "local",
                LocalResidualConfig,
                {
                    "local_head_counts",
                    "cartesian_tensor_product_local_layers",
                    "geometry_aware_local_layers",
                    "residual_layers",
                    "distance_bands",
                },
            ),
            ("neighbor", NeighborConfig, {"relation_cutoffs"}),
        ):
            if key in config:
                config[key] = _construct_nested(
                    nested_type,
                    config[key],
                    tuple_fields=tuple_fields,
                    name=key,
                )
        return cls(**config)

    def _validate_profile(self) -> None:
        hidden = _parse_layout(
            self.representation.hidden_irreps,
            name="hidden_irreps",
        )
        transient = self.representation.transient_max_degree
        has_tensor = _has_irrep(hidden, 2, "e")
        if self.profile == "minimal":
            if has_tensor or hidden.max_degree > 1 or transient != 2:
                raise ValueError(
                    "minimal profile requires persistent 0e+1o and transient l=2"
                )
            if self.symmetry_group != "O3":
                raise ValueError("minimal profile requires symmetry_group='O3'")
        elif self.profile == "standard":
            if not has_tensor or hidden.max_degree != 2 or transient != 2:
                raise ValueError("standard profile requires persistent 0e+1o+2e")
            if self.symmetry_group != "O3":
                raise ValueError("standard profile requires symmetry_group='O3'")
        elif self.profile == "chiral":
            if not has_tensor or hidden.max_degree != 2:
                raise ValueError("chiral profile requires a low-l 0e+1o+2e carrier")
            if self.symmetry_group != "SE3":
                raise ValueError("chiral profile requires symmetry_group='SE3'")
        elif self.profile == "high_order":
            if not has_tensor or hidden.max_degree > 2 or transient != 3:
                raise ValueError(
                    "high_order profile requires a low-l carrier and transient l=3"
                )
            if self.symmetry_group != "O3":
                raise ValueError("high_order profile requires symmetry_group='O3'")
        if self.representation.tensor_product_instructions and self.profile != "expert":
            raise ValueError("raw tensor_product_instructions are expert-only")

    def _validate_input_irreps(self) -> None:
        declared = self.representation.input_irreps
        if declared is None:
            return
        expected = IrrepLayout.parse(
            _input_irrep_spec(
                node_dim=self.node_dim,
                input_vector_dim=self.representation.input_vector_dim,
                input_tensor_dim=self.representation.input_tensor_dim,
            )
        )
        actual = _parse_layout(declared, name="input_irreps")
        if actual != expected:
            raise ValueError(
                "input_irreps must exactly match node_dim, input_vector_dim, "
                "and input_tensor_dim for the optimized Cartesian executor"
            )

    def _validate_schedules(self) -> None:
        if (
            self.local.use_sparse_low_rank_local_residual
            and self.local.residual_stride > self.num_layers
        ):
            raise ValueError(
                "residual_stride cannot exceed num_layers for an active residual"
            )
        for name, schedule in (
            ("local_head_counts", self.local.local_head_counts),
            ("residual_layers", self.local.residual_layers),
            (
                "cartesian_tensor_product_local_layers",
                self.local.cartesian_tensor_product_local_layers,
            ),
            (
                "geometry_aware_local_layers",
                self.local.geometry_aware_local_layers,
            ),
            (
                "transient_workspace_layers",
                self.representation.transient_workspace_layers,
            ),
        ):
            if schedule is None:
                continue
            if name == "local_head_counts":
                if len(schedule) != self.num_layers:
                    raise ValueError("local_head_counts length must equal num_layers")
                if any(value > self.num_heads for value in schedule):
                    raise ValueError(
                        "local_head_counts entries cannot exceed num_heads"
                    )
            else:
                if not schedule:
                    raise ValueError(f"{name} must not be empty")
                if len(set(schedule)) != len(schedule):
                    raise ValueError(f"{name} must not contain duplicates")
                if any(index >= self.num_layers for index in schedule):
                    raise ValueError(f"{name} contains an invalid layer index")

    def _legacy_kwargs(self, *, for_validation: bool) -> dict[str, object]:
        values: dict[str, object] = {
            "node_dim": self.node_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "residual_scale_init": self.residual_scale_init,
            "eps": self.eps,
            "coordinate_updates": self.coordinate_updates,
            "readout_mode": self.readout_mode,
            "scalar_content_mode": self.scalar_content_mode,
            "symmetry_group": self.symmetry_group,
        }
        values.update(self.representation._legacy_kwargs())
        values.update(self.global_transport._legacy_kwargs())
        values.update(
            self.local._legacy_kwargs(
                num_layers=self.num_layers,
                for_validation=for_validation,
            )
        )
        values.update(self.neighbor._legacy_kwargs())
        return values

    def _validate_legacy_compatible_subset(self) -> None:
        hidden = _parse_layout(
            self.representation.hidden_irreps,
            name="hidden_irreps",
        )
        output = _parse_layout(
            self.representation.output_irreps,
            name="output_irreps",
        )
        if not _is_cartesian_layout(hidden) or not _is_cartesian_layout(output):
            return
        from .moment import EquivariantAttentionConfig, _validate_config

        values = self._legacy_kwargs(for_validation=True)
        _assert_legacy_coverage(values)
        _validate_config(EquivariantAttentionConfig(**values))


_GLOBAL_LEGACY_FIELDS = (
    "linear_kernel_init",
    "linear_kernel_max",
    "vector_kernel_init",
    "vector_kernel_max",
    "kernel_floor",
    "kernel_floor_mode",
    "use_alignment_linear_term",
    "use_key_balancing",
    "global_memory_count",
    "use_memory_interaction",
    "memory_assignment_temperature",
    "memory_assignment_scale",
    "memory_interaction_cutoff",
    "use_radial_trace",
    "use_multiscale_spatial_kernel",
    "use_adaptive_multiscale_spatial_kernel",
    "use_whitened_global_read",
    "whitened_global_ridge",
    "whitened_global_rank_gate",
    "use_global_tensor_value_transport",
    "use_global_key_balancing",
)

_LOCAL_LEGACY_FIELDS = (
    "local_head_counts",
    "local_cutoff",
    "num_rbf",
    "learn_local_radial_gate",
    "use_pairwise_local_content",
    "pairwise_residual_scale_init",
    "use_edge_conditioned_local_transport",
    "normalize_edge_conditioned_local_by_sqrt_degree",
    "use_gated_local_transport",
    "use_grouped_invariant_normalization",
    "checkpoint_gated_local_mlp",
    "local_rbf_spacing",
    "use_cartesian_tensor_product_local_transport",
    "cartesian_tensor_product_local_layers",
    "use_geometry_aware_local_attention",
    "use_se3_axial_tensor_product",
    "geometry_aware_local_layers",
    "use_local_key_balancing",
    "use_sparse_low_rank_local_residual",
    "local_residual_rank",
    "sparse_residual_normalization",
    "sparse_residual_score_limit",
    "sparse_residual_balancing",
    "sparse_residual_stream_chunk_size",
)

_NEIGHBOR_LEGACY_FIELDS = (
    "coordinate_neighbor_policy",
    "sparse_residual_neighbor_policy",
    "sparse_residual_complete_fallback_max_nodes",
    "geometry_cache_mode",
    "num_edge_relations",
    "relation_cutoffs",
)

_DIRECT_LEGACY_FIELDS = frozenset(
    {
        "node_dim",
        "num_layers",
        "num_heads",
        "residual_scale_init",
        "eps",
        "coordinate_updates",
        "readout_mode",
        "scalar_content_mode",
        "symmetry_group",
        "hidden_irreps",
        "output_irreps",
        "input_vector_dim",
        "input_tensor_dim",
        "num_node_roles",
        "use_irrep_rms_normalization",
        "angular_feature_rank",
        "use_tensor_product_kernel",
        "tensor_kernel_init",
        "tensor_kernel_max",
        "use_quartic_kernel",
        "quartic_kernel_init",
        "quartic_kernel_max",
        "use_static_tensor_carrier",
        "use_transient_l3_workspace",
        "transient_l3_channels",
        "transient_l3_layers",
        "transient_l3_residual_scale_init",
        "global_transport_mode",
        "global_reduction_backend",
        "local_reduction_backend",
        "local_residual_layers",
        "distance_band_cutoffs",
        "sparse_residual_backend",
    }
)
_KNOWN_LEGACY_FIELDS = frozenset(
    {
        *_DIRECT_LEGACY_FIELDS,
        *_GLOBAL_LEGACY_FIELDS,
        *_LOCAL_LEGACY_FIELDS,
        *_NEIGHBOR_LEGACY_FIELDS,
    }
)


def _assert_legacy_coverage(values: dict[str, object]) -> None:
    names = frozenset(values)
    missing = _KNOWN_LEGACY_FIELDS - names
    unknown = names - _KNOWN_LEGACY_FIELDS
    if missing or unknown:
        raise RuntimeError(
            "structured/legacy field coverage drift: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _select_fields(
    values: dict[str, object],
    names: tuple[str, ...],
) -> dict[str, object]:
    return {name: values[name] for name in names}


def _dataclass_payload(value: object) -> dict[str, object]:
    payload = {
        field.name: getattr(value, field.name) for field in fields(value) if field.init
    }
    for name in ("input_irreps", "hidden_irreps", "output_irreps"):
        if name in payload and isinstance(payload[name], CartesianIrreps):
            payload[name] = {
                "cartesian_irreps": asdict(payload[name]),
            }
    return payload


def _decode_irreps(value: object, *, name: str) -> object:
    if isinstance(value, str):
        return value
    mapping = _strict_mapping(
        value,
        expected={"cartesian_irreps"},
        required={"cartesian_irreps"},
        name=name,
    )
    payload = _strict_mapping(
        mapping["cartesian_irreps"],
        expected={field.name for field in fields(CartesianIrreps)},
        required=set(),
        name=f"{name}.cartesian_irreps",
    )
    return CartesianIrreps(**payload)


def _construct_nested(
    nested_type: type,
    payload: object,
    *,
    tuple_fields: set[str],
    name: str,
) -> object:
    expected = {field.name for field in fields(nested_type) if field.init}
    required = {
        field.name
        for field in fields(nested_type)
        if field.init and field.default is MISSING and field.default_factory is MISSING
    }
    values = _strict_mapping(
        payload,
        expected=expected,
        required=required,
        name=name,
    )
    for key in tuple_fields:
        if key in values and values[key] is not None:
            if not isinstance(values[key], list):
                raise TypeError(f"{name}.{key} must be a JSON array")
            values[key] = tuple(values[key])
    if nested_type is RepresentationConfig:
        for key in ("input_irreps", "hidden_irreps", "output_irreps"):
            if key in values and values[key] is not None:
                values[key] = _decode_irreps(
                    values[key],
                    name=f"{name}.{key}",
                )
    return nested_type(**values)


def _input_irrep_spec(
    *,
    node_dim: int,
    input_vector_dim: int,
    input_tensor_dim: int,
) -> str:
    terms = [f"{node_dim}x0e"]
    if input_vector_dim:
        terms.append(f"{input_vector_dim}x1o")
    if input_tensor_dim:
        terms.append(f"{input_tensor_dim}x2e")
    return " + ".join(terms)


def _strict_mapping(
    payload: object,
    *,
    expected: set[str],
    required: set[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in payload):
        raise TypeError(f"{name} keys must be strings")
    unknown = set(payload) - expected
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {sorted(unknown)}")
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{name} is missing required fields: {sorted(missing)}")
    return dict(payload)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _strict_json_loads(payload: str | bytes | bytearray) -> object:
    if not isinstance(payload, (str, bytes, bytearray)):
        raise TypeError("JSON payload must be str, bytes, or bytearray")
    try:
        return json.loads(payload, object_pairs_hook=_reject_duplicate_pairs)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid architecture JSON") from exc


__all__ = [
    "ArchitectureConfig",
    "GlobalTransportConfig",
    "LocalResidualConfig",
    "NeighborConfig",
    "RepresentationConfig",
]
