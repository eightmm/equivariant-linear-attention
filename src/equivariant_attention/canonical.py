from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt

import torch
from torch import nn

from .branch_fusion import RMSAwareBranchFusion
from .canonical_se3 import CanonicalMultipoleSE3Core
from .equivariant_linear_attention import (
    EquivariantLinearAttentionConfig,
    EquivariantLinearAttentionLayer,
    _bounded_modulate_state,
)
from .irreps import IrrepLayout
from .layered_se3 import (
    LayeredCanonicalSE3Core,
    UnifiedSE3Context,
    _BranchModulation,
    _gate_delta,
    _state_add,
    _state_subtract,
)
from .parity_se3 import (
    _ParityCompleteBlock,
    _ParityState,
    _bounded_scalar,
    _bounded_st,
    _unit_ball,
)
from .unified import Prepared3DGraph, UnifiedEquivariantAttention, prepare_3d_graph


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


@dataclass(frozen=True, slots=True)
class SparseGeometry:
    """Sparse geometry separated from model capacity and representation."""

    cutoff: float = 5.0
    num_rbf: int = 16
    relation_cutoffs: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        cutoff = _positive_real("cutoff", self.cutoff)
        _positive_integer("num_rbf", self.num_rbf)
        if not isinstance(self.relation_cutoffs, tuple):
            raise TypeError("relation_cutoffs must be a tuple")
        for relation_cutoff in self.relation_cutoffs:
            numeric = _positive_real("relation_cutoffs", relation_cutoff)
            if numeric > cutoff:
                raise ValueError("relation cutoffs may only narrow cutoff")

    @property
    def num_edge_relations(self) -> int:
        return len(self.relation_cutoffs)

    def prepare(
        self,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_relation_id: torch.Tensor | None = None,
        prefer_int32: bool = True,
    ) -> Prepared3DGraph:
        """Pack a caller-supplied sparse candidate graph."""

        return prepare_3d_graph(
            batch,
            edge_index,
            edge_relation_id=edge_relation_id,
            prefer_int32=prefer_int32,
        )


@dataclass(frozen=True, slots=True)
class ELAConfig:
    """Minimal canonical configuration for equivariant linear attention.

    Users control input/output representations and total capacity only.
    Attention heads, local rank, normalization, residual scaling, tensor
    closure, and parity-complete hidden multiplicities are deterministic
    implementation choices rather than public architecture search axes.
    """

    input_irreps: str
    output_irreps: str = "1x0e"
    width: int = 128
    depth: int = 8
    geometry: SparseGeometry = field(default_factory=SparseGeometry)

    def __post_init__(self) -> None:
        _positive_integer("width", self.width)
        _positive_integer("depth", self.depth)
        if self.width < 16:
            raise ValueError("width must be at least 16")
        if not isinstance(self.geometry, SparseGeometry):
            raise TypeError("geometry must be a SparseGeometry")
        self.to_advanced_config()

    @property
    def input_layout(self) -> IrrepLayout:
        return IrrepLayout.parse(self.input_irreps)

    @property
    def output_layout(self) -> IrrepLayout:
        return IrrepLayout.parse(self.output_irreps)

    @property
    def hidden_dim(self) -> int:
        return self.width

    @property
    def num_layers(self) -> int:
        return self.depth

    @property
    def num_heads(self) -> int:
        target = max(1, min(8, self.width // 16))
        for candidate in range(target, 0, -1):
            if self.width % candidate == 0:
                return candidate
        return 1

    @property
    def local_rank(self) -> int:
        return min(4, self.num_heads)

    @property
    def local_cutoff(self) -> float:
        return float(self.geometry.cutoff)

    @property
    def num_rbf(self) -> int:
        return self.geometry.num_rbf

    @property
    def relation_cutoffs(self) -> tuple[float, ...]:
        return self.geometry.relation_cutoffs

    @property
    def num_edge_relations(self) -> int:
        return self.geometry.num_edge_relations

    @property
    def num_node_roles(self) -> int:
        return 0

    @property
    def residual_scale_init(self) -> float:
        return 0.1

    @property
    def condition_dim(self) -> int:
        return 0

    @property
    def coordinate_updates(self) -> bool:
        return False

    @property
    def max_coordinate_step(self) -> float:
        return 0.25

    @property
    def residual_dropout(self) -> float:
        return 0.0

    @property
    def drop_path_rate(self) -> float:
        return 0.0

    @property
    def norm_eps(self) -> float:
        return 1e-6

    @property
    def eps(self) -> float:
        return 1e-12

    @property
    def internal_irreps(self) -> IrrepLayout:
        heads = self.num_heads
        return IrrepLayout.parse(
            f"{self.width}x0e + {heads}x0o + {heads}x1o + "
            f"{heads}x1e + {heads}x2e + {heads}x2o"
        )

    def to_advanced_config(self) -> EquivariantLinearAttentionConfig:
        return EquivariantLinearAttentionConfig(
            input_irreps=self.input_irreps,
            output_irreps=self.output_irreps,
            hidden_dim=self.width,
            num_layers=self.depth,
            num_heads=self.num_heads,
            local_rank=self.local_rank,
            local_cutoff=self.geometry.cutoff,
            num_rbf=self.geometry.num_rbf,
            relation_cutoffs=self.geometry.relation_cutoffs,
            residual_scale_init=self.residual_scale_init,
            condition_dim=0,
            coordinate_updates=False,
            residual_dropout=0.0,
            drop_path_rate=0.0,
            norm_eps=self.norm_eps,
            eps=self.eps,
        )

    def canonical_contract(self) -> dict[str, object]:
        return {
            "architecture": "canonical_equivariant_linear_attention",
            "public_options": (
                "input_irreps",
                "output_irreps",
                "width",
                "depth",
                "geometry",
            ),
            "spatial_policy": (
                "exact_global_linear_attention_plus_exact_sparse_short_range"
            ),
            "message_fusion": (
                "identity_initialized_rms_aware_global_local_router"
            ),
            "implicit_spatial": "experimental_not_canonical",
            "attention_residuals": "experimental_deep_stack_only",
            "coordinate_refinement": "external_wrapper",
            "derived_num_heads": self.num_heads,
            "derived_local_rank": self.local_rank,
            "internal_irreps": str(self.internal_irreps),
        }


class ELALayer(EquivariantLinearAttentionLayer):
    """Canonical ELA layer with branch-aware global/local fusion."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.branch_fusion = RMSAwareBranchFusion(
            scalar_width=self.scalar_width,
            eps=max(self.eps, 1e-6),
        )

    def _attention_branch(
        self,
        state: _ParityState,
        context: UnifiedSE3Context,
        modulation: _BranchModulation | None,
    ) -> _ParityState:
        normalized = self.attention_norm(state)
        if modulation is not None:
            normalized = _bounded_modulate_state(normalized, modulation)
        global_message = self._global_message(
            normalized,
            context.normalized_positions,
            context.batch,
            context.graph_layout,
        )
        local_message = self._local_message(normalized, context.geometry)
        routed_global, routed_local = self.branch_fusion(
            normalized,
            global_message,
            local_message,
        )
        candidate = _ParityCompleteBlock._update_state(
            self,
            normalized,
            routed_global,
            routed_local,
        )
        attention_delta = _state_subtract(candidate, normalized)
        if modulation is not None:
            attention_delta = _gate_delta(attention_delta, modulation)
        intermediate = _state_add(state, attention_delta)

        closure = self.tensor_closure(
            self.closure_norm(intermediate),
            context.multipoles,
        )
        closure_delta = _ParityState(
            even_scalar=self.closure_scalar_scale.to(
                dtype=state.even_scalar.dtype
            )
            * closure.even_scalar,
            odd_scalar=self.closure_odd_scale.to(
                dtype=state.odd_scalar.dtype
            )
            * _bounded_scalar(closure.odd_scalar, self.eps),
            polar_vector=self.closure_polar_scale.to(
                dtype=state.polar_vector.dtype
            )
            * _unit_ball(closure.polar_vector, self.eps),
            axial_vector=self.closure_axial_scale.to(
                dtype=state.axial_vector.dtype
            )
            * _unit_ball(closure.axial_vector, self.eps),
            even_tensor=self.closure_even_tensor_scale.to(
                dtype=state.even_tensor.dtype
            )
            * _bounded_st(closure.even_tensor, self.eps),
            odd_tensor=self.closure_odd_tensor_scale.to(
                dtype=state.odd_tensor.dtype
            )
            * _bounded_st(closure.odd_tensor, self.eps),
        )
        if modulation is not None:
            closure_delta = _gate_delta(closure_delta, modulation)
        total_delta = _state_add(attention_delta, closure_delta)
        total_delta = self._regularize(
            total_delta,
            normalized,
            activation=self.attention_activation,
            dropout=self.attention_dropout,
            drop_path=self.attention_drop_path,
            context=context,
        )
        return _state_add(state, total_delta)


class ELACore(LayeredCanonicalSE3Core):
    """Stack of canonical branch-aware ELA layers."""

    def __init__(
        self,
        config: EquivariantLinearAttentionConfig,
        *,
        condition_dim: int | None = None,
    ) -> None:
        # Reuse the admitted multipole core initialization, then replace its
        # block list with branch-aware ELA layers. Compatibility blocks are not
        # retained in the constructed model.
        CanonicalMultipoleSE3Core.__init__(
            self,
            input_irreps=config.input_layout,
            output_irreps=config.output_layout,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            local_rank=config.local_rank,
            local_cutoff=config.local_cutoff,
            num_rbf=config.num_rbf,
            num_node_roles=config.num_node_roles,
            num_edge_relations=config.num_edge_relations,
            relation_cutoffs=config.relation_cutoffs,
            residual_scale_init=config.residual_scale_init,
            eps=config.eps,
        )
        resolved_condition_dim = (
            config.condition_dim if condition_dim is None else condition_dim
        )
        if resolved_condition_dim < 0:
            raise ValueError("condition_dim must be nonnegative")
        self.condition_dim = int(resolved_condition_dim)
        self.coordinate_updates = False
        self.max_coordinate_step = 0.25
        block_scale = config.residual_scale_init / sqrt(
            max(1, config.num_layers)
        )
        self.blocks = nn.ModuleList(
            [
                ELALayer(
                    scalar_width=config.hidden_dim,
                    num_heads=config.num_heads,
                    local_rank=config.local_rank,
                    num_rbf=config.num_rbf,
                    num_edge_relations=config.num_edge_relations,
                    multipole_rank=self.multipole_rank,
                    residual_scale_init=block_scale,
                    condition_dim=self.condition_dim,
                    coordinate_updates=False,
                    residual_dropout=0.0,
                    drop_path_probability=0.0,
                    norm_eps=config.norm_eps,
                    eps=config.eps,
                )
                for _ in range(config.num_layers)
            ]
        )


class ELA(UnifiedEquivariantAttention):
    """Preferred minimal equivariant linear-attention model."""

    attention_kind = "canonical_equivariant_linear_attention"

    def __init__(self, config: ELAConfig) -> None:
        nn.Module.__init__(self)
        if not isinstance(config, ELAConfig):
            raise TypeError("config must be an ELAConfig")
        self.config = config
        self.advanced_config = config.to_advanced_config()
        self.core = ELACore(self.advanced_config)
        self._initialize_chiral_bridge()
        self.internal_irreps = self.core.internal_irreps
        self.output_irreps = self.core.output_irreps

    def extra_repr(self) -> str:
        return (
            f"input_irreps={self.config.input_layout}, "
            f"output_irreps={self.output_irreps}, width={self.config.width}, "
            f"depth={self.config.depth}, heads={self.config.num_heads}, "
            f"local_rank={self.config.local_rank}, "
            f"cutoff={self.config.geometry.cutoff}"
        )


CanonicalEquivariantLinearAttention = ELA


__all__ = [
    "CanonicalEquivariantLinearAttention",
    "ELA",
    "ELAConfig",
    "ELACore",
    "ELALayer",
    "SparseGeometry",
]
