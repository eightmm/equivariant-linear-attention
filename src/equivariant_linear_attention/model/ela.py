from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt

import torch
from torch import nn

from ..context import (
    ELAContext,
    ELAFeatures,
    FourierOrderEncoder,
    RefinementRequest,
)
from ..geometry.prepared import Prepared3DGraph, prepare_3d_graph
from ..irreps import IrrepLayout
from ..nn.core import CanonicalMultipoleSE3Core
from ..nn.fusion import RMSAwareBranchFusion
from ..nn.heads import EquivariantVectorHead
from ..nn.layers import (
    _ELAStackCore,
    _ELALayerContext,
    _ELAHiddenState,
    _BranchModulation,
    _LayerModulation,
    _gate_delta,
    _state_add,
    _state_subtract,
)
from ..nn.parity import (
    _ParityCompleteBlock,
    _ParityState,
    _bounded_scalar,
    _bounded_st,
    _unit_ball,
)
from .runtime import _ELARuntime
from .stack import (
    EquivariantLinearAttentionConfig,
    EquivariantLinearAttentionLayer,
    _bounded_modulate_state,
)


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
        return prepare_3d_graph(
            batch,
            edge_index,
            edge_relation_id=edge_relation_id,
            prefer_int32=prefer_int32,
        )


@dataclass(frozen=True, slots=True)
class ELAConfig:
    """Configuration of the single public equivariant linear-attention model.

    Optional semantic order, invariant conditioning, and coordinate refinement
    are allocated through ``features`` and switched per call with
    :class:`ELAContext`. Heads, local rank, hidden irreps, normalization, tensor
    closure, and chirality remain deterministic implementation choices.
    """

    input_irreps: str
    output_irreps: str = "1x0e"
    width: int = 128
    depth: int = 8
    geometry: SparseGeometry = field(default_factory=SparseGeometry)
    features: ELAFeatures = field(default_factory=ELAFeatures)

    def __post_init__(self) -> None:
        _positive_integer("width", self.width)
        _positive_integer("depth", self.depth)
        if self.width < 16:
            raise ValueError("width must be at least 16")
        if not isinstance(self.geometry, SparseGeometry):
            raise TypeError("geometry must be a SparseGeometry")
        if not isinstance(self.features, ELAFeatures):
            raise TypeError("features must be an ELAFeatures")
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
        return self.features.total_condition_dim(self.width)

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
            condition_dim=self.condition_dim,
            coordinate_updates=False,
            residual_dropout=0.0,
            drop_path_rate=0.0,
            norm_eps=self.norm_eps,
            eps=self.eps,
        )

    def canonical_contract(self) -> dict[str, object]:
        return {
            "architecture": "equivariant_linear_attention",
            "public_model": "ELA",
            "public_layer": "ELALayer",
            "public_options": (
                "input_irreps",
                "output_irreps",
                "width",
                "depth",
                "geometry",
                "features",
            ),
            "spatial_policy": (
                "exact_global_linear_attention_plus_exact_sparse_short_range"
            ),
            "message_fusion": (
                "identity_initialized_rms_aware_global_local_router"
            ),
            "optional_features": self.features.contract(self.width),
            "implicit_spatial": "not_in_canonical_architecture",
            "attention_residuals": "not_in_canonical_architecture",
            "derived_num_heads": self.num_heads,
            "derived_local_rank": self.local_rank,
            "internal_irreps": str(self.internal_irreps),
        }


class ELALayer(EquivariantLinearAttentionLayer):
    """Concrete stack-layer type constructed and exposed by :class:`ELA`."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.branch_fusion = RMSAwareBranchFusion(
            scalar_width=self.scalar_width,
            eps=max(self.eps, 1e-6),
        )

    def _resolve_modulation(
        self,
        condition: torch.Tensor | None,
        context: _ELALayerContext,
        state: _ParityState,
    ) -> _LayerModulation | None:
        # A configured capability is genuinely optional per call. In particular,
        # trained conditioner biases cannot leak into a context-free forward.
        if condition is None:
            return None
        return super()._resolve_modulation(condition, context, state)

    def _attention_branch(
        self,
        state: _ParityState,
        context: _ELALayerContext,
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


class _ELACore(_ELAStackCore):
    """Internal stack of the single public ELALayer implementation."""

    def __init__(self, config: EquivariantLinearAttentionConfig) -> None:
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
        self.condition_dim = int(config.condition_dim)
        self.coordinate_updates = False
        self.max_coordinate_step = 0.25
        block_scale = config.residual_scale_init / sqrt(max(1, config.num_layers))
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


class _ELAEngine(_ELARuntime):
    """Internal engine behind the single public ELA architecture.

    Optional features do not create alternative model classes. They are enabled
    by :class:`ELAFeatures` at construction and activated by fields present in
    :class:`ELAContext` at runtime.
    """

    attention_kind = "equivariant_linear_attention"

    def __init__(self, config: ELAConfig) -> None:
        nn.Module.__init__(self)
        if not isinstance(config, ELAConfig):
            raise TypeError("config must be an ELAConfig")
        self.config = config
        self.advanced_config = config.to_advanced_config()
        self.core = _ELACore(self.advanced_config)
        self._initialize_chiral_bridge()
        self.internal_irreps = self.core.internal_irreps
        self.output_irreps = self.core.output_irreps

        self.order_encoder: FourierOrderEncoder | None = None
        if config.features.order_dim:
            self.order_encoder = FourierOrderEncoder(
                coordinate_dim=config.features.order_dim,
                num_bands=config.features.order_bands(config.width),
                eps=config.norm_eps,
            )

        self.coordinate_head: EquivariantVectorHead | None = None
        self.coordinate_gate: nn.Linear | None = None
        if config.features.coordinate_refinement:
            self.coordinate_head = EquivariantVectorHead(
                config.width,
                config.num_heads,
                output_channels=1,
            )
            self.coordinate_gate = nn.Linear(config.width, 1)
            with torch.no_grad():
                self.coordinate_head.base_weight.zero_()
                final = self.coordinate_head.scalar_mixer[-1]
                if not isinstance(final, nn.Linear):
                    raise RuntimeError("unexpected coordinate-head output module")
                final.weight.zero_()
                final.bias.zero_()
                self.coordinate_gate.weight.zero_()
                self.coordinate_gate.bias.zero_()

    @staticmethod
    def _validate_finite(name: str, value: torch.Tensor) -> None:
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        finite = torch.isfinite(value).all()
        async_assert = getattr(torch, "_assert_async", None)
        if value.device.type == "cuda" and async_assert is not None:
            async_assert(finite, f"{name} must contain only finite values")
        elif not bool(finite):
            raise ValueError(f"{name} must contain only finite values")

    def _broadcast_condition(
        self,
        condition: torch.Tensor,
        graph: Prepared3DGraph,
    ) -> torch.Tensor:
        expected = self.config.features.condition_dim
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
        if condition.ndim != 2 or condition.shape[1] != expected:
            raise ValueError(
                f"condition must have shape (N, {expected}), (G, {expected}), "
                f"(1, {expected}), or ({expected},)"
            )
        if condition.device != graph.device:
            raise ValueError("condition and graph must share one device")
        self._validate_finite("condition", condition)
        if condition.shape[0] == graph.num_nodes:
            return condition
        if condition.shape[0] == graph.graph_layout.num_graphs:
            return condition[graph.batch]
        if condition.shape[0] == 1:
            return condition.expand(graph.num_nodes, -1)
        raise ValueError(
            "condition leading dimension must be one, the node count, or the graph count"
        )

    def encode_context(
        self,
        context: ELAContext | None,
        graph: Prepared3DGraph,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if context is None:
            return None
        if not isinstance(context, ELAContext):
            raise TypeError("context must be an ELAContext")
        features = self.config.features
        condition_active = context.condition is not None
        order_active = context.order is not None
        if condition_active and features.condition_dim == 0:
            raise ValueError(
                "condition was supplied but ELAFeatures.condition_dim is zero"
            )
        if order_active and features.order_dim == 0:
            raise ValueError("order was supplied but ELAFeatures.order_dim is zero")
        if not condition_active and not order_active:
            return None

        parts: list[torch.Tensor] = []
        if features.condition_dim:
            if context.condition is None:
                parts.append(
                    torch.zeros(
                        graph.num_nodes,
                        features.condition_dim,
                        device=graph.device,
                        dtype=dtype,
                    )
                )
            else:
                parts.append(
                    self._broadcast_condition(context.condition, graph).to(dtype=dtype)
                )
        if features.order_dim:
            if context.order is None:
                parts.append(
                    torch.zeros(
                        graph.num_nodes,
                        features.order_encoding_dim(self.config.width),
                        device=graph.device,
                        dtype=dtype,
                    )
                )
            else:
                if self.order_encoder is None:
                    raise RuntimeError("order encoder is not initialized")
                parts.append(self.order_encoder(context.order, graph).to(dtype=dtype))
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _selected_mask(
        request: RefinementRequest,
        *,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = request.update_mask
        if mask is None:
            return torch.ones(num_nodes, device=device, dtype=torch.bool)
        if not isinstance(mask, torch.Tensor):
            raise TypeError("update_mask must be a tensor")
        if mask.dtype != torch.bool:
            raise TypeError("update_mask must use torch.bool")
        if mask.shape != (num_nodes,):
            raise ValueError("update_mask must have shape (N,)")
        if mask.device != device:
            raise ValueError("update_mask and positions must share one device")
        return mask

    @staticmethod
    def _center_and_bound(
        raw: torch.Tensor,
        batch: torch.Tensor,
        selected: torch.Tensor,
        request: RefinementRequest,
    ) -> torch.Tensor:
        batch_index = batch.to(dtype=torch.long)
        if batch_index.numel() == 0:
            raise ValueError("coordinate refinement requires at least one node")
        num_graphs = int(batch_index.max().item()) + 1
        masked = torch.where(selected[:, None], raw, torch.zeros_like(raw))
        if request.centering == "none":
            centered = masked
        elif request.centering == "graph":
            graph_sum = raw.new_zeros((num_graphs, 3)).index_add(
                0, batch_index, raw
            )
            counts = torch.bincount(
                batch_index, minlength=num_graphs
            ).clamp_min(1)
            centered = raw - graph_sum[batch_index] / counts[
                batch_index, None
            ].to(dtype=raw.dtype)
            centered = torch.where(
                selected[:, None], centered, torch.zeros_like(centered)
            )
        else:
            selected_sum = raw.new_zeros((num_graphs, 3)).index_add(
                0, batch_index, masked
            )
            selected_count = torch.bincount(
                batch_index[selected], minlength=num_graphs
            ).clamp_min(1)
            centered = raw - selected_sum[batch_index] / selected_count[
                batch_index, None
            ].to(dtype=raw.dtype)
            centered = torch.where(
                selected[:, None], centered, torch.zeros_like(centered)
            )

        norms = torch.linalg.vector_norm(centered, dim=-1)
        graph_max = norms.new_zeros((num_graphs,))
        graph_max.scatter_reduce_(
            0,
            batch_index,
            norms,
            reduce="amax",
            include_self=True,
        )
        scale = (
            float(request.max_step)
            / graph_max.clamp_min(float(request.max_step))
        ).clamp(max=1.0)
        return centered * scale[batch_index, None]

    def _coordinate_delta(
        self,
        state: _ELAHiddenState,
        graph: Prepared3DGraph,
        request: RefinementRequest,
    ) -> torch.Tensor:
        if self.coordinate_head is None or self.coordinate_gate is None:
            raise RuntimeError("coordinate refinement is not enabled in ELAFeatures")
        selected = self._selected_mask(
            request,
            num_nodes=graph.num_nodes,
            device=graph.device,
        )
        raw = self.coordinate_head(
            state.even_scalar,
            state.polar_vector,
        ).squeeze(1)
        raw = torch.sigmoid(self.coordinate_gate(state.even_scalar)) * raw
        return self._center_and_bound(raw, graph.batch, selected, request)

    def forward_features(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
        *,
        context: ELAContext | None = None,
    ) -> tuple[_ELAHiddenState, torch.Tensor, torch.Tensor]:
        condition = self.encode_context(
            context,
            graph,
            dtype=node_irreps.dtype,
        )
        return super().forward_features(
            node_irreps,
            pos,
            graph,
            condition=condition,
        )

    def forward(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
        *,
        context: ELAContext | None = None,
    ) -> dict[str, torch.Tensor]:
        refinement = None if context is None else context.refinement
        if refinement is None:
            condition = self.encode_context(
                context,
                graph,
                dtype=node_irreps.dtype,
            )
            return super().forward(
                node_irreps,
                pos,
                graph,
                condition=condition,
            )
        if not self.config.features.coordinate_refinement:
            raise ValueError(
                "refinement was requested but ELAFeatures.coordinate_refinement is false"
            )

        current_positions = pos
        current_graph = graph
        total_delta = torch.zeros_like(pos)
        for _ in range(refinement.steps):
            condition = self.encode_context(
                context,
                current_graph,
                dtype=node_irreps.dtype,
            )
            state, _, _ = super().forward_features(
                node_irreps,
                current_positions,
                current_graph,
                condition=condition,
            )
            delta = self._coordinate_delta(
                state,
                current_graph,
                refinement,
            ).to(dtype=current_positions.dtype)
            current_positions = current_positions + delta
            total_delta = total_delta + delta
            if refinement.graph_rebuilder is not None:
                rebuilt = refinement.graph_rebuilder(
                    current_positions,
                    current_graph.batch,
                )
                if not isinstance(rebuilt, Prepared3DGraph):
                    raise TypeError("graph_rebuilder must return Prepared3DGraph")
                if rebuilt.num_nodes != current_graph.num_nodes:
                    raise ValueError("rebuilt graph must preserve node count")
                if rebuilt.device != current_graph.device:
                    raise ValueError("rebuilt graph must preserve device")
                if not torch.equal(rebuilt.batch, current_graph.batch):
                    raise ValueError("rebuilt graph must preserve graph membership")
                current_graph = rebuilt

        condition = self.encode_context(
            context,
            current_graph,
            dtype=node_irreps.dtype,
        )
        output = dict(
            super().forward(
                node_irreps,
                current_positions,
                current_graph,
                condition=condition,
            )
        )
        output["positions"] = current_positions
        output["coordinate_delta"] = total_delta
        return output

    def extra_repr(self) -> str:
        return (
            f"input_irreps={self.config.input_layout}, "
            f"output_irreps={self.output_irreps}, width={self.config.width}, "
            f"depth={self.config.depth}, heads={self.config.num_heads}, "
            f"local_rank={self.config.local_rank}, "
            f"cutoff={self.config.geometry.cutoff}, "
            f"features={self.config.features.contract(self.config.width)}"
        )


__all__ = ["ELAConfig", "ELALayer", "SparseGeometry"]
