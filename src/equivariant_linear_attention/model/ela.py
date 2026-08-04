from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt

import torch
from torch import nn

from ..context import ELAContext, ELAFeatures, FourierOrderEncoder
from ..geometry.prepared import (
    PreparationSpec,
    Prepared3DGraph,
    _prepare_radius_3d_graph,
    prepare_3d_graph,
)
from ..irreps import IrrepLayout, split_irreps
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
    """Sparse topology and radial geometry policy.

    ``max_neighbors`` belongs to the model configuration rather than an
    incidental preparation call, making topology reproducible. ``skin`` builds
    a Verlet candidate shell so moving-coordinate workloads can safely reuse a
    prepared graph until any node moves more than half the skin distance.
    """

    cutoff: float = 5.0
    num_rbf: int = 16
    max_neighbors: int | None = None
    skin: float = 0.0
    relation_cutoffs: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        cutoff = _positive_real("cutoff", self.cutoff)
        _positive_integer("num_rbf", self.num_rbf)
        if self.max_neighbors is not None:
            _positive_integer("max_neighbors", self.max_neighbors)
        if isinstance(self.skin, bool) or not isinstance(self.skin, (int, float)):
            raise TypeError("skin must be a real number")
        if not isfinite(float(self.skin)) or float(self.skin) < 0.0:
            raise ValueError("skin must be finite and nonnegative")
        if self.max_neighbors is not None and float(self.skin) > 0.0:
            raise ValueError(
                "skin and max_neighbors cannot be combined: a cached capped "
                "candidate list cannot preserve exact nearest-neighbor membership"
            )
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
        spec: PreparationSpec | None = None,
    ) -> Prepared3DGraph:
        return prepare_3d_graph(
            batch,
            edge_index,
            edge_relation_id=edge_relation_id,
            prefer_int32=prefer_int32,
            spec=spec,
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
    coordinate_updates: int = 0
    max_coordinate_step: float = 0.25

    def __post_init__(self) -> None:
        _positive_integer("width", self.width)
        _positive_integer("depth", self.depth)
        if self.width < 16:
            raise ValueError("width must be at least 16")
        if not isinstance(self.geometry, SparseGeometry):
            raise TypeError("geometry must be a SparseGeometry")
        if not isinstance(self.features, ELAFeatures):
            raise TypeError("features must be an ELAFeatures")
        if isinstance(self.coordinate_updates, bool) or not isinstance(
            self.coordinate_updates, int
        ):
            raise TypeError("coordinate_updates must be an integer")
        if self.coordinate_updates < 0:
            raise ValueError("coordinate_updates must be nonnegative")
        if self.coordinate_updates > self.depth:
            raise ValueError(
                "coordinate_updates cannot exceed depth: updates occur at "
                "distinct layer boundaries"
            )
        _positive_real("max_coordinate_step", self.max_coordinate_step)
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
        # Preserve the established carrier through width=128 while allowing
        # geometric capacity to scale in larger models instead of saturating.
        target = max(1, min(16, self.width // 16))
        for candidate in range(target, 0, -1):
            if self.width % candidate == 0:
                return candidate
        return 1

    @property
    def local_rank(self) -> int:
        if self.num_heads <= 8:
            return min(4, self.num_heads)
        return min(8, self.num_heads)

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
    def coordinate_update_layers(self) -> tuple[int, ...]:
        """One-based layer boundaries that perform coordinate updates.

        Advanced configurations may request fewer coordinate updates than
        layers.  Those updates are spread deterministically through the stack
        and always include the final layer.  The public ``update_positions``
        switch requests ``depth`` updates, hence one update per layer.
        """

        updates = self.coordinate_updates
        if updates == 0:
            return ()
        return tuple(step * self.depth // updates for step in range(1, updates + 1))

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
            "public_graph": "ELAGraph",
            "public_contract": "ELAGraph -> ELA -> ELAGraph",
            "inspectable_layer": "ELALayer",
            "public_options": (
                "input_irreps",
                "output_irreps",
                "width",
                "depth",
                "geometry",
                "features",
                "coordinate_updates",
                "max_coordinate_step",
            ),
            "spatial_policy": (
                "exact_global_linear_attention_plus_exact_sparse_short_range"
            ),
            "message_fusion": "fixed_exact_global_plus_local_sum",
            "optional_features": self.features.contract(self.width),
            "coordinate_updates": self.coordinate_updates,
            "coordinate_update_layers": self.coordinate_update_layers,
            "max_coordinate_step": self.max_coordinate_step,
            "implicit_spatial": "not_in_canonical_architecture",
            "attention_residuals": "not_in_canonical_architecture",
            "derived_num_heads": self.num_heads,
            "derived_local_rank": self.local_rank,
            "internal_irreps": str(self.internal_irreps),
        }


class ELALayer(EquivariantLinearAttentionLayer):
    """Concrete canonical stack layer.

    Global and local exact messages are combined by the architecture's fixed
    additive update. The previously tested learned branch router remains an
    experimental module but is no longer part of canonical ELA parameters or
    execution.
    """

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
        candidate = _ParityCompleteBlock._update_state(
            self,
            normalized,
            global_message,
            local_message,
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
            even_scalar=self.closure_scalar_scale.to(dtype=state.even_scalar.dtype)
            * closure.even_scalar,
            odd_scalar=self.closure_odd_scale.to(dtype=state.odd_scalar.dtype)
            * _bounded_scalar(closure.odd_scalar, self.eps),
            polar_vector=self.closure_polar_scale.to(dtype=state.polar_vector.dtype)
            * _unit_ball(closure.polar_vector, self.eps),
            axial_vector=self.closure_axial_scale.to(dtype=state.axial_vector.dtype)
            * _unit_ball(closure.axial_vector, self.eps),
            even_tensor=self.closure_even_tensor_scale.to(dtype=state.even_tensor.dtype)
            * _bounded_st(closure.even_tensor, self.eps),
            odd_tensor=self.closure_odd_tensor_scale.to(dtype=state.odd_tensor.dtype)
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
    """One composition-oriented stack of the canonical :class:`ELALayer`."""

    def __init__(self, config: EquivariantLinearAttentionConfig) -> None:
        super().__init__(
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
            condition_dim=config.condition_dim,
            coordinate_updates=False,
            max_coordinate_step=config.max_coordinate_step,
            eps=config.eps,
        )
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
                    condition_dim=config.condition_dim,
                    coordinate_updates=False,
                    residual_dropout=0.0,
                    drop_path_probability=0.0,
                    norm_eps=config.norm_eps,
                    eps=config.eps,
                )
                for _ in range(config.num_layers)
            ]
        )


class _ELAEngine(nn.Module):
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
        if config.coordinate_updates > 0:
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

    @property
    def layers(self) -> nn.ModuleList:
        return self.core.layers

    def _initialize_chiral_bridge(self) -> None:
        with torch.no_grad():
            for layer in self.core.layers:
                weight = layer.local_chiral_scalar_out.weight
                weight.zero_()
                rows = torch.arange(weight.shape[0], device=weight.device)
                weight[rows, rows.remainder(weight.shape[1])] = 1.0

    def _validate_graph_inputs(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
    ) -> None:
        if not isinstance(graph, Prepared3DGraph) or not graph._validated:
            raise TypeError("graph must be a validated Prepared3DGraph")
        if node_irreps.shape[0] != graph.num_nodes:
            raise ValueError("node_irreps node count must match graph")
        if pos.shape != (graph.num_nodes, 3):
            raise ValueError("pos must have shape (N, 3)")
        if node_irreps.device != graph.device or pos.device != graph.device:
            raise ValueError("model inputs and graph must share one device")

    def split_input(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        return split_irreps(self.config.input_layout, value)

    def split_output(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        return split_irreps(self.output_irreps, value)

    def _prepare_context(
        self,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
    ) -> _ELALayerContext:
        dummy = pos.new_zeros((graph.num_nodes, self.config.input_layout.dim))
        self._validate_graph_inputs(dummy, pos, graph)
        return self.core.prepare_context(
            pos,
            graph.batch,
            graph.graph_layout,
            graph.neighbors,
        )

    def _embed_input(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
    ) -> tuple[_ELAHiddenState, _ELALayerContext]:
        self._validate_graph_inputs(node_irreps, pos, graph)
        context = self.core.prepare_context(
            pos,
            graph.batch,
            graph.graph_layout,
            graph.neighbors,
        )
        return self.core.embed_input(node_irreps, context), context

    def _project_state(self, state: _ELAHiddenState) -> torch.Tensor:
        return self.core.project_state(state)

    def _core_forward_features(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
        *,
        condition: torch.Tensor | None,
    ) -> tuple[_ELAHiddenState, torch.Tensor, torch.Tensor]:
        self._validate_graph_inputs(node_irreps, pos, graph)
        return self.core.forward_features(
            node_irreps,
            pos,
            graph.batch,
            graph.graph_layout,
            graph.neighbors,
            condition=condition,
        )

    def _core_forward(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
        *,
        condition: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        self._validate_graph_inputs(node_irreps, pos, graph)
        return self.core(
            node_irreps,
            pos,
            graph.batch,
            graph.graph_layout,
            graph.neighbors,
            condition=condition,
        )

    def _static_forward_unchecked(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
        *,
        condition: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Tensor hot path after the public boundary validated the graph.

        This is intentionally private: callers must first establish the full
        ``ELAGraph -> ELABatch -> Prepared3DGraph`` contract.  Keeping those
        checks out of this method lets ``prepare_for_inference`` compile only
        stable numerical work rather than tracing cache/provenance branches.
        """

        if self.config.coordinate_updates:
            raise RuntimeError("static numerical core cannot update positions")
        layer_context = self.core.prepare_context(
            pos,
            graph.batch,
            graph.graph_layout,
            graph.neighbors,
        )
        state = self.core.embed_input(node_irreps, layer_context)
        for layer in self.layers:
            state = layer(state, layer_context, condition).state
        return {
            "node_irreps": self._project_state(state),
            "positions": pos,
            "coordinate_delta": torch.zeros_like(pos),
        }

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

    def _encode_context(
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
        update_mask: torch.Tensor | None,
        *,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        if update_mask is None:
            return torch.ones(num_nodes, device=device, dtype=torch.bool)
        if not isinstance(update_mask, torch.Tensor):
            raise TypeError("update_mask must be a tensor")
        if update_mask.dtype != torch.bool:
            raise TypeError("update_mask must use torch.bool")
        if update_mask.shape != (num_nodes,):
            raise ValueError("update_mask must have shape (N,)")
        if update_mask.device != device:
            raise ValueError("update_mask and positions must share one device")
        return update_mask

    @staticmethod
    def _center_and_bound(
        raw: torch.Tensor,
        batch: torch.Tensor,
        selected: torch.Tensor,
        *,
        max_step: float,
    ) -> torch.Tensor:
        """Center selected updates per graph and bound the graphwise max step."""

        batch_index = batch.to(dtype=torch.long)
        if batch_index.numel() == 0:
            raise ValueError("coordinate updates require at least one node")
        num_graphs = int(batch_index.max().item()) + 1
        masked = torch.where(selected[:, None], raw, torch.zeros_like(raw))
        selected_sum = raw.new_zeros((num_graphs, 3)).index_add(
            0,
            batch_index,
            masked,
        )
        selected_count = torch.bincount(
            batch_index[selected],
            minlength=num_graphs,
        ).clamp_min(1)
        centered = raw - selected_sum[batch_index] / selected_count[
            batch_index,
            None,
        ].to(dtype=raw.dtype)
        centered = torch.where(
            selected[:, None],
            centered,
            torch.zeros_like(centered),
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
        scale = (max_step / graph_max.clamp_min(max_step)).clamp(max=1.0)
        return centered * scale[batch_index, None]

    def _coordinate_delta(
        self,
        state: _ELAHiddenState,
        graph: Prepared3DGraph,
        update_mask: torch.Tensor | None,
        *,
        max_step: float | None = None,
    ) -> torch.Tensor:
        if self.coordinate_head is None or self.coordinate_gate is None:
            raise RuntimeError("coordinate updates are not enabled on this model")
        selected = self._selected_mask(
            update_mask,
            num_nodes=graph.num_nodes,
            device=graph.device,
        )
        raw = self.coordinate_head(
            state.even_scalar,
            state.polar_vector,
        ).squeeze(1)
        raw = torch.sigmoid(self.coordinate_gate(state.even_scalar)) * raw
        return self._center_and_bound(
            raw,
            graph.batch,
            selected,
            max_step=(
                float(self.config.max_coordinate_step)
                if max_step is None
                else float(max_step)
            ),
        )

    def _refresh_coordinate_graph(
        self,
        positions: torch.Tensor,
        graph: Prepared3DGraph,
    ) -> Prepared3DGraph:
        if graph.spec.source == "explicit" or graph.spec.can_reuse_positions(positions):
            return graph

        relation_count = self.config.geometry.num_edge_relations
        if relation_count > 1:
            raise ValueError(
                "automatic radius updates cannot infer multiple semantic relations"
            )
        graph_counts = (
            graph.graph_layout.graph_counts if graph.graph_layout.is_grouped else None
        )
        return _prepare_radius_3d_graph(
            positions,
            graph.batch,
            cutoff=self.config.geometry.cutoff,
            max_neighbors=graph.spec.max_neighbors,
            include_self=False,
            num_edge_relations=relation_count,
            skin=self.config.geometry.skin,
            graph_counts=graph_counts,
        )

    def _forward_features(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
        *,
        context: ELAContext | None = None,
    ) -> tuple[_ELAHiddenState, torch.Tensor, torch.Tensor]:
        condition = self._encode_context(
            context,
            graph,
            dtype=node_irreps.dtype,
        )
        if self.config.coordinate_updates > 0:
            state, positions, total_delta, _ = self._stagewise_forward_features(
                node_irreps,
                pos,
                graph,
                condition=condition,
                update_mask=None,
            )
            return state, positions, total_delta
        return self._core_forward_features(
            node_irreps,
            pos,
            graph,
            condition=condition,
        )

    def _stagewise_forward_features(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
        *,
        condition: torch.Tensor | None,
        update_mask: torch.Tensor | None,
    ) -> tuple[_ELAHiddenState, torch.Tensor, torch.Tensor, Prepared3DGraph]:
        """Run each canonical layer once while carrying state across geometry updates."""

        self._validate_graph_inputs(node_irreps, pos, graph)
        self.core._validate_inputs(
            node_irreps,
            pos,
            graph.batch,
            graph.graph_layout,
            graph.neighbors,
            node_role_id=None,
        )
        current_positions = pos
        current_graph = graph
        layer_context = self.core.prepare_context(
            current_positions,
            current_graph.batch,
            current_graph.graph_layout,
            current_graph.neighbors,
        )
        state = self.core.embed_input(node_irreps, layer_context)
        total_delta = torch.zeros_like(pos)
        update_layers = frozenset(self.config.coordinate_update_layers)
        stage_max_step = float(self.config.max_coordinate_step) / len(update_layers)

        for layer_number, layer in enumerate(self.layers, start=1):
            layer_output = layer(state, layer_context, condition)
            state = layer_output.state
            if layer_number not in update_layers:
                continue

            delta = self._coordinate_delta(
                state,
                current_graph,
                update_mask,
                max_step=stage_max_step,
            ).to(dtype=current_positions.dtype)
            current_positions = current_positions + delta
            total_delta = total_delta + delta

            # A final displacement affects the returned geometry but no later
            # layer consumes it, so rebuilding there would be pure overhead.
            if layer_number == self.config.depth:
                continue
            current_graph = self._refresh_coordinate_graph(
                current_positions,
                current_graph,
            )
            layer_context = self.core.prepare_context(
                current_positions,
                current_graph.batch,
                current_graph.graph_layout,
                current_graph.neighbors,
            )

        return state, current_positions, total_delta, current_graph

    def _forward_engine(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
        *,
        context: ELAContext | None = None,
        update_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.config.coordinate_updates == 0:
            if update_mask is not None:
                raise ValueError(
                    "update_mask was supplied but update_positions=False on the model"
                )
            condition = self._encode_context(
                context,
                graph,
                dtype=node_irreps.dtype,
            )
            return self._core_forward(
                node_irreps,
                pos,
                graph,
                condition=condition,
            )

        condition = self._encode_context(
            context,
            graph,
            dtype=node_irreps.dtype,
        )
        state, current_positions, total_delta, current_graph = (
            self._stagewise_forward_features(
                node_irreps,
                pos,
                graph,
                condition=condition,
                update_mask=update_mask,
            )
        )
        node_output = self._project_state(state)
        graph_sum = node_output.new_zeros(
            (current_graph.graph_layout.num_graphs, node_output.shape[-1])
        )
        graph_sum.index_add_(
            0,
            current_graph.batch.to(dtype=torch.long),
            node_output,
        )
        graph_output = graph_sum / current_graph.graph_layout.graph_counts.to(
            dtype=node_output.dtype
        ).clamp_min(1.0).unsqueeze(-1)
        return {
            "node_irreps": node_output,
            "graph_irreps": graph_output,
            "positions": current_positions,
            "coordinate_delta": total_delta,
        }

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
