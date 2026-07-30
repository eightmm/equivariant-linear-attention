from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, sqrt
from collections.abc import Sequence

import torch
from torch import nn

from .equivariant_linear_attention import (
    EquivariantLinearAttention,
    EquivariantLinearAttentionConfig,
    EquivariantLinearAttentionCore,
    EquivariantLinearAttentionLayer,
)
from .graph_layout import PackedGraphLayout
from .irreps import IrrepLayout
from .layered_se3 import UnifiedSE3Context, UnifiedSE3State
from .neighbors import PackedNeighborGraph
from .parity_se3 import _ParityState, _state_add, _state_subtract, _st_square


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
class EquivariantAttentionResidualConfig(EquivariantLinearAttentionConfig):
    """Equivariant linear attention with Moonshot-style depth attention.

    ``attention_residual_blocks=1`` is an exact single-source identity path.
    Values greater than one enable block-level depth attention. The depth router
    uses one invariant softmax weight per node and source and applies that same
    weight to every irrep sector, matching the single-head conclusion of the
    Attention Residuals ablation while preserving equivariance.
    """

    attention_residual_blocks: int = 1
    attention_residual_eps: float = 1e-6

    def __post_init__(self) -> None:
        EquivariantLinearAttentionConfig.__post_init__(self)
        blocks = _positive_integer(
            "attention_residual_blocks",
            self.attention_residual_blocks,
        )
        if blocks > self.num_layers:
            raise ValueError(
                "attention_residual_blocks must not exceed num_layers"
            )
        _positive_real("attention_residual_eps", self.attention_residual_eps)

    def canonical_contract(self) -> dict[str, object]:
        contract = EquivariantLinearAttentionConfig.canonical_contract(self)
        contract.update(
            {
                "depth_residual": "equivariant_block_attention_residual",
                "depth_residual_blocks": self.attention_residual_blocks,
                "depth_key": "rms_normalized_invariant_irrep_descriptor",
                "depth_query": "one_zero_initialized_pseudo_query_per_sublayer",
                "depth_mixing": "single_softmax_weight_shared_by_all_irreps",
            }
        )
        return contract


class EquivariantBlockAttentionResidual(nn.Module):
    """Depth-wise softmax attention over parity-complete hidden states.

    A learned pseudo-query scores invariant RMS-normalized descriptors of earlier
    states. The resulting scalar weight is shared by all channels, components,
    and parity sectors. Consequently, the weighted sum commutes with every
    tracked O(3) action and remains SE(3)-equivariant when used with a separate
    affine coordinate stream.
    """

    def __init__(self, *, scalar_width: int, num_heads: int, eps: float) -> None:
        super().__init__()
        if scalar_width <= 0 or num_heads <= 0:
            raise ValueError("scalar_width and num_heads must be positive")
        self.scalar_width = scalar_width
        self.num_heads = num_heads
        self.eps = max(float(eps), 1e-8)
        self.descriptor_dim = scalar_width + 5 * num_heads
        self.pseudo_query = nn.Parameter(torch.zeros(self.descriptor_dim))

    @staticmethod
    def _work_dtype(value: torch.Tensor) -> torch.dtype:
        return torch.float64 if value.dtype == torch.float64 else torch.float32

    def _descriptor(self, state: _ParityState) -> torch.Tensor:
        dtype = self._work_dtype(state.even_scalar)
        even = state.even_scalar.to(dtype=dtype)
        odd = state.odd_scalar.to(dtype=dtype)
        polar = state.polar_vector.to(dtype=dtype)
        axial = state.axial_vector.to(dtype=dtype)
        even_tensor = state.even_tensor.to(dtype=dtype)
        odd_tensor = state.odd_tensor.to(dtype=dtype)

        descriptor = torch.cat(
            [
                even,
                torch.sqrt(odd.square() + self.eps),
                torch.sqrt(polar.square().mean(dim=-1) + self.eps),
                torch.sqrt(axial.square().mean(dim=-1) + self.eps),
                torch.sqrt(_st_square(even_tensor) / 5.0 + self.eps),
                torch.sqrt(_st_square(odd_tensor) / 5.0 + self.eps),
            ],
            dim=-1,
        )
        rms = torch.sqrt(descriptor.square().mean(dim=-1, keepdim=True) + self.eps)
        return descriptor / rms

    def routing_weights(
        self,
        sources: Sequence[_ParityState],
    ) -> torch.Tensor:
        if not sources:
            raise ValueError("attention residual requires at least one source")
        if len(sources) == 1:
            return sources[0].even_scalar.new_ones(
                (1, sources[0].even_scalar.shape[0])
            )
        descriptors = torch.stack([self._descriptor(source) for source in sources])
        query = self.pseudo_query.to(
            device=descriptors.device,
            dtype=descriptors.dtype,
        )
        logits = torch.einsum("d,snd->sn", query, descriptors) / sqrt(
            self.descriptor_dim
        )
        return torch.softmax(logits, dim=0).to(
            dtype=sources[0].even_scalar.dtype
        )

    def forward(self, sources: Sequence[_ParityState]) -> _ParityState:
        weights = self.routing_weights(sources)
        if len(sources) == 1:
            return sources[0]

        def mix(name: str) -> torch.Tensor:
            stacked = torch.stack([getattr(source, name) for source in sources])
            scale = weights.reshape(
                weights.shape[0],
                weights.shape[1],
                *((1,) * (stacked.ndim - 2)),
            )
            return (scale * stacked).sum(dim=0)

        return _ParityState(
            even_scalar=mix("even_scalar"),
            odd_scalar=mix("odd_scalar"),
            polar_vector=mix("polar_vector"),
            axial_vector=mix("axial_vector"),
            even_tensor=mix("even_tensor"),
            odd_tensor=mix("odd_tensor"),
        )


class EquivariantAttentionResidualLayer(EquivariantLinearAttentionLayer):
    """One linear-attention layer with separate attention/FFN depth routers."""

    def __init__(self, *, attention_residual_eps: float = 1e-6, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.attention_depth_router = EquivariantBlockAttentionResidual(
            scalar_width=self.scalar_width,
            num_heads=self.num_heads,
            eps=attention_residual_eps,
        )
        self.ffn_depth_router = EquivariantBlockAttentionResidual(
            scalar_width=self.scalar_width,
            num_heads=self.num_heads,
            eps=attention_residual_eps,
        )

    def route_attention(
        self,
        sources: Sequence[UnifiedSE3State],
    ) -> UnifiedSE3State:
        routed = self.attention_depth_router(
            tuple(source._to_internal() for source in sources)
        )
        return UnifiedSE3State._from_internal(routed)

    def route_ffn(
        self,
        sources: Sequence[UnifiedSE3State],
    ) -> UnifiedSE3State:
        routed = self.ffn_depth_router(
            tuple(source._to_internal() for source in sources)
        )
        return UnifiedSE3State._from_internal(routed)


class EquivariantAttentionResidualCore(EquivariantLinearAttentionCore):
    """Block Attention Residuals adapted to equivariant linear attention."""

    def __init__(
        self,
        *,
        attention_residual_blocks: int,
        attention_residual_eps: float,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.attention_residual_blocks = min(
            _positive_integer(
                "attention_residual_blocks",
                attention_residual_blocks,
            ),
            self.num_layers,
        )
        block_scale = kwargs.get("residual_scale_init", 0.1)
        block_scale = float(block_scale) / sqrt(max(1, self.num_layers))
        layers: list[nn.Module] = []
        for index in range(self.num_layers):
            probability = float(kwargs.get("drop_path_rate", 0.0)) * index / max(
                1,
                self.num_layers - 1,
            )
            layers.append(
                EquivariantAttentionResidualLayer(
                    scalar_width=self.hidden_dim,
                    num_heads=self.num_heads,
                    local_rank=self.local_rank,
                    num_rbf=self.num_rbf,
                    num_edge_relations=self.num_edge_relations,
                    multipole_rank=self.multipole_rank,
                    residual_scale_init=block_scale,
                    condition_dim=int(kwargs.get("condition_dim", 0)),
                    coordinate_updates=bool(kwargs.get("coordinate_updates", False)),
                    max_coordinate_step=float(
                        kwargs.get("max_coordinate_step", 0.25)
                    ),
                    residual_dropout=float(kwargs.get("residual_dropout", 0.0)),
                    drop_path_probability=probability,
                    norm_eps=float(kwargs.get("norm_eps", 1e-6)),
                    attention_residual_eps=attention_residual_eps,
                    eps=float(kwargs.get("eps", 1e-12)),
                )
            )
        self.blocks = nn.ModuleList(layers)

    @staticmethod
    def _sources(
        completed: Sequence[UnifiedSE3State],
        partial: UnifiedSE3State | None,
    ) -> tuple[UnifiedSE3State, ...]:
        if partial is None:
            return tuple(completed)
        return (*completed, partial)

    def forward_features(
        self,
        node_irreps: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        graph_layout: PackedGraphLayout,
        neighbors: PackedNeighborGraph,
        *,
        node_role_id: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[UnifiedSE3State, torch.Tensor, torch.Tensor]:
        self._validate_inputs(
            node_irreps,
            positions,
            batch,
            graph_layout,
            neighbors,
            node_role_id=node_role_id,
        )
        context = self.prepare_context(positions, batch, graph_layout, neighbors)
        embedding = self.embed_input(
            node_irreps,
            context,
            node_role_id=node_role_id,
        )

        completed: list[UnifiedSE3State] = [embedding]
        partial: UnifiedSE3State | None = None
        layers_per_block = ceil(self.num_layers / self.attention_residual_blocks)
        current_positions = positions
        total_delta = positions.new_zeros(positions.shape)

        for index, layer_module in enumerate(self.blocks):
            layer = layer_module
            if not isinstance(layer, EquivariantAttentionResidualLayer):
                raise RuntimeError("unexpected attention-residual layer type")

            if index and index % layers_per_block == 0:
                if partial is None:
                    raise RuntimeError("completed block has no partial state")
                completed.append(partial)
                completed = completed[-self.attention_residual_blocks :]
                partial = None

            attention_input = layer.route_attention(
                self._sources(completed, partial)
            )
            attention_internal = attention_input._to_internal()
            modulation = layer._resolve_modulation(
                condition,
                context,
                attention_internal,
            )
            attention_output = layer._attention_branch(
                attention_internal,
                context,
                None if modulation is None else modulation.attention,
            )
            attention_delta = _state_subtract(
                attention_output,
                attention_internal,
            )
            partial_internal = (
                attention_delta
                if partial is None
                else _state_add(partial._to_internal(), attention_delta)
            )
            partial = UnifiedSE3State._from_internal(partial_internal)

            ffn_input = layer.route_ffn(self._sources(completed, partial))
            ffn_internal = ffn_input._to_internal()
            ffn_output = layer._ffn_branch(
                ffn_internal,
                context,
                None if modulation is None else modulation.ffn,
            )
            ffn_delta = _state_subtract(ffn_output, ffn_internal)
            partial = UnifiedSE3State._from_internal(
                _state_add(partial._to_internal(), ffn_delta)
            )

            coordinate_delta = layer._coordinate_delta_internal(
                partial._to_internal(),
                modulation,
            )
            current_positions = context.positions + coordinate_delta
            total_delta = total_delta + coordinate_delta.to(dtype=total_delta.dtype)

            if self.coordinate_updates and index + 1 < len(self.blocks):
                context = self.prepare_context(
                    current_positions,
                    batch,
                    graph_layout,
                    neighbors,
                )
            elif not self.coordinate_updates:
                current_positions = positions

        if partial is None:
            raise RuntimeError("attention-residual stack produced no state")
        return partial, current_positions, total_delta


class EquivariantAttentionResiduals(EquivariantLinearAttention):
    """Preferred opt-in block AttnRes variant of equivariant linear attention."""

    attention_kind = "equivariant_linear_attention_with_depth_attention_residuals"

    def __init__(self, config: EquivariantAttentionResidualConfig) -> None:
        nn.Module.__init__(self)
        if not isinstance(config, EquivariantAttentionResidualConfig):
            raise TypeError("config must be an EquivariantAttentionResidualConfig")
        self.config = config
        self.core = EquivariantAttentionResidualCore(
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
            coordinate_updates=config.coordinate_updates,
            max_coordinate_step=config.max_coordinate_step,
            residual_dropout=config.residual_dropout,
            drop_path_rate=config.drop_path_rate,
            norm_eps=config.norm_eps,
            eps=config.eps,
            attention_residual_blocks=config.attention_residual_blocks,
            attention_residual_eps=config.attention_residual_eps,
        )
        self._initialize_chiral_bridge()
        self.internal_irreps = self.core.internal_irreps
        self.output_irreps = self.core.output_irreps


__all__ = [
    "EquivariantAttentionResidualConfig",
    "EquivariantAttentionResidualCore",
    "EquivariantAttentionResidualLayer",
    "EquivariantAttentionResiduals",
    "EquivariantBlockAttentionResidual",
]
