from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import torch
from torch import nn

from .equivariant_linear_attention import (
    EquivariantLinearAttention,
    EquivariantLinearAttentionConfig,
)
from .implicit_spatial import (
    ImplicitGaussianSpatialKernel,
    ImplicitSpatialKernelConfig,
)
from .implicit_spatial_residual import ImplicitSpatialResidual
from .unified import Prepared3DGraph, prepare_3d_graph


SpatialOperatorArm = Literal["explicit", "implicit", "hybrid"]


@dataclass(frozen=True, slots=True)
class SpatialOperatorAblationConfig:
    """Resource-matched comparison of three spatial-operator arms.

    All arms contain the same modules and therefore have the same parameter
    schema. The arm changes only execution:

    - ``explicit``: exact global linear attention + explicit sparse local;
    - ``implicit``: exact global linear attention + edge-free implicit residual;
    - ``hybrid``: exact global linear attention + both spatial residuals.

    Coordinate updates are intentionally excluded from this comparison layer so
    every arm sees one frozen geometry and the operator attribution is clean.
    """

    model: EquivariantLinearAttentionConfig
    implicit: ImplicitSpatialKernelConfig = ImplicitSpatialKernelConfig()
    implicit_residual_scale_init: float = 0.0
    implicit_every: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.model, EquivariantLinearAttentionConfig):
            raise TypeError("model must be an EquivariantLinearAttentionConfig")
        if not isinstance(self.implicit, ImplicitSpatialKernelConfig):
            raise TypeError("implicit must be an ImplicitSpatialKernelConfig")
        if self.model.coordinate_updates:
            raise ValueError(
                "spatial operator ablation requires coordinate_updates=False"
            )
        if self.implicit_residual_scale_init < 0.0:
            raise ValueError("implicit_residual_scale_init must be nonnegative")
        if isinstance(self.implicit_every, bool) or not isinstance(
            self.implicit_every,
            int,
        ):
            raise TypeError("implicit_every must be an integer")
        if self.implicit_every <= 0:
            raise ValueError("implicit_every must be positive")

    def contract(self) -> dict[str, object]:
        return {
            "arms": ("explicit", "implicit", "hybrid"),
            "common_global_operator": "exact_equivariant_linear_attention",
            "explicit_local": "single_positive_receiver_csr",
            "implicit_local": "gaussian_taylor_edge_free_residual",
            "same_parameter_schema": True,
            "same_input_output_irreps": True,
            "coordinate_updates": False,
            "implicit_every": self.implicit_every,
            "implicit_zero_init": self.implicit_residual_scale_init == 0.0,
        }


def empty_prepared_graph_like(graph: Prepared3DGraph) -> Prepared3DGraph:
    """Create the no-edge topology used by the implicit-only arm."""

    edge_index = torch.empty(
        (2, 0),
        device=graph.device,
        dtype=torch.long,
    )
    return prepare_3d_graph(graph.batch, edge_index)


def state_dict_sha256(module: nn.Module) -> str:
    """Hash parameter and buffer names, dtypes, shapes, and exact bytes."""

    digest = sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class SpatialOperatorAblationModel(nn.Module):
    """One schema-matched model used for explicit/implicit/hybrid comparisons."""

    def __init__(
        self,
        config: SpatialOperatorAblationConfig,
        *,
        arm: SpatialOperatorArm,
    ) -> None:
        super().__init__()
        if not isinstance(config, SpatialOperatorAblationConfig):
            raise TypeError("config must be a SpatialOperatorAblationConfig")
        if arm not in {"explicit", "implicit", "hybrid"}:
            raise ValueError("arm must be explicit, implicit, or hybrid")
        self.config = config
        self.arm: SpatialOperatorArm = arm
        self.backbone = EquivariantLinearAttention(config.model)
        self.implicit_residuals = nn.ModuleList(
            [
                ImplicitSpatialResidual(
                    ImplicitGaussianSpatialKernel(config.implicit),
                    scalar_width=config.model.hidden_dim,
                    num_heads=config.model.num_heads,
                    residual_scale_init=config.implicit_residual_scale_init,
                )
                for _ in range(config.model.num_layers)
            ]
        )

    @property
    def uses_explicit_local(self) -> bool:
        return self.arm in {"explicit", "hybrid"}

    @property
    def uses_implicit_local(self) -> bool:
        return self.arm in {"implicit", "hybrid"}

    @property
    def input_irreps(self):  # type: ignore[no-untyped-def]
        return self.backbone.config.input_layout

    @property
    def output_irreps(self):  # type: ignore[no-untyped-def]
        return self.backbone.output_irreps

    def _execution_graph(self, graph: Prepared3DGraph) -> Prepared3DGraph:
        return graph if self.uses_explicit_local else empty_prepared_graph_like(graph)

    @staticmethod
    def _graph_mean(
        value: torch.Tensor,
        graph: Prepared3DGraph,
    ) -> torch.Tensor:
        output = value.new_zeros(
            (graph.graph_layout.num_graphs, value.shape[-1])
        )
        output.index_add_(0, graph.batch, value)
        counts = graph.graph_layout.graph_counts.to(
            device=value.device,
            dtype=value.dtype,
        ).clamp_min(1.0)
        return output / counts.unsqueeze(-1)

    def forward(
        self,
        node_irreps: torch.Tensor,
        positions: torch.Tensor,
        graph: Prepared3DGraph,
        *,
        node_role_id: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        execution_graph = self._execution_graph(graph)
        state, context = self.backbone.embed_input(
            node_irreps,
            positions,
            execution_graph,
            node_role_id=node_role_id,
        )
        for index, layer in enumerate(self.backbone.layers):
            state = layer.attention_residual(state, context, condition)
            if self.uses_implicit_local and index % self.config.implicit_every == 0:
                state = self.implicit_residuals[index](
                    state,
                    positions,
                    graph.batch,
                )
            state = layer.ffn_residual(state, context, condition)

        node_output = self.backbone.project_state(state)
        return {
            "node_irreps": node_output,
            "graph_irreps": self._graph_mean(node_output, graph),
            "positions": positions,
            "coordinate_delta": torch.zeros_like(positions),
        }

    def audit(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "uses_explicit_local": self.uses_explicit_local,
            "uses_implicit_local": self.uses_implicit_local,
            "parameter_count": sum(
                parameter.numel() for parameter in self.parameters()
            ),
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "state_dict_sha256": state_dict_sha256(self),
            "contract": self.config.contract(),
        }


__all__ = [
    "SpatialOperatorAblationConfig",
    "SpatialOperatorAblationModel",
    "SpatialOperatorArm",
    "empty_prepared_graph_like",
    "state_dict_sha256",
]
