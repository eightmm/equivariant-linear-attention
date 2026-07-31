from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .canonical import ELAConfig, ELACore
from .layered_se3 import UnifiedSE3State
from .unified import Prepared3DGraph, UnifiedEquivariantAttention


@dataclass(frozen=True, slots=True)
class InvariantConditioningConfig:
    """Invariant DiT-style conditioning attached outside minimal ELAConfig."""

    condition_dim: int

    def __post_init__(self) -> None:
        if isinstance(self.condition_dim, bool) or not isinstance(
            self.condition_dim, int
        ):
            raise TypeError("condition_dim must be an integer")
        if self.condition_dim <= 0:
            raise ValueError("condition_dim must be positive")


class ConditionedELA(UnifiedEquivariantAttention):
    """Canonical branch-aware ELA with explicit invariant conditioning.

    Conditioning is a wrapper-level capability rather than a boolean or width in
    the minimal model config. The condition is an ordinary invariant `0e`
    feature and may be shared, graph-level, or node-level. Each layer applies
    bounded adaptive modulation and independent attention/FFN residual gates.
    Conditioner output projections are zero initialized, so the initial model
    exactly reproduces the unconditioned ELA function after shared weights are
    loaded.
    """

    attention_kind = "conditioned_canonical_equivariant_linear_attention"

    def __init__(
        self,
        base_config: ELAConfig,
        conditioning: InvariantConditioningConfig,
    ) -> None:
        nn.Module.__init__(self)
        if not isinstance(base_config, ELAConfig):
            raise TypeError("base_config must be an ELAConfig")
        if not isinstance(conditioning, InvariantConditioningConfig):
            raise TypeError(
                "conditioning must be an InvariantConditioningConfig"
            )
        self.config = base_config
        self.conditioning = conditioning
        self.advanced_config = base_config.to_advanced_config()
        self.core = ELACore(
            self.advanced_config,
            condition_dim=conditioning.condition_dim,
        )
        self._initialize_chiral_bridge()
        self.internal_irreps = self.core.internal_irreps
        self.output_irreps = self.core.output_irreps

    @staticmethod
    def _validate_condition(condition: torch.Tensor | None) -> None:
        if condition is None:
            return
        if not isinstance(condition, torch.Tensor):
            raise TypeError("condition must be a tensor")
        if not condition.is_floating_point():
            raise TypeError("condition must use a floating-point dtype")
        finite = torch.isfinite(condition).all()
        async_assert = getattr(torch, "_assert_async", None)
        if condition.device.type == "cuda" and async_assert is not None:
            async_assert(finite, "condition must contain only finite values")
        elif not bool(finite):
            raise ValueError("condition must contain only finite values")

    def forward_features(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
        *,
        node_role_id: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[UnifiedSE3State, torch.Tensor, torch.Tensor]:
        self._validate_condition(condition)
        return super().forward_features(
            node_irreps,
            pos,
            graph,
            node_role_id=node_role_id,
            condition=condition,
        )

    def forward(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
        *,
        node_role_id: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_condition(condition)
        return super().forward(
            node_irreps,
            pos,
            graph,
            node_role_id=node_role_id,
            condition=condition,
        )

    def extra_repr(self) -> str:
        return (
            f"input_irreps={self.config.input_layout}, "
            f"output_irreps={self.output_irreps}, width={self.config.width}, "
            f"depth={self.config.depth}, "
            f"condition_dim={self.conditioning.condition_dim}"
        )


__all__ = [
    "ConditionedELA",
    "InvariantConditioningConfig",
]
