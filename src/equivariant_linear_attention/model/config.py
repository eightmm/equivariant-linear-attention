"""Derived configuration and public mathematical contract for ELA."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from ..context import ELAFeatures
from ..irreps import IrrepLayout


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
class ELAConfig:
    input_irreps: str
    output_irreps: str = "1x0e"
    width: int = 128
    depth: int = 8
    features: ELAFeatures = field(default_factory=ELAFeatures)
    update_positions: bool = False
    max_coordinate_step: float = 0.25

    def __post_init__(self) -> None:
        _positive_integer("width", self.width)
        _positive_integer("depth", self.depth)
        if self.width < 16:
            raise ValueError("width must be at least 16")
        if not isinstance(self.features, ELAFeatures):
            raise TypeError("features must be ELAFeatures")
        if not isinstance(self.update_positions, bool):
            raise TypeError("update_positions must be a bool")
        _positive_real("max_coordinate_step", self.max_coordinate_step)
        if self.input_layout.max_degree > 2 or self.output_layout.max_degree > 2:
            raise ValueError("persistent input and output irreps must have l<=2")

    @property
    def input_layout(self) -> IrrepLayout:
        return IrrepLayout.parse(self.input_irreps)

    @property
    def output_layout(self) -> IrrepLayout:
        return IrrepLayout.parse(self.output_irreps)

    @property
    def num_heads(self) -> int:
        target = max(1, min(16, self.width // 16))
        for candidate in range(target, 0, -1):
            if self.width % candidate == 0:
                return candidate
        return 1

    @property
    def moment_rank(self) -> int:
        return max(4, min(12, self.width // 16))

    @property
    def relation_width(self) -> int:
        return max(8, self.width // self.num_heads)

    @property
    def num_charts(self) -> int:
        return max(2, min(8, self.width // 24))

    @property
    def local_probe_rank(self) -> int:
        return max(2, min(8, self.width // 32))

    @property
    def local_scales(self) -> int:
        return 3

    @property
    def local_points(self) -> int:
        return max(8, min(32, self.width // 4))

    @property
    def local_chunk_size(self) -> int:
        return 128

    @property
    def eps(self) -> float:
        return 1e-8

    def contract(self) -> dict[str, object]:
        return {
            "architecture": "edge_free_equivariant_linear_attention",
            "execution_architecture": "pointwise_local_global_equivariant_linear_attention",
            "public_contract": "ELAGraph -> ELA -> ELAGraph",
            "persistent_irreps": "0e + 0o + 1o + 1e + 2e + 2o",
            "global_relative_moment_order": 4,
            "local_cumulant_order": 4,
            "local_jet_degree": 2,
            "local_body_order": 4,
            "transient_irreps": ("3o", "4e"),
            "local_support": "transient_bounded_wendland",
            "relation_operator": "single_fused_self_adjoint_gram_factor",
            "relation_value_layout": "packed_all_irreps",
            "coordinate_basis": "35d_compact_symmetric_degree_0_to_4",
            "krylov_basis": "three_term_graphwise_irrep_orthogonal",
            "coordinate_manifold": "natural_gradient_SE3_quotient_plus_shape",
            "explicit_edges": False,
            "pair_state": False,
            "persistent_pair_state": False,
            "transient_local_support": True,
            "derived_num_heads": self.num_heads,
            "derived_moment_rank": self.moment_rank,
            "derived_num_charts": self.num_charts,
            "derived_local_probe_rank": self.local_probe_rank,
            "derived_local_scales": self.local_scales,
            "derived_local_points": self.local_points,
        }


__all__ = ["ELAConfig"]
