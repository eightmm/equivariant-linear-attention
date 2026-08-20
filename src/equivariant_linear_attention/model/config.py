"""Configuration for the one canonical pair-centric TriELA model."""

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


def _non_negative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    if numeric >= 1.0:
        raise ValueError(f"{name} must be smaller than one")
    return numeric


@dataclass(frozen=True, slots=True)
class TriELAConfig:
    """All shape-defining choices for the exact dense TriELA architecture."""

    input_irreps: str
    output_irreps: str = "1x0e"
    width: int = 128
    pair_width: int = 64
    triangle_hidden: int = 64
    num_stages: int = 3
    pair_blocks_per_stage: int = 4
    local_blocks_per_stage: int = 2
    pair_transition_factor: int = 4
    pair_dropout: float = 0.1
    local_points: int = 32
    max_pair_tokens: int = 512
    condition_dim: int = 0
    order_dim: int = 0
    pair_feature_dim: int = 0
    distance_rbf_bins: int = 32
    distance_max: float = 32.0
    distogram_bins: int = 64
    distogram_max: float = 32.0
    update_positions: bool = False
    max_coordinate_step: float = 0.25
    features: ELAFeatures = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _positive_integer("width", self.width)
        if self.width < 16:
            raise ValueError("width must be at least 16")
        _positive_integer("pair_width", self.pair_width)
        if self.pair_width < 2:
            raise ValueError("pair_width must be at least 2")
        _positive_integer("triangle_hidden", self.triangle_hidden)
        if self.triangle_hidden < 2:
            raise ValueError("triangle_hidden must be at least 2")
        _positive_integer("num_stages", self.num_stages)
        _positive_integer("pair_blocks_per_stage", self.pair_blocks_per_stage)
        _positive_integer("local_blocks_per_stage", self.local_blocks_per_stage)
        _positive_integer("pair_transition_factor", self.pair_transition_factor)
        _probability("pair_dropout", self.pair_dropout)
        _positive_integer("local_points", self.local_points)
        _positive_integer("max_pair_tokens", self.max_pair_tokens)
        _non_negative_integer("condition_dim", self.condition_dim)
        _non_negative_integer("order_dim", self.order_dim)
        _non_negative_integer("pair_feature_dim", self.pair_feature_dim)
        _positive_integer("distance_rbf_bins", self.distance_rbf_bins)
        _positive_real("distance_max", self.distance_max)
        _positive_integer("distogram_bins", self.distogram_bins)
        if self.distogram_bins < 2:
            raise ValueError("distogram_bins must be at least 2")
        _positive_real("distogram_max", self.distogram_max)
        if not isinstance(self.update_positions, bool):
            raise TypeError("update_positions must be a bool")
        _positive_real("max_coordinate_step", self.max_coordinate_step)
        if self.input_layout.dim == 0 or self.output_layout.dim == 0:
            raise ValueError("input and output irreps must have positive dimension")
        if self.input_layout.max_degree > 2 or self.output_layout.max_degree > 2:
            raise ValueError("persistent input and output irreps must have l<=2")
        object.__setattr__(
            self,
            "features",
            ELAFeatures(
                condition_dim=self.condition_dim,
                order_dim=self.order_dim,
            ),
        )

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
    def local_chunk_size(self) -> int:
        return 128

    @property
    def eps(self) -> float:
        return 1e-8

    def contract(self) -> dict[str, object]:
        return {
            "architecture": "pair_centric_tri_ela",
            "public_contract": (
                "ELAGraph + optional BiomolecularPairContext -> TriELA -> ELAGraph"
            ),
            "pair_backend": "dense_exact",
            "pair_state": "persistent_ordered_invariant",
            "triangle_block": "outgoing_then_incoming_then_swiglu",
            "global_transport": "equivariant_linear_attention",
            "local_transport": "pair_conditioned_bounded_geometry",
            "coordinate_update_location": (
                "after_local_only" if self.update_positions else "disabled"
            ),
            "persistent_irreps": "0e + 0o + 1o + 1e + 2e + 2o",
            "explicit_dense_pair": True,
            "whole_model_linear_scaling": False,
            "legacy_path": False,
            "fallback_backend": False,
            "num_heads": self.num_heads,
            "max_pair_tokens": self.max_pair_tokens,
        }


__all__ = ["TriELAConfig"]
