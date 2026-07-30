from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from collections.abc import Sequence


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class ComplexityEstimate:
    """Symbolic contract plus dimensionless arithmetic/memory proxies.

    The proxies are intended for relative sweep validation only; they are not
    FLOP counts and do not predict wall-clock time without device measurements.
    """

    name: str
    arithmetic_proxy: int
    inference_memory_proxy: int
    training_memory_proxy: int
    formula: str
    assumptions: tuple[str, ...]
    node_linear: bool
    depth_linear: bool


@dataclass(frozen=True, slots=True)
class ScalingFit:
    """Least-squares fit of ``measurement = coefficient * size**slope``."""

    slope: float
    coefficient: float
    r_squared: float


def estimate_base_linear_attention(
    *,
    nodes: int,
    edges: int,
    layers: int,
    channel_factor: int = 1,
) -> ComplexityEstimate:
    """Estimate the base spatial stack with a precomputed candidate graph."""

    nodes = _positive_integer("nodes", nodes)
    layers = _positive_integer("layers", layers)
    channel_factor = _positive_integer("channel_factor", channel_factor)
    if isinstance(edges, bool) or not isinstance(edges, int) or edges < 0:
        raise ValueError("edges must be a nonnegative integer")
    per_layer = channel_factor * (nodes + edges)
    return ComplexityEstimate(
        name="equivariant_linear_attention",
        arithmetic_proxy=layers * per_layer,
        inference_memory_proxy=channel_factor * (nodes + edges),
        training_memory_proxy=layers * channel_factor * (nodes + edges),
        formula="O(L * (N + E))",
        assumptions=(
            "fixed hidden width, head count, radial rank, and local rank",
            "candidate graph is prepared outside the layer",
        ),
        node_linear=edges <= 256 * nodes,
        depth_linear=True,
    )


def estimate_attention_residuals(
    *,
    nodes: int,
    edges: int,
    layers: int,
    blocks: int,
    channel_factor: int = 1,
) -> ComplexityEstimate:
    """Estimate spatial work plus two depth routers per layer."""

    base = estimate_base_linear_attention(
        nodes=nodes,
        edges=edges,
        layers=layers,
        channel_factor=channel_factor,
    )
    blocks = _positive_integer("blocks", blocks)
    if blocks > layers:
        raise ValueError("blocks must not exceed layers")
    depth_routing = 2 * layers * blocks * nodes * channel_factor
    return ComplexityEstimate(
        name="equivariant_attention_residuals",
        arithmetic_proxy=base.arithmetic_proxy + depth_routing,
        inference_memory_proxy=base.inference_memory_proxy
        + blocks * nodes * channel_factor,
        training_memory_proxy=base.training_memory_proxy + depth_routing,
        formula="O(L * (N + E) + L * B * N)",
        assumptions=base.assumptions
        + ("B is the number of retained block-level depth sources",),
        node_linear=base.node_linear,
        depth_linear=blocks < layers or blocks == 1,
    )


def estimate_implicit_spatial_kernel(
    *,
    nodes: int,
    feature_rank: int,
    value_width: int,
    applications: int = 1,
) -> ComplexityEstimate:
    """Estimate an edge-free finite-feature spatial transport."""

    nodes = _positive_integer("nodes", nodes)
    feature_rank = _positive_integer("feature_rank", feature_rank)
    value_width = _positive_integer("value_width", value_width)
    applications = _positive_integer("applications", applications)
    # One Phi^T V and one Phi statistic contraction per application.
    arithmetic = 2 * applications * nodes * feature_rank * value_width
    working = nodes * (feature_rank + value_width) + feature_rank * value_width
    return ComplexityEstimate(
        name="implicit_spatial_kernel",
        arithmetic_proxy=arithmetic,
        inference_memory_proxy=working,
        training_memory_proxy=applications * working,
        formula="O(A * N * F * D)",
        assumptions=(
            "fixed finite feature rank F",
            "fixed transported value width D",
            "no explicit edge list or pair matrix",
        ),
        node_linear=True,
        depth_linear=True,
    )


def fit_log_log_slope(
    sizes: Sequence[float],
    measurements: Sequence[float],
) -> ScalingFit:
    """Fit a power-law slope without requiring NumPy."""

    if len(sizes) != len(measurements):
        raise ValueError("sizes and measurements must have equal length")
    if len(sizes) < 2:
        raise ValueError("at least two observations are required")
    x = []
    y = []
    for size, measurement in zip(sizes, measurements, strict=True):
        if not isfinite(float(size)) or float(size) <= 0.0:
            raise ValueError("sizes must be finite and positive")
        if not isfinite(float(measurement)) or float(measurement) <= 0.0:
            raise ValueError("measurements must be finite and positive")
        x.append(log(float(size)))
        y.append(log(float(measurement)))

    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    variance = sum((value - x_mean) ** 2 for value in x)
    if variance == 0.0:
        raise ValueError("sizes must not all be identical")
    covariance = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y, strict=True)
    )
    slope = covariance / variance
    intercept = y_mean - slope * x_mean
    prediction = [intercept + slope * value for value in x]
    residual = sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(y, prediction, strict=True)
    )
    total = sum((actual - y_mean) ** 2 for actual in y)
    r_squared = 1.0 if total == 0.0 else 1.0 - residual / total
    return ScalingFit(
        slope=slope,
        coefficient=float(__import__("math").exp(intercept)),
        r_squared=r_squared,
    )


__all__ = [
    "ComplexityEstimate",
    "ScalingFit",
    "estimate_attention_residuals",
    "estimate_base_linear_attention",
    "estimate_implicit_spatial_kernel",
    "fit_log_log_slope",
]
