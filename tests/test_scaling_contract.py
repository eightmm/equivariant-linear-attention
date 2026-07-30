from __future__ import annotations

import pytest

from equivariant_attention.scaling_contract import (
    estimate_attention_residuals,
    estimate_base_linear_attention,
    estimate_implicit_spatial_kernel,
    fit_log_log_slope,
)


def test_base_node_linearity_requires_explicit_edge_scaling_assumption() -> None:
    unknown = estimate_base_linear_attention(
        nodes=1024,
        edges=32768,
        layers=8,
        edge_scaling="unknown",
    )
    sparse_family = estimate_base_linear_attention(
        nodes=1024,
        edges=32768,
        layers=8,
        edge_scaling="linear",
    )
    dense_family = estimate_base_linear_attention(
        nodes=1024,
        edges=1024 * 1024,
        layers=8,
        edge_scaling="quadratic",
    )

    assert unknown.node_linear is False
    assert sparse_family.node_linear is True
    assert dense_family.node_linear is False
    assert sparse_family.formula == "O(L * (N + E))"


def test_attnres_is_depth_linear_only_for_fixed_block_count() -> None:
    fixed = estimate_attention_residuals(
        nodes=512,
        edges=8192,
        layers=24,
        blocks=8,
        edge_scaling="linear",
        blocks_fixed_with_depth=True,
    )
    growing = estimate_attention_residuals(
        nodes=512,
        edges=8192,
        layers=24,
        blocks=24,
        edge_scaling="linear",
        blocks_fixed_with_depth=False,
    )

    assert fixed.depth_linear is True
    assert growing.depth_linear is False
    assert growing.formula == "O(L * (N + E) + L * B * N)"
    assert growing.arithmetic_proxy > fixed.arithmetic_proxy


def test_implicit_kernel_proxy_is_exactly_linear_in_nodes() -> None:
    small = estimate_implicit_spatial_kernel(
        nodes=256,
        feature_rank=30,
        value_width=64,
        applications=4,
    )
    large = estimate_implicit_spatial_kernel(
        nodes=1024,
        feature_rank=30,
        value_width=64,
        applications=4,
    )

    assert large.arithmetic_proxy == 4 * small.arithmetic_proxy
    assert small.node_linear is True
    assert small.formula == "O(A * N * F * D)"


def test_log_log_slope_recovers_linear_and_quadratic_series() -> None:
    sizes = [128.0, 256.0, 512.0, 1024.0]
    linear = fit_log_log_slope(sizes, [3.0 * value for value in sizes])
    quadratic = fit_log_log_slope(
        sizes,
        [0.5 * value**2 for value in sizes],
    )

    assert linear.slope == pytest.approx(1.0, abs=1e-12)
    assert linear.r_squared == pytest.approx(1.0, abs=1e-12)
    assert quadratic.slope == pytest.approx(2.0, abs=1e-12)
    assert quadratic.r_squared == pytest.approx(1.0, abs=1e-12)


def test_scaling_fit_rejects_nonpositive_measurements() -> None:
    with pytest.raises(ValueError, match="measurements"):
        fit_log_log_slope([1.0, 2.0], [1.0, 0.0])
