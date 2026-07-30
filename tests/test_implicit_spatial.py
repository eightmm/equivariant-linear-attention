from __future__ import annotations

import torch

from equivariant_attention.implicit_spatial import (
    ImplicitGaussianSpatialKernel,
    ImplicitSpatialKernelConfig,
    ImplicitSpatialStateTransport,
)
from equivariant_attention.layered_se3 import UnifiedSE3State


def _st_to_matrix(value: torch.Tensor) -> torch.Tensor:
    xx, yy, xy, xz, yz = value.unbind(dim=-1)
    zz = -xx - yy
    return torch.stack(
        [
            torch.stack([xx, xy, xz], dim=-1),
            torch.stack([xy, yy, yz], dim=-1),
            torch.stack([xz, yz, zz], dim=-1),
        ],
        dim=-2,
    )


def _matrix_to_st(value: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            value[..., 0, 0],
            value[..., 1, 1],
            value[..., 0, 1],
            value[..., 0, 2],
            value[..., 1, 2],
        ],
        dim=-1,
    )


def _transform_state(
    state: UnifiedSE3State,
    orthogonal: torch.Tensor,
) -> UnifiedSE3State:
    determinant = torch.linalg.det(orthogonal)

    def vector(value: torch.Tensor, *, axial: bool) -> torch.Tensor:
        result = torch.einsum("...c,dc->...d", value, orthogonal)
        return determinant * result if axial else result

    def tensor(value: torch.Tensor, *, odd: bool) -> torch.Tensor:
        matrix = _st_to_matrix(value)
        result = torch.einsum(
            "ab,...bc,dc->...ad",
            orthogonal,
            matrix,
            orthogonal,
        )
        compact = _matrix_to_st(result)
        return determinant * compact if odd else compact

    return UnifiedSE3State(
        even_scalar=state.even_scalar,
        odd_scalar=determinant * state.odd_scalar,
        polar_vector=vector(state.polar_vector, axial=False),
        axial_vector=vector(state.axial_vector, axial=True),
        even_tensor=tensor(state.even_tensor, odd=False),
        odd_tensor=tensor(state.odd_tensor, odd=True),
    )


def _assert_state_close(
    actual: UnifiedSE3State,
    expected: UnifiedSE3State,
) -> None:
    for name in (
        "even_scalar",
        "odd_scalar",
        "polar_vector",
        "axial_vector",
        "even_tensor",
        "odd_tensor",
    ):
        torch.testing.assert_close(
            getattr(actual, name),
            getattr(expected, name),
            atol=3e-9,
            rtol=3e-9,
        )


def test_factorized_transport_matches_dense_feature_reference() -> None:
    torch.manual_seed(3)
    kernel = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(
            scales=(1.5, 3.0),
            order=2,
            exclude_self=False,
            normalization="none",
        )
    ).double()
    positions = torch.randn(7, 3, dtype=torch.float64)
    values = torch.randn(7, 5, dtype=torch.float64)
    batch = torch.zeros(7, dtype=torch.long)

    context = kernel.prepare(positions, batch)
    dense_kernel = context.features @ context.features.T
    expected = dense_kernel @ values
    actual = kernel.transport_prepared(values, context).output

    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)


def test_order_two_features_approximate_local_gaussian() -> None:
    torch.manual_seed(5)
    scale = 2.0
    kernel = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(
            scales=(scale,),
            order=2,
            exclude_self=False,
            normalization="none",
        )
    ).double()
    # Small centered coordinates keep the omitted cubic and higher terms small.
    positions = 0.15 * torch.randn(8, 3, dtype=torch.float64)
    batch = torch.zeros(8, dtype=torch.long)
    context = kernel.prepare(positions, batch)
    approximate = context.features @ context.features.T
    difference = positions[:, None, :] - positions[None, :, :]
    exact = torch.exp(-difference.square().sum(dim=-1) / (2.0 * scale**2))

    relative_error = (approximate - exact).abs().max() / exact.abs().max()
    assert relative_error < 2e-4


def test_single_node_self_exclusion_has_zero_message() -> None:
    kernel = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(
            scales=(1.0,),
            exclude_self=True,
            normalization="one_plus_mass",
        )
    ).double()
    positions = torch.zeros(1, 3, dtype=torch.float64)
    values = torch.randn(1, 4, dtype=torch.float64)
    result = kernel(values, positions, torch.zeros(1, dtype=torch.long))

    torch.testing.assert_close(result.output, torch.zeros_like(result.output))
    torch.testing.assert_close(result.mass, torch.zeros_like(result.mass))


def test_implicit_moments_are_translation_and_o3_equivariant() -> None:
    torch.manual_seed(7)
    kernel = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(
            scales=(0.8, 1.6, 3.2),
            exclude_self=True,
            normalization="one_plus_mass",
        )
    ).double()
    positions = torch.randn(9, 3, dtype=torch.float64)
    batch = torch.zeros(9, dtype=torch.long)
    reflection = torch.diag(
        torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    )
    translation = torch.tensor([1.1, -0.4, 0.7], dtype=torch.float64)

    reference = kernel.moments(positions, batch)
    transformed = kernel.moments(
        positions @ reflection.T + translation,
        batch,
    )

    torch.testing.assert_close(transformed.mass, reference.mass, atol=2e-10, rtol=2e-10)
    expected_vector = torch.einsum(
        "...c,dc->...d",
        reference.relative_vector,
        reflection,
    )
    torch.testing.assert_close(
        transformed.relative_vector,
        expected_vector,
        atol=3e-9,
        rtol=3e-9,
    )
    reference_matrix = _st_to_matrix(reference.relative_tensor)
    expected_matrix = torch.einsum(
        "ab,...bc,dc->...ad",
        reflection,
        reference_matrix,
        reflection,
    )
    torch.testing.assert_close(
        transformed.relative_tensor,
        _matrix_to_st(expected_matrix),
        atol=4e-9,
        rtol=4e-9,
    )


def test_state_transport_preserves_every_parity_sector() -> None:
    torch.manual_seed(11)
    kernel = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(
            scales=(1.0, 2.0),
            exclude_self=True,
            normalization="one_plus_mass",
        )
    ).double()
    transport = ImplicitSpatialStateTransport(kernel)
    nodes = 7
    state = UnifiedSE3State(
        even_scalar=torch.randn(nodes, 5, dtype=torch.float64),
        odd_scalar=torch.randn(nodes, 3, dtype=torch.float64),
        polar_vector=torch.randn(nodes, 3, 3, dtype=torch.float64),
        axial_vector=torch.randn(nodes, 3, 3, dtype=torch.float64),
        even_tensor=torch.randn(nodes, 3, 5, dtype=torch.float64),
        odd_tensor=torch.randn(nodes, 3, 5, dtype=torch.float64),
    )
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    batch = torch.zeros(nodes, dtype=torch.long)
    reflection = torch.diag(
        torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    )
    translation = torch.tensor([-0.5, 0.8, 1.2], dtype=torch.float64)

    reference = transport(state, positions, batch)
    actual = transport(
        _transform_state(state, reflection),
        positions @ reflection.T + translation,
        batch,
    )
    expected = _transform_state(reference, reflection)
    _assert_state_close(actual, expected)


def test_graphs_do_not_interact_without_edges() -> None:
    torch.manual_seed(13)
    kernel = ImplicitGaussianSpatialKernel(
        ImplicitSpatialKernelConfig(scales=(1.0, 2.0))
    ).double()
    positions = torch.randn(8, 3, dtype=torch.float64)
    values = torch.randn(8, 4, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    reference = kernel(values, positions, batch).output[:4]
    changed_positions = positions.clone()
    changed_positions[4:] = 100.0 * torch.randn(4, 3, dtype=torch.float64)
    changed_values = values.clone()
    changed_values[4:] = 100.0 * torch.randn(4, 4, dtype=torch.float64)
    actual = kernel(changed_values, changed_positions, batch).output[:4]

    torch.testing.assert_close(actual, reference, atol=2e-10, rtol=2e-10)


def test_complexity_contract_contains_no_edge_or_pair_input() -> None:
    config = ImplicitSpatialKernelConfig(scales=(1.0, 2.0, 4.0), order=2)
    contract = config.complexity_contract()

    assert config.feature_rank == 30
    assert contract["edge_input"] is False
    assert contract["neighbor_construction"] is False
    assert contract["pair_matrix"] is False
    assert contract["exact_radius_graph"] is False
