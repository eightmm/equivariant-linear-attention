from __future__ import annotations

import torch

from equivariant_linear_attention.geometry import prepare_3d_graph
from equivariant_linear_attention.model.runtime import (
    _BaseStackConfig,
    _ELARuntime,
)
from equivariant_linear_attention.nn.multipoles import _c2_cutoff


def _complete_edge_index(num_nodes: int) -> torch.Tensor:
    receiver = torch.arange(num_nodes).repeat_interleave(num_nodes)
    sender = torch.arange(num_nodes).repeat(num_nodes)
    return torch.stack([receiver, sender])


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


def _transform_st(
    value: torch.Tensor,
    orthogonal: torch.Tensor,
    *,
    parity_sign: torch.Tensor,
) -> torch.Tensor:
    matrix = _st_to_matrix(value)
    transformed = torch.einsum(
        "ab,...bc,dc->...ad",
        orthogonal,
        matrix,
        orthogonal,
    )
    return parity_sign * _matrix_to_st(transformed)


def _noncoplanar_positions() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, -0.1],
            [-0.3, 1.2, 0.4],
            [0.4, -0.6, 1.4],
            [-1.1, -0.2, 0.7],
            [0.8, 1.1, -0.9],
        ],
        dtype=torch.float64,
    )


def test_unified_contract_records_multipole_complete_path() -> None:
    config = _BaseStackConfig(
        input_irreps="5x0e",
        output_irreps="1x0e",
        hidden_dim=16,
        num_heads=4,
        local_rank=3,
    )
    contract = config.canonical_contract()

    assert contract["node_geometry"] == "static_l0_l1_l2_radial_multipoles"
    assert contract["global_operator"] == "exact_positive_feature_gemm_l0_l1_l2"
    assert contract["local_operator"] == "single_positive_mass_damped_rank_r"
    assert contract["local_reduction"] == "single_receiver_csr"
    assert contract["local_routing"] == (
        "scalar_vector_axial_tensor_parity_complete"
    )
    assert contract["tensor_product_closure"] == "low_rank_lte2_cartesian"
    assert contract["irrep_normalization"] == "sector_rms_pre_norm"
    assert contract["residual_scaling"] == "per_copy_layerscale"
    assert contract["cutoff_regularity"] == "C2"
    assert contract["fallbacks"] == ()


def test_c2_cutoff_has_zero_value_slope_and_curvature_at_boundary() -> None:
    squared_ratio = torch.tensor(
        1.0,
        dtype=torch.float64,
        requires_grad=True,
    )
    value = _c2_cutoff(squared_ratio)
    slope = torch.autograd.grad(
        value,
        squared_ratio,
        create_graph=True,
    )[0]
    curvature = torch.autograd.grad(slope, squared_ratio)[0]

    torch.testing.assert_close(value, torch.zeros_like(value))
    torch.testing.assert_close(slope, torch.zeros_like(slope))
    torch.testing.assert_close(curvature, torch.zeros_like(curvature))


def test_node_multipoles_obey_full_parity_transform() -> None:
    torch.manual_seed(17)
    config = _BaseStackConfig(
        input_irreps="3x0e",
        output_irreps="1x0e",
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        local_rank=3,
        local_cutoff=10.0,
        num_rbf=8,
    )
    model = _ELARuntime(config).double().eval()
    positions = _noncoplanar_positions()
    batch = torch.zeros(positions.shape[0], dtype=torch.long)
    graph = prepare_3d_graph(
        batch,
        _complete_edge_index(positions.shape[0]),
    )

    geometry = model.core._build_geometry(positions, graph.neighbors)
    multipoles = model.core.node_multipoles(geometry)

    reflection = torch.diag(
        torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    )
    transformed_geometry = model.core._build_geometry(
        positions @ reflection.T + torch.tensor([2.0, -1.0, 0.5]),
        graph.neighbors,
    )
    transformed = model.core.node_multipoles(transformed_geometry)
    determinant = torch.linalg.det(reflection)

    torch.testing.assert_close(transformed.mass, multipoles.mass)
    torch.testing.assert_close(
        transformed.mass_square,
        multipoles.mass_square,
    )
    torch.testing.assert_close(
        transformed.polar,
        torch.einsum("...c,dc->...d", multipoles.polar, reflection),
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        transformed.axial,
        determinant
        * torch.einsum("...c,dc->...d", multipoles.axial, reflection),
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        transformed.odd_scalar,
        determinant * multipoles.odd_scalar,
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        transformed.even_tensor,
        _transform_st(
            multipoles.even_tensor,
            reflection,
            parity_sign=torch.ones((), dtype=torch.float64),
        ),
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        transformed.odd_tensor,
        _transform_st(
            multipoles.odd_tensor,
            reflection,
            parity_sign=determinant,
        ),
        atol=1e-10,
        rtol=1e-10,
    )


def test_even_objective_reaches_new_multipole_and_tensor_paths() -> None:
    torch.manual_seed(29)
    config = _BaseStackConfig(
        input_irreps="4x0e",
        output_irreps="1x0e",
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        local_rank=3,
        local_cutoff=10.0,
        num_rbf=8,
    )
    model = _ELARuntime(config).double()
    positions = _noncoplanar_positions().requires_grad_(True)
    node_features = torch.randn(
        positions.shape[0],
        4,
        dtype=torch.float64,
        requires_grad=True,
    )
    graph = prepare_3d_graph(
        torch.zeros(positions.shape[0], dtype=torch.long),
        _complete_edge_index(positions.shape[0]),
    )

    output = model(node_features, positions, graph)["node_irreps"]
    output.square().mean().backward()

    first = model.core.blocks[0]
    checked = (
        model.core.scale_density_projection.weight,
        model.core.initial_multipole_polar.weight,
        model.core.initial_multipole_even_tensor.weight,
        first.raw_tensor_alignment,
        first.local_tensor_mix,
        first.local_tensor_radial_score.weight,
        first.tensor_closure.even_scalar_out.weight,
        first.pre_norm.even_tensor_gain,
    )
    for parameter in checked:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    assert positions.grad is not None
    assert node_features.grad is not None
    assert torch.isfinite(positions.grad).all()
    assert torch.isfinite(node_features.grad).all()


def test_local_tensor_routing_uses_one_mass_and_one_value_lane() -> None:
    config = _BaseStackConfig(
        input_irreps="3x0e",
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        local_rank=3,
    )
    block = _ELARuntime(config).core.blocks[0]

    assert not hasattr(block, "tensor_local_scalar_out")
    assert hasattr(block, "local_tensor_mix")
    assert hasattr(block, "local_scalar_out")


def test_layerscale_is_copy_specific_for_every_irrep_sector() -> None:
    config = _BaseStackConfig(
        input_irreps="3x0e",
        hidden_dim=24,
        num_layers=1,
        num_heads=6,
        local_rank=3,
    )
    block = _ELARuntime(config).core.blocks[0]

    assert block.scalar_scale.shape == (24,)
    assert block.odd_scale.shape == (6,)
    assert block.polar_scale.shape == (6, 1)
    assert block.axial_scale.shape == (6, 1)
    assert block.even_tensor_scale.shape == (6, 1)
    assert block.odd_tensor_scale.shape == (6, 1)
    assert block.ffn_scalar_scale.shape == (24,)
    assert block.ffn_polar_scale.shape == (6, 1)
