from __future__ import annotations

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention.config import ArchitectureConfig
from equivariant_attention.high_order import TransientL3Workspace
from equivariant_attention.neighbors import build_receiver_csr


def _fixture(
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260821)
    vectors = torch.randn(5, 2, 3, generator=generator, dtype=dtype)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.1, 0.2],
            [-0.2, 1.1, 0.3],
            [0.4, -0.5, 1.2],
            [1.3, 0.7, -0.4],
        ],
        dtype=dtype,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            [0, 1, 2, 3, 4, 1, 2, 0, 3, 1, 4, 2, 4, 0, 3],
        ],
        dtype=torch.long,
    )
    invariants = torch.randn(
        edge_index.shape[1],
        3,
        generator=generator,
        dtype=dtype,
    )
    return vectors, positions, edge_index, invariants


def _orthogonal_with_reflection() -> torch.Tensor:
    matrix = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    assert torch.linalg.det(matrix) < 0
    return matrix


def test_transient_l3_is_o3_translation_permutation_and_edge_order_equivariant() -> (
    None
):
    vectors, positions, edge_index, invariants = _fixture()
    torch.manual_seed(20260822)
    module = TransientL3Workspace(
        input_vector_channels=2,
        workspace_channels=3,
        output_vector_channels=2,
        edge_scalar_dim=3,
    ).double()

    expected = module(vectors, positions, edge_index, invariants)
    orthogonal = _orthogonal_with_reflection()
    translation = torch.tensor([3.0, -2.0, 0.5], dtype=torch.float64)
    transformed = module(
        torch.einsum("ab,ncb->nca", orthogonal, vectors),
        positions @ orthogonal.mT + translation,
        edge_index,
        invariants,
    )
    torch.testing.assert_close(
        transformed,
        torch.einsum("ab,ncb->nca", orthogonal, expected),
        rtol=3e-11,
        atol=3e-11,
    )

    permutation = torch.tensor([3, 0, 4, 1, 2])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    edge_order = torch.arange(edge_index.shape[1] - 1, -1, -1)
    permuted = module(
        vectors[permutation],
        positions[permutation],
        inverse[edge_index[:, edge_order]],
        invariants[edge_order],
    )
    torch.testing.assert_close(
        permuted,
        expected[permutation],
        rtol=2e-12,
        atol=2e-12,
    )


def test_transient_l3_projects_only_after_receiver_aggregation() -> None:
    module = TransientL3Workspace(
        input_vector_channels=1,
        workspace_channels=1,
        output_vector_channels=1,
        normalization="none",
    ).double()
    with torch.no_grad():
        module.lift_weight.fill_(1.0)
        module.project_weight.fill_(1.0)
    vectors = torch.tensor(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.2, -0.1]],
            [[-0.3, 0.7, 0.4]],
        ],
        dtype=torch.float64,
    )
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.2, 1.1, 0.3]],
        dtype=torch.float64,
    )
    first_edge = torch.tensor([[0], [1]])
    second_edge = torch.tensor([[0], [2]])
    both_edges = torch.cat([first_edge, second_edge], dim=1)

    combined = module(vectors, positions, both_edges)
    edgewise_sum = (
        module(vectors, positions, first_edge)
        + module(vectors, positions, second_edge)
    )

    assert not torch.allclose(combined[0], edgewise_sum[0])
    assert torch.count_nonzero(combined[1:]) == 0
    assert module.workspace_degree == 3
    assert not hasattr(module, "persistent_workspace")
    assert all(tensor.shape[-1] != 7 for tensor in module.state_dict().values())


def test_transient_l3_supports_finite_second_derivatives() -> None:
    vectors, positions, edge_index, invariants = _fixture()
    vectors.requires_grad_(True)
    positions.requires_grad_(True)
    invariants.requires_grad_(True)
    module = TransientL3Workspace(
        input_vector_channels=2,
        workspace_channels=2,
        output_vector_channels=2,
        edge_scalar_dim=3,
    ).double()

    output = module(vectors, positions, edge_index, invariants)
    first = torch.autograd.grad(
        output.square().sum(),
        (vectors, positions, invariants, *module.parameters()),
        create_graph=True,
    )
    second = torch.autograd.grad(
        sum(gradient.square().sum() for gradient in first),
        (vectors, positions, invariants),
    )

    assert all(torch.isfinite(gradient).all() for gradient in (*first, *second))


def test_transient_l3_keeps_bfloat16_projection_and_fp32_geometry_safe() -> None:
    vectors, positions, edge_index, invariants = _fixture(dtype=torch.float32)
    vectors = vectors.to(dtype=torch.bfloat16).requires_grad_(True)
    positions.requires_grad_(True)
    invariants = invariants.to(dtype=torch.bfloat16).requires_grad_(True)
    module = TransientL3Workspace(
        input_vector_channels=2,
        workspace_channels=2,
        output_vector_channels=2,
        edge_scalar_dim=3,
    ).to(dtype=torch.bfloat16)

    output = module(vectors, positions, edge_index, invariants)
    loss = output.float().square().sum()
    vector_gradient, position_gradient, invariant_gradient = torch.autograd.grad(
        loss,
        (vectors, positions, invariants),
    )

    assert output.dtype == torch.bfloat16
    assert vector_gradient.dtype == torch.bfloat16
    assert position_gradient.dtype == torch.float32
    assert invariant_gradient.dtype == torch.bfloat16
    assert all(
        torch.isfinite(value).all()
        for value in (
            output,
            vector_gradient,
            position_gradient,
            invariant_gradient,
        )
    )


def test_high_order_profile_executes_transient_workspace_in_public_model() -> None:
    structured = ArchitectureConfig.for_profile(
        "high_order",
        node_dim=4,
        width=16,
        num_heads=2,
        num_layers=2,
    )
    config = structured.to_legacy()
    model = EquivariantAttention(config).double()
    features = torch.randn(
        5,
        4,
        generator=torch.Generator().manual_seed(20260823),
        dtype=torch.float64,
    ).requires_grad_(True)
    positions = _fixture()[1].requires_grad_(True)
    edge_index = _fixture()[2]
    batch = torch.zeros(5, dtype=torch.long)

    output = model(
        features,
        positions,
        batch=batch,
        edge_index=edge_index,
    )
    feature_gradient, position_gradient = torch.autograd.grad(
        output["graph_scalars"].square().sum(),
        (features, positions),
    )

    assert config.use_transient_l3_workspace
    assert len(model.transient_l3_layers) == config.num_layers
    assert output["graph_scalars"].shape == (1, 1)
    assert torch.isfinite(feature_gradient).all()
    assert torch.isfinite(position_gradient).all()
    assert all(
        "persistent_workspace" not in name
        for name in model.state_dict()
    )

    with torch.no_grad():
        try:
            model(features, positions, batch=batch)
        except ValueError as exc:
            assert "explicit neighbors" in str(exc)
        else:
            raise AssertionError("high-order model accepted missing neighbors")


@pytest.mark.parametrize("packed", [False, True])
def test_transient_l3_rejects_outside_cutoff_candidates_exactly(
    packed: bool,
) -> None:
    torch.manual_seed(20260827)
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            hidden_irreps="16x0e + 2x1o + 1x2e",
            output_irreps="1x0e + 1x1o",
            num_layers=1,
            num_heads=2,
            local_cutoff=1.0,
            use_transient_l3_workspace=True,
            transient_l3_channels=1,
            transient_l3_layers=(0,),
        )
    ).double()
    features = torch.randn(2, 4, dtype=torch.float64)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    self_edges = torch.tensor([[0, 1], [0, 1]])
    outside_candidates = torch.tensor(
        [[0, 1, 0, 1], [0, 1, 1, 0]]
    )
    kwargs = (
        {
            "packed_neighbors": build_receiver_csr(
                self_edges,
                num_nodes=2,
            )
        }
        if packed
        else {"edge_index": self_edges}
    )
    expected = model(features, positions, **kwargs)
    kwargs = (
        {
            "packed_neighbors": build_receiver_csr(
                outside_candidates,
                num_nodes=2,
            )
        }
        if packed
        else {"edge_index": outside_candidates}
    )
    actual = model(features, positions, **kwargs)

    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_every_configured_transient_l3_layer_reaches_scalar_loss() -> None:
    structured = ArchitectureConfig.for_profile(
        "high_order",
        node_dim=4,
        width=16,
        num_heads=2,
        num_layers=2,
    )
    model = EquivariantAttention(structured.to_legacy()).double()
    features = torch.randn(
        5,
        4,
        generator=torch.Generator().manual_seed(20260828),
        dtype=torch.float64,
    )
    positions = _fixture()[1]
    edge_index = _fixture()[2]

    loss = model(
        features,
        positions,
        edge_index=edge_index,
    )["graph_scalars"].square().sum()
    loss.backward()

    for layer in model.transient_l3_layers.values():
        assert layer.lift_weight.grad is not None
        assert layer.project_weight.grad is not None
        assert torch.isfinite(layer.lift_weight.grad).all()
        assert torch.isfinite(layer.project_weight.grad).all()


def test_transient_l3_rejects_cross_graph_candidates() -> None:
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            hidden_irreps="16x0e + 2x1o + 1x2e",
            output_irreps="1x0e",
            num_layers=1,
            num_heads=2,
            use_transient_l3_workspace=True,
            transient_l3_layers=(0,),
        )
    ).double()
    features = torch.randn(2, 4, dtype=torch.float64)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=torch.float64,
    )
    batch = torch.tensor([0, 1])
    cross_graph_edges = torch.tensor(
        [[0, 1, 0, 1], [0, 1, 1, 0]]
    )

    with pytest.raises(ValueError, match="same graph"):
        model(
            features,
            positions,
            batch=batch,
            edge_index=cross_graph_edges,
        )


def test_disabled_transient_l3_preserves_rng_state_schema_and_outputs() -> None:
    common = dict(
        node_dim=4,
        hidden_irreps="16x0e + 2x1o + 2x2e",
        output_irreps="1x0e + 1x1o",
        num_layers=2,
        num_heads=2,
    )
    torch.manual_seed(20260824)
    incumbent = EquivariantAttention(EquivariantAttentionConfig(**common)).double()
    incumbent_rng = torch.random.get_rng_state()
    torch.manual_seed(20260824)
    explicit_disabled = EquivariantAttention(
        EquivariantAttentionConfig(
            **common,
            use_transient_l3_workspace=False,
        )
    ).double()

    assert torch.equal(torch.random.get_rng_state(), incumbent_rng)
    assert incumbent.state_dict().keys() == explicit_disabled.state_dict().keys()
    for name, expected in incumbent.state_dict().items():
        torch.testing.assert_close(
            explicit_disabled.state_dict()[name],
            expected,
            rtol=0,
            atol=0,
        )
    features = torch.randn(5, 4, dtype=torch.float64)
    positions = _fixture()[1]
    batch = torch.zeros(5, dtype=torch.long)
    incumbent_output = incumbent(features, positions, batch=batch)
    disabled_output = explicit_disabled(features, positions, batch=batch)
    for name in incumbent_output:
        torch.testing.assert_close(
            disabled_output[name],
            incumbent_output[name],
            rtol=0,
            atol=0,
        )
