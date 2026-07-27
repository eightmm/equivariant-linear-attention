from __future__ import annotations

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
import equivariant_attention.moment as moment
from equivariant_attention.training import build_regression_model


def _model(
    *,
    geometry_aware: bool,
    symmetry_group: str = "O3",
    axial: bool = False,
) -> EquivariantAttention:
    return EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=5,
            hidden_irreps="12x0e + 3x1o",
            output_irreps="2x0e + 1x1o + 1x2e",
            num_layers=3,
            num_heads=3,
            local_head_counts=(3, 0, 3),
            use_key_balancing=False,
            use_gated_local_transport=True,
            use_grouped_invariant_normalization=True,
            symmetry_group=symmetry_group,
            use_geometry_aware_local_attention=geometry_aware,
            use_se3_axial_tensor_product=axial,
        )
    ).double()


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260728)
    node_feats = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    pos = 0.3 * torch.randn(7, 3, generator=generator, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    graph_nodes = (torch.arange(3), torch.arange(3, 7))
    receiver = torch.cat(
        [nodes.repeat_interleave(nodes.numel()) for nodes in graph_nodes]
    )
    sender = torch.cat([nodes.repeat(nodes.numel()) for nodes in graph_nodes])
    edge_index = torch.stack([receiver, sender])
    return node_feats, pos, batch, edge_index


def _proper_rotation() -> torch.Tensor:
    rotation, _ = torch.linalg.qr(
        torch.tensor(
            [
                [0.7, -0.2, 0.4],
                [0.1, 0.9, 0.3],
                [-0.5, 0.2, 0.8],
            ],
            dtype=torch.float64,
        )
    )
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0].neg_()
    return rotation


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        (
            {"symmetry_group": 3},
            TypeError,
            "symmetry_group must be a string",
        ),
        (
            {"symmetry_group": "SO3"},
            ValueError,
            "symmetry_group must be one of",
        ),
        (
            {"use_geometry_aware_local_attention": True},
            ValueError,
            "geometry-aware local attention requires gated local transport",
        ),
        (
            {
                "use_gated_local_transport": True,
                "use_se3_axial_tensor_product": True,
            },
            ValueError,
            "axial tensor product requires geometry-aware local attention",
        ),
        (
            {
                "use_gated_local_transport": True,
                "use_geometry_aware_local_attention": True,
                "use_se3_axial_tensor_product": True,
            },
            ValueError,
            "axial tensor product requires symmetry_group='SE3'",
        ),
        (
            {
                "use_gated_local_transport": True,
                "use_geometry_aware_local_attention": True,
                "geometry_aware_local_layers": (1,),
            },
            ValueError,
            "geometry_aware_local_layers must select local stages",
        ),
    ],
)
def test_geometry_attention_rejects_invalid_symmetry_contract(
    overrides: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "node_dim": 4,
        "hidden_irreps": "8x0e + 2x1o",
        "num_layers": 2,
        "num_heads": 2,
        "local_head_counts": (2, 0),
    }
    kwargs.update(overrides)
    with pytest.raises(error, match=message):
        EquivariantAttention(EquivariantAttentionConfig(**kwargs))  # type: ignore[arg-type]


def test_geometry_attention_builder_wires_selectable_se3_path() -> None:
    model = build_regression_model(
        node_dim=5,
        hidden_dim=48,
        num_layers=3,
        num_heads=3,
        local_head_counts=(3, 0, 3),
        use_gated_local_transport=True,
        symmetry_group="SE3",
        use_geometry_aware_local_attention=True,
        use_se3_axial_tensor_product=True,
    )

    assert model.symmetry == "SE3"
    for layer, local_heads in zip(model.layers, (3, 0, 3), strict=True):
        if local_heads:
            assert layer.gated_local is not None
            assert layer.gated_local.geometry_attention is not None
            assert layer.gated_local.geometry_attention.axial_gate is not None
        else:
            assert layer.gated_local is None


def test_disabled_geometry_attention_is_exactly_backward_compatible() -> None:
    torch.manual_seed(2801)
    incumbent = _model(geometry_aware=False)
    torch.manual_seed(2801)
    explicit = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=5,
            hidden_irreps="12x0e + 3x1o",
            output_irreps="2x0e + 1x1o + 1x2e",
            num_layers=3,
            num_heads=3,
            local_head_counts=(3, 0, 3),
            use_key_balancing=False,
            use_gated_local_transport=True,
            use_grouped_invariant_normalization=True,
            symmetry_group="O3",
            use_geometry_aware_local_attention=False,
            use_se3_axial_tensor_product=False,
        )
    ).double()
    node_feats, pos, batch, edge_index = _inputs()

    assert incumbent.state_dict().keys() == explicit.state_dict().keys()
    for name, value in incumbent.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name]), name
    incumbent_output = incumbent(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index,
    )
    explicit_output = explicit(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index,
    )
    for name in incumbent_output:
        assert torch.equal(incumbent_output[name], explicit_output[name]), name


def test_geometry_attention_preserves_all_common_seeded_parameters() -> None:
    torch.manual_seed(2802)
    incumbent = _model(geometry_aware=False)
    torch.manual_seed(2802)
    candidate = _model(geometry_aware=True)

    incumbent_state = incumbent.state_dict()
    candidate_state = candidate.state_dict()
    common = incumbent_state.keys() & candidate_state.keys()
    assert common
    assert any("geometry_attention" in name for name in candidate_state)
    assert not any("geometry_attention" in name for name in incumbent_state)
    for name in common:
        assert torch.equal(incumbent_state[name], candidate_state[name]), name


def test_receiver_softmax_is_normalized_per_receiver_and_head() -> None:
    receiver = torch.tensor([0, 0, 1, 1, 1])
    logits = torch.tensor(
        [
            [-2.0, 0.5],
            [1.0, -1.5],
            [0.2, 0.3],
            [-0.4, 1.1],
            [0.7, -0.8],
        ],
        dtype=torch.float64,
    )
    mass = torch.tensor([1.0, 0.4, 0.8, 1.0, 0.2], dtype=torch.float64)

    weights = moment._receiver_softmax(
        logits,
        receiver,
        num_nodes=3,
        mass=mass,
    )
    totals = moment._fused_index_sum(receiver, 3, weights)[0]

    assert torch.allclose(totals[:2], torch.ones_like(totals[:2]), atol=1e-12)
    assert torch.equal(totals[2], torch.zeros_like(totals[2]))
    assert torch.all(weights > 0)


def test_geometry_attention_singleton_value_vanishes_at_cutoff() -> None:
    torch.manual_seed(2806)
    transport = moment._GatedEquivariantLocalTransport(
        scalars=4,
        vectors=1,
        num_heads=1,
        num_rbf=4,
        eps=1e-12,
        use_geometry_aware_local_attention=True,
        residual_scale_init=0.1,
    ).double()
    scalars = torch.randn(2, 4, dtype=torch.float64)
    vectors = torch.randn(2, 1, 3, dtype=torch.float64)
    batch = torch.zeros(2, dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 0], [0, 1, 1]])

    def evaluate(distance: float) -> torch.Tensor:
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
            dtype=torch.float64,
        )
        geometry = moment._local_geometry(
            pos,
            batch,
            num_graphs=1,
            cutoff=2.5,
            num_rbf=4,
            edge_index=edge_index,
        )
        return torch.cat(
            [
                value[0].reshape(-1)
                for value in transport(scalars, vectors, geometry, num_nodes=2)
            ]
        )

    near_boundary = evaluate(2.5 - 1e-5)
    outside = evaluate(2.5 + 1e-5)

    assert torch.allclose(near_boundary, outside, atol=2e-4, rtol=2e-4)


def test_active_geometry_attention_public_path_preserves_full_o3_and_permutations() -> (
    None
):
    torch.manual_seed(2803)
    model = _model(geometry_aware=True)
    node_feats, pos, batch, edge_index = _inputs()
    proper = _proper_rotation()
    reflection = torch.diag(
        torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    )
    orthogonal = reflection @ proper
    translation = torch.tensor([[0.3, -0.2, 0.4]], dtype=torch.float64)
    permutation = torch.tensor([4, 0, 6, 2, 3, 1, 5])
    inverse = torch.argsort(permutation)

    reference = model(node_feats, pos, batch=batch, edge_index=edge_index)
    moved = model(
        node_feats,
        pos @ orthogonal.T + translation,
        batch=batch,
        edge_index=edge_index,
    )
    permuted = model(
        node_feats[permutation],
        pos[permutation],
        batch=batch[permutation],
        edge_index=inverse[edge_index],
    )
    reordered = model(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index.flip(1),
    )

    assert torch.allclose(moved["node_scalars"], reference["node_scalars"], atol=1e-9)
    assert torch.allclose(
        moved["node_vectors"],
        torch.einsum("nca,ba->ncb", reference["node_vectors"], orthogonal),
        atol=1e-9,
    )
    expected_tensor = torch.einsum(
        "ab,nkbc,dc->nkad",
        orthogonal,
        reference["node_tensors"],
        orthogonal,
    )
    assert torch.allclose(moved["node_tensors"], expected_tensor, atol=1e-9)
    for name in ("node_scalars", "node_vectors", "node_tensors"):
        assert torch.allclose(permuted[name][inverse], reference[name], atol=1e-9)
        assert torch.allclose(reordered[name], reference[name], atol=1e-9)


def test_se3_axial_helper_obeys_proper_rotation_and_separates_reflection_parity() -> (
    None
):
    first = torch.tensor(
        [[[0.7, -0.2, 0.3, 0.4, -0.1]]],
        dtype=torch.float64,
    )
    second = torch.tensor(
        [[[-0.3, 0.6, 0.1, -0.5, 0.2]]],
        dtype=torch.float64,
    )
    proper = _proper_rotation()
    reflection = torch.diag(
        torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    )

    def transform(value: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        full = moment._st_features_to_matrix(value)
        moved = torch.einsum("ab,...bc,dc->...ad", matrix, full, matrix)
        return moment._st_matrix_to_features(moved)

    reference = moment._st_commutator_axial(first, second)
    proper_output = moment._st_commutator_axial(
        transform(first, proper),
        transform(second, proper),
    )
    reflected_output = moment._st_commutator_axial(
        transform(first, reflection),
        transform(second, reflection),
    )

    expected_proper = torch.einsum("...a,ba->...b", reference, proper)
    expected_axial_reflection = -torch.einsum(
        "...a,ba->...b",
        reference,
        reflection,
    )
    polar_reflection = torch.einsum("...a,ba->...b", reference, reflection)
    assert torch.allclose(proper_output, expected_proper, atol=1e-12)
    assert torch.allclose(
        reflected_output,
        expected_axial_reflection,
        atol=1e-12,
    )
    assert not torch.allclose(reflected_output, polar_reflection, atol=1e-8)


def test_active_se3_public_path_preserves_proper_rigid_motion() -> None:
    torch.manual_seed(2804)
    model = _model(
        geometry_aware=True,
        symmetry_group="SE3",
        axial=True,
    )
    node_feats, pos, batch, edge_index = _inputs()
    rotation = _proper_rotation()
    translation = torch.tensor([[-0.4, 0.2, 0.1]], dtype=torch.float64)

    reference = model(node_feats, pos, batch=batch, edge_index=edge_index)
    moved = model(
        node_feats,
        pos @ rotation.T + translation,
        batch=batch,
        edge_index=edge_index,
    )

    assert torch.allclose(moved["node_scalars"], reference["node_scalars"], atol=1e-9)
    assert torch.allclose(
        moved["node_vectors"],
        torch.einsum("nca,ba->ncb", reference["node_vectors"], rotation),
        atol=1e-9,
    )
    expected_tensor = torch.einsum(
        "ab,nkbc,dc->nkad",
        rotation,
        reference["node_tensors"],
        rotation,
    )
    assert torch.allclose(moved["node_tensors"], expected_tensor, atol=1e-9)


def test_geometry_attention_parameters_and_coordinates_receive_gradients() -> None:
    torch.manual_seed(2805)
    model = _model(
        geometry_aware=True,
        symmetry_group="SE3",
        axial=True,
    )
    node_feats, pos, batch, edge_index = _inputs()
    node_feats.requires_grad_()
    pos.requires_grad_()

    output = model(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index,
    )
    loss = output["graph_scalars"].square().sum()
    loss.backward()

    new_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "geometry_attention" in name
    ]
    assert new_gradients
    assert all(gradient is not None for gradient in new_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in new_gradients)
    assert any(torch.count_nonzero(gradient) for gradient in new_gradients)
    assert pos.grad is not None
    assert torch.isfinite(pos.grad).all()
    assert torch.count_nonzero(pos.grad)
