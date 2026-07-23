import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
import equivariant_attention.moment as moment


def _orthogonal(*, reflection: bool) -> torch.Tensor:
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if (torch.linalg.det(matrix) < 0) != reflection:
        matrix[:, 0].neg_()
    return matrix


def _max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).detach().abs().max())


def _transform_tensor(value: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ab,...bc,dc->...ad", transform, value, transform)


def _config(
    *,
    coordinate_updates: bool = False,
) -> EquivariantAttentionConfig:
    return EquivariantAttentionConfig(
        node_dim=5,
        hidden_irreps="12x0e + 3x1o",
        output_irreps="2x0e + 2x1o + 1x2e",
        num_layers=3,
        num_heads=3,
        local_head_counts=(0, 0, 0),
        coordinate_updates=coordinate_updates,
        use_multiscale_spatial_kernel=True,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1703)
    return (
        torch.randn(9, 5, dtype=torch.float64),
        torch.randn(9, 3, dtype=torch.float64),
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1]),
    )


def test_quadratic_gaussian_features_have_positive_o3_invariant_dot_products() -> (
    None
):
    torch.manual_seed(1701)
    pos = torch.randn(7, 3, dtype=torch.float64)
    scales = torch.tensor([0.125, 0.5, 1.0], dtype=torch.float64)
    transform = _orthogonal(reflection=True)

    features = moment._quadratic_gaussian_spatial_features(pos, scales)
    moved = moment._quadratic_gaussian_spatial_features(pos @ transform.T, scales)
    kernel = torch.einsum("ihf,jhf->hij", features, features)
    moved_kernel = torch.einsum("ihf,jhf->hij", moved, moved)

    assert features.shape == (7, 3, 10)
    assert torch.all(kernel > 0.0)
    assert torch.allclose(moved_kernel, kernel, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("single_graph", [False, True])
def test_spatial_factorization_matches_materialized_dense_kernel(
    balanced: bool,
    single_graph: bool,
) -> None:
    torch.manual_seed(1705)
    query_scalar = moment._normalize_positive_features(
        torch.rand(8, 2, 4, dtype=torch.float64),
        eps=1e-12,
    )
    key_scalar = moment._normalize_positive_features(
        torch.rand(8, 2, 4, dtype=torch.float64),
        eps=1e-12,
    )
    query_vector = moment._unit_ball(
        torch.randn(8, 2, 3, dtype=torch.float64),
        eps=1e-12,
    )
    key_vector = moment._unit_ball(
        torch.randn(8, 2, 3, dtype=torch.float64),
        eps=1e-12,
    )
    pos = torch.randn(8, 3, dtype=torch.float64)
    spatial = moment._quadratic_gaussian_spatial_features(
        pos,
        torch.tensor([0.25, 1.0], dtype=torch.float64),
    )
    linear_scale = torch.tensor([0.1, 0.2], dtype=torch.float64)
    quadratic_scale = torch.tensor([0.3, 0.6], dtype=torch.float64)
    value = torch.randn(8, 2, 5, dtype=torch.float64)
    batch = (
        torch.zeros(8, dtype=torch.long)
        if single_graph
        else torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
    )
    num_graphs = int(batch.max().item()) + 1

    actual = moment._factorized_moment_attention(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        quadratic_scale,
        value,
        batch,
        num_graphs=num_graphs,
        balanced=balanced,
        alignment_scale=linear_scale,
        alignment_dot_scale=linear_scale,
        kernel_floor=0.5,
        spatial_features=spatial,
    )
    expected = torch.empty_like(value)
    for graph in range(num_graphs):
        index = batch == graph
        content = torch.einsum(
            "ihd,jhd->hij",
            query_scalar[index],
            key_scalar[index],
        )
        angular = torch.einsum(
            "iha,jha->hij",
            query_vector[index],
            key_vector[index],
        )
        spatial_kernel = torch.einsum(
            "ihf,jhf->hij",
            spatial[index],
            spatial[index],
        )
        kernel = (
            0.5
            + content
            + linear_scale[:, None, None] * (1.0 + angular)
            + quadratic_scale[:, None, None] * angular.square()
            + spatial_kernel
        )
        if balanced:
            kernel = kernel / kernel.sum(dim=1, keepdim=True)
        weights = kernel / kernel.sum(dim=2, keepdim=True)
        expected[index] = torch.einsum("hij,jhf->ihf", weights, value[index])

    assert _max_error(actual, expected) < 1e-10


@pytest.mark.parametrize("reflection", [False, True])
@pytest.mark.parametrize("coordinate_updates", [False, True])
def test_edge_free_spatial_model_preserves_symmetry_batch_and_coordinates(
    reflection: bool,
    coordinate_updates: bool,
) -> None:
    torch.manual_seed(1707)
    model = EquivariantAttention(
        _config(coordinate_updates=coordinate_updates)
    ).double().eval()
    node_feats, pos, batch = _inputs()
    transform = _orthogonal(reflection=reflection)
    translation = torch.randn(1, 3, dtype=torch.float64)
    permutation = torch.tensor([3, 0, 2, 1, 7, 5, 4, 8, 6])
    inverse = torch.argsort(permutation)

    reference = model(node_feats, pos, batch=batch)
    moved = model(node_feats, pos @ transform.T + translation, batch=batch)
    permuted = model(
        node_feats[permutation],
        pos[permutation],
        batch=batch[permutation],
    )
    independent = [
        model(node_feats[batch == graph], pos[batch == graph])
        for graph in range(2)
    ]

    for scope in ("node", "graph"):
        assert (
            _max_error(
                moved[f"{scope}_scalars"],
                reference[f"{scope}_scalars"],
            )
            < 1e-6
        )
        assert (
            _max_error(
                moved[f"{scope}_vectors"],
                torch.einsum(
                    "...a,ba->...b",
                    reference[f"{scope}_vectors"],
                    transform,
                ),
            )
            < 1e-6
        )
        assert (
            _max_error(
                moved[f"{scope}_tensors"],
                _transform_tensor(reference[f"{scope}_tensors"], transform),
            )
            < 1e-6
        )

    for key in ("node_scalars", "node_vectors", "node_tensors"):
        assert _max_error(permuted[key][inverse], reference[key]) < 1e-6
        assert _max_error(
            reference[key],
            torch.cat([output[key] for output in independent]),
        ) < 1e-6
    for key in ("graph_scalars", "graph_vectors", "graph_tensors"):
        assert _max_error(permuted[key], reference[key]) < 1e-6
        assert _max_error(
            reference[key],
            torch.cat([output[key] for output in independent]),
        ) < 1e-6
    if coordinate_updates:
        assert torch.allclose(
            moved["node_positions"],
            reference["node_positions"] @ transform.T + translation,
            atol=1e-9,
            rtol=1e-9,
        )
        assert torch.allclose(
            reference["node_positions"],
            torch.cat([output["node_positions"] for output in independent]),
            atol=1e-9,
            rtol=1e-9,
        )
        assert torch.allclose(
            permuted["node_positions"][inverse],
            reference["node_positions"],
            atol=1e-9,
            rtol=1e-9,
        )


def test_edge_free_spatial_option_is_opt_in_and_rejects_edge_routes() -> None:
    torch.manual_seed(1709)
    default = EquivariantAttention(EquivariantAttentionConfig(node_dim=5)).double()
    torch.manual_seed(1709)
    explicit_off = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=5,
            use_multiscale_spatial_kernel=False,
        )
    ).double()
    torch.manual_seed(1709)
    spatial = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=5,
            use_multiscale_spatial_kernel=True,
        )
    ).double()
    node_feats = torch.randn(5, 5, dtype=torch.float64)
    pos = torch.randn(5, 3, dtype=torch.float64)

    assert list(default.state_dict()) == list(explicit_off.state_dict())
    assert list(default.state_dict()) == list(spatial.state_dict())
    for name, value in default.state_dict().items():
        assert torch.equal(value, explicit_off.state_dict()[name])
        assert torch.equal(value, spatial.state_dict()[name])
    for name, value in default(node_feats, pos).items():
        assert torch.equal(value, explicit_off(node_feats, pos)[name])
    assert not torch.equal(
        default(node_feats, pos)["node_scalars"],
        spatial(node_feats, pos)["node_scalars"],
    )

    with pytest.raises(TypeError, match="use_multiscale_spatial_kernel"):
        EquivariantAttention(
            EquivariantAttentionConfig(  # type: ignore[arg-type]
                node_dim=5,
                use_multiscale_spatial_kernel=1,
            )
        )
    with pytest.raises(ValueError, match="all-global"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=5,
                num_layers=2,
                num_heads=2,
                local_head_counts=(2, 0),
                use_multiscale_spatial_kernel=True,
            )
        )
    with pytest.raises(ValueError, match="learned global transport"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=5,
                global_transport_mode="uniform",
                use_multiscale_spatial_kernel=True,
            )
        )


def test_edge_free_spatial_training_path_has_finite_input_and_parameter_gradients() -> (
    None
):
    torch.manual_seed(1711)
    model = EquivariantAttention(_config(coordinate_updates=True)).double()
    node_feats, pos, batch = _inputs()
    node_feats.requires_grad_()
    pos.requires_grad_()

    output = model(node_feats, pos, batch=batch)
    loss = (
        output["node_scalars"].square().mean()
        + output["node_vectors"].square().mean()
        + output["node_tensors"].square().mean()
        + output["node_positions"].square().mean()
    )
    loss.backward()

    assert node_feats.grad is not None and torch.isfinite(node_feats.grad).all()
    assert pos.grad is not None and torch.isfinite(pos.grad).all()
    parameter_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert parameter_gradients
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)
    assert any(torch.count_nonzero(gradient) for gradient in parameter_gradients)
