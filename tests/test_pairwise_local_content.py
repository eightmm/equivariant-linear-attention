import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
import equivariant_attention.moment as moment


def _config(*, pairwise: bool) -> EquivariantAttentionConfig:
    return EquivariantAttentionConfig(
        node_dim=5,
        hidden_irreps="16x0e + 4x1o",
        output_irreps="2x0e + 1x1o + 1x2e",
        num_layers=3,
        num_heads=2,
        local_head_counts=(2, 0, 2),
        use_pairwise_local_content=pairwise,
    )


def _model() -> EquivariantAttention:
    torch.manual_seed(811)
    model = EquivariantAttention(_config(pairwise=True)).double()
    with torch.no_grad():
        model.scalar_out.weight.normal_()
        model.scalar_out.bias.normal_()
    return model.eval()


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(813)
    return (
        torch.randn(8, 5, dtype=torch.float64),
        torch.randn(8, 3, dtype=torch.float64),
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
    )


def _orthogonal(*, reflection: bool) -> torch.Tensor:
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if (torch.linalg.det(matrix) < 0) != reflection:
        matrix[:, 0].neg_()
    return matrix


def test_pairwise_local_content_is_opt_in_and_preserves_default_state() -> None:
    torch.manual_seed(817)
    default = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=5,
            local_head_counts=(2, 0, 2),
        )
    ).double()
    torch.manual_seed(817)
    explicit_off = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=5,
            local_head_counts=(2, 0, 2),
            use_pairwise_local_content=False,
        )
    ).double()
    node_feats, pos, batch = _inputs()

    default_output = default(node_feats, pos, batch=batch)
    explicit_output = explicit_off(node_feats, pos, batch=batch)

    assert list(default.state_dict()) == list(explicit_off.state_dict())
    for name, value in default.state_dict().items():
        assert torch.equal(value, explicit_off.state_dict()[name])
    for name, value in default_output.items():
        assert torch.equal(value, explicit_output[name])
    assert not any("local_pairwise_content" in name for name in default.state_dict())


@pytest.mark.parametrize("reflection", [False, True])
def test_pairwise_local_content_preserves_o3_translation_and_permutation(
    reflection: bool,
) -> None:
    model = _model()
    node_feats, pos, batch = _inputs()
    transform = _orthogonal(reflection=reflection)
    translation = torch.randn(1, 3, dtype=torch.float64)
    permutation = torch.tensor([3, 0, 2, 1, 7, 5, 4, 6])
    inverse = torch.argsort(permutation)

    reference = model(node_feats, pos, batch=batch)
    moved = model(node_feats, pos @ transform.T + translation, batch=batch)
    permuted = model(
        node_feats[permutation],
        pos[permutation],
        batch=batch[permutation],
    )

    assert torch.allclose(
        moved["graph_scalars"], reference["graph_scalars"], atol=1e-9, rtol=1e-9
    )
    assert torch.allclose(
        moved["graph_vectors"],
        reference["graph_vectors"] @ transform.T,
        atol=1e-9,
        rtol=1e-9,
    )
    assert torch.allclose(
        permuted["node_scalars"][inverse],
        reference["node_scalars"],
        atol=1e-9,
        rtol=1e-9,
    )


def test_pairwise_local_content_preserves_batch_isolation() -> None:
    model = _model()
    node_feats, pos, batch = _inputs()

    mixed = model(node_feats, pos, batch=batch)["graph_scalars"][0]
    isolated = model(node_feats[:4], pos[:4])["graph_scalars"][0]

    assert torch.allclose(mixed, isolated, atol=1e-9, rtol=1e-9)


def test_pairwise_local_parameters_are_active_and_parameter_bounded() -> None:
    model = _model().train()
    baseline = EquivariantAttention(_config(pairwise=False)).double()
    node_feats, pos, batch = _inputs()

    loss = model(node_feats, pos, batch=batch)["graph_scalars"].square().sum()
    loss.backward()

    pairwise_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "local_pairwise_content" in name
    ]
    assert pairwise_parameters
    assert all(parameter.grad is not None for parameter in pairwise_parameters)
    assert all(
        torch.isfinite(parameter.grad).all() for parameter in pairwise_parameters
    )
    assert all(torch.count_nonzero(parameter.grad) for parameter in pairwise_parameters)
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_count = sum(parameter.numel() for parameter in model.parameters())
    assert (candidate_count - baseline_count) / baseline_count < 0.05


def test_pairwise_local_content_requires_exact_bool_and_local_heads() -> None:
    with pytest.raises(TypeError, match="use_pairwise_local_content must be a bool"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=5,
                use_pairwise_local_content=1,  # type: ignore[arg-type]
            )
        )
    with pytest.raises(ValueError, match="requires at least one local head"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=5,
                use_pairwise_local_content=True,
            )
        )


def test_pairwise_local_content_supports_exact_baseline_initialization() -> None:
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=5,
            hidden_irreps="16x0e + 4x1o",
            num_heads=2,
            local_head_counts=(2, 0, 2),
            use_pairwise_local_content=True,
            pairwise_residual_scale_init=0.0,
        )
    )

    assert model.local_pairwise_content is not None
    assert model.local_pairwise_content.residual_scale.item() == 0.0

    with pytest.raises(ValueError, match="pairwise_residual_scale_init"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=5,
                local_head_counts=(2, 0, 2),
                use_pairwise_local_content=True,
                pairwise_residual_scale_init=-0.1,
            )
        )


def test_pairwise_local_content_is_continuous_at_cutoff() -> None:
    torch.manual_seed(823)
    module = moment._LocalPairwiseContent(
        head_dim=4,
        num_rbf=4,
        residual_scale_init=0.1,
        eps=1e-12,
    ).double()
    query = torch.randn(3, 1, 4, dtype=torch.float64)
    key = torch.randn(3, 1, 4, dtype=torch.float64)
    batch = torch.zeros(3, dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 1, 2]])

    def evaluate(distance: float) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [0.8, 0.2, 0.0], [distance, 0.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        geometry = moment._local_geometry(
            pos,
            batch,
            num_graphs=1,
            cutoff=2.5,
            num_rbf=4,
            edge_index=edge_index,
        )
        output = module(query, key, geometry, num_nodes=3)
        gradient = torch.autograd.grad(output.square().sum(), pos)[0]
        return output.detach(), gradient.detach()

    inside, inside_gradient = evaluate(2.5 - 1e-5)
    outside, outside_gradient = evaluate(2.5 + 1e-5)

    assert torch.allclose(inside, outside, atol=2e-4, rtol=2e-4)
    assert torch.allclose(
        inside_gradient,
        outside_gradient,
        atol=2e-3,
        rtol=2e-3,
    )
