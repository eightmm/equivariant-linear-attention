import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
import equivariant_attention.moment as moment


def _config(*, coordinate_updates: bool, routing: str = "lgl") -> EquivariantAttentionConfig:
    local_head_counts = {
        "ggg": (0, 0, 0),
        "lgl": (2, 0, 2),
    }[routing]
    return EquivariantAttentionConfig(
        node_dim=4,
        hidden_irreps="12x0e + 3x1o",
        output_irreps="2x0e + 2x1o + 1x2e",
        num_layers=3,
        num_heads=2,
        local_head_counts=local_head_counts,
        coordinate_updates=coordinate_updates,
    )


def _model(*, routing: str = "lgl") -> EquivariantAttention:
    torch.manual_seed(701)
    model = EquivariantAttention(
        _config(coordinate_updates=True, routing=routing)
    ).double()
    with torch.no_grad():
        model.scalar_out.weight.normal_()
        model.scalar_out.bias.normal_()
    return model.eval()


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(703)
    return (
        torch.randn(7, 4, dtype=torch.float64),
        torch.randn(7, 3, dtype=torch.float64),
        torch.tensor([0, 0, 0, 1, 1, 1, 1]),
    )


def _orthogonal(*, reflection: bool) -> torch.Tensor:
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if (torch.linalg.det(matrix) < 0) != reflection:
        matrix[:, 0].neg_()
    return matrix


def _graph_means(value: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    means = []
    for graph in range(int(batch.max()) + 1):
        means.append(value[batch == graph].mean(dim=0))
    return torch.stack(means)


def test_coordinate_updates_are_opt_in_without_changing_default_state_or_output() -> None:
    torch.manual_seed(709)
    default = EquivariantAttention(
        EquivariantAttentionConfig(node_dim=4)
    ).double()
    torch.manual_seed(709)
    explicit_off = EquivariantAttention(
        EquivariantAttentionConfig(node_dim=4, coordinate_updates=False)
    ).double()
    node_feats = torch.randn(5, 4, dtype=torch.float64)
    pos = torch.randn(5, 3, dtype=torch.float64)

    default_output = default(node_feats, pos)
    explicit_output = explicit_off(node_feats, pos)

    assert list(default.state_dict()) == list(explicit_off.state_dict())
    for name, tensor in default.state_dict().items():
        assert torch.equal(tensor, explicit_off.state_dict()[name])
    assert set(default_output) == {
        "node_scalars",
        "node_vectors",
        "node_tensors",
        "graph_scalars",
        "graph_vectors",
        "graph_tensors",
    }
    for name in default_output:
        assert torch.equal(default_output[name], explicit_output[name])

    dynamic = _model()
    dynamic_output = dynamic(node_feats, pos)
    assert set(dynamic_output) == {*default_output, "node_positions"}
    assert any("coordinate_updaters" in name for name in dynamic.state_dict())
    assert dynamic_output["node_positions"].shape == pos.shape


@pytest.mark.parametrize("reflection", [False, True])
def test_attention_updated_positions_are_o3_translation_and_permutation_equivariant(
    reflection: bool,
) -> None:
    model = _model()
    node_feats, pos, batch = _inputs()
    transform = _orthogonal(reflection=reflection)
    translation = torch.randn(1, 3, dtype=torch.float64)
    permutation = torch.tensor([2, 0, 1, 6, 4, 3, 5])
    inverse = torch.argsort(permutation)

    reference = model(node_feats, pos, batch=batch)
    moved = model(
        node_feats,
        pos @ transform.T + translation,
        batch=batch,
    )
    permuted = model(
        node_feats[permutation],
        pos[permutation],
        batch=batch[permutation],
    )

    assert not torch.equal(reference["node_positions"], pos)
    assert torch.allclose(
        moved["node_positions"],
        reference["node_positions"] @ transform.T + translation,
        atol=1e-9,
        rtol=1e-9,
    )
    assert torch.allclose(
        permuted["node_positions"][inverse],
        reference["node_positions"],
        atol=1e-9,
        rtol=1e-9,
    )
    assert torch.allclose(
        moved["graph_scalars"], reference["graph_scalars"], atol=1e-9, rtol=1e-9
    )
    assert torch.allclose(
        permuted["graph_scalars"],
        reference["graph_scalars"],
        atol=1e-9,
        rtol=1e-9,
    )


def test_attention_coordinate_steps_are_bounded_and_preserve_graph_centroids() -> None:
    model = _model()
    node_feats, pos, batch = _inputs()
    steps: list[torch.Tensor] = []
    handles = [
        updater.register_forward_hook(
            lambda _module, _inputs, output: steps.append(output.detach())
        )
        for updater in model.coordinate_updaters
    ]
    try:
        output = model(node_feats, pos, batch=batch)
    finally:
        for handle in handles:
            handle.remove()

    assert len(steps) == 2
    for step in steps:
        assert float(torch.linalg.vector_norm(step, dim=-1).max()) <= 0.25 + 1e-12
        assert torch.allclose(
            _graph_means(step, batch), torch.zeros(2, 3, dtype=torch.float64),
            atol=1e-12, rtol=0.0,
        )
    assert torch.allclose(
        _graph_means(output["node_positions"], batch),
        _graph_means(pos, batch),
        atol=1e-12,
        rtol=0.0,
    )


def test_dynamic_attention_recomputes_local_and_global_geometry_per_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_calls = 0
    global_calls = 0
    original_local = moment._local_geometry
    original_global = moment._scale_first_geometry

    def counted_local(*args: object, **kwargs: object) -> tuple[torch.Tensor, ...]:
        nonlocal local_calls
        local_calls += 1
        return original_local(*args, **kwargs)

    def counted_global(*args: object, **kwargs: object) -> tuple[torch.Tensor, ...]:
        nonlocal global_calls
        global_calls += 1
        return original_global(*args, **kwargs)

    monkeypatch.setattr(moment, "_local_geometry", counted_local)
    monkeypatch.setattr(moment, "_scale_first_geometry", counted_global)
    node_feats, pos, batch = _inputs()

    _model(routing="lgl")(node_feats, pos, batch=batch)
    assert local_calls == 2
    assert global_calls == 1

    local_calls = 0
    global_calls = 0
    _model(routing="ggg")(node_feats, pos, batch=batch)
    assert local_calls == 0
    assert global_calls == 3


def test_attention_coordinate_update_handles_singletons_coincident_nodes_and_gradients() -> None:
    model = _model().train()
    node_feats = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    batch = torch.tensor([0, 1, 1, 1])

    output = model(node_feats, pos, batch=batch)
    loss = output["graph_scalars"].square().sum() + output[
        "node_positions"
    ].square().sum()
    loss.backward()

    assert torch.equal(output["node_positions"][0], pos[0])
    assert torch.isfinite(output["node_positions"]).all()
    assert pos.grad is not None and torch.isfinite(pos.grad).all()
    coordinate_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "coordinate_updaters" in name
    ]
    assert coordinate_parameters
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in coordinate_parameters
    )
    assert any(torch.count_nonzero(parameter.grad) for parameter in coordinate_parameters)


def test_dynamic_attention_scalar_coordinate_gradients_are_o3_covariant() -> None:
    model = _model()
    node_feats, pos, batch = _inputs()
    pos.requires_grad_()
    transform = _orthogonal(reflection=True)
    translation = torch.randn(1, 3, dtype=torch.float64)

    reference = model(node_feats, pos, batch=batch)["graph_scalars"].square().sum()
    reference_gradient = torch.autograd.grad(reference, pos)[0]
    moved_pos = (pos.detach() @ transform.T + translation).requires_grad_()
    moved = model(node_feats, moved_pos, batch=batch)["graph_scalars"].square().sum()
    moved_gradient = torch.autograd.grad(moved, moved_pos)[0]

    assert torch.allclose(moved, reference, atol=1e-9, rtol=1e-9)
    assert torch.allclose(
        moved_gradient,
        reference_gradient @ transform.T,
        atol=1e-8,
        rtol=1e-8,
    )


def test_coordinate_updates_requires_an_exact_bool() -> None:
    with pytest.raises(TypeError, match="coordinate_updates must be a bool"):
        EquivariantAttention(
            EquivariantAttentionConfig(node_dim=4, coordinate_updates=1)  # type: ignore[arg-type]
        )


def test_coordinate_updates_require_two_layers_and_a_hidden_vector_channel() -> None:
    with pytest.raises(ValueError, match="at least two layers"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=4,
                num_layers=1,
                coordinate_updates=True,
            )
        )
    with pytest.raises(ValueError, match="scalar and vector channels"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=4,
                hidden_irreps="8x0e",
                num_layers=2,
                coordinate_updates=True,
            )
        )


def test_coordinate_updates_keep_bfloat16_features_and_float32_geometry_finite() -> None:
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            hidden_irreps="8x0e + 2x1o",
            output_irreps="1x0e",
            num_layers=2,
            num_heads=2,
            coordinate_updates=True,
        )
    ).to(dtype=torch.bfloat16)
    node_feats = torch.randn(5, 4, dtype=torch.bfloat16)
    pos = torch.randn(5, 3, dtype=torch.float32, requires_grad=True)

    output = model(node_feats, pos)
    loss = output["graph_scalars"].float().square().sum() + output[
        "node_positions"
    ].square().sum()
    loss.backward()

    assert output["node_positions"].dtype == torch.float32
    assert all(torch.isfinite(value).all() for value in output.values())
    assert pos.grad is not None and torch.isfinite(pos.grad).all()
