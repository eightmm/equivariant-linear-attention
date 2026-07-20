import torch

import equivariant_attention as package
from equivariant_attention._egnn_baseline import (
    _DynamicEGNNBaseline,
    _StaticEGNNBaseline,
    _StaticEGNNLayer,
    _directed_complete_edges_without_self as _model_edges,
)
from equivariant_attention.benchmarking import GraphBatch
from equivariant_attention.training import build_regression_model, predict_graph_scalar


def _directed_complete_edges_without_self(
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    receiver = []
    sender = []
    for i in range(batch.numel()):
        for j in range(batch.numel()):
            if i != j and int(batch[i]) == int(batch[j]):
                receiver.append(i)
                sender.append(j)
    return (
        torch.tensor(receiver, dtype=torch.long, device=batch.device),
        torch.tensor(sender, dtype=torch.long, device=batch.device),
    )


def _loop_layer_reference(
    layer: _StaticEGNNLayer,
    value: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    aggregates = []
    for i in range(value.shape[0]):
        messages = []
        for j in range(value.shape[0]):
            if i == j or int(batch[i]) != int(batch[j]):
                continue
            squared_distance = (pos[i] - pos[j]).square().sum().reshape(1)
            edge = layer.edge_mlp(
                torch.cat([value[i], value[j], squared_distance], dim=0)
            )
            messages.append(edge * layer.edge_gate(edge))
        aggregates.append(
            torch.stack(messages).sum(dim=0)
            if messages
            else value.new_zeros(value.shape[1])
        )
    aggregate = torch.stack(aggregates)
    return value + layer.node_mlp(torch.cat([value, aggregate], dim=-1))


def _nontrivial_model() -> _StaticEGNNBaseline:
    model = _StaticEGNNBaseline(node_dim=4, hidden_dim=8, num_layers=2).double()
    with torch.no_grad():
        model.scalar_out.weight.normal_()
        model.scalar_out.bias.normal_()
    return model.eval()


def test_static_egnn_edges_are_same_graph_directed_complete_without_self() -> None:
    batch = torch.tensor([0, 1, 0, 1, 1])
    graph_counts = torch.bincount(batch)

    receiver, sender = _model_edges(batch, graph_counts)
    expected_receiver, expected_sender = _directed_complete_edges_without_self(batch)

    assert set(zip(receiver.tolist(), sender.tolist(), strict=True)) == set(
        zip(expected_receiver.tolist(), expected_sender.tolist(), strict=True)
    )
    assert receiver.numel() == sum(
        int(count) * (int(count) - 1) for count in graph_counts
    )


def test_static_egnn_layer_matches_explicit_loop_forward_and_gradients() -> None:
    torch.manual_seed(101)
    layer = _StaticEGNNLayer(hidden_dim=5).double()
    value = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
    pos = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    batch = torch.tensor([0, 1, 0, 1, 0])
    receiver, sender = _directed_complete_edges_without_self(batch)

    actual = layer(value, pos, receiver, sender)
    expected = _loop_layer_reference(layer, value, pos, batch)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    differentiated = (value, pos, *tuple(layer.parameters()))
    actual_gradients = torch.autograd.grad(
        actual.square().sum(), differentiated, retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected.square().sum(), differentiated)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.isfinite(actual_gradient).all()
        assert torch.allclose(
            actual_gradient, expected_gradient, atol=1e-11, rtol=1e-10
        )
    assert actual_gradients[1].abs().sum() > 0.0


def test_static_egnn_graph_scalar_is_o3_translation_and_permutation_invariant() -> (
    None
):
    torch.manual_seed(103)
    model = _nontrivial_model()
    node_feats = torch.randn(7, 4, dtype=torch.float64)
    pos = torch.randn(7, 3, dtype=torch.float64)
    batch = torch.tensor([0, 1, 0, 1, 0, 1, 1])
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    orthogonal[:, 0].neg_()
    translation = torch.tensor([2.0, -3.0, 0.5], dtype=torch.float64)
    permutation = torch.tensor([5, 2, 0, 6, 3, 1, 4])

    reference = model(node_feats, pos, batch=batch)["graph_scalars"]
    transformed = model(
        node_feats,
        pos @ orthogonal.T + translation,
        batch=batch,
    )["graph_scalars"]
    permuted = model(
        node_feats[permutation],
        pos[permutation],
        batch=batch[permutation],
    )["graph_scalars"]

    assert torch.det(orthogonal) < 0.0
    assert torch.allclose(reference, transformed, atol=1e-10, rtol=1e-10)
    assert torch.allclose(reference, permuted, atol=1e-10, rtol=1e-10)


def test_static_egnn_coordinate_gradients_are_o3_covariant() -> None:
    torch.manual_seed(104)
    model = _nontrivial_model()
    node_feats = torch.randn(7, 4, dtype=torch.float64)
    pos = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    orthogonal[:, 0].neg_()
    translation = torch.randn(1, 3, dtype=torch.float64)

    reference = model(node_feats, pos, batch=batch)["graph_scalars"].square().sum()
    reference_gradient = torch.autograd.grad(reference, pos)[0]
    moved_pos = (pos.detach() @ orthogonal.T + translation).requires_grad_()
    moved = model(node_feats, moved_pos, batch=batch)["graph_scalars"].square().sum()
    moved_gradient = torch.autograd.grad(moved, moved_pos)[0]

    assert torch.allclose(moved, reference, atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        moved_gradient,
        reference_gradient @ orthogonal.T,
        atol=1e-9,
        rtol=1e-9,
    )


def test_static_egnn_keeps_graphs_isolated() -> None:
    torch.manual_seed(107)
    model = _nontrivial_model()
    first_feats = torch.randn(4, 4, dtype=torch.float64)
    first_pos = torch.randn(4, 3, dtype=torch.float64)
    second_feats = torch.randn(3, 4, dtype=torch.float64)
    second_pos = torch.randn(3, 3, dtype=torch.float64)

    alone = model(first_feats, first_pos)["graph_scalars"]
    together = model(
        torch.cat([first_feats, second_feats]),
        torch.cat([first_pos, second_pos]),
        batch=torch.tensor([0, 0, 0, 0, 1, 1, 1]),
    )["graph_scalars"]

    assert torch.allclose(alone[0], together[0], atol=1e-12, rtol=1e-12)


def test_static_egnn_singleton_has_finite_backward() -> None:
    torch.manual_seed(109)
    model = _nontrivial_model().train()
    node_feats = torch.randn(1, 4, dtype=torch.float64, requires_grad=True)
    pos = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)

    output = model(node_feats, pos)["graph_scalars"]
    output.square().sum().backward()

    assert torch.isfinite(output).all()
    assert node_feats.grad is not None and torch.isfinite(node_feats.grad).all()
    assert pos.grad is None or torch.isfinite(pos.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_qm9_parameter_match_zero_init_minimal_output_and_private_api() -> None:
    model = _StaticEGNNBaseline(node_dim=11)
    lgl = build_regression_model(
        node_dim=11,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        local_head_counts=(4, 0, 4),
    )
    batch = GraphBatch(
        node_feats=torch.randn(5, 11),
        pos=torch.randn(5, 3),
        batch=torch.tensor([0, 0, 0, 1, 1]),
        target=torch.randn(2, 1),
        sample_ids=("first", "second"),
    )

    output = model(batch.node_feats, batch.pos, batch=batch.batch)

    egnn_total = sum(parameter.numel() for parameter in model.parameters())
    egnn_trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    lgl_total = sum(parameter.numel() for parameter in lgl.parameters())
    lgl_trainable = sum(
        parameter.numel() for parameter in lgl.parameters() if parameter.requires_grad
    )
    assert (egnn_total, egnn_trainable) == (152_065, 152_065)
    assert (lgl_total, lgl_trainable) == (153_285, 153_081)
    assert abs(egnn_trainable - lgl_trainable) / lgl_trainable < 0.01
    assert len(model.layers) == 3
    assert model.hidden_dim == 91
    assert torch.count_nonzero(model.scalar_out.weight) == 0
    assert torch.count_nonzero(model.scalar_out.bias) == 0
    assert set(output) == {"graph_scalars"}
    assert all(isinstance(value, torch.Tensor) for value in output.values())
    assert output["graph_scalars"].shape == (2, 1)
    assert predict_graph_scalar(model, batch).shape == (2, 1)
    assert not hasattr(package, "EGNN")
    assert not hasattr(package, "StaticEGNNBaseline")
    assert not hasattr(package, "build_egnn_baseline")


def _dynamic_model() -> _DynamicEGNNBaseline:
    torch.manual_seed(131)
    model = _DynamicEGNNBaseline(node_dim=4, hidden_dim=8, num_layers=3).double()
    with torch.no_grad():
        model.scalar_out.weight.normal_()
        model.scalar_out.bias.normal_()
    return model.eval()


def test_dynamic_egnn_positions_are_o3_translation_and_permutation_equivariant() -> None:
    model = _dynamic_model()
    torch.manual_seed(137)
    node_feats = torch.randn(7, 4, dtype=torch.float64)
    pos = torch.randn(7, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    orthogonal[:, 0].neg_()
    translation = torch.randn(1, 3, dtype=torch.float64)
    permutation = torch.tensor([2, 0, 1, 6, 4, 3, 5])
    inverse = torch.argsort(permutation)

    reference = model(node_feats, pos, batch=batch)
    moved = model(
        node_feats, pos @ orthogonal.T + translation, batch=batch
    )
    permuted = model(
        node_feats[permutation], pos[permutation], batch=batch[permutation]
    )

    assert set(reference) == {"graph_scalars", "node_positions"}
    assert not torch.equal(reference["node_positions"], pos)
    assert torch.allclose(
        moved["node_positions"],
        reference["node_positions"] @ orthogonal.T + translation,
        atol=1e-10,
        rtol=1e-10,
    )
    assert torch.allclose(
        permuted["node_positions"][inverse],
        reference["node_positions"],
        atol=1e-10,
        rtol=1e-10,
    )
    assert torch.allclose(
        moved["graph_scalars"], reference["graph_scalars"], atol=1e-10, rtol=1e-10
    )


def test_dynamic_egnn_steps_are_bounded_centered_and_batch_isolated() -> None:
    model = _dynamic_model()
    torch.manual_seed(139)
    first_feats = torch.randn(4, 4, dtype=torch.float64)
    first_pos = torch.randn(4, 3, dtype=torch.float64)
    second_feats = torch.randn(3, 4, dtype=torch.float64)
    second_pos = torch.randn(3, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    steps: list[torch.Tensor] = []
    handles = [
        updater.register_forward_hook(
            lambda _module, _inputs, output: steps.append(output.detach())
        )
        for updater in model.coordinate_updaters
    ]
    try:
        together = model(
            torch.cat([first_feats, second_feats]),
            torch.cat([first_pos, second_pos]),
            batch=batch,
        )
        alone = model(first_feats, first_pos)
    finally:
        for handle in handles:
            handle.remove()

    assert len(steps) >= 2
    for step in steps[:2]:
        assert float(torch.linalg.vector_norm(step, dim=-1).max()) <= 0.25 + 1e-12
        for graph in range(2):
            assert torch.allclose(
                step[batch == graph].mean(dim=0),
                torch.zeros(3, dtype=torch.float64),
                atol=1e-12,
                rtol=0.0,
            )
    assert torch.allclose(
        together["node_positions"][:4],
        alone["node_positions"],
        atol=1e-12,
        rtol=1e-12,
    )
    for graph in range(2):
        index = batch == graph
        original = torch.cat([first_pos, second_pos])[index]
        assert torch.allclose(
            together["node_positions"][index].mean(dim=0),
            original.mean(dim=0),
            atol=1e-12,
            rtol=0.0,
        )


def test_dynamic_egnn_singleton_and_coordinate_parameters_have_finite_gradients() -> None:
    model = _dynamic_model().train()
    node_feats = torch.randn(1, 4, dtype=torch.float64, requires_grad=True)
    pos = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)

    output = model(node_feats, pos)
    loss = output["graph_scalars"].square().sum() + output[
        "node_positions"
    ].square().sum()
    loss.backward()

    assert torch.equal(output["node_positions"], pos)
    assert pos.grad is not None and torch.isfinite(pos.grad).all()
    coordinate_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "coordinate_updaters" in name
    ]
    assert coordinate_parameters
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in coordinate_parameters
    )


def test_dynamic_egnn_coordinate_parameters_receive_nonzero_gradients() -> None:
    model = _dynamic_model().train()
    node_feats = torch.randn(4, 4, dtype=torch.float64)
    pos = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)

    output = model(node_feats, pos)
    output["node_positions"].square().sum().backward()

    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "coordinate_updaters" in name
    ]
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


def test_dynamic_egnn_scalar_coordinate_gradients_are_o3_covariant() -> None:
    model = _dynamic_model()
    node_feats = torch.randn(7, 4, dtype=torch.float64)
    pos = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    transform, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    transform[:, 0].neg_()
    translation = torch.randn(1, 3, dtype=torch.float64)

    reference = model(node_feats, pos, batch=batch)["graph_scalars"].square().sum()
    reference_gradient = torch.autograd.grad(reference, pos)[0]
    moved_pos = (pos.detach() @ transform.T + translation).requires_grad_()
    moved = model(node_feats, moved_pos, batch=batch)["graph_scalars"].square().sum()
    moved_gradient = torch.autograd.grad(moved, moved_pos)[0]

    assert torch.allclose(moved, reference, atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        moved_gradient,
        reference_gradient @ transform.T,
        atol=1e-9,
        rtol=1e-9,
    )
