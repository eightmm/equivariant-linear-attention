from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.advanced import ELAConfig


def _complete_edges(nodes: int) -> torch.Tensor:
    sender = torch.arange(nodes).repeat(nodes)
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    return torch.stack([sender, receiver])


def _activate_coordinate_head(model: ELA) -> None:
    assert model.coordinate_head is not None
    with torch.no_grad():
        model.coordinate_head.base_weight.fill_(0.2)


def test_public_coordinate_update_is_one_hidden_state_preserving_stack() -> None:
    torch.manual_seed(701)
    depth = 3
    model = ELA(
        "4x0e",
        "1x0e",
        width=16,
        depth=depth,
        cutoff=10.0,
        update_positions=True,
        max_coordinate_step=0.1,
    ).double()
    _activate_coordinate_head(model)
    graph = ELAGraph(
        torch.randn(6, 4, dtype=torch.float64),
        torch.randn(6, 3, dtype=torch.float64),
        edge_index=_complete_edges(6),
    )

    layer_inputs: list[object] = []
    layer_outputs: list[object] = []
    handles = []
    for layer in model.layers:
        handles.append(
            layer.register_forward_pre_hook(
                lambda _module, args: layer_inputs.append(args[0])
            )
        )
        handles.append(
            layer.register_forward_hook(
                lambda _module, _args, output: layer_outputs.append(output)
            )
        )
    try:
        output = model(graph)
    finally:
        for handle in handles:
            handle.remove()

    assert model.config.coordinate_updates == depth
    assert model.config.coordinate_update_layers == (1, 2, 3)
    assert len(layer_inputs) == depth
    assert len(layer_outputs) == depth
    for index in range(1, depth):
        previous = layer_outputs[index - 1]
        current = layer_inputs[index]
        assert current.even_scalar is previous.state.even_scalar
        assert current.polar_vector is previous.state.polar_vector
    assert output.delta is not None
    assert torch.count_nonzero(output.delta) > 0
    torch.testing.assert_close(output.pos, graph.pos + output.delta)


def test_advanced_coordinate_updates_have_distinct_deterministic_boundaries() -> None:
    config = ELAConfig(
        input_irreps="3x0e",
        width=16,
        depth=5,
        coordinate_updates=2,
    )
    assert config.coordinate_update_layers == (2, 5)
    model = ELA.from_config(config).double()
    _activate_coordinate_head(model)
    calls = 0

    def count_coordinate_calls(_module: object, _args: object, _output: object) -> None:
        nonlocal calls
        calls += 1

    assert model.coordinate_head is not None
    handle = model.coordinate_head.register_forward_hook(count_coordinate_calls)
    try:
        model(
            ELAGraph(
                torch.randn(5, 3, dtype=torch.float64),
                torch.randn(5, 3, dtype=torch.float64),
                edge_index=_complete_edges(5),
            )
        )
    finally:
        handle.remove()
    assert calls == 2

    with pytest.raises(ValueError, match="cannot exceed depth"):
        ELAConfig(
            input_irreps="3x0e",
            width=16,
            depth=2,
            coordinate_updates=3,
        )


@pytest.mark.parametrize("determinant_sign", [1, -1], ids=["rotation", "reflection"])
def test_stagewise_coordinate_update_obeys_o3_translation_and_permutation(
    determinant_sign: int,
) -> None:
    torch.manual_seed(709 + determinant_sign)
    model = ELA(
        "3x0e",
        "1x0e + 1x1o",
        width=16,
        depth=3,
        cutoff=10.0,
        update_positions=True,
        max_coordinate_step=0.05,
    ).double().eval()
    _activate_coordinate_head(model)
    nodes = 6
    features = torch.randn(nodes, 3, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    edges = _complete_edges(nodes)
    update_mask = torch.tensor([True, True, True, True, False, False])
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if int(torch.linalg.det(orthogonal).sign().item()) != determinant_sign:
        orthogonal[:, 0].neg_()
    translation = torch.tensor([0.8, -1.2, 0.4], dtype=torch.float64)

    with torch.inference_mode():
        reference = model(
            ELAGraph(
                features,
                positions,
                edge_index=edges,
                update_mask=update_mask,
            )
        )
        transformed = model(
            ELAGraph(
                features,
                positions @ orthogonal.T + translation,
                edge_index=edges,
                update_mask=update_mask,
            )
        )

    torch.testing.assert_close(
        transformed.x[:, :1],
        reference.x[:, :1],
        atol=8e-8,
        rtol=8e-8,
    )
    torch.testing.assert_close(
        transformed.x[:, 1:].reshape(nodes, 1, 3),
        reference.x[:, 1:].reshape(nodes, 1, 3) @ orthogonal.T,
        atol=1e-7,
        rtol=1e-7,
    )
    torch.testing.assert_close(
        transformed.pos,
        reference.pos @ orthogonal.T + translation,
        atol=1e-7,
        rtol=1e-7,
    )
    assert transformed.delta is not None and reference.delta is not None
    torch.testing.assert_close(
        transformed.delta,
        reference.delta @ orthogonal.T,
        atol=1e-7,
        rtol=1e-7,
    )

    permutation = torch.randperm(nodes)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(nodes)
    with torch.inference_mode():
        permuted = model(
            ELAGraph(
                features[permutation],
                positions[permutation],
                edge_index=inverse[edges],
                update_mask=update_mask[permutation],
            )
        )
    torch.testing.assert_close(
        permuted.x,
        reference.x[permutation],
        atol=1e-7,
        rtol=1e-7,
    )
    torch.testing.assert_close(
        permuted.pos,
        reference.pos[permutation],
        atol=1e-7,
        rtol=1e-7,
    )


def test_stagewise_coordinate_update_supports_double_backward() -> None:
    torch.manual_seed(719)
    model = ELA(
        "3x0e",
        "1x0e",
        width=16,
        depth=2,
        cutoff=10.0,
        update_positions=True,
        max_coordinate_step=1.0,
    ).double()
    _activate_coordinate_head(model)
    features = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    positions = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    output = model(
        ELAGraph(features, positions, edge_index=_complete_edges(4))
    )
    first = torch.autograd.grad(
        output.x.square().mean() + output.pos.square().mean(),
        (features, positions),
        create_graph=True,
    )
    second = torch.autograd.grad(
        first[0].square().sum() + first[1].square().sum(),
        (features, positions),
    )
    for derivative in (*first, *second):
        assert torch.isfinite(derivative).all()
