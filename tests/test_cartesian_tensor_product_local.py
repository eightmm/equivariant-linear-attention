from __future__ import annotations

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
import equivariant_attention.moment as moment
from equivariant_attention.training import build_regression_model


def _model(*, ctp: bool) -> EquivariantAttention:
    return EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=5,
            hidden_irreps="12x0e + 3x1o + 3x2e",
            output_irreps="2x0e + 1x1o + 1x2e",
            num_layers=3,
            num_heads=3,
            local_head_counts=(3, 0, 3),
            use_key_balancing=False,
            use_gated_local_transport=True,
            use_grouped_invariant_normalization=True,
            use_static_tensor_carrier=True,
            use_cartesian_tensor_product_local_transport=ctp,
            cartesian_tensor_product_local_layers=(2,) if ctp else None,
        )
    ).double()


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260727)
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


def _activate_ctp(model: EquivariantAttention) -> None:
    with torch.no_grad():
        for layer in model.layers:
            transport = layer.gated_local
            if transport is None or transport.tensor_product_gate is None:
                continue
            output = transport.tensor_product_gate[-1]
            assert isinstance(output, torch.nn.Linear)
            output.bias.copy_(
                torch.tensor([0.35, -0.25, 0.4], dtype=output.bias.dtype)
            )


@pytest.mark.parametrize(
    ("hidden_irreps", "gated", "ctp", "error", "message"),
    [
        (
            "8x0e + 2x1o + 2x2e",
            False,
            True,
            ValueError,
            "requires gated local transport",
        ),
        (
            "8x0e + 2x1o",
            True,
            True,
            ValueError,
            "persistent 2e",
        ),
        (
            "8x0e + 2x1o + 2x2e",
            True,
            1,
            TypeError,
            "must be a bool",
        ),
    ],
)
def test_ctp_rejects_invalid_activation_contract(
    hidden_irreps: str,
    gated: bool,
    ctp: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=4,
                hidden_irreps=hidden_irreps,
                num_layers=2,
                num_heads=2,
                local_head_counts=(2, 0),
                use_gated_local_transport=gated,
                use_cartesian_tensor_product_local_transport=ctp,  # type: ignore[arg-type]
            )
        )


def test_ctp_plan_is_static_and_builder_wires_only_local_stages() -> None:
    model = build_regression_model(
        node_dim=5,
        hidden_dim=48,
        num_layers=3,
        num_heads=3,
        local_head_counts=(3, 0, 3),
        use_gated_local_transport=True,
        hidden_tensor_dim=3,
        use_cartesian_tensor_product_local_transport=True,
    )

    expected = (
        ("tensor_direction", "2e", "1o", "1o"),
        ("tensor_passthrough", "2e", "0e", "2e"),
        ("vector_direction", "1o", "1o", "2e"),
    )
    for layer, local_heads in zip(model.layers, (3, 0, 3), strict=True):
        if local_heads:
            assert layer.gated_local is not None
            assert tuple(
                (
                    path.name,
                    path.input_irrep,
                    path.geometry_irrep,
                    path.output_irrep,
                )
                for path in layer.gated_local.tensor_product_paths
            ) == expected
            assert layer.gated_local.tensor_product_gate is not None
        else:
            assert layer.gated_local is None


def test_ctp_zero_initialization_exactly_matches_persistent_tensor_control() -> None:
    torch.manual_seed(2701)
    control = _model(ctp=False)
    torch.manual_seed(2701)
    candidate = _model(ctp=True)
    node_feats, pos, batch, edge_index = _inputs()

    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    common = control_state.keys() & candidate_state.keys()
    assert common
    assert any("tensor_product_gate" in name for name in candidate_state)
    assert not any("tensor_product_gate" in name for name in control_state)
    for name in common:
        assert torch.equal(control_state[name], candidate_state[name]), name

    control_output = control(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index,
    )
    candidate_output = candidate(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index,
    )
    for name in control_output:
        assert torch.equal(control_output[name], candidate_output[name]), name


def test_ctp_sender_tensor_path_separates_rotated_quadrupoles_exactly() -> None:
    transport = moment._GatedEquivariantLocalTransport(
        scalars=4,
        vectors=1,
        tensors=1,
        num_heads=1,
        num_rbf=4,
        eps=1e-12,
        use_cartesian_tensor_product_local_transport=True,
    ).double()
    assert transport.tensor_product_gate is not None
    with torch.no_grad():
        for parameter in transport.parameters():
            parameter.zero_()
        output = transport.tensor_product_gate[-1]
        assert isinstance(output, torch.nn.Linear)
        output.bias[0] = 1.0

    scalars = torch.zeros(2, 4, dtype=torch.float64)
    vectors = torch.zeros(2, 1, 3, dtype=torch.float64)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        dtype=torch.float64,
    )
    batch = torch.zeros(2, dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 0], [0, 1, 1]])
    geometry = moment._local_geometry(
        pos,
        batch,
        num_graphs=1,
        cutoff=1.0,
        num_rbf=4,
        edge_index=edge_index,
    )
    x = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    y = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    tensor_x = torch.zeros(2, 1, 5, dtype=torch.float64)
    tensor_y = tensor_x.clone()
    tensor_x[1, 0] = 2.0 * moment._symmetric_traceless_features(x)
    tensor_y[1, 0] = 2.0 * moment._symmetric_traceless_features(y)

    message_x = transport(
        scalars,
        vectors,
        geometry,
        num_nodes=2,
        persistent_tensor=tensor_x,
    )[1]
    message_y = transport(
        scalars,
        vectors,
        geometry,
        num_nodes=2,
        persistent_tensor=tensor_y,
    )[1]
    displacement = geometry.nonself_displacement[0]
    cutoff = geometry.nonself_cutoff[0]
    expected_scale = torch.tanh(torch.tensor(1.0, dtype=torch.float64))
    expected_scale = expected_scale * cutoff / torch.sqrt(1.0 + cutoff)
    expected_x = expected_scale * moment._st_matrix_vector(
        tensor_x[1, 0],
        displacement,
    )
    expected_y = expected_scale * moment._st_matrix_vector(
        tensor_y[1, 0],
        displacement,
    )

    assert torch.allclose(message_x[0, 0], expected_x, atol=1e-12, rtol=1e-11)
    assert torch.allclose(message_y[0, 0], expected_y, atol=1e-12, rtol=1e-11)
    assert not torch.allclose(message_x[0, 0], message_y[0, 0])


def test_active_ctp_public_path_preserves_o3_permutation_edge_order_and_batch() -> None:
    torch.manual_seed(2702)
    model = _model(ctp=True)
    _activate_ctp(model)
    node_feats, pos, batch, edge_index = _inputs()
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    orthogonal[:, 0].neg_()
    translation = torch.randn(1, 3, dtype=torch.float64)
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
    first = model(
        node_feats[:3],
        pos[:3],
        edge_index=edge_index[:, :9],
    )
    second = model(
        node_feats[3:],
        pos[3:],
        edge_index=edge_index[:, 9:] - 3,
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
        assert torch.allclose(reference[name][:3], first[name], atol=1e-9)
        assert torch.allclose(reference[name][3:], second[name], atol=1e-9)


def test_ctp_parameters_and_coordinates_receive_finite_nonzero_gradients() -> None:
    torch.manual_seed(2703)
    model = _model(ctp=True)
    _activate_ctp(model)
    node_feats, pos, batch, edge_index = _inputs()
    node_feats.requires_grad_()
    pos.requires_grad_()

    loss = model(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index,
    )["graph_scalars"].square().sum()
    loss.backward()

    ctp_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "tensor_product_gate" in name
    ]
    assert ctp_gradients
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in ctp_gradients
    )
    assert any(torch.count_nonzero(gradient) for gradient in ctp_gradients)
    for gradient in (node_feats.grad, pos.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient)
