from __future__ import annotations

import torch

from equivariant_attention.moment import (
    EquivariantAttention,
    EquivariantAttentionConfig,
)
from equivariant_attention.training import build_regression_model


def _model() -> EquivariantAttention:
    return EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=5,
            hidden_irreps="12x0e + 3x1o + 2x2e",
            output_irreps="2x0e + 1x1o + 2x2e",
            num_layers=3,
            num_heads=3,
            use_key_balancing=False,
            use_multiscale_spatial_kernel=True,
        )
    ).double()


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260723)
    node_feats = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    pos = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    return node_feats, pos, batch


def test_persistent_2e_outputs_are_symmetric_and_traceless() -> None:
    model = _model()
    node_feats, pos, batch = _inputs()

    output = model(node_feats, pos, batch=batch)

    assert output["node_tensors"].shape == (7, 2, 3, 3)
    assert torch.allclose(
        output["node_tensors"],
        output["node_tensors"].transpose(-1, -2),
        atol=1e-12,
    )
    assert torch.allclose(
        output["node_tensors"].diagonal(dim1=-2, dim2=-1).sum(dim=-1),
        torch.zeros(7, 2, dtype=torch.float64),
        atol=1e-12,
    )


def test_persistent_2e_is_o3_translation_and_permutation_equivariant() -> None:
    model = _model()
    node_feats, pos, batch = _inputs()
    orthogonal = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=torch.float64,
    )
    translation = torch.tensor([2.0, -3.0, 0.5], dtype=torch.float64)
    permutation = torch.tensor([2, 0, 3, 1, 6, 4, 5])

    reference = model(node_feats, pos, batch=batch)
    moved = model(
        node_feats,
        pos @ orthogonal.T + translation,
        batch=batch,
    )
    permuted = model(
        node_feats[permutation],
        pos[permutation],
        batch=batch[permutation],
    )
    inverse = torch.argsort(permutation)

    expected_tensor = torch.einsum(
        "ab,nkbc,dc->nkad",
        orthogonal,
        reference["node_tensors"],
        orthogonal,
    )
    expected_graph_tensor = torch.einsum(
        "ab,gkbc,dc->gkad",
        orthogonal,
        reference["graph_tensors"],
        orthogonal,
    )
    expected_node_vector = torch.einsum(
        "ab,nkb->nka",
        orthogonal,
        reference["node_vectors"],
    )
    expected_graph_vector = torch.einsum(
        "ab,gkb->gka",
        orthogonal,
        reference["graph_vectors"],
    )
    assert torch.allclose(moved["node_scalars"], reference["node_scalars"], atol=1e-9)
    assert torch.allclose(moved["graph_scalars"], reference["graph_scalars"], atol=1e-9)
    assert torch.allclose(moved["node_vectors"], expected_node_vector, atol=1e-9)
    assert torch.allclose(moved["graph_vectors"], expected_graph_vector, atol=1e-9)
    assert torch.allclose(moved["node_tensors"], expected_tensor, atol=1e-9)
    assert torch.allclose(moved["graph_tensors"], expected_graph_tensor, atol=1e-9)
    for key in ("node_scalars", "node_vectors", "node_tensors"):
        assert torch.allclose(
            permuted[key][inverse],
            reference[key],
            atol=1e-9,
            rtol=1e-9,
        )
    for key in ("graph_scalars", "graph_vectors", "graph_tensors"):
        assert torch.allclose(
            permuted[key],
            reference[key],
            atol=1e-9,
            rtol=1e-9,
        )


def test_persistent_2e_path_receives_finite_nonzero_gradients() -> None:
    model = _model()
    node_feats, pos, batch = _inputs()
    node_feats.requires_grad_()
    pos.requires_grad_()

    loss = model(node_feats, pos, batch=batch)["graph_scalars"].square().sum()
    loss.backward()

    tensor_parameters = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "persistent_tensor" in name
    ]
    assert tensor_parameters
    assert all(gradient is not None for gradient in tensor_parameters)
    assert all(torch.isfinite(gradient).all() for gradient in tensor_parameters)
    assert any(torch.count_nonzero(gradient).item() for gradient in tensor_parameters)
    assert node_feats.grad is not None and torch.isfinite(node_feats.grad).all()
    assert pos.grad is not None and torch.isfinite(pos.grad).all()


def test_scalar_vector_hidden_state_allocates_no_persistent_tensor_parameters() -> None:
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=3,
            hidden_irreps="8x0e + 2x1o",
            output_irreps="1x0e + 1x2e",
            num_layers=2,
            num_heads=2,
        )
    )

    assert not any(
        "persistent_tensor" in name for name, _parameter in model.named_parameters()
    )
    assert model.tensor_out.weight.shape[1] == 2


def test_persistent_2e_preserves_batch_isolation_for_every_output() -> None:
    model = _model()
    node_feats, pos, batch = _inputs()

    combined = model(node_feats, pos, batch=batch)
    first = model(node_feats[:4], pos[:4])
    second = model(node_feats[4:], pos[4:])

    for key in ("node_scalars", "node_vectors", "node_tensors"):
        assert torch.allclose(combined[key][:4], first[key], atol=1e-9, rtol=1e-9)
        assert torch.allclose(combined[key][4:], second[key], atol=1e-9, rtol=1e-9)
    for key in ("graph_scalars", "graph_vectors", "graph_tensors"):
        assert torch.allclose(combined[key][0], first[key][0], atol=1e-9, rtol=1e-9)
        assert torch.allclose(combined[key][1], second[key][0], atol=1e-9, rtol=1e-9)


def test_disabled_persistent_2e_is_exactly_the_default_model() -> None:
    kwargs = {
        "node_dim": 5,
        "hidden_dim": 16,
        "num_layers": 2,
        "num_heads": 4,
    }
    torch.manual_seed(91)
    default = build_regression_model(**kwargs).double()
    torch.manual_seed(91)
    explicit_off = build_regression_model(**kwargs, hidden_tensor_dim=0).double()
    node_feats, pos, batch = _inputs()

    assert list(default.state_dict()) == list(explicit_off.state_dict())
    for name, value in default.state_dict().items():
        assert torch.equal(value, explicit_off.state_dict()[name])
        assert "persistent_tensor" not in name
    default_output = default(node_feats, pos, batch=batch)
    explicit_output = explicit_off(node_feats, pos, batch=batch)
    assert set(default_output) == {
        "node_scalars",
        "node_vectors",
        "node_tensors",
        "graph_scalars",
        "graph_vectors",
        "graph_tensors",
    }
    for key in default_output:
        assert torch.equal(default_output[key], explicit_output[key])
