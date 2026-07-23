from __future__ import annotations

import pytest
import torch

from equivariant_attention.moment import (
    EquivariantAttention,
    EquivariantAttentionConfig,
    _factorized_moment_attention,
    _positive_scalar_features,
    _st_features_to_matrix,
    _tensor_product_features,
    _unit_frobenius_st,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260723)
    return (
        torch.randn(8, 5, generator=generator, dtype=torch.float64),
        torch.randn(8, 3, generator=generator, dtype=torch.float64),
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
    )


def _model(**overrides: object) -> EquivariantAttention:
    config = {
        "node_dim": 5,
        "hidden_irreps": "12x0e + 3x1o + 2x2e",
        "output_irreps": "2x0e + 1x1o + 1x2e",
        "num_layers": 3,
        "num_heads": 3,
        "use_key_balancing": False,
        "scalar_content_mode": "bounded",
        "use_tensor_product_kernel": True,
    }
    config.update(overrides)
    return EquivariantAttention(EquivariantAttentionConfig(**config)).double()


def test_bounded_positive_content_retains_magnitude_with_finite_bound() -> None:
    raw = torch.tensor(
        [[[-3.0, -3.0, -3.0, -3.0]], [[3.0, 3.0, 3.0, 3.0]]],
        dtype=torch.float64,
    )

    unit = _positive_scalar_features(raw, 1e-12, mode="unit")
    bounded = _positive_scalar_features(raw, 1e-12, mode="bounded")
    unit_norm = torch.linalg.vector_norm(unit, dim=-1)
    bounded_norm = torch.linalg.vector_norm(bounded, dim=-1)

    assert torch.all(unit > 0)
    assert torch.all(bounded > 0)
    assert torch.allclose(unit_norm[0], unit_norm[1], atol=1e-8)
    assert bounded_norm[1].item() > bounded_norm[0].item()
    assert bounded_norm.max().item() < 2.0


def test_shifted_tensor_features_match_nonnegative_frobenius_kernel() -> None:
    query = torch.tensor(
        [[[0.4, -0.1, 0.2, -0.3, 0.1]]],
        dtype=torch.float64,
    )
    key = torch.tensor(
        [[[-0.2, 0.3, -0.1, 0.25, -0.15]]],
        dtype=torch.float64,
    )
    scale = torch.tensor([0.3], dtype=torch.float64)

    query_features, key_features = _tensor_product_features(
        query,
        key,
        scale,
        eps=1e-12,
    )
    kernel = (query_features * key_features).sum(dim=-1)
    bounded_query = _unit_frobenius_st(query, 1e-12)
    bounded_key = _unit_frobenius_st(key, 1e-12)
    expected = scale * (
        1.0
        + (
            _st_features_to_matrix(bounded_query) * _st_features_to_matrix(bounded_key)
        ).sum(dim=(-2, -1))
    )

    assert kernel.item() >= 0.0
    assert kernel.item() <= 2.0 * scale.item()
    assert torch.allclose(kernel, expected, atol=1e-12, rtol=1e-12)
    assert query_features.shape[-1] == 10
    assert key_features.shape == query_features.shape


def test_augmented_tensor_content_matches_dense_factorized_attention() -> None:
    generator = torch.Generator().manual_seed(19)
    query_scalar = torch.rand(6, 2, 3, generator=generator, dtype=torch.float64)
    key_scalar = torch.rand(6, 2, 3, generator=generator, dtype=torch.float64)
    query_tensor = torch.randn(6, 2, 5, generator=generator, dtype=torch.float64)
    key_tensor = torch.randn(6, 2, 5, generator=generator, dtype=torch.float64)
    tensor_scale = torch.tensor([0.2, 0.4], dtype=torch.float64)
    query_extra, key_extra = _tensor_product_features(
        query_tensor,
        key_tensor,
        tensor_scale,
        eps=1e-12,
    )
    query = torch.cat([query_scalar, query_extra], dim=-1)
    key = torch.cat([key_scalar, key_extra], dim=-1)
    query_vector = 0.2 * torch.randn(6, 2, 3, generator=generator, dtype=torch.float64)
    key_vector = 0.2 * torch.randn(6, 2, 3, generator=generator, dtype=torch.float64)
    value = torch.randn(6, 2, 4, generator=generator, dtype=torch.float64)
    quadratic_scale = torch.tensor([0.1, 0.2], dtype=torch.float64)
    alignment_scale = torch.tensor([0.05, 0.07], dtype=torch.float64)
    batch = torch.zeros(6, dtype=torch.long)

    factorized = _factorized_moment_attention(
        query,
        key,
        query_vector,
        key_vector,
        quadratic_scale,
        value,
        batch,
        num_graphs=1,
        balanced=False,
        alignment_scale=alignment_scale,
        alignment_dot_scale=alignment_scale,
    )
    content = torch.einsum("ihd,jhd->ijh", query, key)
    angular = torch.einsum("iha,jha->ijh", query_vector, key_vector)
    kernel = (
        1.0
        + content
        + alignment_scale[None, None, :]
        + alignment_scale[None, None, :] * angular
        + quadratic_scale[None, None, :] * angular.square()
    )
    dense = torch.einsum(
        "ijh,jhf->ihf",
        kernel / kernel.sum(dim=1, keepdim=True),
        value,
    )

    assert torch.allclose(factorized, dense, atol=1e-10, rtol=1e-9)


def test_architecture_v2_preserves_o3_translation_permutation_and_batch() -> None:
    torch.manual_seed(31)
    model = _model()
    node_feats, pos, batch = _inputs()
    orthogonal = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=torch.float64,
    )
    translation = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    permutation = torch.tensor([2, 0, 3, 1, 6, 4, 7, 5])

    reference = model(node_feats, pos, batch=batch)
    moved = model(node_feats, pos @ orthogonal.T + translation, batch=batch)
    permuted = model(
        node_feats[permutation],
        pos[permutation],
        batch=batch[permutation],
    )
    inverse = torch.argsort(permutation)

    assert torch.allclose(
        moved["graph_scalars"], reference["graph_scalars"], atol=1e-9, rtol=1e-9
    )
    assert torch.allclose(
        permuted["node_scalars"][inverse],
        reference["node_scalars"],
        atol=1e-9,
        rtol=1e-9,
    )
    first = model(node_feats[:4], pos[:4])
    second = model(node_feats[4:], pos[4:])
    assert torch.allclose(
        reference["graph_scalars"][0], first["graph_scalars"][0], atol=1e-9
    )
    assert torch.allclose(
        reference["graph_scalars"][1], second["graph_scalars"][0], atol=1e-9
    )


def test_architecture_v2_defaults_are_exactly_backward_compatible() -> None:
    kwargs = {
        "node_dim": 5,
        "hidden_irreps": "12x0e + 3x1o + 2x2e",
        "num_layers": 2,
        "num_heads": 3,
    }
    torch.manual_seed(44)
    default = EquivariantAttention(EquivariantAttentionConfig(**kwargs)).double()
    torch.manual_seed(44)
    explicit = EquivariantAttention(
        EquivariantAttentionConfig(
            **kwargs,
            scalar_content_mode="unit",
            use_tensor_product_kernel=False,
        )
    ).double()
    node_feats, pos, batch = _inputs()

    assert list(default.state_dict()) == list(explicit.state_dict())
    for name, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name])
        assert "tensor_kernel" not in name
    default_output = default(node_feats, pos, batch=batch)
    explicit_output = explicit(node_feats, pos, batch=batch)
    for key in default_output:
        assert torch.equal(default_output[key], explicit_output[key])


def test_tensor_option_preserves_all_common_parameter_initializations() -> None:
    kwargs = {
        "node_dim": 5,
        "hidden_irreps": "12x0e + 3x1o + 2x2e",
        "num_layers": 3,
        "num_heads": 3,
    }
    torch.manual_seed(59)
    persistent_only = EquivariantAttention(EquivariantAttentionConfig(**kwargs))
    torch.manual_seed(59)
    tensor_kernel = EquivariantAttention(
        EquivariantAttentionConfig(
            **kwargs,
            use_tensor_product_kernel=True,
        )
    )

    persistent_state = persistent_only.state_dict()
    tensor_state = tensor_kernel.state_dict()
    common_names = persistent_state.keys() & tensor_state.keys()

    assert common_names
    for name in common_names:
        assert torch.equal(persistent_state[name], tensor_state[name]), name


def test_tensor_product_kernel_requires_persistent_2e_state() -> None:
    with pytest.raises(ValueError, match="persistent 2e"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=4,
                hidden_irreps="8x0e + 2x1o",
                num_heads=2,
                use_tensor_product_kernel=True,
            )
        )


def test_tensor_product_kernel_receives_finite_nonzero_gradients() -> None:
    torch.manual_seed(71)
    model = _model()
    node_feats, pos, batch = _inputs()

    model(node_feats, pos, batch=batch)["graph_scalars"].square().sum().backward()

    scales = [
        layer.raw_tensor_kernel.grad
        for layer in model.layers
        if layer.raw_tensor_kernel is not None
    ]
    projected = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "tensor_kernel_query.weight" in name or "tensor_kernel_key.weight" in name
    ]
    assert all(gradient is not None for gradient in scales)
    assert all(torch.isfinite(gradient).all() for gradient in scales)
    assert any(float(gradient.abs().sum()) > 0.0 for gradient in scales)
    assert any(
        gradient is not None
        and torch.isfinite(gradient).all()
        and float(gradient.abs().sum()) > 0.0
        for gradient in projected
    )


def test_persistent_tensor_candidate_stays_finite_across_updates() -> None:
    torch.manual_seed(73)
    model = _model().float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    node_feats, pos, batch = _inputs()
    node_feats = node_feats.float()
    pos = pos.float()

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(node_feats, pos, batch=batch)["graph_scalars"].square().mean()
        loss.backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        optimizer.step()

    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
