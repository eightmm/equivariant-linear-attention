from __future__ import annotations

import math

import pytest
import torch

from equivariant_attention.moment import (
    EquivariantAttention,
    EquivariantAttentionConfig,
    _QuarticFeatureMap,
    _factorized_moment_attention,
    _symmetric_quadratic_features,
)
from equivariant_attention.training import gradient_l2_norms_by_path


def _orthogonal_reflection(dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator().manual_seed(2507)
    matrix = torch.randn(3, 3, generator=generator, dtype=dtype)
    orthogonal, _ = torch.linalg.qr(matrix)
    if torch.linalg.det(orthogonal) > 0:
        orthogonal[:, 0] = -orthogonal[:, 0]
    return orthogonal


def _symmetric_traceless(value: torch.Tensor) -> torch.Tensor:
    symmetric = 0.5 * (value + value.transpose(-1, -2))
    trace = symmetric.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    eye = torch.eye(3, dtype=value.dtype, device=value.device)
    return symmetric - trace[..., None, None] * eye / 3.0


def _v3_model(**overrides: object) -> EquivariantAttention:
    config: dict[str, object] = {
        "node_dim": 5,
        "input_vector_dim": 2,
        "input_tensor_dim": 1,
        "hidden_irreps": "12x0e + 3x1o + 2x2e",
        "output_irreps": "2x0e + 2x1o + 1x2e",
        "num_layers": 2,
        "num_heads": 3,
        "local_head_counts": (0, 0),
        "use_key_balancing": False,
        "angular_feature_rank": 2,
        "use_quartic_kernel": True,
        "use_irrep_rms_normalization": True,
    }
    config.update(overrides)
    return EquivariantAttention(EquivariantAttentionConfig(**config)).double()


def _equivariant_inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(2511)
    scalars = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    pos = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    vectors = torch.randn(7, 2, 3, generator=generator, dtype=torch.float64)
    raw_tensors = torch.randn(
        7,
        1,
        3,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    tensors = _symmetric_traceless(raw_tensors)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    return scalars, pos, vectors, tensors, batch


def test_quartic_feature_map_is_exact_and_fixed_width() -> None:
    generator = torch.Generator().manual_seed(2501)
    query = torch.randn(5, 2, 6, generator=generator, dtype=torch.float64)
    key = torch.randn(5, 2, 6, generator=generator, dtype=torch.float64)
    feature_map = _QuarticFeatureMap(6).double()

    query_feature = feature_map(query)
    key_feature = feature_map(key)
    actual = (query_feature * key_feature).sum(dim=-1)
    expected = (query * key).sum(dim=-1).pow(4)

    assert query_feature.shape == (5, 2, math.comb(9, 4))
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("dimension", [3, 6])
def test_symmetric_quadratic_map_is_exact_and_compressed(
    dimension: int,
) -> None:
    generator = torch.Generator().manual_seed(2520 + dimension)
    query = torch.randn(7, 2, dimension, generator=generator, dtype=torch.float64)
    key = torch.randn(7, 2, dimension, generator=generator, dtype=torch.float64)

    query_feature = _symmetric_quadratic_features(query, left_factor=True)
    key_feature = _symmetric_quadratic_features(key, left_factor=False)
    actual = (query_feature * key_feature).sum(dim=-1)
    expected = (query * key).sum(dim=-1).square()

    assert query_feature.shape[-1] == dimension * (dimension + 1) // 2
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_quartic_features_separate_a_degree_two_moment_collision() -> None:
    root_two = math.sqrt(2.0)
    first = torch.tensor(
        [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    second = torch.tensor(
        [
            [-root_two, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [root_two, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    quadratic_first = _symmetric_quadratic_features(
        first,
        left_factor=True,
    ).sum(dim=0)
    quadratic_second = _symmetric_quadratic_features(
        second,
        left_factor=True,
    ).sum(dim=0)
    feature_map = _QuarticFeatureMap(3).double()
    quartic_first = feature_map(first).sum(dim=0)
    quartic_second = feature_map(second).sum(dim=0)

    assert torch.equal(first.sum(dim=0), second.sum(dim=0))
    assert torch.allclose(quadratic_first, quadratic_second)
    assert not torch.allclose(quartic_first, quartic_second)


def test_quartic_kernel_matches_explicit_dense_attention() -> None:
    generator = torch.Generator().manual_seed(2502)
    nodes, heads, scalar_dim, value_dim = 6, 2, 3, 4
    query_scalar = torch.rand(
        nodes, heads, scalar_dim, generator=generator, dtype=torch.float64
    )
    key_scalar = torch.rand(
        nodes, heads, scalar_dim, generator=generator, dtype=torch.float64
    )
    query_vector = torch.randn(
        nodes, heads, 3, generator=generator, dtype=torch.float64
    )
    key_vector = torch.randn(
        nodes, heads, 3, generator=generator, dtype=torch.float64
    )
    query_vector = query_vector / torch.linalg.vector_norm(
        query_vector, dim=-1, keepdim=True
    ).clamp_min(1.0)
    key_vector = key_vector / torch.linalg.vector_norm(
        key_vector, dim=-1, keepdim=True
    ).clamp_min(1.0)
    value = torch.randn(
        nodes, heads, value_dim, generator=generator, dtype=torch.float64
    )
    quartic_scale = torch.tensor([0.2, 0.4], dtype=torch.float64)
    feature_map = _QuarticFeatureMap(3).double()
    query_quartic = feature_map(query_vector) * quartic_scale.sqrt()[None, :, None]
    key_quartic = feature_map(key_vector) * quartic_scale.sqrt()[None, :, None]
    augmented_query = torch.cat([query_scalar, query_quartic], dim=-1)
    augmented_key = torch.cat([key_scalar, key_quartic], dim=-1)
    quadratic_scale = torch.tensor([0.3, 0.1], dtype=torch.float64)
    linear_scale = torch.tensor([0.15, 0.25], dtype=torch.float64)
    batch = torch.zeros(nodes, dtype=torch.long)

    actual = _factorized_moment_attention(
        augmented_query,
        augmented_key,
        query_vector,
        key_vector,
        quadratic_scale,
        value,
        batch,
        num_graphs=1,
        balanced=False,
        alignment_scale=linear_scale,
        alignment_dot_scale=linear_scale,
        kernel_floor=1.0,
    )
    dot = torch.einsum("ihd,jhd->hij", query_vector, key_vector)
    content = torch.einsum("ihd,jhd->hij", query_scalar, key_scalar)
    kernel = (
        1.0
        + content
        + linear_scale[:, None, None] * (1.0 + dot)
        + quadratic_scale[:, None, None] * dot.square()
        + quartic_scale[:, None, None] * dot.pow(4)
    )
    expected = torch.einsum("hij,jhv->ihv", kernel, value)
    expected = expected / kernel.sum(dim=-1).transpose(0, 1).unsqueeze(-1)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_v3_external_irreps_are_o3_equivariant_and_permutation_safe() -> None:
    torch.manual_seed(2503)
    model = _v3_model()
    scalars, pos, vectors, tensors, batch = _equivariant_inputs()
    transform = _orthogonal_reflection(torch.float64)

    reference = model(
        scalars,
        pos,
        batch=batch,
        node_vectors=vectors,
        node_tensors=tensors,
    )
    moved = model(
        scalars,
        torch.einsum("ab,nb->na", transform, pos),
        batch=batch,
        node_vectors=torch.einsum("ab,ncb->nca", transform, vectors),
        node_tensors=torch.einsum(
            "ab,nkbc,dc->nkad",
            transform,
            tensors,
            transform,
        ),
    )

    for name in ("node_scalars", "graph_scalars"):
        assert torch.allclose(moved[name], reference[name], atol=1e-8, rtol=1e-8)
    for name in ("node_vectors", "graph_vectors"):
        expected = torch.einsum("ab,ncb->nca", transform, reference[name])
        assert torch.allclose(moved[name], expected, atol=1e-8, rtol=1e-8)
    for name in ("node_tensors", "graph_tensors"):
        expected = torch.einsum(
            "ab,nkbc,dc->nkad",
            transform,
            reference[name],
            transform,
        )
        assert torch.allclose(moved[name], expected, atol=1e-8, rtol=1e-8)

    permutation = torch.tensor([2, 0, 1, 6, 3, 5, 4])
    inverse = torch.argsort(permutation)
    permuted = model(
        scalars[permutation],
        pos[permutation],
        batch=batch[permutation],
        node_vectors=vectors[permutation],
        node_tensors=tensors[permutation],
    )
    for name in ("node_scalars", "node_vectors", "node_tensors"):
        assert torch.allclose(
            permuted[name][inverse],
            reference[name],
            atol=1e-8,
            rtol=1e-8,
        )
    for name in ("graph_scalars", "graph_vectors", "graph_tensors"):
        assert torch.allclose(permuted[name], reference[name], atol=1e-8, rtol=1e-8)


def test_v3_external_irreps_affect_outputs_and_receive_gradients() -> None:
    torch.manual_seed(2504)
    model = _v3_model()
    scalars, pos, vectors, tensors, batch = _equivariant_inputs()
    vectors = vectors.requires_grad_()
    tensors = tensors.requires_grad_()

    output = model(
        scalars,
        pos,
        batch=batch,
        node_vectors=vectors,
        node_tensors=tensors,
    )
    loss = sum(value.square().mean() for value in output.values())
    loss.backward()

    assert vectors.grad is not None
    assert tensors.grad is not None
    assert torch.isfinite(vectors.grad).all()
    assert torch.isfinite(tensors.grad).all()
    assert float(vectors.grad.abs().sum()) > 0.0
    assert float(tensors.grad.abs().sum()) > 0.0


def test_v3_external_irreps_are_cast_to_the_hidden_feature_dtype() -> None:
    torch.manual_seed(2529)
    model = _v3_model().float()
    scalars, pos, vectors, tensors, batch = _equivariant_inputs()

    output = model(
        scalars.float(),
        pos,
        batch=batch,
        node_vectors=vectors,
        node_tensors=tensors,
    )

    assert output["node_scalars"].dtype == torch.float32
    assert output["node_vectors"].dtype == torch.float32
    assert torch.isfinite(output["graph_scalars"]).all()


def test_v3_disabled_paths_preserve_default_state_and_output() -> None:
    inputs = _equivariant_inputs()
    scalars, pos, _, _, batch = inputs
    torch.manual_seed(2505)
    default = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=5,
            hidden_irreps="12x0e + 3x1o",
            num_layers=2,
            num_heads=3,
        )
    ).double()
    torch.manual_seed(2505)
    explicit = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=5,
            hidden_irreps="12x0e + 3x1o",
            num_layers=2,
            num_heads=3,
            input_vector_dim=0,
            input_tensor_dim=0,
            angular_feature_rank=1,
            use_quartic_kernel=False,
            use_irrep_rms_normalization=False,
            checkpoint_gated_local_mlp=False,
        )
    ).double()

    assert list(default.state_dict()) == list(explicit.state_dict())
    for name, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name]), name
    reference = default(scalars, pos, batch=batch)
    actual = explicit(scalars, pos, batch=batch)
    for name in reference:
        assert torch.equal(actual[name], reference[name]), name


def test_gated_checkpoint_preserves_forward_and_backward() -> None:
    base = dict(
        node_dim=5,
        hidden_irreps="8x0e + 2x1o",
        output_irreps="1x0e",
        num_layers=1,
        num_heads=2,
        local_head_counts=(2,),
        use_key_balancing=False,
        use_gated_local_transport=True,
        local_cutoff=5.0,
    )
    torch.manual_seed(2506)
    reference_model = EquivariantAttention(
        EquivariantAttentionConfig(**base)
    ).double()
    torch.manual_seed(2506)
    checkpointed_model = EquivariantAttention(
        EquivariantAttentionConfig(**base, checkpoint_gated_local_mlp=True)
    ).double()
    scalars, pos, _, _, batch = _equivariant_inputs()
    first_graph = torch.cartesian_prod(torch.arange(3), torch.arange(3)).T
    second_graph = torch.cartesian_prod(torch.arange(3, 7), torch.arange(3, 7)).T
    edge_index = torch.cat([first_graph, second_graph], dim=1)

    reference = reference_model(scalars, pos, batch=batch, edge_index=edge_index)
    checkpointed = checkpointed_model(
        scalars,
        pos,
        batch=batch,
        edge_index=edge_index,
    )
    reference["graph_scalars"].square().sum().backward()
    checkpointed["graph_scalars"].square().sum().backward()

    assert torch.equal(
        checkpointed["graph_scalars"],
        reference["graph_scalars"],
    )
    for (left_name, left), (right_name, right) in zip(
        reference_model.named_parameters(),
        checkpointed_model.named_parameters(),
        strict=True,
    ):
        assert left_name == right_name
        if left.grad is None or right.grad is None:
            assert left.grad is None and right.grad is None
            continue
        assert torch.allclose(left.grad, right.grad, atol=1e-10, rtol=1e-9), left_name


def test_pathwise_gradient_norms_partition_the_total_norm() -> None:
    torch.manual_seed(2508)
    model = _v3_model()
    scalars, pos, vectors, tensors, batch = _equivariant_inputs()
    output = model(
        scalars,
        pos,
        batch=batch,
        node_vectors=vectors,
        node_tensors=tensors,
    )
    output["graph_scalars"].square().mean().backward()

    norms = gradient_l2_norms_by_path(model)

    assert {"input", "global", "ffn", "readout"} <= set(norms)
    assert all(math.isfinite(value) and value >= 0.0 for value in norms.values())
    total = sum(
        float(parameter.grad.detach().double().square().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    ) ** 0.5
    reconstructed = sum(value * value for value in norms.values()) ** 0.5
    assert reconstructed == pytest.approx(total, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"angular_feature_rank": 0}, "angular_feature_rank"),
        ({"input_vector_dim": -1}, "input_vector_dim"),
        ({"input_tensor_dim": 1, "hidden_irreps": "12x0e + 3x1o"}, "2e"),
        (
            {"checkpoint_gated_local_mlp": True, "use_gated_local_transport": False},
            "checkpoint",
        ),
    ],
)
def test_v3_rejects_invalid_configuration(
    override: dict[str, object],
    message: str,
) -> None:
    base: dict[str, object] = {
        "node_dim": 5,
        "hidden_irreps": "12x0e + 3x1o",
        "num_heads": 3,
    }
    base.update(override)
    with pytest.raises((TypeError, ValueError), match=message):
        EquivariantAttention(EquivariantAttentionConfig(**base))
