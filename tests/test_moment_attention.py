import pytest
import torch

from equivariant_attention import EquivariantMomentAttention, EquivariantMomentAttentionConfig
from equivariant_attention.moment import (
    _factorized_attention,
    _gram_invariants,
    _normalized_st_cubic_trace,
    _normalized_st_square_vector,
    _radial_distance_features,
    _relative_radial_trace,
    _shifted_angular_features,
    _st_features_to_matrix,
    _symmetric_outer_features,
)


def _random_rotation(dtype: torch.dtype) -> torch.Tensor:
    q = torch.randn(4, dtype=dtype)
    q = q / q.norm()
    w, x, y, z = q
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
        ]
    )


def _rotate_tensor(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ab,...bc,dc->...ad", rotation, tensor, rotation)


def _make_inputs(dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(211)
    node_feats = torch.randn(9, 6, dtype=dtype)
    pos = torch.randn(9, 3, dtype=dtype)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    return node_feats, pos, batch


def _make_model(
    dtype: torch.dtype = torch.float64,
    *,
    enhanced: bool = False,
    feature: str | None = None,
    sinkhorn_iterations: int = 1,
) -> EquivariantMomentAttention:
    torch.manual_seed(223)
    feature_flags = {
        "radial_trace": enhanced,
        "full_gram_invariants": enhanced,
        "shifted_angular_kernel": enhanced,
        "learnable_balance_exponent": enhanced,
        "radial_distance_kernel": enhanced,
        "dynamic_moment_routing": enhanced,
        "equivariant_ffn": enhanced,
    }
    if feature is not None:
        feature_flags[feature] = True
    model = EquivariantMomentAttention(
        EquivariantMomentAttentionConfig(
            node_dim=6,
            hidden_irreps="12x0e + 3x1o",
            output_irreps="2x0e + 2x1o + 1x2e",
            num_layers=2,
            num_heads=3,
            balance_attention=True,
            sinkhorn_iterations=sinkhorn_iterations,
            **feature_flags,
        )
    )
    return model.to(dtype=dtype).eval()


def _max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return (left - right).abs().max().item()


def _matrix_to_st_features(value: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [value[..., 0, 0], value[..., 1, 1], value[..., 0, 1], value[..., 0, 2], value[..., 1, 2]],
        dim=-1,
    )


def test_squared_vector_features_factorize_squared_dot_product() -> None:
    torch.manual_seed(227)
    query = torch.randn(5, 3, dtype=torch.float64)
    key = torch.randn(7, 3, dtype=torch.float64)

    factorized = _symmetric_outer_features(query) @ _symmetric_outer_features(key).T
    expected = (query @ key.T).square()

    assert _max_error(factorized, expected) < 1e-12


def test_shifted_angular_features_factorize_shifted_square() -> None:
    torch.manual_seed(228)
    query = torch.randn(5, 2, 3, dtype=torch.float64)
    key = torch.randn(7, 2, 3, dtype=torch.float64)
    kernel_scale = torch.tensor([0.2, 0.7], dtype=torch.float64)
    shift = torch.tensor([1.1, 1.8], dtype=torch.float64)

    query_features = _shifted_angular_features(query, kernel_scale, shift)
    key_features = _shifted_angular_features(key, kernel_scale, shift)
    factorized = torch.einsum("ihd,jhd->hij", query_features, key_features)
    expected = kernel_scale[:, None, None] * (
        shift[:, None, None] + torch.einsum("iha,jha->hij", query, key)
    ).square()

    assert _max_error(factorized, expected) < 1e-12


def test_radial_distance_features_factorize_positive_distance_kernel() -> None:
    torch.manual_seed(229)
    pos = torch.randn(9, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    shift = torch.tensor([1.05, 1.4], dtype=torch.float64)

    query, key = _radial_distance_features(pos, batch, shift, eps=1e-12)
    actual = torch.einsum("ihd,jhd->hij", query[:4], key[:4])
    graph_pos = pos[:4]
    length = 2.0 * graph_pos.norm(dim=-1).max()
    expected = shift[:, None, None] - (
        (graph_pos[:, None] - graph_pos[None, :]).square().sum(dim=-1) / length.square()
    )[None]

    assert _max_error(actual, expected) < 1e-12
    assert actual.min().item() > 0.0


def test_radial_distance_features_define_zero_radius_singleton_graph() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    batch = torch.tensor([0, 1, 1], dtype=torch.long)
    shift = torch.tensor([1.1, 1.4], dtype=torch.float64)

    query, key = _radial_distance_features(pos, batch, shift, eps=1e-12)
    singleton_kernel = (query[0] * key[0]).sum(dim=-1)
    second_graph_kernel = torch.einsum("ihd,jhd->hij", query[1:], key[1:])

    assert torch.allclose(singleton_kernel, shift)
    assert second_graph_kernel.min().item() > 0.0


def test_radial_content_product_attention_matches_dense_kernel() -> None:
    torch.manual_seed(230)
    nodes, heads, content_dim, value_dim = 7, 2, 4, 3
    pos = torch.randn(nodes, 3, dtype=torch.float64)
    batch = torch.zeros(nodes, dtype=torch.long)
    shift = torch.tensor([1.05, 1.3], dtype=torch.float64)
    content_query = torch.rand(nodes, heads, content_dim, dtype=torch.float64)
    content_key = torch.rand(nodes, heads, content_dim, dtype=torch.float64)
    value = torch.randn(nodes, heads, value_dim, dtype=torch.float64)
    radial_query, radial_key = _radial_distance_features(pos, batch, shift, eps=1e-12)
    query = torch.einsum("nhd,nhr->nhdr", content_query, radial_query).flatten(start_dim=2)
    key = torch.einsum("nhd,nhr->nhdr", content_key, radial_key).flatten(start_dim=2)

    actual = _factorized_attention(query, key, value, batch, balanced=True, eps=1e-12)
    content_kernel = torch.einsum("ihd,jhd->hij", content_query, content_key)
    radial_kernel = torch.einsum("ihr,jhr->hij", radial_query, radial_key)
    kernel = content_kernel * radial_kernel
    kernel = kernel / kernel.sum(dim=1, keepdim=True)
    weights = kernel / kernel.sum(dim=2, keepdim=True)
    expected = torch.einsum("hij,jhf->ihf", weights, value)

    assert _max_error(actual, expected) < 1e-10


@pytest.mark.parametrize("balanced", [False, True])
def test_factorized_attention_matches_dense_kernel(balanced: bool) -> None:
    torch.manual_seed(229)
    query = torch.rand(6, 2, 7, dtype=torch.float64)
    key = torch.rand(6, 2, 7, dtype=torch.float64)
    value = torch.randn(6, 2, 5, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    actual = _factorized_attention(query, key, value, batch, balanced=balanced, eps=1e-12)
    expected = torch.empty_like(value)
    for graph_id in range(2):
        index = batch == graph_id
        kernel = torch.einsum("ihd,jhd->hij", query[index], key[index])
        if balanced:
            kernel = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-12)
        weights = kernel / kernel.sum(dim=2, keepdim=True).clamp_min(1e-12)
        expected[index] = torch.einsum("hij,jhf->ihf", weights, value[index])

    assert _max_error(actual, expected) < 1e-10


def test_factorized_attention_matches_dense_fractional_balancing() -> None:
    torch.manual_seed(230)
    query = torch.rand(6, 2, 7, dtype=torch.float64)
    key = torch.rand(6, 2, 7, dtype=torch.float64)
    value = torch.randn(6, 2, 5, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    exponent = torch.tensor([0.25, 0.8], dtype=torch.float64)

    actual = _factorized_attention(
        query,
        key,
        value,
        batch,
        balanced=True,
        balance_exponent=exponent,
        eps=1e-12,
    )
    expected = torch.empty_like(value)
    for graph_id in range(2):
        index = batch == graph_id
        kernel = torch.einsum("ihd,jhd->hij", query[index], key[index])
        key_mass = kernel.sum(dim=1).clamp_min(1e-12)
        kernel = kernel * key_mass.pow(-exponent[:, None])[:, None, :]
        weights = kernel / kernel.sum(dim=2, keepdim=True).clamp_min(1e-12)
        expected[index] = torch.einsum("hij,jhf->ihf", weights, value[index])

    assert _max_error(actual, expected) < 1e-10


def test_factorized_attention_two_sinkhorn_iterations_match_dense() -> None:
    torch.manual_seed(238)
    query = torch.rand(9, 3, 6, dtype=torch.float64)
    key = torch.rand(9, 3, 6, dtype=torch.float64)
    value = torch.randn(9, 3, 4, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)

    actual = _factorized_attention(
        query,
        key,
        value,
        batch,
        balanced=True,
        sinkhorn_iterations=2,
        eps=1e-12,
    )
    expected = torch.empty_like(value)
    for graph_id in range(2):
        index = batch == graph_id
        kernel = torch.einsum("ihd,jhd->hij", query[index], key[index])
        row_scale = torch.ones_like(kernel[:, :, 0])
        for _ in range(2):
            key_mass = torch.einsum("hij,hi->hj", kernel, row_scale).clamp_min(1e-12)
            key_scale = key_mass.reciprocal()
            row_mass = torch.einsum("hij,hj->hi", kernel, key_scale).clamp_min(1e-12)
            row_scale = row_mass.reciprocal()
        weights = kernel * key_scale[:, None, :]
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        expected[index] = torch.einsum("hij,jhf->ihf", weights, value[index])

    assert _max_error(actual, expected) < 1e-10


def test_factorized_attention_one_iteration_is_the_compatible_default() -> None:
    torch.manual_seed(240)
    query = torch.rand(6, 2, 7, dtype=torch.float64)
    key = torch.rand(6, 2, 7, dtype=torch.float64)
    value = torch.randn(6, 2, 5, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    default = _factorized_attention(query, key, value, batch, balanced=True, eps=1e-12)
    num_graphs = 2
    query_sum = torch.zeros(num_graphs, *query.shape[1:], dtype=query.dtype).index_add(0, batch, query)
    key_mass = (key * query_sum[batch]).sum(dim=-1).clamp_min(1e-12)
    weighted_key = key * key_mass.reciprocal().unsqueeze(-1)
    key_sum = torch.zeros(num_graphs, *key.shape[1:], dtype=key.dtype).index_add(0, batch, weighted_key)
    denominator = (query * key_sum[batch]).sum(dim=-1).clamp_min(1e-12)
    summary_values = weighted_key.unsqueeze(-1) * value.unsqueeze(-2)
    summary = torch.zeros(num_graphs, *summary_values.shape[1:], dtype=value.dtype).index_add(
        0, batch, summary_values
    )
    numerator = torch.einsum("nhd,nhdf->nhf", query, summary[batch])
    legacy = numerator / denominator.unsqueeze(-1)

    assert torch.equal(default, legacy)


def test_factorized_attention_rejects_nonpositive_sinkhorn_iterations() -> None:
    query = torch.ones(2, 1, 2)
    key = torch.ones(2, 1, 2)
    value = torch.ones(2, 1, 1)
    batch = torch.zeros(2, dtype=torch.long)

    with pytest.raises(ValueError, match="sinkhorn_iterations must be positive"):
        _factorized_attention(
            query,
            key,
            value,
            batch,
            balanced=True,
            sinkhorn_iterations=0,
            eps=1e-12,
        )


def test_relative_radial_trace_matches_dense_pairwise_sum() -> None:
    torch.manual_seed(231)
    weights = torch.rand(2, 4, 4, dtype=torch.float64)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    gate = torch.randn(2, 4, dtype=torch.float64)
    pos = torch.randn(4, 3, dtype=torch.float64)
    mass = torch.einsum("hij,hj->ih", weights, gate)
    first = torch.einsum("hij,hj,ja->iha", weights, gate, pos)
    second = torch.einsum("hij,hj,j->ih", weights, gate, pos.square().sum(dim=-1))

    actual = _relative_radial_trace(mass, first, second, pos[:, None, :])
    expected = torch.einsum(
        "hij,hj,ija->ih",
        weights,
        gate,
        (pos[None, :, :] - pos[:, None, :]).square(),
    )

    assert _max_error(actual, expected) < 1e-10


def test_gram_invariants_are_rotation_invariant() -> None:
    torch.manual_seed(232)
    state = torch.randn(5, 3, 3, dtype=torch.float64)
    message = torch.randn(5, 2, 3, dtype=torch.float64)
    vector_base = torch.randn(5, 2, 3, dtype=torch.float64)
    relative = torch.randn(5, 2, 3, dtype=torch.float64)
    rotation = _random_rotation(torch.float64)

    actual = _gram_invariants(state, message, vector_base, relative)
    rotated = _gram_invariants(
        torch.einsum("nca,ba->ncb", state, rotation),
        torch.einsum("nha,ba->nhb", message, rotation),
        torch.einsum("nha,ba->nhb", vector_base, rotation),
        torch.einsum("nha,ba->nhb", relative, rotation),
    )

    assert _max_error(actual, rotated) < 1e-12


def test_normalized_tensor_square_route_and_cubic_trace_transform_correctly() -> None:
    torch.manual_seed(236)
    tensor = torch.randn(6, 2, 5, dtype=torch.float64)
    query = torch.randn(6, 2, 3, dtype=torch.float64)
    rotation = _random_rotation(torch.float64)
    tensor_matrix = _st_features_to_matrix(tensor)
    rotated_tensor = _matrix_to_st_features(_rotate_tensor(tensor_matrix, rotation))
    rotated_query = torch.einsum("nha,ba->nhb", query, rotation)

    route = _normalized_st_square_vector(tensor, query, eps=1e-12)
    cubic = _normalized_st_cubic_trace(tensor, eps=1e-12)
    moved_route = _normalized_st_square_vector(rotated_tensor, rotated_query, eps=1e-12)
    moved_cubic = _normalized_st_cubic_trace(rotated_tensor, eps=1e-12)

    assert _max_error(moved_route, torch.einsum("nha,ba->nhb", route, rotation)) < 1e-12
    assert _max_error(moved_cubic, cubic) < 1e-12


def test_moment_attention_forward_contract() -> None:
    model = _make_model()
    node_feats, pos, batch = _make_inputs()

    out = model(node_feats, pos, batch=batch)

    assert out["node_scalars"].shape == (9, 2)
    assert out["node_vectors"].shape == (9, 2, 3)
    assert out["node_tensors"].shape == (9, 1, 3, 3)
    assert out["graph_scalars"].shape == (2, 2)
    assert out["graph_vectors"].shape == (2, 2, 3)
    assert out["graph_tensors"].shape == (2, 1, 3, 3)
    assert out["hidden_irreps"] == "12x0e + 3x1o"
    assert out["output_irreps"] == "2x0e + 2x1o + 1x2e"
    assert out["attention_mode"] == "moment_linear"


def test_moment_attention_rotation_translation_equivariance() -> None:
    model = _make_model()
    node_feats, pos, batch = _make_inputs()
    out = model(node_feats, pos, batch=batch)

    torch.manual_seed(233)
    rotation = _random_rotation(pos.dtype)
    translation = torch.randn(1, 3, dtype=pos.dtype)
    moved = model(node_feats, pos @ rotation.T + translation, batch=batch)

    assert _max_error(moved["node_scalars"], out["node_scalars"]) < 1e-6
    assert _max_error(moved["graph_scalars"], out["graph_scalars"]) < 1e-6
    assert _max_error(moved["node_vectors"], torch.einsum("nca,ba->ncb", out["node_vectors"], rotation)) < 1e-6
    assert _max_error(moved["graph_vectors"], torch.einsum("gca,ba->gcb", out["graph_vectors"], rotation)) < 1e-6
    assert _max_error(moved["node_tensors"], _rotate_tensor(out["node_tensors"], rotation)) < 1e-6
    assert _max_error(moved["graph_tensors"], _rotate_tensor(out["graph_tensors"], rotation)) < 1e-6


def test_enhanced_moment_attention_rotation_translation_equivariance() -> None:
    model = _make_model(enhanced=True)
    node_feats, pos, batch = _make_inputs()
    out = model(node_feats, pos, batch=batch)

    torch.manual_seed(234)
    rotation = _random_rotation(pos.dtype)
    translation = torch.randn(1, 3, dtype=pos.dtype)
    moved = model(node_feats, pos @ rotation.T + translation, batch=batch)

    assert _max_error(moved["node_scalars"], out["node_scalars"]) < 1e-6
    assert _max_error(moved["node_vectors"], torch.einsum("nca,ba->ncb", out["node_vectors"], rotation)) < 1e-6
    assert _max_error(moved["node_tensors"], _rotate_tensor(out["node_tensors"], rotation)) < 1e-6


@pytest.mark.parametrize("ffn_hidden_ratio", [0.25, 1.0, 2.0])
def test_ffn_moment_attention_reflection_equivariance(ffn_hidden_ratio: float) -> None:
    torch.manual_seed(235)
    model = EquivariantMomentAttention(
        EquivariantMomentAttentionConfig(
            node_dim=6,
            hidden_irreps="12x0e + 3x1o",
            output_irreps="2x0e + 2x1o + 1x2e",
            num_layers=2,
            num_heads=3,
            equivariant_ffn=True,
            ffn_hidden_ratio=ffn_hidden_ratio,
        )
    ).to(dtype=torch.float64).eval()
    node_feats, pos, batch = _make_inputs()
    out = model(node_feats, pos, batch=batch)
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=pos.dtype))
    moved = model(node_feats, pos @ reflection.T, batch=batch)

    assert _max_error(moved["node_scalars"], out["node_scalars"]) < 1e-6
    assert _max_error(moved["node_vectors"], torch.einsum("nca,ba->ncb", out["node_vectors"], reflection)) < 1e-6
    assert _max_error(moved["node_tensors"], _rotate_tensor(out["node_tensors"], reflection)) < 1e-6
    assert _max_error(moved["graph_scalars"], out["graph_scalars"]) < 1e-6
    assert _max_error(moved["graph_vectors"], torch.einsum("gca,ba->gcb", out["graph_vectors"], reflection)) < 1e-6
    assert _max_error(moved["graph_tensors"], _rotate_tensor(out["graph_tensors"], reflection)) < 1e-6


def test_radial_distance_moment_attention_reflection_equivariance() -> None:
    model = _make_model(feature="radial_distance_kernel")
    node_feats, pos, batch = _make_inputs()
    out = model(node_feats, pos, batch=batch)
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=pos.dtype))
    moved = model(node_feats, pos @ reflection.T, batch=batch)

    assert _max_error(moved["node_scalars"], out["node_scalars"]) < 1e-6
    assert _max_error(moved["node_vectors"], torch.einsum("nca,ba->ncb", out["node_vectors"], reflection)) < 1e-6
    assert _max_error(moved["node_tensors"], _rotate_tensor(out["node_tensors"], reflection)) < 1e-6


def test_dynamic_moment_routing_reflection_equivariance() -> None:
    model = _make_model(feature="dynamic_moment_routing")
    node_feats, pos, batch = _make_inputs()
    out = model(node_feats, pos, batch=batch)
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=pos.dtype))
    moved = model(node_feats, pos @ reflection.T, batch=batch)

    assert _max_error(moved["node_scalars"], out["node_scalars"]) < 1e-6
    assert _max_error(moved["node_vectors"], torch.einsum("nca,ba->ncb", out["node_vectors"], reflection)) < 1e-6
    assert _max_error(moved["node_tensors"], _rotate_tensor(out["node_tensors"], reflection)) < 1e-6


def test_iterative_sinkhorn_reflection_and_permutation_consistency() -> None:
    model = _make_model(sinkhorn_iterations=2)
    node_feats, pos, batch = _make_inputs()
    out = model(node_feats, pos, batch=batch)
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=pos.dtype))
    moved = model(node_feats, pos @ reflection.T, batch=batch)

    assert _max_error(moved["node_scalars"], out["node_scalars"]) < 1e-6
    assert _max_error(moved["node_vectors"], torch.einsum("nca,ba->ncb", out["node_vectors"], reflection)) < 1e-6
    assert _max_error(moved["node_tensors"], _rotate_tensor(out["node_tensors"], reflection)) < 1e-6

    permutation = torch.tensor([3, 0, 7, 5, 1, 8, 2, 6, 4], dtype=torch.long)
    inverse = torch.argsort(permutation)
    permuted = model(node_feats[permutation], pos[permutation], batch=batch[permutation])
    assert _max_error(permuted["node_scalars"][inverse], out["node_scalars"]) < 1e-6
    assert _max_error(permuted["node_vectors"][inverse], out["node_vectors"]) < 1e-6
    assert _max_error(permuted["node_tensors"][inverse], out["node_tensors"]) < 1e-6


def test_zero_initialized_dynamic_routing_matches_static_model() -> None:
    static = _make_model()
    dynamic = _make_model(feature="dynamic_moment_routing")
    node_feats, pos, batch = _make_inputs()

    static_out = static(node_feats, pos, batch=batch)
    dynamic_out = dynamic(node_feats, pos, batch=batch)

    for key in ["node_scalars", "node_vectors", "node_tensors"]:
        assert _max_error(dynamic_out[key], static_out[key]) < 1e-12


def test_zero_initialized_dynamic_routing_receives_nonzero_gradients() -> None:
    torch.manual_seed(237)
    model = EquivariantMomentAttention(
        EquivariantMomentAttentionConfig(
            node_dim=6,
            hidden_irreps="12x0e + 3x1o",
            output_irreps="1x0e",
            num_layers=2,
            num_heads=3,
            dynamic_moment_routing=True,
            equivariant_ffn=True,
        )
    ).to(dtype=torch.float64)
    node_feats, pos, batch = _make_inputs()

    loss = model(node_feats, pos, batch=batch)["graph_scalars"].square().mean()
    loss.backward()

    routing_grads = [
        parameter.grad
        for layer in model.layers
        for parameter in [layer.routing_mlp[-1].weight, layer.routing_context.weight]
    ]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in routing_grads)
    assert sum(float(gradient.abs().sum()) for gradient in routing_grads) > 0.0


@pytest.mark.parametrize(
    "feature",
    [
        "radial_trace",
        "full_gram_invariants",
        "shifted_angular_kernel",
        "learnable_balance_exponent",
        "radial_distance_kernel",
        "dynamic_moment_routing",
        "equivariant_ffn",
    ],
)
def test_each_enhancement_is_individually_equivariant(feature: str) -> None:
    model = _make_model(feature=feature)
    node_feats, pos, batch = _make_inputs()
    out = model(node_feats, pos, batch=batch)
    rotation = _random_rotation(pos.dtype)
    moved = model(node_feats, pos @ rotation.T, batch=batch)

    assert _max_error(moved["node_scalars"], out["node_scalars"]) < 1e-6
    assert _max_error(moved["node_vectors"], torch.einsum("nca,ba->ncb", out["node_vectors"], rotation)) < 1e-6
    assert _max_error(moved["node_tensors"], _rotate_tensor(out["node_tensors"], rotation)) < 1e-6


@pytest.mark.parametrize("enhanced", [False, True])
def test_moment_attention_permutation_and_batch_consistency(enhanced: bool) -> None:
    model = _make_model(enhanced=enhanced)
    node_feats, pos, batch = _make_inputs()
    batched = model(node_feats, pos, batch=batch)

    perm = torch.tensor([3, 0, 7, 5, 1, 8, 2, 6, 4], dtype=torch.long)
    inverse = torch.argsort(perm)
    permuted = model(node_feats[perm], pos[perm], batch=batch[perm])
    assert _max_error(permuted["node_scalars"][inverse], batched["node_scalars"]) < 1e-6
    assert _max_error(permuted["node_vectors"][inverse], batched["node_vectors"]) < 1e-6
    assert _max_error(permuted["node_tensors"][inverse], batched["node_tensors"]) < 1e-6

    independent = [model(node_feats[batch == graph_id], pos[batch == graph_id]) for graph_id in range(2)]
    for key in ["node_scalars", "node_vectors", "node_tensors"]:
        assert _max_error(batched[key], torch.cat([out[key] for out in independent], dim=0)) < 1e-6
    for key in ["graph_scalars", "graph_vectors", "graph_tensors"]:
        assert _max_error(batched[key], torch.cat([out[key] for out in independent], dim=0)) < 1e-6


def test_moment_attention_preserves_graph_scale_information() -> None:
    model = _make_model()
    node_feats, pos, batch = _make_inputs()

    original = model(node_feats, pos, batch=batch)["node_scalars"]
    scaled = model(node_feats, pos * 2.5, batch=batch)["node_scalars"]

    assert _max_error(original, scaled) > 1e-6


def test_moment_attention_backward_is_finite_for_single_node_graphs() -> None:
    torch.manual_seed(239)
    model = EquivariantMomentAttention(
        EquivariantMomentAttentionConfig(
            node_dim=4,
            hidden_irreps="8x0e + 2x1o",
            output_irreps="1x0e",
            num_layers=2,
            num_heads=2,
            radial_distance_kernel=True,
            dynamic_moment_routing=True,
            equivariant_ffn=True,
        )
    )
    node_feats = torch.randn(3, 4)
    pos = torch.randn(3, 3)
    batch = torch.arange(3)

    loss = model(node_feats, pos, batch=batch)["graph_scalars"].square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_moment_attention_rejects_nonpositive_sinkhorn_iterations() -> None:
    with pytest.raises(ValueError, match="sinkhorn_iterations must be positive"):
        EquivariantMomentAttention(EquivariantMomentAttentionConfig(node_dim=4, sinkhorn_iterations=0))


def test_moment_attention_rejects_persistent_tensor_hidden_state() -> None:
    with pytest.raises(ValueError, match="hidden_irreps supports only scalar and vector channels"):
        EquivariantMomentAttention(
            EquivariantMomentAttentionConfig(node_dim=4, hidden_irreps="8x0e + 2x1o + 1x2e")
        )
