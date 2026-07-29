from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
import equivariant_attention.moment as moment


def _orthogonal() -> torch.Tensor:
    transform, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.linalg.det(transform) > 0:
        transform[:, 0].neg_()
    return transform


def _config(
    *,
    adaptive: bool,
) -> EquivariantAttentionConfig:
    return EquivariantAttentionConfig(
        node_dim=5,
        hidden_irreps="8x0e + 2x1o",
        output_irreps="2x0e + 1x1o + 1x2e",
        num_layers=3,
        num_heads=2,
        local_head_counts=(2, 0, 2),
        local_cutoff=5.0,
        use_gated_local_transport=True,
        use_grouped_invariant_normalization=True,
        use_adaptive_multiscale_spatial_kernel=adaptive,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(2801)
    node_feats = torch.randn(8, 5, dtype=torch.float64)
    pos = 0.4 * torch.randn(8, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7],
            [0, 1, 2, 3, 1, 2, 3, 0, 4, 5, 6, 7, 5, 6, 7, 4],
        ]
    )
    return node_feats, pos, batch, edge_index


def test_adaptive_features_define_positive_o3_invariant_multiscale_kernel() -> None:
    torch.manual_seed(2802)
    pos = torch.randn(7, 3, dtype=torch.float64)
    scales = torch.tensor([0.125, 0.25, 0.5, 1.0], dtype=torch.float64)
    query_gate_logits = torch.randn(7, 3, 4, dtype=torch.float64)
    key_gate_logits = torch.randn(7, 3, 4, dtype=torch.float64)
    transform = _orthogonal()

    query_features = moment._adaptive_multiscale_spatial_features(
        pos,
        scales,
        query_gate_logits,
    )
    key_features = moment._adaptive_multiscale_spatial_features(
        pos,
        scales,
        key_gate_logits,
    )
    moved_query = moment._adaptive_multiscale_spatial_features(
        pos @ transform.T,
        scales,
        query_gate_logits,
    )
    moved_key = moment._adaptive_multiscale_spatial_features(
        pos @ transform.T,
        scales,
        key_gate_logits,
    )
    kernel = torch.einsum("ihf,jhf->hij", query_features, key_features)
    moved_kernel = torch.einsum(
        "ihf,jhf->hij",
        moved_query,
        moved_key,
    )

    assert query_features.shape == (7, 3, 40)
    assert torch.all(kernel > 0.0)
    assert torch.allclose(moved_kernel, kernel, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_adaptive_profiles_remain_positive_for_opposing_finite_logits(
    dtype: torch.dtype,
) -> None:
    pos = torch.zeros(1, 3, dtype=dtype)
    scales = torch.tensor([0.125, 0.25, 0.5, 1.0], dtype=dtype)
    query_logits = torch.tensor(
        [[[100.0, -100.0, -100.0, -100.0]]],
        dtype=dtype,
    )
    key_logits = torch.tensor(
        [[[-100.0, 100.0, -100.0, -100.0]]],
        dtype=dtype,
    )

    query_features = moment._adaptive_multiscale_spatial_features(
        pos,
        scales,
        query_logits,
    )
    key_features = moment._adaptive_multiscale_spatial_features(
        pos,
        scales,
        key_logits,
    )
    kernel = torch.einsum("ihf,jhf->hij", query_features, key_features)

    assert torch.isfinite(query_features).all()
    assert torch.isfinite(key_features).all()
    assert torch.all(kernel > 0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_adaptive_profiles_are_unit_norm_shift_invariant_and_finite_at_extremes(
    dtype: torch.dtype,
) -> None:
    pos = torch.zeros(1, 3, dtype=dtype)
    scales = torch.tensor([0.125, 0.25, 0.5, 1.0], dtype=dtype)
    moderate_logits = torch.tensor(
        [[[-3.0, -0.5, 0.25, 2.0]]],
        dtype=dtype,
        requires_grad=True,
    )
    extreme_logits = torch.tensor(
        [
            [
                [
                    torch.finfo(dtype).min,
                    -100.0,
                    100.0,
                    torch.finfo(dtype).max,
                ]
            ]
        ],
        dtype=dtype,
        requires_grad=True,
    )

    moderate = moment._adaptive_multiscale_spatial_features(
        pos,
        scales,
        moderate_logits,
    ).reshape(1, 1, 4, 10)
    shifted = moment._adaptive_multiscale_spatial_features(
        pos,
        scales,
        moderate_logits + 19.0,
    ).reshape(1, 1, 4, 10)
    extreme = moment._adaptive_multiscale_spatial_features(
        pos,
        scales,
        extreme_logits,
    ).reshape(1, 1, 4, 10)
    moderate_profile = moderate[..., 0]
    extreme_profile = extreme[..., 0]

    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    assert torch.allclose(moderate, shifted, atol=tolerance, rtol=tolerance)
    assert torch.allclose(
        moderate_profile.square().sum(dim=-1),
        torch.ones(1, 1, dtype=dtype),
        atol=tolerance,
        rtol=tolerance,
    )
    assert torch.isfinite(extreme).all()
    assert torch.all(extreme_profile > 0.0)
    (moderate.sum() + extreme.sum()).backward()
    assert moderate_logits.grad is not None
    assert extreme_logits.grad is not None
    assert torch.isfinite(moderate_logits.grad).all()
    assert torch.isfinite(extreme_logits.grad).all()


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("single_graph", [False, True])
def test_adaptive_spatial_factorization_matches_materialized_dense_kernel(
    balanced: bool,
    single_graph: bool,
) -> None:
    torch.manual_seed(2803)
    nodes = 7
    heads = 2
    query_scalar = moment._normalize_positive_features(
        torch.rand(nodes, heads, 4, dtype=torch.float64),
        eps=1e-12,
    )
    key_scalar = moment._normalize_positive_features(
        torch.rand(nodes, heads, 4, dtype=torch.float64),
        eps=1e-12,
    )
    query_vector = moment._unit_ball(
        torch.randn(nodes, heads, 3, dtype=torch.float64),
        eps=1e-12,
    )
    key_vector = moment._unit_ball(
        torch.randn(nodes, heads, 3, dtype=torch.float64),
        eps=1e-12,
    )
    pos = torch.randn(nodes, 3, dtype=torch.float64)
    scales = torch.tensor([0.125, 0.25, 0.5, 1.0], dtype=torch.float64)
    query_spatial = moment._adaptive_multiscale_spatial_features(
        pos,
        scales,
        torch.randn(nodes, heads, 4, dtype=torch.float64),
    )
    key_spatial = moment._adaptive_multiscale_spatial_features(
        pos,
        scales,
        torch.randn(nodes, heads, 4, dtype=torch.float64),
    )
    value = torch.randn(nodes, heads, 5, dtype=torch.float64)
    linear_scale = torch.tensor([0.1, 0.2], dtype=torch.float64)
    quadratic_scale = torch.tensor([0.3, 0.6], dtype=torch.float64)
    batch = (
        torch.zeros(nodes, dtype=torch.long)
        if single_graph
        else torch.tensor([0, 0, 0, 1, 1, 1, 1])
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
        spatial_features=query_spatial,
        spatial_key_features=key_spatial,
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
            query_spatial[index],
            key_spatial[index],
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

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-9)


def test_adaptive_spatial_lgl_is_opt_in_and_preserves_common_initialization() -> None:
    torch.manual_seed(2804)
    default = EquivariantAttention(_config(adaptive=False)).double()
    torch.manual_seed(2804)
    explicit_off = EquivariantAttention(_config(adaptive=False)).double()
    torch.manual_seed(2804)
    candidate = EquivariantAttention(_config(adaptive=True)).double()
    node_feats, pos, batch, edge_index = _inputs()

    assert list(default.state_dict()) == list(explicit_off.state_dict())
    for name, value in default.state_dict().items():
        assert torch.equal(value, explicit_off.state_dict()[name]), name
    common = default.state_dict().keys() & candidate.state_dict().keys()
    assert common
    for name in common:
        assert torch.equal(default.state_dict()[name], candidate.state_dict()[name]), name
    adaptive_keys = [
        name for name in candidate.state_dict() if "adaptive_spatial_gate" in name
    ]
    assert adaptive_keys
    assert set(candidate.state_dict()) - set(default.state_dict()) == set(adaptive_keys)
    assert all("layers.1." in name for name in adaptive_keys)
    assert all(
        layer.adaptive_spatial_gate is None
        for index, layer in enumerate(candidate.layers)
        if index != 1
    )

    default_output = default(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index,
    )
    explicit_output = explicit_off(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index,
    )
    for name in default_output:
        assert torch.equal(default_output[name], explicit_output[name]), name


def test_adaptive_spatial_lgl_validates_its_narrow_route() -> None:
    with pytest.raises(TypeError, match="use_adaptive_multiscale_spatial_kernel"):
        EquivariantAttention(
            EquivariantAttentionConfig(  # type: ignore[arg-type]
                node_dim=5,
                use_adaptive_multiscale_spatial_kernel=1,
            )
        )
    with pytest.raises(ValueError, match="three-layer LGL"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=5,
                use_adaptive_multiscale_spatial_kernel=True,
            )
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=5,
                num_layers=3,
                num_heads=2,
                local_head_counts=(2, 0, 2),
                use_multiscale_spatial_kernel=True,
                use_adaptive_multiscale_spatial_kernel=True,
            )
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"global_transport_mode": "uniform"}, "learned global transport"),
        (
            {"global_memory_count": 4, "use_memory_interaction": True},
            "cannot use memory interaction",
        ),
        (
            {"use_key_balancing": False, "use_whitened_global_read": True},
            "cannot use whitened global read",
        ),
    ],
)
def test_adaptive_spatial_lgl_rejects_incompatible_global_mechanisms(
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        EquivariantAttention(replace(_config(adaptive=True), **updates))


def test_adaptive_spatial_lgl_preserves_o3_translation_and_permutation() -> None:
    torch.manual_seed(2805)
    model = EquivariantAttention(_config(adaptive=True)).double().eval()
    node_feats, pos, batch, edge_index = _inputs()
    transform = _orthogonal()
    translation = torch.randn(1, 3, dtype=torch.float64)
    permutation = torch.tensor([3, 0, 2, 1, 7, 5, 4, 6])
    inverse = torch.argsort(permutation)

    reference = model(node_feats, pos, batch=batch, edge_index=edge_index)
    moved = model(
        node_feats,
        pos @ transform.T + translation,
        batch=batch,
        edge_index=edge_index,
    )
    permuted = model(
        node_feats[permutation],
        pos[permutation],
        batch=batch[permutation],
        edge_index=inverse[edge_index],
    )

    assert torch.allclose(moved["node_scalars"], reference["node_scalars"], atol=1e-9)
    assert torch.allclose(
        moved["node_vectors"],
        torch.einsum("nca,ba->ncb", reference["node_vectors"], transform),
        atol=1e-9,
    )
    expected_tensor = torch.einsum(
        "ab,nkbc,dc->nkad",
        transform,
        reference["node_tensors"],
        transform,
    )
    assert torch.allclose(moved["node_tensors"], expected_tensor, atol=1e-9)
    for name in ("node_scalars", "node_vectors", "node_tensors"):
        assert torch.allclose(permuted[name][inverse], reference[name], atol=1e-9)


def test_adaptive_spatial_gate_receives_finite_nonzero_gradients() -> None:
    torch.manual_seed(2806)
    model = EquivariantAttention(_config(adaptive=True)).double()
    node_feats, pos, batch, edge_index = _inputs()
    pos.requires_grad_()

    output = model(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index,
    )
    output["graph_scalars"].square().sum().backward()

    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "adaptive_spatial_gate" in name
    ]
    assert gradients
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
    gate = model.layers[1].adaptive_spatial_gate
    assert gate is not None
    assert gate.weight.grad is not None
    midpoint = gate.out_features // 2
    assert torch.count_nonzero(gate.weight.grad[:midpoint])
    assert torch.count_nonzero(gate.weight.grad[midpoint:])
    assert pos.grad is not None and torch.isfinite(pos.grad).all()


def test_adaptive_spatial_gate_has_finite_gradients_for_opposing_logits() -> None:
    torch.manual_seed(2808)
    model = EquivariantAttention(_config(adaptive=True)).float()
    node_feats, pos, batch, edge_index = _inputs()
    node_feats = node_feats.float()
    pos = pos.float()
    gate = model.layers[1].adaptive_spatial_gate
    assert gate is not None
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.reshape(2, model.config.num_heads, -1)[0].copy_(
            torch.tensor([100.0, -100.0, -100.0, -100.0])
        )
        gate.bias.reshape(2, model.config.num_heads, -1)[1].copy_(
            torch.tensor([-100.0, 100.0, -100.0, -100.0])
        )

    output = model(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index,
    )
    assert all(torch.isfinite(value).all() for value in output.values())
    output["graph_scalars"].square().sum().backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert gate.weight.grad is not None
    assert gate.bias.grad is not None


def test_multiscale_spatial_summary_separates_degree_two_moment_collision() -> None:
    root_five = torch.tensor(5.0, dtype=torch.float64).sqrt()
    cloud_a_x = (
        torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=torch.float64) / root_five
    )
    root_eight_fifths = torch.tensor(8.0 / 5.0, dtype=torch.float64).sqrt()
    root_two_fifths = torch.tensor(2.0 / 5.0, dtype=torch.float64).sqrt()
    cloud_b_x = torch.stack(
        [
            -root_eight_fifths,
            -root_two_fifths,
            root_two_fifths,
            root_eight_fifths,
        ]
    )
    cloud_a = torch.zeros(4, 3, dtype=torch.float64)
    cloud_b = torch.zeros(4, 3, dtype=torch.float64)
    cloud_a[:, 0] = cloud_a_x
    cloud_b[:, 0] = cloud_b_x
    scales = torch.tensor([0.125, 0.25, 0.5, 1.0], dtype=torch.float64)
    zero_logits = torch.zeros(4, 1, 4, dtype=torch.float64)

    assert torch.allclose(cloud_a.sum(dim=0), cloud_b.sum(dim=0), atol=1e-12)
    assert torch.allclose(
        torch.einsum("ni,nj->ij", cloud_a, cloud_a),
        torch.einsum("ni,nj->ij", cloud_b, cloud_b),
        atol=1e-12,
    )
    summary_a = moment._adaptive_multiscale_spatial_features(
        cloud_a,
        scales,
        zero_logits,
    ).sum(dim=0)
    summary_b = moment._adaptive_multiscale_spatial_features(
        cloud_b,
        scales,
        zero_logits,
    ).sum(dim=0)

    assert torch.linalg.vector_norm(summary_a - summary_b) > 1e-6
