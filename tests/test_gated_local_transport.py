from __future__ import annotations

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
import equivariant_attention.moment as moment


def _model(
    *,
    gated: bool = False,
    grouped: bool = False,
) -> EquivariantAttention:
    return EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=5,
            hidden_irreps="8x0e + 2x1o",
            output_irreps="2x0e + 1x1o + 1x2e",
            num_layers=2,
            num_heads=2,
            local_head_counts=(2, 0),
            use_key_balancing=False,
            use_gated_local_transport=gated,
            use_grouped_invariant_normalization=grouped,
        )
    ).double()


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_feats = torch.randn(7, 5, dtype=torch.float64)
    pos = 0.35 * torch.randn(7, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6, 1, 2, 0, 4, 5, 6, 3],
        ]
    )
    return node_feats, pos, batch, edge_index


def _constant_sender_transport() -> moment._GatedEquivariantLocalTransport:
    transport = moment._GatedEquivariantLocalTransport(
        scalars=4,
        vectors=1,
        num_heads=1,
        num_rbf=4,
        eps=1e-12,
    ).double()
    with torch.no_grad():
        for parameter in transport.parameters():
            parameter.zero_()
        transport.edge_mlp[-1].bias[transport.head_dim + 2] = 1.0
    return transport


def test_gated_local_defaults_are_exactly_backward_compatible() -> None:
    torch.manual_seed(2401)
    default = _model()
    torch.manual_seed(2401)
    explicit = _model(gated=False, grouped=False)
    node_feats, pos, batch, edge_index = _inputs()

    assert list(default.state_dict()) == list(explicit.state_dict())
    for name, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name]), name
    default_output = default(node_feats, pos, batch=batch, edge_index=edge_index)
    explicit_output = explicit(node_feats, pos, batch=batch, edge_index=edge_index)
    for name in default_output:
        assert torch.equal(default_output[name], explicit_output[name]), name


def test_gated_local_preserves_every_common_initialization() -> None:
    torch.manual_seed(2402)
    baseline = _model()
    torch.manual_seed(2402)
    candidate = _model(gated=True, grouped=True)

    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    common = baseline_state.keys() & candidate_state.keys()

    assert common
    assert any("gated_local" in name for name in candidate_state)
    for name in common:
        assert torch.equal(baseline_state[name], candidate_state[name]), name


def test_gated_local_public_path_is_o3_translation_and_permutation_consistent() -> None:
    torch.manual_seed(2403)
    model = _model(gated=True, grouped=True)
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

    assert torch.allclose(moved["node_scalars"], reference["node_scalars"], atol=1e-9)
    assert torch.allclose(
        moved["node_vectors"],
        torch.einsum("nca,ba->ncb", reference["node_vectors"], orthogonal),
        atol=1e-9,
    )
    expected_tensor = torch.einsum(
        "ab,nkbc,dc->nkad", orthogonal, reference["node_tensors"], orthogonal
    )
    assert torch.allclose(moved["node_tensors"], expected_tensor, atol=1e-9)
    for name in ("node_scalars", "node_vectors", "node_tensors"):
        assert torch.allclose(permuted[name][inverse], reference[name], atol=1e-9)


def test_gated_local_receives_finite_nonzero_state_and_coordinate_gradients() -> None:
    torch.manual_seed(2404)
    model = _model(gated=True, grouped=True)
    node_feats, pos, batch, edge_index = _inputs()
    node_feats.requires_grad_()
    pos.requires_grad_()

    output = model(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index,
    )
    output["graph_scalars"].square().sum().backward()

    gated_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "gated_local" in name
    ]
    assert gated_gradients
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gated_gradients
    )
    assert any(torch.count_nonzero(gradient) for gradient in gated_gradients)
    for gradient in (node_feats.grad, pos.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient)


def test_all_local_gated_layer_skips_route_inactive_projections() -> None:
    torch.manual_seed(2408)
    model = _model(gated=True, grouped=True)
    local_layer = model.layers[0]
    inactive_modules = (
        local_layer.query_scalar,
        local_layer.key_scalar,
        local_layer.value_scalar,
        local_layer.key_vector,
        local_layer.value_vector,
        local_layer.key_vector_gate,
        local_layer.relative_gate,
        local_layer.tensor_gate,
        local_layer.radial_trace_gate,
    )

    def reject_inactive_projection(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
    ) -> None:
        raise AssertionError("route-inactive projection executed")

    handles = [
        module.register_forward_pre_hook(reject_inactive_projection)
        for module in inactive_modules
    ]
    try:
        node_feats, pos, batch, edge_index = _inputs()
        output = model(node_feats, pos, batch=batch, edge_index=edge_index)
    finally:
        for handle in handles:
            handle.remove()

    assert torch.isfinite(output["graph_scalars"]).all()


def test_all_local_gated_fast_path_matches_zero_pairwise_reference() -> None:
    class ZeroPairwise(torch.nn.Module):
        def forward(
            self,
            query_scalar: torch.Tensor,
            _key_scalar: torch.Tensor,
            _local_geometry: object,
            *,
            num_nodes: int,
        ) -> torch.Tensor:
            return query_scalar.new_zeros(
                (num_nodes, query_scalar.shape[1], query_scalar.shape[2])
            )

    torch.manual_seed(2410)
    fast = _model(gated=True, grouped=True)
    torch.manual_seed(2410)
    reference = _model(gated=True, grouped=True)
    reference.local_pairwise_content = ZeroPairwise()

    node_feats, pos, batch, edge_index = _inputs()
    fast_node_feats = node_feats.detach().clone().requires_grad_()
    fast_pos = pos.detach().clone().requires_grad_()
    reference_node_feats = node_feats.detach().clone().requires_grad_()
    reference_pos = pos.detach().clone().requires_grad_()
    fast_output = fast(
        fast_node_feats,
        fast_pos,
        batch=batch,
        edge_index=edge_index,
    )
    reference_output = reference(
        reference_node_feats,
        reference_pos,
        batch=batch,
        edge_index=edge_index,
    )

    for name in fast_output:
        assert torch.allclose(
            fast_output[name],
            reference_output[name],
            atol=1e-12,
            rtol=1e-11,
        ), name
    fast_loss = sum(value.square().sum() for value in fast_output.values())
    reference_loss = sum(value.square().sum() for value in reference_output.values())
    fast_gradients = torch.autograd.grad(
        fast_loss,
        (fast_node_feats, fast_pos),
    )
    reference_gradients = torch.autograd.grad(
        reference_loss,
        (reference_node_feats, reference_pos),
    )
    for fast_gradient, reference_gradient in zip(
        fast_gradients,
        reference_gradients,
        strict=True,
    ):
        assert torch.allclose(
            fast_gradient,
            reference_gradient,
            atol=1e-11,
            rtol=1e-10,
        )


def test_factorized_gated_edge_projection_matches_expanded_edge_mlp() -> None:
    torch.manual_seed(2409)
    legacy = moment._GatedEquivariantLocalTransport(
        scalars=8,
        vectors=2,
        num_heads=2,
        num_rbf=4,
        eps=1e-12,
    ).double()
    factorized = moment._GatedEquivariantLocalTransport(
        scalars=8,
        vectors=2,
        num_heads=2,
        num_rbf=4,
        eps=1e-12,
    ).double()
    factorized.load_state_dict(legacy.state_dict())
    receiver = torch.tensor([0, 0, 1, 2, 3, 3])
    sender = torch.tensor([1, 2, 0, 3, 0, 2])

    legacy_inputs = (
        torch.randn(4, 2, 4, dtype=torch.float64, requires_grad=True),
        torch.randn(6, 4, dtype=torch.float64, requires_grad=True),
        torch.randn(6, 2, 5, dtype=torch.float64, requires_grad=True),
    )
    factorized_inputs = tuple(
        value.detach().clone().requires_grad_() for value in legacy_inputs
    )
    scalar_heads, rbf, vector_invariants = legacy_inputs
    expanded_rbf = rbf.unsqueeze(1).expand(-1, 2, -1)
    expected = legacy.edge_mlp(
        torch.cat(
            [
                scalar_heads[receiver],
                scalar_heads[sender],
                expanded_rbf,
                vector_invariants,
            ],
            dim=-1,
        )
    )
    actual = factorized._factorized_edge_mlp(
        factorized_inputs[0],
        receiver,
        sender,
        factorized_inputs[1],
        factorized_inputs[2],
    )
    probe = torch.randn_like(expected)
    expected_targets = (*legacy_inputs, *legacy.edge_mlp.parameters())
    actual_targets = (*factorized_inputs, *factorized.edge_mlp.parameters())
    expected_gradients = torch.autograd.grad(
        (expected * probe).sum(),
        expected_targets,
    )
    actual_gradients = torch.autograd.grad(
        (actual * probe).sum(),
        actual_targets,
    )

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-11)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        assert torch.allclose(
            actual_gradient,
            expected_gradient,
            atol=1e-12,
            rtol=1e-11,
        )


def test_local_geometry_caches_nonself_views_without_changing_tuple_contract() -> None:
    _node_feats, pos, batch, edge_index = _inputs()
    geometry = moment._local_geometry(
        pos,
        batch,
        num_graphs=2,
        cutoff=2.5,
        num_rbf=4,
        edge_index=edge_index,
    )
    receiver, sender, displacement, squared_distance, rbf = geometry
    nonself = receiver != sender

    assert len(geometry) == 5
    assert torch.equal(geometry[0], receiver)
    assert torch.equal(geometry.nonself_receiver, receiver[nonself])
    assert torch.equal(geometry.nonself_sender, sender[nonself])
    assert torch.equal(geometry.nonself_displacement, displacement[nonself])
    assert torch.equal(geometry.nonself_squared_distance, squared_distance[nonself])
    assert torch.equal(geometry.nonself_rbf, rbf[nonself])
    assert torch.equal(
        geometry.nonself_cutoff,
        moment._cosine_of_squared_distance_cutoff(squared_distance[nonself]),
    )
    assert torch.equal(
        geometry.nonself_tensor_features,
        moment._symmetric_traceless_features(displacement[nonself]),
    )


def test_gated_local_is_output_and_gradient_continuous_at_cutoff() -> None:
    torch.manual_seed(2405)
    transport = moment._GatedEquivariantLocalTransport(
        scalars=4,
        vectors=1,
        num_heads=1,
        num_rbf=4,
        eps=1e-12,
    ).double()
    scalars = torch.randn(3, 4, dtype=torch.float64)
    vectors = torch.randn(3, 1, 3, dtype=torch.float64)
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
        outputs = transport(scalars, vectors, geometry, num_nodes=3)
        flat = torch.cat([value.reshape(-1) for value in outputs])
        coefficients = torch.linspace(
            0.25,
            1.25,
            flat.numel(),
            dtype=flat.dtype,
        )
        gradient = torch.autograd.grad((flat * coefficients).sum(), pos)[0]
        return flat.detach(), gradient.detach()

    inside, inside_gradient = evaluate(2.5 - 1e-5)
    outside, outside_gradient = evaluate(2.5 + 1e-5)

    assert torch.allclose(inside, outside, atol=2e-4, rtol=2e-4)
    assert torch.allclose(
        inside_gradient,
        outside_gradient,
        atol=2e-3,
        rtol=2e-3,
    )


def test_gated_local_singleton_message_attenuates_to_cutoff_with_finite_gradients() -> (
    None
):
    transport = _constant_sender_transport()
    scalars = torch.zeros(2, 4, dtype=torch.float64)
    vectors = torch.zeros(2, 1, 3, dtype=torch.float64)
    vectors[1, 0, 0] = 1.0
    batch = torch.zeros(2, dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 0], [0, 1, 1]])
    ratios = (0.5, 0.7, 0.9, 0.95, 0.99, 1.0)
    observed: list[float] = []

    for ratio in ratios:
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [2.5 * ratio, 0.0, 0.0]],
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
        vector_message = transport(
            scalars,
            vectors,
            geometry,
            num_nodes=2,
        )[1]
        magnitude = vector_message[0, 0, 0]
        gradient = torch.autograd.grad(magnitude, pos)[0]
        cutoff = moment._cosine_of_squared_distance_cutoff(
            torch.tensor(ratio**2, dtype=torch.float64)
        )
        expected = torch.tanh(torch.tensor(1.0, dtype=torch.float64))
        expected = expected * cutoff / torch.sqrt(1.0 + cutoff)

        assert torch.allclose(magnitude, expected, atol=1e-12, rtol=1e-11)
        assert torch.isfinite(vector_message).all()
        assert torch.isfinite(gradient).all()
        observed.append(magnitude.detach().item())

    assert all(left > right for left, right in zip(observed, observed[1:]))
    assert observed[-1] == 0.0


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_gated_local_unit_weight_neighbors_scale_as_soft_mass(
    degree: int,
) -> None:
    transport = _constant_sender_transport()
    num_nodes = degree + 1
    scalars = torch.zeros(num_nodes, 4, dtype=torch.float64)
    vectors = torch.zeros(num_nodes, 1, 3, dtype=torch.float64)
    vectors[1:, 0, 0] = 1.0
    pos = torch.zeros(num_nodes, 3, dtype=torch.float64)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    nodes = torch.arange(num_nodes)
    edge_index = torch.stack(
        [
            torch.cat([nodes, torch.zeros(degree, dtype=torch.long)]),
            torch.cat([nodes, torch.arange(1, num_nodes)]),
        ]
    )
    geometry = moment._local_geometry(
        pos,
        batch,
        num_graphs=1,
        cutoff=2.5,
        num_rbf=4,
        edge_index=edge_index,
    )

    vector_message = transport(
        scalars,
        vectors,
        geometry,
        num_nodes=num_nodes,
    )[1]
    expected = (
        degree
        * torch.tanh(torch.tensor(1.0, dtype=torch.float64))
        / torch.sqrt(torch.tensor(1.0 + degree, dtype=torch.float64))
    )

    assert torch.allclose(vector_message[0, 0, 0], expected, atol=1e-12, rtol=1e-11)


@pytest.mark.parametrize(
    "overrides",
    [
        {"local_head_counts": (0, 0), "use_gated_local_transport": True},
        {"local_head_counts": (1, 0), "use_gated_local_transport": True},
        {
            "local_head_counts": (2, 0),
            "use_gated_local_transport": True,
            "use_edge_conditioned_local_transport": True,
        },
    ],
)
def test_gated_local_rejects_inactive_partial_or_conflicting_routes(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="gated local"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=5,
                hidden_irreps="8x0e + 2x1o",
                num_layers=2,
                num_heads=2,
                **overrides,
            )
        )
