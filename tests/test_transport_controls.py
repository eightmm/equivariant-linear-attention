import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
import equivariant_attention.moment as moment


def _uniform_message_inputs(seed: int = 907) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    nodes, heads, head_dim = 5, 2, 3
    tensors = [
        torch.randn(nodes, heads, head_dim, dtype=torch.float64),
        torch.randn(nodes, heads, head_dim, dtype=torch.float64),
        torch.randn(nodes, heads, 3, dtype=torch.float64),
        torch.randn(nodes, heads, 3, dtype=torch.float64),
        torch.rand(heads, dtype=torch.float64),
        torch.randn(nodes, heads, head_dim, dtype=torch.float64),
        torch.randn(nodes, heads, 3, dtype=torch.float64),
        torch.randn(nodes, heads, dtype=torch.float64),
        torch.randn(nodes, heads, dtype=torch.float64),
        torch.randn(nodes, heads, dtype=torch.float64),
        torch.randn(nodes, 3, dtype=torch.float64),
    ]
    return tuple(value.detach().requires_grad_() for value in tensors)


def _dense_uniform_messages(
    scalar_value: torch.Tensor,
    vector_value: torch.Tensor,
    relative_gate: torch.Tensor,
    tensor_gate: torch.Tensor,
    radial_trace_gate: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    scalar_messages = []
    vector_messages = []
    relative_messages = []
    tensor_messages = []
    radial_messages = []
    for receiver in range(pos.shape[0]):
        same_graph = batch == batch[receiver]
        relative = pos[same_graph] - pos[receiver]
        count = int(same_graph.sum())
        scalar_messages.append(scalar_value[same_graph].sum(dim=0) / count)
        vector_messages.append(vector_value[same_graph].sum(dim=0) / count)
        relative_messages.append(
            (
                relative_gate[same_graph].unsqueeze(-1)
                * relative.unsqueeze(1)
            ).sum(dim=0)
            / count
        )
        tensor_messages.append(
            (
                tensor_gate[same_graph].unsqueeze(-1)
                * moment._symmetric_traceless_features(relative).unsqueeze(1)
            ).sum(dim=0)
            / count
        )
        radial_messages.append(
            (
                radial_trace_gate[same_graph]
                * relative.square().sum(dim=-1, keepdim=True)
            ).sum(dim=0)
            / count
        )
    return tuple(
        torch.stack(values)
        for values in (
            scalar_messages,
            vector_messages,
            relative_messages,
            tensor_messages,
            radial_messages,
        )
    )


def test_uniform_global_messages_match_dense_reference_and_gradients() -> None:
    (
        q0,
        k0,
        q1,
        k1,
        gamma,
        scalar_value,
        vector_value,
        relative_gate,
        tensor_gate,
        radial_trace_gate,
        pos,
    ) = _uniform_message_inputs()
    batch = torch.tensor([0, 0, 1, 1, 1])
    beta = torch.rand(2, dtype=torch.float64, requires_grad=True)
    actual = moment._global_moment_messages(
        q0,
        k0,
        q1,
        k1,
        gamma,
        scalar_value,
        vector_value,
        relative_gate,
        tensor_gate,
        radial_trace_gate,
        pos,
        batch,
        num_graphs=2,
        graph_counts=torch.tensor([2, 3]),
        balanced=True,
        alignment_scale=beta,
        alignment_dot_scale=beta,
        kernel_floor=1.0,
        kernel_floor_mode="fixed",
        memory_count=1,
        memory_temperature=1.0,
        memory_assignment_scale=2.5,
        memory_interaction_cutoff=2.5,
        use_memory_interaction=False,
        use_radial_trace=True,
        global_transport_mode="uniform",
    )
    expected = _dense_uniform_messages(
        scalar_value,
        vector_value,
        relative_gate,
        tensor_gate,
        radial_trace_gate,
        pos,
        batch,
    )
    probes = tuple(torch.randn_like(value) for value in actual)
    actual_loss = sum((left * probe).sum() for left, probe in zip(actual, probes))
    expected_loss = sum((right * probe).sum() for right, probe in zip(expected, probes))
    targets = (
        scalar_value,
        vector_value,
        relative_gate,
        tensor_gate,
        radial_trace_gate,
        pos,
    )
    actual_gradients = torch.autograd.grad(actual_loss, targets, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected_loss, targets)

    for left, right in zip(actual, expected, strict=True):
        assert torch.allclose(left, right, atol=1e-10, rtol=1e-9)
    for left, right in zip(actual_gradients, expected_gradients, strict=True):
        assert torch.allclose(left, right, atol=1e-10, rtol=1e-9)
    assert torch.autograd.grad(actual_loss, (q0, k0, q1, k1, gamma, beta), allow_unused=True) == (
        None,
        None,
        None,
        None,
        None,
        None,
    )


@pytest.mark.parametrize(
    "routing, expected",
    [
        ("ggg", (0, 0, 0)),
        ("lgg", (4, 0, 0)),
        ("ggl", (0, 0, 4)),
        ("lgl", (4, 0, 4)),
        ("lll", (4, 4, 4)),
    ],
)
def test_core_routing_presets(routing: str, expected: tuple[int, ...]) -> None:
    assert moment.routing_head_counts(routing, num_layers=3, num_heads=4) == expected


def test_default_and_explicit_learned_modes_are_exactly_identical() -> None:
    torch.manual_seed(911)
    default = EquivariantAttention(EquivariantAttentionConfig(node_dim=4)).double()
    torch.manual_seed(911)
    explicit = EquivariantAttention(
        EquivariantAttentionConfig(node_dim=4, global_transport_mode="learned")
    ).double()
    node_feats = torch.randn(7, 4, dtype=torch.float64)
    pos = torch.randn(7, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])

    assert list(default.state_dict()) == list(explicit.state_dict())
    for name in default.state_dict():
        assert torch.equal(default.state_dict()[name], explicit.state_dict()[name])
    for key, value in default(node_feats, pos, batch=batch).items():
        assert torch.equal(value, explicit(node_feats, pos, batch=batch)[key])


def test_transport_mode_preserves_legacy_positional_config_arguments() -> None:
    config = EquivariantAttentionConfig(
        4,
        "8x0e + 2x1o",
        "1x0e",
        1,
        2,
        0.05,
        1.0,
        0.05,
        1.0,
        1.0,
        "fixed",
        True,
        True,
        None,
        2.5,
        16,
        False,
        1,
        False,
        1.0,
        2.5,
        2.5,
        False,
        0.1,
        1e-8,
    )

    assert config.eps == 1e-8
    assert config.global_transport_mode == "learned"


@pytest.mark.parametrize(
    "route, mode, expected_calls",
    [
        ((2, 2, 2), "learned", 0),
        ((0, 0, 0), "none", 0),
        ((2, 0, 2), "learned", 1),
        ((2, 0, 2), "uniform", 1),
    ],
)
def test_global_geometry_is_lazy_and_computed_once(
    monkeypatch: pytest.MonkeyPatch,
    route: tuple[int, ...],
    mode: str,
    expected_calls: int,
) -> None:
    original = moment._scale_first_geometry
    calls = 0

    def counted(*args: object, **kwargs: object) -> tuple[torch.Tensor, ...]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(moment, "_scale_first_geometry", counted)
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            hidden_irreps="8x0e + 2x1o",
            num_heads=2,
            local_head_counts=route,
            global_transport_mode=mode,
            use_key_balancing=False,
        )
    )
    model(torch.randn(6, 4), torch.randn(6, 3))

    assert calls == expected_calls


def test_all_global_none_bypasses_attention_updater_and_keeps_ffn() -> None:
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            hidden_irreps="8x0e + 2x1o",
            num_layers=1,
            num_heads=2,
            global_transport_mode="none",
        )
    ).double()
    layer = model.layers[0]
    with torch.no_grad():
        layer.scalar_update[-1].bias.fill_(7.0)
        layer.scalar_residual_scale.fill_(1.0)
        layer.ffn_scalar_residual_scale.zero_()
        layer.ffn_vector_residual_scale.zero_()
    scalars = torch.randn(5, 8, dtype=torch.float64)
    vectors = torch.randn(5, 2, 3, dtype=torch.float64)
    pos = torch.randn(5, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 1, 1, 1])

    out_scalars, out_vectors, tensor = layer(
        scalars,
        vectors,
        None,
        pos,
        batch,
        2,
        torch.tensor([2, 3]),
        None,
    )

    assert torch.equal(out_scalars, scalars)
    assert torch.equal(out_vectors, vectors)
    assert torch.equal(tensor, tensor.new_zeros(tensor.shape))


def test_lgl_none_does_not_communicate_between_distant_fragments() -> None:
    torch.manual_seed(919)
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            hidden_irreps="8x0e + 2x1o",
            output_irreps="2x0e + 1x1o + 1x2e",
            num_heads=2,
            local_head_counts=(2, 0, 2),
            global_transport_mode="none",
            use_key_balancing=False,
        )
    ).double()
    node_feats = torch.randn(6, 4, dtype=torch.float64)
    fragment = torch.tensor(
        [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 0.7, 0.0]],
        dtype=torch.float64,
    )
    pos = torch.cat([fragment, fragment + torch.tensor([10.0, 0.0, 0.0])])
    changed = node_feats.clone()
    changed[3:] += 3.0

    reference = model(node_feats, pos)["node_scalars"][:3]
    influenced = model(changed, pos)["node_scalars"][:3]

    assert torch.equal(influenced, reference)


@pytest.mark.parametrize("mode", ["learned", "uniform", "none"])
def test_transport_modes_keep_state_schema(mode: str) -> None:
    model = EquivariantAttention(
        EquivariantAttentionConfig(node_dim=4, global_transport_mode=mode)
    )
    reference = EquivariantAttention(EquivariantAttentionConfig(node_dim=4))
    assert {
        name: tensor.shape for name, tensor in model.state_dict().items()
    } == {
        name: tensor.shape for name, tensor in reference.state_dict().items()
    }


def test_transport_mode_validation_and_hemm_warning() -> None:
    with pytest.raises(ValueError, match="global_transport_mode"):
        EquivariantAttention(
            EquivariantAttentionConfig(node_dim=4, global_transport_mode="invalid")
        )
    with pytest.raises(ValueError, match="learned global transport"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=4,
                local_head_counts=(4, 0, 4),
                global_transport_mode="uniform",
                global_memory_count=4,
                use_memory_interaction=True,
            )
        )
    with pytest.warns(RuntimeWarning, match="Stage-0"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=4,
                local_head_counts=(4, 0, 4),
                global_memory_count=4,
                use_memory_interaction=True,
            )
        )


@pytest.mark.parametrize("mode", ["uniform", "none"])
def test_nonlearned_modes_preserve_o3_translation_and_permutation(mode: str) -> None:
    torch.manual_seed(929)
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            hidden_irreps="8x0e + 2x1o",
            output_irreps="2x0e + 1x1o + 1x2e",
            num_heads=2,
            local_head_counts=(2, 0, 2),
            global_transport_mode=mode,
            use_key_balancing=False,
        )
    ).double()
    node_feats = torch.randn(7, 4, dtype=torch.float64)
    pos = torch.randn(7, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    reference = model(node_feats, pos, batch=batch)
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    orthogonal[:, 0] *= -1
    translation = torch.randn(1, 3, dtype=torch.float64)
    moved = model(node_feats, pos @ orthogonal.T + translation, batch=batch)
    permutation = torch.tensor([2, 0, 1, 6, 4, 3, 5])
    inverse = torch.argsort(permutation)
    permuted = model(
        node_feats[permutation], pos[permutation], batch=batch[permutation]
    )

    assert torch.allclose(moved["node_scalars"], reference["node_scalars"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        moved["node_vectors"],
        torch.einsum("nca,ba->ncb", reference["node_vectors"], orthogonal),
        atol=1e-6,
        rtol=1e-6,
    )
    expected_tensor = torch.einsum(
        "ab,nkbc,dc->nkad", orthogonal, reference["node_tensors"], orthogonal
    )
    assert torch.allclose(moved["node_tensors"], expected_tensor, atol=1e-6, rtol=1e-6)
    for key in ("node_scalars", "node_vectors", "node_tensors"):
        assert torch.allclose(permuted[key][inverse], reference[key], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", ["uniform", "none"])
def test_nonlearned_scalar_coordinate_gradients_are_o3_covariant(mode: str) -> None:
    torch.manual_seed(937)
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            hidden_irreps="8x0e + 2x1o",
            num_heads=2,
            local_head_counts=(2, 0, 2),
            global_transport_mode=mode,
            use_key_balancing=False,
        )
    ).double()
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
