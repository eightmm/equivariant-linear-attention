from __future__ import annotations

import pytest
import torch

import equivariant_attention.moment as moment
from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention.training import build_regression_model


def _symmetric_traceless(value: torch.Tensor) -> torch.Tensor:
    symmetric = 0.5 * (value + value.transpose(-1, -2))
    trace = symmetric.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    identity = torch.eye(3, dtype=value.dtype, device=value.device)
    return symmetric - trace[..., None, None] * identity / 3.0


def _model(
    *,
    transport: bool,
    static_carrier: bool = False,
) -> EquivariantAttention:
    return EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            input_tensor_dim=1,
            hidden_irreps="8x0e + 2x1o + 2x2e",
            output_irreps="2x0e + 1x1o + 1x2e",
            num_layers=1,
            num_heads=2,
            local_head_counts=(0,),
            use_key_balancing=False,
            use_global_tensor_value_transport=transport,
            use_static_tensor_carrier=static_carrier,
        )
    ).double()


def _inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(280701)
    scalars = torch.randn(7, 4, generator=generator, dtype=torch.float64)
    pos = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    raw_tensors = torch.randn(
        7,
        1,
        3,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    tensors = _symmetric_traceless(raw_tensors)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    return scalars, pos, tensors, batch


@pytest.mark.parametrize("use_radial_trace", [False, True])
def test_global_tensor_value_matches_the_explicit_dense_kernel(
    use_radial_trace: bool,
) -> None:
    generator = torch.Generator().manual_seed(280702)
    nodes, heads, head_dim = 7, 2, 3
    query_scalar = torch.rand(
        nodes, heads, head_dim, generator=generator, dtype=torch.float64
    )
    key_scalar = torch.rand(
        nodes, heads, head_dim, generator=generator, dtype=torch.float64
    )
    query_vector = 0.2 * torch.randn(
        nodes, heads, 3, generator=generator, dtype=torch.float64
    )
    key_vector = 0.2 * torch.randn(
        nodes, heads, 3, generator=generator, dtype=torch.float64
    )
    scalar_value = torch.randn(
        nodes, heads, head_dim, generator=generator, dtype=torch.float64
    )
    vector_value = torch.randn(
        nodes, heads, 3, generator=generator, dtype=torch.float64
    )
    tensor_value = torch.randn(
        nodes, heads, 5, generator=generator, dtype=torch.float64
    ).requires_grad_()
    zeros = torch.zeros(nodes, heads, dtype=torch.float64)
    pos = torch.randn(nodes, 3, generator=generator, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    linear_scale = torch.tensor([0.07, 0.11], dtype=torch.float64)
    quadratic_scale = torch.tensor([0.13, 0.17], dtype=torch.float64)

    actual = moment._global_moment_messages(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        quadratic_scale,
        scalar_value,
        vector_value,
        zeros,
        zeros,
        zeros,
        pos,
        batch,
        num_graphs=2,
        graph_counts=torch.tensor([3, 4]),
        balanced=False,
        alignment_scale=linear_scale,
        alignment_dot_scale=linear_scale,
        kernel_floor=1.0,
        kernel_floor_mode="fixed",
        memory_count=1,
        memory_temperature=1.0,
        memory_assignment_scale=2.5,
        memory_interaction_cutoff=2.5,
        use_memory_interaction=False,
        use_radial_trace=use_radial_trace,
        persistent_tensor_value=tensor_value,
    )[3]

    expected = torch.empty_like(tensor_value)
    for graph in range(2):
        index = batch == graph
        content = torch.einsum(
            "ihf,jhf->hij",
            query_scalar[index],
            key_scalar[index],
        )
        angular = torch.einsum(
            "iha,jha->hij",
            query_vector[index],
            key_vector[index],
        )
        kernel = (
            1.0
            + content
            + linear_scale[:, None, None] * (1.0 + angular)
            + quadratic_scale[:, None, None] * angular.square()
        )
        weights = kernel / kernel.sum(dim=-1, keepdim=True)
        expected[index] = torch.einsum(
            "hij,jhf->ihf",
            weights,
            tensor_value[index],
        )

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-9)
    actual.square().sum().backward()
    assert tensor_value.grad is not None
    assert torch.isfinite(tensor_value.grad).all()
    assert torch.count_nonzero(tensor_value.grad).item() > 0


@pytest.mark.parametrize("static_carrier", [False, True])
def test_global_tensor_transport_is_opt_in_and_remote_sender_active(
    static_carrier: bool,
) -> None:
    torch.manual_seed(280703)
    disabled = _model(transport=False, static_carrier=static_carrier)
    disabled_rng_state = torch.random.get_rng_state()
    torch.manual_seed(280703)
    enabled = _model(transport=True, static_carrier=static_carrier)
    enabled_rng_state = torch.random.get_rng_state()
    scalars, pos, tensors, batch = _inputs()
    changed = tensors.clone()
    changed[3, 0] = _symmetric_traceless(
        torch.tensor(
            [[2.0, -0.4, 0.3], [-0.4, -1.0, 0.2], [0.3, 0.2, -1.0]],
            dtype=torch.float64,
        )
    )

    assert list(disabled.state_dict()) == list(enabled.state_dict())
    assert torch.equal(disabled_rng_state, enabled_rng_state)
    for name, value in disabled.state_dict().items():
        assert torch.equal(value, enabled.state_dict()[name])
    load_result = enabled.load_state_dict(disabled.state_dict(), strict=True)
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys

    disabled_before = disabled(
        scalars,
        pos,
        batch=batch,
        node_tensors=tensors,
    )["node_tensors"]
    disabled_after = disabled(
        scalars,
        pos,
        batch=batch,
        node_tensors=changed,
    )["node_tensors"]
    enabled_before = enabled(
        scalars,
        pos,
        batch=batch,
        node_tensors=tensors,
    )["node_tensors"]
    enabled_after = enabled(
        scalars,
        pos,
        batch=batch,
        node_tensors=changed,
    )["node_tensors"]

    # Node 0 and changed sender 3 share a graph. With the route disabled, a
    # sender tensor is not in any global value payload.
    assert torch.equal(disabled_before[0], disabled_after[0])
    assert not torch.allclose(
        enabled_before[0],
        enabled_after[0],
        atol=1e-10,
        rtol=1e-10,
    )
    # The other graph must remain isolated in both arms.
    assert torch.equal(disabled_before[4:], disabled_after[4:])
    assert torch.equal(enabled_before[4:], enabled_after[4:])


@pytest.mark.parametrize("static_carrier", [False, True])
def test_global_tensor_transport_preserves_o3_translation_and_permutation(
    static_carrier: bool,
) -> None:
    torch.manual_seed(280704)
    model = _model(transport=True, static_carrier=static_carrier)
    scalars, pos, tensors, batch = _inputs()
    orthogonal = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    assert torch.linalg.det(orthogonal) < 0
    translation = torch.tensor([3.0, -2.0, 1.5], dtype=torch.float64)
    permutation = torch.tensor([2, 0, 3, 1, 6, 4, 5])

    reference = model(scalars, pos, batch=batch, node_tensors=tensors)
    moved = model(
        scalars,
        pos @ orthogonal.T + translation,
        batch=batch,
        node_tensors=torch.einsum(
            "ab,nkbc,dc->nkad",
            orthogonal,
            tensors,
            orthogonal,
        ),
    )
    permuted = model(
        scalars[permutation],
        pos[permutation],
        batch=batch[permutation],
        node_tensors=tensors[permutation],
    )
    inverse = torch.argsort(permutation)

    for name in ("node_scalars", "graph_scalars"):
        assert torch.allclose(moved[name], reference[name], atol=1e-9, rtol=1e-9)
    for name in ("node_vectors", "graph_vectors"):
        expected = torch.einsum("ab,nkb->nka", orthogonal, reference[name])
        assert torch.allclose(moved[name], expected, atol=1e-9, rtol=1e-9)
    for name in ("node_tensors", "graph_tensors"):
        expected = torch.einsum(
            "ab,nkbc,dc->nkad",
            orthogonal,
            reference[name],
            orthogonal,
        )
        assert torch.allclose(moved[name], expected, atol=1e-9, rtol=1e-9)
    for name in ("node_scalars", "node_vectors", "node_tensors"):
        assert torch.allclose(
            permuted[name][inverse],
            reference[name],
            atol=1e-9,
            rtol=1e-9,
        )
    for name in ("graph_scalars", "graph_vectors", "graph_tensors"):
        assert torch.allclose(
            permuted[name],
            reference[name],
            atol=1e-9,
            rtol=1e-9,
        )


def test_uniform_global_tensor_transport_is_the_exact_graph_mean() -> None:
    generator = torch.Generator().manual_seed(280705)
    nodes, heads, head_dim = 5, 2, 3
    query_scalar = torch.randn(
        nodes, heads, head_dim, generator=generator, dtype=torch.float64
    )
    key_scalar = torch.randn(
        nodes, heads, head_dim, generator=generator, dtype=torch.float64
    )
    query_vector = torch.randn(
        nodes, heads, 3, generator=generator, dtype=torch.float64
    )
    key_vector = torch.randn(
        nodes, heads, 3, generator=generator, dtype=torch.float64
    )
    tensor_value = torch.randn(
        nodes, heads, 5, generator=generator, dtype=torch.float64
    )
    scalar_value = torch.zeros(nodes, heads, head_dim, dtype=torch.float64)
    vector_value = torch.zeros(nodes, heads, 3, dtype=torch.float64)
    zeros = torch.zeros(nodes, heads, dtype=torch.float64)
    pos = torch.randn(nodes, 3, generator=generator, dtype=torch.float64)
    batch = torch.tensor([0, 0, 1, 1, 1])
    scales = torch.ones(heads, dtype=torch.float64)

    tensor = moment._global_moment_messages(
        query_scalar,
        key_scalar,
        query_vector,
        key_vector,
        scales,
        scalar_value,
        vector_value,
        zeros,
        zeros,
        zeros,
        pos,
        batch,
        num_graphs=2,
        graph_counts=torch.tensor([2, 3]),
        balanced=True,
        alignment_scale=scales,
        alignment_dot_scale=scales,
        kernel_floor=1.0,
        kernel_floor_mode="fixed",
        memory_count=1,
        memory_temperature=1.0,
        memory_assignment_scale=2.5,
        memory_interaction_cutoff=2.5,
        use_memory_interaction=False,
        use_radial_trace=False,
        global_transport_mode="uniform",
        persistent_tensor_value=tensor_value,
    )[3]
    expected = torch.stack(
        [
            tensor_value[:2].mean(dim=0),
            tensor_value[:2].mean(dim=0),
            tensor_value[2:].mean(dim=0),
            tensor_value[2:].mean(dim=0),
            tensor_value[2:].mean(dim=0),
        ]
    )

    assert torch.allclose(tensor, expected, atol=1e-12, rtol=1e-12)


def test_global_tensor_transport_config_and_training_builder_contract() -> None:
    assert not EquivariantAttentionConfig(
        node_dim=4
    ).use_global_tensor_value_transport
    with pytest.raises(TypeError, match="use_global_tensor_value_transport"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=4,
                use_global_tensor_value_transport=1,  # type: ignore[arg-type]
            )
        )
    with pytest.raises(ValueError, match="persistent 2e"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=4,
                use_global_tensor_value_transport=True,
            )
        )
    with pytest.raises(ValueError, match="global head"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=4,
                hidden_irreps="8x0e + 2x1o + 2x2e",
                num_layers=1,
                num_heads=2,
                local_head_counts=(2,),
                use_global_tensor_value_transport=True,
            )
        )

    model = build_regression_model(
        node_dim=4,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        hidden_tensor_dim=4,
        use_global_tensor_value_transport=True,
    )
    assert isinstance(model, EquivariantAttention)
    assert model.config.use_global_tensor_value_transport


def test_explicitly_disabled_transport_is_bitwise_the_default() -> None:
    common = {
        "node_dim": 4,
        "input_tensor_dim": 1,
        "hidden_irreps": "8x0e + 2x1o + 2x2e",
        "output_irreps": "2x0e + 1x1o + 1x2e",
        "num_layers": 1,
        "num_heads": 2,
        "local_head_counts": (0,),
        "use_key_balancing": False,
    }
    torch.manual_seed(280706)
    default = EquivariantAttention(
        EquivariantAttentionConfig(**common)
    ).double()
    default_rng_state = torch.random.get_rng_state()
    torch.manual_seed(280706)
    explicit = EquivariantAttention(
        EquivariantAttentionConfig(
            **common,
            use_global_tensor_value_transport=False,
        )
    ).double()
    explicit_rng_state = torch.random.get_rng_state()
    scalars, pos, tensors, batch = _inputs()

    assert list(default.state_dict()) == list(explicit.state_dict())
    assert torch.equal(default_rng_state, explicit_rng_state)
    for name, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name])
    default_output = default(
        scalars,
        pos,
        batch=batch,
        node_tensors=tensors,
    )
    explicit_output = explicit(
        scalars,
        pos,
        batch=batch,
        node_tensors=tensors,
    )
    for name in default_output:
        assert torch.equal(default_output[name], explicit_output[name])
