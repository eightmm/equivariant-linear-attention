from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig


def _config(
    *,
    enabled: bool,
    num_layers: int = 3,
    residual_rank: int = 4,
    residual_layers: tuple[int, ...] | None = None,
) -> EquivariantAttentionConfig:
    return EquivariantAttentionConfig(
        node_dim=5,
        hidden_irreps="12x0e + 3x1o",
        output_irreps="2x0e + 1x1o + 1x2e",
        num_layers=num_layers,
        num_heads=3,
        local_head_counts=(0,) * num_layers,
        local_cutoff=5.0,
        use_key_balancing=False,
        use_sparse_low_rank_local_residual=enabled,
        local_residual_rank=residual_rank,
        local_residual_layers=residual_layers,
    )


def _complete_same_graph_edges(batch: torch.Tensor) -> torch.Tensor:
    receiver: list[int] = []
    sender: list[int] = []
    for graph_index in range(int(batch.max().item()) + 1):
        nodes = torch.nonzero(batch == graph_index, as_tuple=False).flatten().tolist()
        for target in nodes:
            for source in nodes:
                receiver.append(target)
                sender.append(source)
    return torch.tensor([receiver, sender], dtype=torch.long)


def _ring_edges(nodes: int, width: int) -> torch.Tensor:
    receiver: list[int] = []
    sender: list[int] = []
    for target in range(nodes):
        for offset in range(width):
            receiver.append(target)
            sender.append((target + offset) % nodes)
    return torch.tensor([receiver, sender], dtype=torch.long)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(290101)
    node_feats = torch.randn(8, 5, generator=generator, dtype=torch.float64)
    pos = torch.tensor(
        [
            [-0.8, -0.2, 0.1],
            [0.7, -0.3, -0.2],
            [0.2, 0.9, 0.4],
            [-0.1, -0.6, 0.8],
            [3.2, 0.1, -0.4],
            [4.1, -0.5, 0.2],
            [3.7, 0.8, 0.6],
            [2.9, -0.7, 0.5],
        ],
        dtype=torch.float64,
    )
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    edge_index = _complete_same_graph_edges(batch)
    return node_feats, pos, batch, edge_index


def _candidate_parameters(
    model: EquivariantAttention,
) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "sparse_low_rank_local_residual" in name
    ]


def _activate_local_residual(
    model: EquivariantAttention,
    node_feats: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    edge_index: torch.Tensor,
) -> tuple[int, int]:
    local_parameters = _candidate_parameters(model)
    assert local_parameters
    local_ids = {id(parameter) for _name, parameter in local_parameters}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in local_ids)
    optimizer = torch.optim.SGD(
        [parameter for _name, parameter in local_parameters],
        lr=5e-2,
    )
    nonzero_gradient_counts: list[int] = []
    for _step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            node_feats,
            pos,
            batch=batch,
            edge_index=edge_index,
        )
        loss = (
            output["node_scalars"].square().mean()
            + 0.1 * output["node_vectors"].square().mean()
            + 0.1 * output["node_tensors"].square().mean()
        )
        loss.backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for _name, parameter in local_parameters
        )
        nonzero_gradient_counts.append(
            sum(
                int(torch.count_nonzero(parameter.grad).item() > 0)
                for _name, parameter in local_parameters
                if parameter.grad is not None
            )
        )
        optimizer.step()
    return tuple(nonzero_gradient_counts)  # type: ignore[return-value]


def _assert_equivariant_outputs(
    reference: dict[str, torch.Tensor],
    moved: dict[str, torch.Tensor],
    transform: torch.Tensor,
) -> None:
    for name in ("node_scalars", "graph_scalars"):
        assert torch.allclose(moved[name], reference[name], atol=2e-9, rtol=2e-9)
    for name in ("node_vectors", "graph_vectors"):
        expected = torch.einsum("ab,nkb->nka", transform, reference[name])
        assert torch.allclose(moved[name], expected, atol=2e-9, rtol=2e-9)
    for name in ("node_tensors", "graph_tensors"):
        expected = torch.einsum(
            "ab,nkbc,dc->nkad",
            transform,
            reference[name],
            transform,
        )
        assert torch.allclose(moved[name], expected, atol=2e-9, rtol=2e-9)


def test_sparse_low_rank_residual_defaults_are_disabled_and_bitwise_compatible() -> (
    None
):
    common = {
        "node_dim": 5,
        "hidden_irreps": "12x0e + 3x1o",
        "output_irreps": "2x0e + 1x1o + 1x2e",
        "num_heads": 3,
        "use_key_balancing": False,
    }
    torch.manual_seed(290102)
    default = EquivariantAttention(
        EquivariantAttentionConfig(**common)
    ).double()
    default_rng = torch.random.get_rng_state()
    torch.manual_seed(290102)
    explicit_off = EquivariantAttention(
        EquivariantAttentionConfig(
            **common,
            use_sparse_low_rank_local_residual=False,
        )
    ).double()
    explicit_rng = torch.random.get_rng_state()
    node_feats, pos, batch, _edge_index = _inputs()

    assert default.config.use_sparse_low_rank_local_residual is False
    assert list(default.state_dict()) == list(explicit_off.state_dict())
    assert torch.equal(default_rng, explicit_rng)
    for name, value in default.state_dict().items():
        assert torch.equal(value, explicit_off.state_dict()[name]), name
    default_output = default(node_feats, pos, batch=batch)
    explicit_output = explicit_off(node_feats, pos, batch=batch)
    for name in default_output:
        assert torch.equal(default_output[name], explicit_output[name]), name


@pytest.mark.parametrize(
    ("updates", "exception", "message"),
    [
        (
            {"use_sparse_low_rank_local_residual": 1},
            TypeError,
            "use_sparse_low_rank_local_residual",
        ),
        ({"local_residual_rank": True}, TypeError, "local_residual_rank"),
        ({"local_residual_rank": 0}, ValueError, "local_residual_rank"),
        ({"local_residual_layers": [0]}, TypeError, "local_residual_layers"),
        ({"local_residual_layers": ()}, ValueError, "local_residual_layers"),
        ({"local_residual_layers": (0, 0)}, ValueError, "duplicate"),
        ({"local_residual_layers": (-1,)}, ValueError, "invalid"),
        ({"local_residual_layers": (3,)}, ValueError, "invalid"),
    ],
)
def test_sparse_low_rank_residual_validates_public_controls(
    updates: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        EquivariantAttention(replace(_config(enabled=True), **updates))


def test_sparse_low_rank_residual_rejects_inert_or_legacy_partitioned_routes() -> None:
    with pytest.raises(
        ValueError,
        match="local_residual_layers.*use_sparse_low_rank_local_residual",
    ):
        EquivariantAttention(
            replace(
                _config(enabled=False),
                local_residual_layers=(0,),
            )
        )
    with pytest.raises(ValueError, match="global transport"):
        EquivariantAttention(
            replace(
                _config(enabled=True),
                global_transport_mode="none",
            )
        )
    with pytest.raises(ValueError, match="local_head_counts"):
        EquivariantAttention(
            replace(
                _config(enabled=True),
                local_head_counts=(3, 0, 3),
            )
        )


@pytest.mark.parametrize(
    ("num_layers", "residual_layers", "expected_active"),
    [
        (5, None, {0, 1, 2, 3, 4}),
        (6, (1, 4), {1, 4}),
    ],
)
def test_sparse_low_rank_residual_schedule_keeps_every_base_head_global(
    num_layers: int,
    residual_layers: tuple[int, ...] | None,
    expected_active: set[int],
) -> None:
    model = EquivariantAttention(
        _config(
            enabled=True,
            num_layers=num_layers,
            residual_layers=residual_layers,
        )
    ).double()
    node_feats, pos, batch, edge_index = _inputs()

    assert len(model.layers) == num_layers
    for index, layer in enumerate(model.layers):
        assert layer.local_head_count == 0
        assert layer.global_head_count == model.config.num_heads
        residual = layer.sparse_low_rank_local_residual
        assert (residual is not None) is (index in expected_active)
    output = model(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index,
    )
    assert all(torch.isfinite(value).all() for value in output.values())


def test_enabled_zero_initialized_residual_preserves_function_and_common_rng() -> None:
    torch.manual_seed(290103)
    baseline = EquivariantAttention(_config(enabled=False)).double()
    baseline_rng = torch.random.get_rng_state()
    torch.manual_seed(290103)
    candidate = EquivariantAttention(_config(enabled=True)).double()
    candidate_rng = torch.random.get_rng_state()
    node_feats, pos, batch, edge_index = _inputs()

    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    common_names = baseline_state.keys() & candidate_state.keys()
    extra_names = candidate_state.keys() - baseline_state.keys()
    assert common_names
    assert extra_names
    assert all("sparse_low_rank_local_residual" in name for name in extra_names)
    assert torch.equal(baseline_rng, candidate_rng)
    for name in common_names:
        assert torch.equal(baseline_state[name], candidate_state[name]), name

    baseline_output = baseline(node_feats, pos, batch=batch)
    candidate_output = candidate(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index,
    )
    for name in baseline_output:
        assert torch.equal(baseline_output[name], candidate_output[name]), name


def test_sparse_low_rank_residual_receives_deeper_gradients_after_two_steps() -> None:
    torch.manual_seed(290104)
    model = EquivariantAttention(
        _config(enabled=True, residual_layers=(0, 2))
    ).double()
    node_feats, pos, batch, edge_index = _inputs()

    first_count, second_count = _activate_local_residual(
        model,
        node_feats,
        pos,
        batch,
        edge_index,
    )

    assert first_count > 0
    assert second_count > first_count


def test_active_sparse_low_rank_residual_preserves_o3_translation_permutation_and_edge_order() -> (
    None
):
    torch.manual_seed(290105)
    model = EquivariantAttention(
        _config(enabled=True, residual_layers=(0, 2))
    ).double()
    node_feats, pos, batch, edge_index = _inputs()
    _activate_local_residual(model, node_feats, pos, batch, edge_index)
    model.eval()
    reflection = torch.diag(
        torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    )
    assert torch.linalg.det(reflection) < 0
    translation = torch.tensor([2.0, -3.0, 1.5], dtype=torch.float64)
    permutation = torch.tensor([3, 0, 2, 1, 7, 5, 4, 6])
    inverse = torch.argsort(permutation)

    reference = model(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index,
    )
    moved = model(
        node_feats,
        pos @ reflection.T + translation,
        batch=batch,
        edge_index=edge_index,
    )
    permuted = model(
        node_feats[permutation],
        pos[permutation],
        batch=batch[permutation],
        edge_index=inverse[edge_index],
    )
    edge_reordered = model(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index.flip(dims=(1,)),
    )

    _assert_equivariant_outputs(reference, moved, reflection)
    for name in ("node_scalars", "node_vectors", "node_tensors"):
        assert torch.allclose(
            permuted[name][inverse],
            reference[name],
            atol=2e-9,
            rtol=2e-9,
        )
    for name in ("graph_scalars", "graph_vectors", "graph_tensors"):
        assert torch.allclose(
            permuted[name],
            reference[name],
            atol=2e-9,
            rtol=2e-9,
        )
    for name in reference:
        assert torch.allclose(
            edge_reordered[name],
            reference[name],
            atol=2e-9,
            rtol=2e-9,
        )


def test_active_sparse_low_rank_residual_preserves_graph_batch_isolation() -> None:
    torch.manual_seed(290106)
    model = EquivariantAttention(_config(enabled=True)).double()
    node_feats, pos, batch, edge_index = _inputs()
    _activate_local_residual(model, node_feats, pos, batch, edge_index)
    model.eval()
    changed_feats = node_feats.clone()
    changed_pos = pos.clone()
    changed_feats[4:] = 5.0 * changed_feats[4:] + 3.0
    changed_pos[4:] = changed_pos[4:] + torch.tensor(
        [7.0, -4.0, 2.0],
        dtype=torch.float64,
    )

    reference = model(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index,
    )
    changed = model(
        changed_feats,
        changed_pos,
        batch=batch,
        edge_index=edge_index,
    )

    for name in ("node_scalars", "node_vectors", "node_tensors"):
        assert torch.allclose(
            changed[name][:4],
            reference[name][:4],
            atol=2e-9,
            rtol=2e-9,
        )
    for name in ("graph_scalars", "graph_vectors", "graph_tensors"):
        assert torch.allclose(
            changed[name][0],
            reference[name][0],
            atol=2e-9,
            rtol=2e-9,
        )


def test_sparse_low_rank_residual_keeps_no_pair_matrix_or_persistent_edge_state() -> (
    None
):
    torch.manual_seed(290107)
    nodes = 11
    model = EquivariantAttention(
        _config(enabled=True, num_layers=1, residual_rank=4)
    ).double()
    node_feats = torch.randn(nodes, 5, dtype=torch.float64)
    pos = 0.2 * torch.randn(nodes, 3, dtype=torch.float64)
    batch = torch.zeros(nodes, dtype=torch.long)
    sparse_edges = _ring_edges(nodes, 3)
    alternate_edges = _ring_edges(nodes, 2)
    residual = model.layers[0].sparse_low_rank_local_residual
    assert residual is not None
    state_shapes_before = {
        name: tuple(value.shape)
        for name, value in residual.state_dict().items()
    }
    saved_shapes: list[tuple[int, ...]] = []

    def record_saved_tensor(value: torch.Tensor) -> torch.Tensor:
        saved_shapes.append(tuple(value.shape))
        return value

    hooks = getattr(torch.autograd.graph, "saved_tensors_hooks", None)
    context = (
        hooks(record_saved_tensor, lambda value: value)
        if hooks is not None
        else nullcontext()
    )
    with context:
        output = model(
            node_feats,
            pos,
            batch=batch,
            edge_index=sparse_edges,
        )
        output["node_scalars"].square().sum().backward()
    model.zero_grad(set_to_none=True)
    model(
        node_feats,
        pos,
        batch=batch,
        edge_index=alternate_edges,
    )

    state_shapes_after = {
        name: tuple(value.shape)
        for name, value in residual.state_dict().items()
    }
    assert state_shapes_after == state_shapes_before
    assert not any(
        len(shape) >= 2
        and (
            shape[:2] == (nodes, nodes)
            or shape[-2:] == (nodes, nodes)
        )
        for shape in saved_shapes
    )
    edge_counts = {sparse_edges.shape[1], alternate_edges.shape[1]}
    for _name, value in residual.named_buffers(recurse=True):
        assert value.ndim == 0 or value.shape[0] not in edge_counts
    for value in vars(residual).values():
        if isinstance(value, torch.Tensor) and value.ndim:
            assert value.shape[0] not in edge_counts
