from __future__ import annotations

import torch
from conftest import orthogonal

from equivariant_linear_attention import (
    BiomolecularPairContext,
    ELAGraph,
    TriELA,
)
from equivariant_linear_attention.heads import DistogramHead
from equivariant_linear_attention.nn.ops import matrix_to_st, st_to_matrix
from equivariant_linear_attention.nn.pair_adapters import (
    PairContextInjection,
    PairToNodeSummary,
)
from equivariant_linear_attention.nn.pair_embedding import (
    NodeGeometryToPair,
    PairEmbedding,
)
from equivariant_linear_attention.nn.pair_state import build_dense_pair_layout
from equivariant_linear_attention.nn.state import ParityState


def _state(nodes: int, *, seed: int = 551) -> ParityState:
    generator = torch.Generator().manual_seed(seed)
    return ParityState(
        torch.randn(nodes, 16, generator=generator, dtype=torch.float64),
        torch.randn(nodes, 1, generator=generator, dtype=torch.float64),
        torch.randn(nodes, 1, 3, generator=generator, dtype=torch.float64),
        torch.randn(nodes, 1, 3, generator=generator, dtype=torch.float64),
        torch.randn(nodes, 1, 5, generator=generator, dtype=torch.float64),
        torch.randn(nodes, 1, 5, generator=generator, dtype=torch.float64),
    )


def _transform_state(state: ParityState, transform: torch.Tensor) -> ParityState:
    determinant = torch.linalg.det(transform)

    def tensor(value: torch.Tensor) -> torch.Tensor:
        matrix = st_to_matrix(value)
        moved = torch.einsum("ia,...ab,jb->...ij", transform, matrix, transform)
        return matrix_to_st(moved)

    return ParityState(
        state.even_scalar,
        determinant * state.odd_scalar,
        state.polar_vector @ transform.T,
        determinant * (state.axial_vector @ transform.T),
        tensor(state.even_tensor),
        determinant * tensor(state.odd_tensor),
    )


def _pair_modules() -> tuple[PairEmbedding, NodeGeometryToPair]:
    arguments = {
        "scalar_width": 16,
        "num_heads": 1,
        "pair_width": 8,
        "rbf_bins": 4,
        "max_distance": 8.0,
        "pair_feature_dim": 2,
        "eps": 1e-8,
    }
    return PairEmbedding(**arguments).double(), NodeGeometryToPair(**arguments).double()


def test_pair_embedding_is_o3_invariant_and_keeps_directed_metadata() -> None:
    nodes = 4
    state = _state(nodes)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.1, 1.1, 0.3], [0.0, 0.2, 1.2]],
        dtype=torch.float64,
    )
    layout = build_dense_pair_layout(
        torch.zeros(nodes, dtype=torch.long),
        max_pair_tokens=nodes,
    )
    external = torch.arange(nodes * nodes * 2, dtype=torch.float64).reshape(
        nodes, nodes, 2
    )
    context = BiomolecularPairContext(
        token_index=torch.arange(nodes),
        chain_id=torch.tensor([0, 0, 1, 1]),
        entity_id=torch.tensor([0, 0, 1, 1]),
        molecule_type=torch.tensor([0, 1, 2, 3]),
        residue_type=torch.tensor([4, 5, 6, 7]),
        pair_features=external,
    )
    embedding, refresh = _pair_modules()
    reference = embedding(state, positions, layout, context)
    transform = orthogonal(reflection=True, seed=553)
    moved = embedding(
        _transform_state(state, transform),
        positions @ transform.T + torch.tensor([2.0, -1.0, 0.5]),
        layout,
        context,
    )
    torch.testing.assert_close(moved.z, reference.z, atol=3e-10, rtol=3e-10)
    assert not torch.equal(reference.z[0, 0, 1], reference.z[0, 1, 0])

    initial_delta = refresh(state, positions, reference, context)
    assert torch.count_nonzero(initial_delta) == 0
    upstream = torch.randn_like(initial_delta)
    (initial_delta * upstream).sum().backward()
    assert refresh.projection.weight.grad is not None
    assert float(refresh.projection.weight.grad.abs().sum()) > 0.0

    with torch.no_grad():
        refresh.projection.weight.add_(0.01 * refresh.projection.weight.grad)
    reference_delta = refresh(state, positions, reference, context)
    moved_delta = refresh(
        _transform_state(state, transform),
        positions @ transform.T,
        moved,
        context,
    )
    torch.testing.assert_close(
        moved_delta,
        reference_delta,
        atol=3e-10,
        rtol=3e-10,
    )


def test_pair_to_node_route_is_initially_noop_but_not_gradient_dead() -> None:
    state = _state(4, seed=557)
    layout = build_dense_pair_layout(
        torch.zeros(4, dtype=torch.long),
        max_pair_tokens=4,
    )
    z = torch.randn(1, 4, 4, 8, dtype=torch.float64, requires_grad=True)
    pair = layout.with_z(z)
    summary = PairToNodeSummary(pair_width=8, context_width=8, eps=1e-8).double()
    injection = PairContextInjection(
        context_width=8,
        scalar_width=16,
        num_heads=1,
    ).double()
    context = summary(pair)
    assert float(context.detach().abs().sum()) > 0.0
    output = injection(state, context)
    for actual, expected in zip(output.as_tuple(), state.as_tuple(), strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    loss = sum(
        (index + 1.0) * value.square().mean()
        for index, value in enumerate(output.as_tuple())
    )
    loss.backward()
    assert injection.even_residual.weight.grad is not None
    assert injection.gates.weight.grad is not None
    assert float(injection.even_residual.weight.grad.abs().sum()) > 0.0
    assert float(injection.gates.weight.grad.abs().sum()) > 0.0

    optimizer = torch.optim.SGD(injection.parameters(), lr=0.01)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    opened_z = z.detach().clone().requires_grad_()
    opened_context = summary(layout.with_z(opened_z))
    opened = injection(state, opened_context)
    sum(value.square().mean() for value in opened.as_tuple()).backward()
    assert opened_z.grad is not None and bool(torch.isfinite(opened_z.grad).all())
    assert float(opened_z.grad.abs().sum()) > 0.0


def test_distogram_symmetrizes_only_its_head_and_masks_padding() -> None:
    layout = build_dense_pair_layout(
        torch.tensor([0, 0, 1]),
        max_pair_tokens=2,
    )
    z = torch.arange(32, dtype=torch.float64).reshape(2, 2, 2, 4)
    z = z.requires_grad_()
    pair = layout.with_z(z)
    head = DistogramHead(pair_width=4, num_bins=5, max_distance=10.0).double()
    logits = head(pair)
    assert not torch.equal(pair.z[0, 0, 1], pair.z[0, 1, 0])
    torch.testing.assert_close(logits, logits.transpose(1, 2))
    assert torch.count_nonzero(logits[1, 1]) == 0
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=torch.float64,
    )
    target = head.targets(pair, positions)
    assert target.shape == pair.pair_mask.shape
    assert bool((target[~pair.pair_mask] == -1).all())
    logits.square().sum().backward()
    assert z.grad is not None
    torch.testing.assert_close(z.grad, z.grad.transpose(1, 2))


def test_canonical_model_always_contains_pair_conditioned_local_blocks() -> None:
    model = TriELA(
        "3x0e",
        "1x0e",
        width=16,
        pair_width=8,
        triangle_hidden=8,
        num_stages=1,
        pair_blocks_per_stage=1,
        local_blocks_per_stage=1,
        pair_transition_factor=2,
        pair_dropout=0.0,
        local_points=3,
        max_pair_tokens=8,
        distance_rbf_bins=4,
        distogram_bins=6,
    )
    assert len(model.stages) == 1
    assert len(model.stages[0].local_blocks) == 1
    assert model.stages[0].local_blocks[0].pair_width == 8
    assert model.describe()["local_transport"] == "pair_conditioned_bounded_geometry"


def test_local_geometry_receives_finite_training_gradients() -> None:
    generator = torch.Generator().manual_seed(559)
    model = TriELA(
        "3x0e",
        "1x0e",
        width=16,
        pair_width=8,
        triangle_hidden=8,
        num_stages=1,
        pair_blocks_per_stage=1,
        local_blocks_per_stage=1,
        pair_transition_factor=2,
        pair_dropout=0.0,
        local_points=3,
        max_pair_tokens=8,
        distance_rbf_bins=4,
        distogram_bins=6,
    )
    graph = ELAGraph(
        torch.randn(5, 3, generator=generator, requires_grad=True),
        torch.randn(5, 3, generator=generator, requires_grad=True),
    )
    output = model(graph)
    output.x.square().mean().backward()
    geometry = model.stages[0].local_blocks[0].geometry
    gradients = [
        parameter.grad
        for parameter in geometry.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert any(float(gradient.abs().sum()) > 0.0 for gradient in gradients)
