from __future__ import annotations

import torch

from equivariant_linear_attention.nn.geometry import GeometryContext
from equivariant_linear_attention.nn.relation import (
    RelationMessage,
    SelfAdjointRelation,
    message_inner,
    orthogonalize,
)
from equivariant_linear_attention.nn.state import ParityState


def _state(
    nodes: int, width: int, heads: int, generator: torch.Generator
) -> ParityState:
    return ParityState(
        torch.randn(nodes, width, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, 3, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, 3, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, 5, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, 5, generator=generator, dtype=torch.float64),
    )


def _message(
    nodes: int, heads: int, dim: int, generator: torch.Generator
) -> RelationMessage:
    return RelationMessage(
        torch.randn(nodes, heads, dim, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, 3, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, 3, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, 5, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, 5, generator=generator, dtype=torch.float64),
    )


def test_relation_operator_is_symmetric_psd_and_matches_dense_action() -> None:
    generator = torch.Generator().manual_seed(201)
    nodes, width, heads = 7, 16, 2
    index = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    geometry = GeometryContext.build(
        torch.randn(nodes, 3, generator=generator, dtype=torch.float64),
        index,
        num_segments=2,
        eps=1e-10,
    )
    state = _state(nodes, width, heads, generator)
    relation = SelfAdjointRelation(
        scalar_width=width,
        num_heads=heads,
        feature_width=5,
        num_charts=3,
        eps=1e-10,
    ).double()
    factors = relation.build(state, geometry)
    value = torch.randn(nodes, heads, 4, generator=generator, dtype=torch.float64)
    actual = relation.apply_tensor(factors, value)

    expected = torch.zeros_like(actual)
    for graph in range(2):
        selected = index == graph
        for head in range(heads):
            content = factors.content_feature[selected, head]
            mercer = factors.mercer_feature[selected, head]
            content_matrix = content @ content.T / factors.content_trace[graph, head]
            mercer_matrix = mercer @ mercer.T / factors.mercer_trace[graph, head]
            assignment = factors.atlas.assignment[selected]
            atlas_matrix = (
                assignment
                @ torch.diag(
                    factors.atlas.chart_weight[graph, head] / factors.atlas.mass[graph]
                )
                @ assignment.T
            )
            weight = factors.mixture[graph, head]
            dense = (
                weight[0] * content_matrix
                + weight[1] * mercer_matrix
                + weight[2] * atlas_matrix
            )
            torch.testing.assert_close(dense, dense.T, atol=2e-12, rtol=0.0)
            eigenvalue = torch.linalg.eigvalsh(dense)
            assert float(eigenvalue.detach().min()) >= -2e-10
            expected[selected, head] = dense @ value[selected, head]
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)


def test_graphwise_irrep_orthogonalization() -> None:
    generator = torch.Generator().manual_seed(203)
    index = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    counts = torch.tensor([3, 4])
    first = _message(7, 2, 4, generator)
    candidate = _message(7, 2, 4, generator)
    output = orthogonalize(
        candidate,
        (first,),
        index=index,
        num_segments=2,
        counts=counts,
        eps=1e-10,
    )
    inner = message_inner(first, output)
    graph_inner = inner.new_zeros(2, 2).index_add(0, index, inner)
    torch.testing.assert_close(
        graph_inner, torch.zeros_like(graph_inner), atol=2e-9, rtol=0.0
    )


def test_atlas_assignment_is_rotation_invariant_and_metric_equivariant() -> None:
    generator = torch.Generator().manual_seed(205)
    nodes, width, heads = 6, 16, 2
    index = torch.tensor([0, 0, 0, 1, 1, 1])
    position = torch.randn(nodes, 3, generator=generator, dtype=torch.float64)
    state = _state(nodes, width, heads, generator)
    relation = SelfAdjointRelation(
        scalar_width=width,
        num_heads=heads,
        feature_width=4,
        num_charts=3,
        eps=1e-10,
    ).double()
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    if torch.linalg.det(q) > 0:
        q[:, 0].neg_()
    reference = relation.build(
        state,
        GeometryContext.build(position, index, num_segments=2, eps=1e-10),
    )
    moved = relation.build(
        state,
        GeometryContext.build(position @ q.T, index, num_segments=2, eps=1e-10),
    )
    torch.testing.assert_close(
        moved.atlas.assignment,
        reference.atlas.assignment,
        atol=3e-10,
        rtol=3e-10,
    )
    expected_metric = torch.einsum(
        "ia,...ab,jb->...ij",
        q,
        reference.atlas.node_metric,
        q,
    )
    torch.testing.assert_close(
        moved.atlas.node_metric,
        expected_metric,
        atol=4e-10,
        rtol=4e-10,
    )
