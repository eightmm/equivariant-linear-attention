from __future__ import annotations

import torch

from equivariant_linear_attention.nn.geometry import GeometryContext, chart_density
from equivariant_linear_attention.nn.relation import (
    LocalChartMercer,
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


def test_local_chart_mercer_matches_truncated_gaussian_pou_kernel() -> None:
    generator = torch.Generator().manual_seed(211)
    nodes, width, heads, charts = 9, 16, 2, 3
    index = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])
    geometry = GeometryContext.build(
        4.0 * torch.randn(nodes, 3, generator=generator, dtype=torch.float64),
        index,
        num_segments=2,
        eps=1e-10,
        length_scale=10.0,
    )
    state = _state(nodes, width, heads, generator)
    local = LocalChartMercer(
        scalar_width=width,
        num_heads=heads,
        num_charts=charts,
        eps=1e-10,
    ).double()
    assignment, delta, gamma = local.charts(state, geometry)
    feature = local.features(assignment, delta, gamma)
    torch.testing.assert_close(feature, local(state, geometry))

    for head in range(heads):
        g = gamma[head]
        for i in range(nodes):
            for j in range(nodes):
                if index[i] != index[j]:
                    continue
                expected = torch.zeros((), dtype=torch.float64)
                for chart in range(charts):
                    di = delta[i, head, chart]
                    dj = delta[j, head, chart]
                    z = 2.0 * g * di.dot(dj)
                    expected = expected + (
                        torch.sqrt(assignment[i, head, chart] + local.eps)
                        * torch.sqrt(assignment[j, head, chart] + local.eps)
                        * torch.exp(-g * (di.dot(di) + dj.dot(dj)))
                        * (1.0 + z + 0.5 * z.square())
                    )
                actual = feature[i, head].dot(feature[j, head])
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_relation_with_local_sector_is_symmetric_psd() -> None:
    generator = torch.Generator().manual_seed(213)
    nodes, width, heads = 8, 16, 2
    index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    geometry = GeometryContext.build(
        5.0 * torch.randn(nodes, 3, generator=generator, dtype=torch.float64),
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
        num_local_charts=4,
        eps=1e-10,
    ).double()
    factors = relation.build(state, geometry)
    assert factors.local_feature is not None
    assert factors.mixture.shape[-1] == 4
    for graph in range(2):
        selected = index == graph
        for head in range(heads):
            feature = factors.feature[selected, head]
            operator = feature @ feature.T
            torch.testing.assert_close(operator, operator.T)
            eigenvalues = torch.linalg.eigvalsh(operator)
            assert eigenvalues.min() >= -1e-10


def test_chart_seeds_are_equivariant_and_spread() -> None:
    generator = torch.Generator().manual_seed(217)
    nodes, seeds = 400, 16
    index = torch.zeros(nodes, dtype=torch.long)
    direction = torch.randn(nodes, 3, generator=generator, dtype=torch.float64)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    radius = 20.0 * torch.rand(nodes, generator=generator, dtype=torch.float64) ** (
        1.0 / 3.0
    )
    position = radius[:, None] * direction

    reference = GeometryContext.build(
        position, index, num_segments=1, eps=1e-10, num_seeds=seeds
    )
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    if torch.linalg.det(q) > 0:
        q[:, 0].neg_()
    moved = GeometryContext.build(
        position @ q.T + torch.tensor([4.0, -7.0, 2.0], dtype=torch.float64),
        index,
        num_segments=1,
        eps=1e-10,
        num_seeds=seeds,
    )
    assert reference.chart_seeds is not None and moved.chart_seeds is not None
    torch.testing.assert_close(
        moved.chart_seeds, reference.chart_seeds @ q.T, atol=1e-9, rtol=1e-9
    )

    permutation = torch.randperm(nodes, generator=generator)
    permuted = GeometryContext.build(
        position[permutation], index, num_segments=1, eps=1e-10, num_seeds=seeds
    )
    torch.testing.assert_close(
        permuted.chart_seeds, reference.chart_seeds, atol=1e-9, rtol=1e-9
    )

    # The seeds exist to spread charts through the structure: learned logits
    # alone left chart centres clustered near the centroid.
    centre = reference.chart_seeds.mean(dim=1, keepdim=True)
    dispersion = (
        (reference.chart_seeds - centre).square().sum(dim=-1).mean().sqrt()
    )
    structure = reference.absolute.square().sum(dim=-1).mean().sqrt()
    assert dispersion > 0.5 * structure


def test_chart_seeds_stay_inside_their_own_segment() -> None:
    generator = torch.Generator().manual_seed(219)
    index = torch.tensor([0] * 200 + [1] * 200)
    first = torch.randn(200, 3, generator=generator, dtype=torch.float64)
    second = 3.0 * torch.randn(200, 3, generator=generator, dtype=torch.float64)
    geometry = GeometryContext.build(
        torch.cat((first, second)), index, num_segments=2, eps=1e-10, num_seeds=8
    )
    assert geometry.chart_seeds is not None
    isolated = GeometryContext.build(first, torch.zeros(200, dtype=torch.long),
                                     num_segments=1, eps=1e-10, num_seeds=8)
    assert isolated.chart_seeds is not None
    torch.testing.assert_close(
        geometry.chart_seeds[0], isolated.chart_seeds[0], atol=1e-9, rtol=1e-9
    )


def test_chart_density_is_invariant_and_node_linear_in_form() -> None:
    generator = torch.Generator().manual_seed(223)
    nodes = 300
    index = torch.zeros(nodes, dtype=torch.long)
    position = 12.0 * torch.randn(nodes, 3, generator=generator, dtype=torch.float64)
    bandwidths = (3.5, 5.0)
    reference = chart_density(
        position, index, 1, num_charts=8, bandwidths=bandwidths, length_scale=10.0
    )
    assert reference.shape == (nodes, len(bandwidths))
    assert bool(reference.isfinite().all()) and bool((reference >= 0.0).all())

    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    if torch.linalg.det(q) > 0:
        q[:, 0].neg_()
    moved = chart_density(
        position @ q.T + torch.tensor([5.0, 1.0, -3.0], dtype=torch.float64),
        index,
        1,
        num_charts=8,
        bandwidths=bandwidths,
        length_scale=10.0,
    )
    torch.testing.assert_close(moved, reference, atol=1e-9, rtol=1e-9)

    permutation = torch.randperm(nodes, generator=generator)
    permuted = chart_density(
        position[permutation], index, 1, num_charts=8,
        bandwidths=bandwidths, length_scale=10.0,
    )
    torch.testing.assert_close(permuted, reference[permutation], atol=1e-9, rtol=1e-9)

    # Denser structures must report higher counts at fixed bandwidth.
    sparse = chart_density(
        3.0 * position, index, 1, num_charts=8,
        bandwidths=bandwidths, length_scale=10.0,
    )
    assert float(sparse.mean()) < float(reference.mean())
