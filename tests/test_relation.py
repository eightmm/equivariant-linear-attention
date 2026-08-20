from __future__ import annotations

import torch

import equivariant_linear_attention.nn.relation as relation_module
from equivariant_linear_attention.nn.geometry import GeometryContext
from equivariant_linear_attention.nn.relation import (
    RelationMessage,
    SelfAdjointRelation,
    message_inner,
    orthogonalize,
)
from equivariant_linear_attention.nn.state import ParityState


def _state(
    nodes: int,
    width: int,
    heads: int,
    generator: torch.Generator,
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
    nodes: int,
    heads: int,
    dim: int,
    generator: torch.Generator,
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
    relation = SelfAdjointRelation(
        scalar_width=width,
        num_heads=heads,
        feature_width=5,
        num_charts=3,
        eps=1e-10,
    ).double()
    factors = relation.build(_state(nodes, width, heads, generator), geometry)
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
            assert float(torch.linalg.eigvalsh(dense).detach().min()) >= -2e-10
            expected[selected, head] = dense @ value[selected, head]
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)


def test_segmented_gram_is_chunked_exact_and_twice_differentiable(
    monkeypatch,
) -> None:
    generator = torch.Generator().manual_seed(202)
    nodes, heads, feature_width, value_width = 9, 2, 69, 4
    index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1])
    feature = torch.randn(
        nodes,
        heads,
        feature_width,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    value = torch.randn(
        nodes,
        heads,
        value_width,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    reduction_shapes: list[torch.Size] = []
    original_segment_sum = relation_module.segment_sum

    def recording_segment_sum(
        tensor: torch.Tensor,
        segment: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        reduction_shapes.append(tensor.shape)
        return original_segment_sum(tensor, segment, count)

    monkeypatch.setattr(relation_module, "segment_sum", recording_segment_sum)
    actual = SelfAdjointRelation._gram_apply(feature, value, index, 2)
    expected = torch.zeros_like(value)
    for graph in range(2):
        selected = index == graph
        for head in range(heads):
            local = feature[selected, head]
            expected[selected, head] = local @ (local.T @ value[selected, head])
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    assert len(reduction_shapes) == 3
    assert sum(shape[-2] for shape in reduction_shapes) == feature_width
    assert max(shape[-2] for shape in reduction_shapes) <= 32

    actual_first = torch.autograd.grad(
        actual.square().sum(),
        (feature, value),
        create_graph=True,
    )
    expected_first = torch.autograd.grad(
        expected.square().sum(),
        (feature, value),
        create_graph=True,
    )
    for observed, oracle in zip(actual_first, expected_first, strict=True):
        torch.testing.assert_close(observed, oracle, atol=2e-10, rtol=2e-10)
    actual_second = torch.autograd.grad(
        sum(gradient.square().sum() for gradient in actual_first),
        (feature, value),
    )
    expected_second = torch.autograd.grad(
        sum(gradient.square().sum() for gradient in expected_first),
        (feature, value),
    )
    for observed, oracle in zip(actual_second, expected_second, strict=True):
        torch.testing.assert_close(observed, oracle, atol=3e-9, rtol=3e-9)


def test_segmented_gram_training_does_not_retain_outer_products() -> None:
    generator = torch.Generator().manual_seed(204)
    nodes, heads, feature_width, value_width = 32, 2, 69, 8
    index = torch.arange(nodes) % 3
    feature = torch.randn(
        nodes,
        heads,
        feature_width,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    value = torch.randn(
        nodes,
        heads,
        value_width,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    saved_shapes: list[torch.Size] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved_shapes.append(tensor.shape)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        SelfAdjointRelation._gram_apply(
            feature, value, index, 3
        ).square().sum().backward()
    assert not any(
        len(shape) == 4
        and shape[0] == nodes
        and shape[1] == heads
        and shape[-1] == value_width
        for shape in saved_shapes
    )


def test_segmented_gram_chunk_boundaries_match_dense_oracle() -> None:
    generator = torch.Generator().manual_seed(205)
    nodes, heads, value_width = 7, 2, 3
    index = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    value = torch.randn(
        nodes,
        heads,
        value_width,
        generator=generator,
        dtype=torch.float64,
    )
    for feature_width in (17, 32, 64):
        feature = torch.randn(
            nodes,
            heads,
            feature_width,
            generator=generator,
            dtype=torch.float64,
        )
        actual = SelfAdjointRelation._gram_apply(feature, value, index, 2)
        expected = torch.zeros_like(value)
        for segment in range(2):
            selected = index == segment
            for head in range(heads):
                local = feature[selected, head]
                expected[selected, head] = local @ (local.T @ value[selected, head])
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


def test_segmented_gram_compiled_forward_backward_smoke() -> None:
    generator = torch.Generator().manual_seed(206)
    feature = torch.randn(
        9,
        2,
        69,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    value = torch.randn(
        9,
        2,
        4,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1])
    compiled = torch.compile(
        SelfAdjointRelation._gram_apply,
        backend="eager",
        dynamic=True,
    )
    output = compiled(feature, value, index, 2)
    gradients = torch.autograd.grad(output.square().sum(), (feature, value))
    assert bool(torch.isfinite(output).all())
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


def test_graphwise_irrep_orthogonalization() -> None:
    generator = torch.Generator().manual_seed(203)
    index = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    counts = torch.tensor([3, 4])
    first = _message(7, 2, 4, generator)
    output = orthogonalize(
        _message(7, 2, 4, generator),
        (first,),
        index=index,
        num_segments=2,
        counts=counts,
        eps=1e-10,
    )
    inner = message_inner(first, output)
    graph_inner = inner.new_zeros(2, 2).index_add(0, index, inner)
    torch.testing.assert_close(
        graph_inner,
        torch.zeros_like(graph_inner),
        atol=2e-9,
        rtol=0.0,
    )


def test_atlas_assignment_is_rotation_invariant_and_metric_equivariant() -> None:
    generator = torch.Generator().manual_seed(207)
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
    transform, _ = torch.linalg.qr(
        torch.randn(3, 3, generator=generator, dtype=torch.float64)
    )
    if torch.linalg.det(transform) > 0:
        transform[:, 0].neg_()
    reference = relation.build(
        state,
        GeometryContext.build(position, index, num_segments=2, eps=1e-10),
    )
    moved = relation.build(
        state,
        GeometryContext.build(
            position @ transform.T,
            index,
            num_segments=2,
            eps=1e-10,
        ),
    )
    torch.testing.assert_close(
        moved.atlas.assignment,
        reference.atlas.assignment,
        atol=3e-10,
        rtol=3e-10,
    )
    expected_metric = torch.einsum(
        "ia,...ab,jb->...ij",
        transform,
        reference.atlas.node_metric,
        transform,
    )
    torch.testing.assert_close(
        moved.atlas.node_metric,
        expected_metric,
        atol=4e-10,
        rtol=4e-10,
    )
