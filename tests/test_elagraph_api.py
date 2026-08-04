from __future__ import annotations

import pytest
import torch

import equivariant_linear_attention as ela
from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.advanced import OrderContext


def test_root_surface_and_asymmetric_edge_orientation() -> None:
    assert ela.__all__ == ["ELA", "ELAGraph"]
    assert ELA is ela.ELA
    assert ELAGraph is ela.ELAGraph

    graph = ELAGraph(
        x=torch.randn(3, 4),
        pos=torch.randn(3, 3),
        # Public edges are 0 -> 1 and 2 -> 1.
        edge_index=torch.tensor([[0, 2], [1, 1]]),
    )
    packed = graph._to_packed()
    assert packed.edge_index is not None
    torch.testing.assert_close(
        packed.edge_index,
        torch.tensor([[1, 1], [0, 2]]),
    )

    model = ELA("4x0e", width=16, depth=1, cutoff=4.0)
    prepared = model._prepare_graph(graph)
    assert prepared.edge_index is graph.edge_index
    assert prepared._prepared_graph is not None
    torch.testing.assert_close(
        prepared._prepared_graph.neighbors.original_edge_index().long(),
        torch.tensor([[1, 1], [0, 2]]),
    )


def test_collate_offsets_public_edges_and_preserves_metadata() -> None:
    first = ELAGraph(
        x=torch.randn(2, 3),
        pos=torch.randn(2, 3),
        edge_index=torch.tensor([[0], [1]]),
        edge_type=torch.tensor([2]),
        condition=torch.tensor([0.25, 1.0]),
        y=torch.tensor(1.5),
        ids=("first",),
    )
    second = ELAGraph(
        x=torch.randn(3, 3),
        pos=torch.randn(3, 3),
        edge_index=torch.tensor([[1, 2], [0, 1]]),
        edge_type=torch.tensor([1, 0]),
        condition=torch.tensor([0.75, -1.0]),
        y=torch.tensor(2.5),
        ids=("second",),
    )

    batch = ELAGraph.collate([first, second])

    torch.testing.assert_close(batch.batch, torch.tensor([0, 0, 1, 1, 1]))
    torch.testing.assert_close(
        batch.edge_index,
        torch.tensor([[0, 3, 4], [1, 2, 3]]),
    )
    torch.testing.assert_close(batch.edge_type, torch.tensor([2, 1, 0]))
    torch.testing.assert_close(
        batch.condition,
        torch.tensor([[0.25, 1.0], [0.75, -1.0]]),
    )
    torch.testing.assert_close(batch.y, torch.tensor([1.5, 2.5]))
    assert batch.ids == ("first", "second")


def test_collate_rejects_mixed_optional_topology() -> None:
    first = ELAGraph(
        torch.randn(2, 2),
        torch.randn(2, 3),
        edge_index=torch.tensor([[0], [1]]),
    )
    second = ELAGraph(torch.randn(2, 2), torch.randn(2, 3))
    with pytest.raises(ValueError, match="all samples must provide edge_index"):
        ELAGraph.collate([first, second])


def test_collate_preserves_all_supported_optional_fields() -> None:
    periods = torch.tensor([12.0])
    first = ELAGraph(
        x=torch.randn(2, 3),
        pos=torch.randn(2, 3),
        edge_index=torch.tensor([[0], [1]]),
        edge_type=torch.tensor([0]),
        group=torch.tensor([4, 4]),
        condition=torch.randn(2, 2),
        order=OrderContext(
            coordinates=torch.tensor([[0], [1]]),
            group_index=torch.tensor([0, 0]),
            periods=periods,
            enabled=torch.tensor([True, False]),
        ),
        update_mask=torch.tensor([True, False]),
        y=torch.tensor(1.0),
        ids=("first",),
        graph_x=torch.randn(1, 2),
        graph_sum=torch.randn(1, 2),
        delta=torch.randn(2, 3),
    )
    second = ELAGraph(
        x=torch.randn(3, 3),
        pos=torch.randn(3, 3),
        edge_index=torch.tensor([[0, 2], [1, 2]]),
        edge_type=torch.tensor([1, 0]),
        group=torch.tensor([7, 7, 8]),
        condition=torch.randn(3, 2),
        order=OrderContext(
            coordinates=torch.tensor([[2], [3], [4]]),
            group_index=torch.tensor([0, 1, 1]),
            periods=periods.clone(),
            enabled=torch.tensor([True, True, False]),
        ),
        update_mask=torch.tensor([False, True, True]),
        y=torch.tensor(2.0),
        ids=("second",),
        graph_x=torch.randn(1, 2),
        graph_sum=torch.randn(1, 2),
        delta=torch.randn(3, 3),
    )

    packed = ELAGraph.collate([first, second])

    assert packed.condition is not None and packed.condition.shape == (5, 2)
    assert packed.order is not None
    torch.testing.assert_close(
        packed.order.coordinates.squeeze(-1),
        torch.arange(5),
    )
    torch.testing.assert_close(packed.order.periods, periods)
    assert packed.order.group_index is not None
    assert packed.order.enabled is not None
    torch.testing.assert_close(
        packed.update_mask,
        torch.tensor([True, False, False, True, True]),
    )
    assert packed.ids == ("first", "second")
    assert packed.graph_x is not None and packed.graph_x.shape == (2, 2)
    assert packed.graph_sum is not None and packed.graph_sum.shape == (2, 2)
    assert packed.delta is not None and packed.delta.shape == (5, 3)


def test_output_is_same_graph_type_and_reuses_fixed_topology() -> None:
    model = ELA("3x0e", "1x0e", width=16, depth=1, cutoff=4.0).double()
    graph = ELAGraph(
        torch.randn(4, 3, dtype=torch.float64),
        torch.randn(4, 3, dtype=torch.float64),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
    )

    output = model(graph)

    assert isinstance(output, ELAGraph)
    assert output.graph_x is not None and output.graph_x.shape == (1, 1)
    assert output.graph_sum is not None and output.graph_sum.shape == (1, 1)
    assert output.delta is not None
    torch.testing.assert_close(output.pos, graph.pos)
    torch.testing.assert_close(output.delta, torch.zeros_like(graph.pos))
    assert graph._prepared_graph is not None
    assert output._prepared_graph is graph._prepared_graph


def test_integer_semantic_order_is_accepted_by_the_public_graph() -> None:
    model = ELA(
        "2x0e",
        width=16,
        depth=1,
        cutoff=2.0,
        order_dim=1,
    ).double()
    graph = ELAGraph(
        torch.randn(4, 2, dtype=torch.float64),
        torch.randn(4, 3, dtype=torch.float64),
        order=OrderContext.sequence(torch.arange(4)),
    )

    output = model(graph)

    assert output.x.shape == (4, 1)
    assert torch.isfinite(output.x).all()
