from __future__ import annotations

import torch

from equivariant_linear_attention import ELA, ELAGraph


def test_radius_cache_rebuilds_after_in_place_coordinate_change() -> None:
    model = ELA("2x0e", width=16, depth=1, cutoff=1.0).double()
    graph = ELAGraph(
        torch.randn(3, 2, dtype=torch.float64),
        torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
    )

    model(graph)
    assert graph._prepared_graph is not None
    assert graph._prepared_graph.num_edges == 0
    old_cache = graph._prepared_graph

    with torch.no_grad():
        graph.pos.copy_(
            torch.tensor(
                [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.7, 0.0, 0.0]],
                dtype=torch.float64,
            )
        )
    model(graph)

    assert graph._prepared_graph is not None
    assert graph._prepared_graph is not old_cache
    assert graph._prepared_graph.num_edges == 6


def test_radius_cache_is_not_reused_across_geometry_dtype_or_cutoff() -> None:
    first = ELA("2x0e", width=16, depth=1, cutoff=0.75).double()
    graph = ELAGraph(
        torch.randn(3, 2, dtype=torch.float64),
        torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.2, 0.0, 0.0]],
            dtype=torch.float64,
        ),
    )
    first(graph)
    assert graph._prepared_graph is not None
    assert graph._prepared_graph.spec.cutoff == 0.75

    converted = graph.to("cpu", dtype=torch.float32, geometry_dtype=torch.float32)
    assert converted._prepared_graph is None

    second = ELA("2x0e", width=16, depth=1, cutoff=1.5).double()
    second(graph)
    assert graph._prepared_graph is not None
    assert graph._prepared_graph.spec.cutoff == 1.5
    assert graph._prepared_graph.num_edges == 6


def test_explicit_cache_survives_positions_but_recomputes_geometry() -> None:
    model = ELA("2x0e", width=16, depth=1, cutoff=2.0).double()
    graph = ELAGraph(
        torch.randn(3, 2, dtype=torch.float64),
        torch.randn(3, 3, dtype=torch.float64),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
    )
    first = model(graph)
    assert graph._prepared_graph is not None
    cache = graph._prepared_graph

    with torch.no_grad():
        graph.pos.add_(torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64))
    second = model(graph)

    assert graph._prepared_graph is cache
    assert first._prepared_graph is cache
    assert second._prepared_graph is cache


def test_max_neighbors_counts_nonself_neighbors() -> None:
    model = ELA(
        "2x0e",
        width=16,
        depth=1,
        cutoff=2.0,
        max_neighbors=1,
    )
    graph = ELAGraph(
        torch.randn(3, 2),
        torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [1.2, 0.0, 0.0]]),
    )
    prepared = model._prepare_graph(graph)._prepared_graph

    assert prepared is not None
    receivers = prepared.neighbors.receiver_index().long()
    senders = prepared.neighbors.sender.long()
    assert torch.equal(
        torch.bincount(receivers, minlength=3),
        torch.ones(3, dtype=torch.long),
    )
    assert torch.all(receivers != senders)


def test_interleaved_components_are_permutation_safe() -> None:
    model = ELA("2x0e", width=16, depth=1, cutoff=2.0).double().eval()
    x = torch.randn(4, 2, dtype=torch.float64)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.3, 0.0, 0.0], [0.4, 0.0, 0.0]],
        dtype=torch.float64,
    )
    group = torch.tensor([0, 1, 0, 1])
    graph = ELAGraph(x, pos, group=group)
    permutation = torch.tensor([2, 0, 3, 1])

    with torch.inference_mode():
        reference = model(graph)
        actual = model(
            ELAGraph(
                x[permutation],
                pos[permutation],
                group=group[permutation],
            )
        )

    torch.testing.assert_close(
        actual.x,
        reference.x[permutation],
        atol=2e-9,
        rtol=2e-9,
    )
    assert graph._prepared_graph is not None
    receiver = graph._prepared_graph.neighbors.receiver_index().long()
    sender = graph._prepared_graph.neighbors.sender.long()
    assert torch.equal(group[receiver], group[sender])


def test_in_place_group_change_invalidates_cache_before_attachment() -> None:
    model = ELA("2x0e", width=16, depth=1, cutoff=2.0).double()
    graph = ELAGraph(
        torch.randn(4, 2, dtype=torch.float64),
        torch.tensor(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.6, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        group=torch.tensor([0, 0, 1, 1]),
    )
    model(graph)
    previous = graph._prepared_graph
    assert previous is not None

    graph.group.copy_(torch.tensor([0, 1, 0, 1]))
    model(graph)

    assert graph._prepared_graph is not None
    assert graph._prepared_graph is not previous
    interaction = torch.tensor([0, 1, 0, 1])
    receiver = graph._prepared_graph.neighbors.receiver_index().long()
    sender = graph._prepared_graph.neighbors.sender.long()
    assert torch.equal(interaction[receiver], interaction[sender])
