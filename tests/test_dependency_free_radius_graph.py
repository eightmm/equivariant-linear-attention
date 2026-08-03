from __future__ import annotations

import torch

from equivariant_attention.radius import radius_graph


def _dense_reference(
    positions: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
    *,
    include_self: bool,
) -> torch.Tensor:
    displacement = positions[:, None, :] - positions[None, :, :]
    valid = displacement.square().sum(dim=-1) < cutoff**2
    valid &= batch[:, None] == batch[None, :]
    if not include_self:
        valid.fill_diagonal_(False)
    receiver, sender = valid.nonzero(as_tuple=True)
    return torch.stack([receiver, sender])


def _pair_codes(edge_index: torch.Tensor, nodes: int) -> torch.Tensor:
    return torch.sort(edge_index[0] * nodes + edge_index[1]).values


def test_dense_radius_path_matches_reference() -> None:
    torch.manual_seed(17)
    positions = torch.randn(11, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    cutoff = 1.5

    expected = _dense_reference(
        positions,
        batch,
        cutoff,
        include_self=True,
    )
    actual = radius_graph(
        positions,
        batch=batch,
        cutoff=cutoff,
        include_self=True,
        dense_threshold=32,
        chunk_size=2,
    )
    torch.testing.assert_close(
        _pair_codes(actual, positions.shape[0]),
        _pair_codes(expected, positions.shape[0]),
    )


def test_cell_list_path_matches_dense_reference() -> None:
    torch.manual_seed(18)
    positions = torch.randn(37, 3, dtype=torch.float64)
    batch = torch.tensor([0] * 17 + [1] * 20)
    cutoff = 1.25
    expected = _dense_reference(
        positions,
        batch,
        cutoff,
        include_self=False,
    )
    actual = radius_graph(
        positions,
        batch=batch,
        cutoff=cutoff,
        include_self=False,
        dense_threshold=4,
    )
    torch.testing.assert_close(
        _pair_codes(actual, positions.shape[0]),
        _pair_codes(expected, positions.shape[0]),
    )


def test_low_precision_topology_does_not_depend_on_radius_backend() -> None:
    cases = (
        (torch.bfloat16, 0, 1.2, 138),
        (torch.float16, 17, 1.5, 232),
    )
    for dtype, seed, cutoff, expected_edges in cases:
        positions = (
            2.0
            * torch.rand(
                (20, 3),
                generator=torch.Generator().manual_seed(seed),
            )
        ).to(dtype=dtype)
        dense = radius_graph(
            positions,
            cutoff=cutoff,
            include_self=False,
            dense_threshold=32,
        )
        cell = radius_graph(
            positions,
            cutoff=cutoff,
            include_self=False,
            dense_threshold=4,
        )
        torch.testing.assert_close(
            _pair_codes(cell, positions.shape[0]),
            _pair_codes(dense, positions.shape[0]),
        )
        assert dense.shape[1] == expected_edges


def test_radius_graph_never_connects_different_graphs() -> None:
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
        ]
    )
    batch = torch.tensor([0, 0, 1, 1])
    edge_index = radius_graph(positions, batch=batch, cutoff=1.0)
    assert torch.equal(batch[edge_index[0]], batch[edge_index[1]])


def test_radius_graph_max_neighbors_bounds_each_receiver() -> None:
    torch.manual_seed(19)
    positions = 0.1 * torch.randn(12, 3)
    edge_index = radius_graph(
        positions,
        cutoff=10.0,
        max_neighbors=4,
        include_self=True,
        dense_threshold=4,
    )
    counts = torch.bincount(edge_index[0], minlength=12)
    assert int(counts.max().item()) <= 4
    self_edges = edge_index[0] == edge_index[1]
    assert int(self_edges.sum().item()) == 12


def test_max_neighbors_keeps_self_and_complete_boundary_ties() -> None:
    positions = torch.zeros(12, 3)
    expected_self = torch.arange(12)

    for dense_threshold in (4, 32):
        edge_index = radius_graph(
            positions,
            cutoff=1.0,
            max_neighbors=1,
            include_self=True,
            dense_threshold=dense_threshold,
        )
        self_edges = edge_index[:, edge_index[0] == edge_index[1]]
        torch.testing.assert_close(self_edges[0], expected_self)
        torch.testing.assert_close(self_edges[1], expected_self)
        # All candidates lie on the same zero-distance shell. Keeping that
        # shell intact is the permutation-equivariant meaning of the cap.
        assert edge_index.shape[1] == 12 * 12


def test_max_neighbors_ties_match_dense_cell_and_node_permutation() -> None:
    axis = torch.arange(3, dtype=torch.float64)
    positions = torch.cartesian_prod(axis, axis, axis)
    nodes = positions.shape[0]
    kwargs = {
        "cutoff": 1.1,
        "max_neighbors": 2,
        "include_self": False,
    }
    dense = radius_graph(positions, dense_threshold=64, **kwargs)
    cell = radius_graph(positions, dense_threshold=4, **kwargs)
    torch.testing.assert_close(
        _pair_codes(cell, nodes),
        _pair_codes(dense, nodes),
    )

    permutation = torch.randperm(nodes, generator=torch.Generator().manual_seed(5))
    permuted = radius_graph(
        positions[permutation],
        dense_threshold=4,
        **kwargs,
    )
    restored = permutation[permuted]
    torch.testing.assert_close(
        _pair_codes(restored, nodes),
        _pair_codes(dense, nodes),
    )


def test_radius_graph_can_exclude_self() -> None:
    positions = torch.zeros(3, 3)
    edge_index = radius_graph(
        positions,
        cutoff=1.0,
        include_self=False,
    )
    assert not bool((edge_index[0] == edge_index[1]).any().item())
    assert edge_index.shape[1] == 6
