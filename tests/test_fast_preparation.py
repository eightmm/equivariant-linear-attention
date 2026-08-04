from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest
import torch

from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.geometry.neighbors import build_receiver_csr
from equivariant_linear_attention.geometry.radius import (
    _radius_graph_csr,
    radius_graph,
)


@pytest.mark.parametrize(
    ("counts", "dense_threshold", "max_neighbors"),
    [
        ([13], 64, None),
        ([37], 4, None),
        ([17, 20], 4, None),
        ([17, 20], 4, 4),
    ],
)
def test_direct_radius_csr_exactly_matches_coo_reference(
    counts: list[int],
    dense_threshold: int,
    max_neighbors: int | None,
) -> None:
    nodes = sum(counts)
    positions = torch.randn(
        nodes,
        3,
        generator=torch.Generator().manual_seed(nodes),
        dtype=torch.float64,
    )
    count_tensor = torch.tensor(counts)
    batch = torch.repeat_interleave(torch.arange(len(counts)), count_tensor)
    edge_index = radius_graph(
        positions,
        cutoff=1.5,
        batch=batch,
        max_neighbors=max_neighbors,
        dense_threshold=dense_threshold,
    )
    reference = build_receiver_csr(
        edge_index,
        num_nodes=nodes,
        edge_relation_id=torch.zeros(edge_index.shape[1], dtype=torch.long),
    )

    direct = _radius_graph_csr(
        positions,
        cutoff=1.5,
        batch=batch,
        max_neighbors=max_neighbors,
        dense_threshold=dense_threshold,
        num_edge_relations=1,
    )

    torch.testing.assert_close(direct.row_ptr, reference.row_ptr)
    torch.testing.assert_close(direct.sender, reference.sender)
    torch.testing.assert_close(
        direct.packed_edge_index(),
        reference.packed_edge_index(),
    )
    torch.testing.assert_close(direct.relation_id, reference.relation_id)


def test_direct_radius_csr_does_not_add_a_receiver_argsort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import equivariant_linear_attention.geometry.radius as radius_module

    positions = torch.randn(
        37,
        3,
        generator=torch.Generator().manual_seed(8),
        dtype=torch.float64,
    )
    batch = torch.tensor([0] * 17 + [1] * 20)
    original_argsort = torch.argsort
    calls = 0

    def counted_argsort(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_argsort(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch, "argsort", counted_argsort)
    reference_edges = radius_graph(
        positions,
        cutoff=1.5,
        batch=batch,
        dense_threshold=4,
    )
    build_receiver_csr(reference_edges, num_nodes=positions.shape[0])
    coo_pack_calls = calls

    def forbidden_repack(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("receiver-major radius output must not be repacked")

    monkeypatch.setattr(radius_module, "build_receiver_csr", forbidden_repack)
    calls = 0
    _radius_graph_csr(
        positions,
        cutoff=1.5,
        batch=batch,
        dense_threshold=4,
    )

    assert calls == coo_pack_calls


def test_collated_graph_wires_known_counts_to_grouped_radius_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import equivariant_linear_attention.api as api_module

    first = ELAGraph(torch.randn(3, 2), torch.randn(3, 3))
    second = ELAGraph(torch.randn(5, 2), torch.randn(5, 3))
    graph = ELAGraph.collate((first, second))
    observed: dict[str, torch.Tensor | None] = {}
    original = api_module._prepare_radius_3d_graph

    def wrapped(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        observed["graph_counts"] = kwargs.get("graph_counts")  # type: ignore[assignment]
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(api_module, "_prepare_radius_3d_graph", wrapped)
    prepared = ELA("2x0e", width=16, depth=1, cutoff=2.0)._prepare_graph(graph)

    assert prepared._prepared_graph is not None
    assert prepared._prepared_graph.graph_layout.is_grouped
    torch.testing.assert_close(
        observed["graph_counts"],
        torch.tensor([3, 5]),
    )


def test_grouped_radius_preparation_forwards_counts_without_batch_rescan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import equivariant_linear_attention.geometry.radius as radius_module

    first = ELAGraph(torch.randn(3, 2), torch.randn(3, 3))
    second = ELAGraph(torch.randn(5, 2), torch.randn(5, 3))
    graph = ELAGraph.collate((first, second))
    observed: dict[str, object] = {}
    original = radius_module._radius_edges

    def wrapped(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        observed["batch"] = kwargs.get("batch")
        observed["counts"] = kwargs.get("_trusted_graph_counts")
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(radius_module, "_radius_edges", wrapped)
    ELA("2x0e", width=16, depth=1, cutoff=2.0)._prepare_graph(graph)

    assert observed["batch"] is None
    torch.testing.assert_close(observed["counts"], torch.tensor([3, 5]))


def test_tiny_grouped_radius_uses_vectorized_dense_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import equivariant_linear_attention.geometry.radius as radius_module

    positions = torch.randn(
        20,
        3,
        generator=torch.Generator().manual_seed(91),
        dtype=torch.float64,
    )
    ptr = torch.tensor([0, 6, 15, 20])

    def forbidden_cell(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("tiny grouped graphs should avoid the cell-list path")

    monkeypatch.setattr(radius_module, "_batched_cell_edges", forbidden_cell)
    receiver, sender = radius_module._radius_edges(
        positions,
        cutoff=1.25,
        ptr=ptr,
        dense_threshold=16,
        receiver_major=True,
    )

    reference_receiver: list[torch.Tensor] = []
    reference_sender: list[torch.Tensor] = []
    for start, stop in zip(ptr[:-1].tolist(), ptr[1:].tolist(), strict=True):
        local_receiver, local_sender = radius_module._dense_edges(
            positions[start:stop],
            cutoff_squared=1.25**2,
            include_self=False,
            max_neighbors=None,
            chunk_size=1024,
        )
        reference_receiver.append(local_receiver + start)
        reference_sender.append(local_sender + start)
    torch.testing.assert_close(receiver, torch.cat(reference_receiver))
    torch.testing.assert_close(sender, torch.cat(reference_sender))


def test_grouped_radius_pair_budget_falls_back_to_cell_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import equivariant_linear_attention.geometry.radius as radius_module

    graphs = 17
    nodes_per_graph = 256
    positions = torch.randn(graphs * nodes_per_graph, 3)
    ptr = torch.arange(graphs + 1) * nodes_per_graph
    called = False

    def observed(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        del args, kwargs
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty

    monkeypatch.setattr(radius_module, "_batched_cell_edges", observed)
    radius_module._radius_edges(
        positions,
        cutoff=1.0,
        ptr=ptr,
        dense_threshold=nodes_per_graph,
    )
    assert called


def test_public_cache_reuse_exactly_revalidates_explicit_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = ELAGraph(
        torch.randn(4, 2),
        torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
    )
    model = ELA("2x0e", width=16, depth=1, cutoff=2.0)
    graph = model._prepare_graph(graph)
    cached = graph._prepared_graph
    assert cached is not None

    original_restore = type(cached.neighbors).original_edge_index
    calls = 0

    def counted_restore(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_restore(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        type(cached.neighbors),
        "original_edge_index",
        counted_restore,
    )
    reused = model._prepare_packed(graph._to_packed())

    assert reused._prepared_graph is cached
    assert not reused._trusted_prepared
    assert calls == 1


def test_assume_immutable_clones_topology_and_admits_trusted_reuse() -> None:
    original = ELAGraph(
        torch.randn(4, 2),
        torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        batch=torch.tensor([0, 0, 1, 1]),
        edge_type=torch.tensor([0, 1, 0, 1]),
        group=torch.tensor([0, 0, 1, 1]),
    )
    graph = original.assume_immutable()

    assert graph.pos.data_ptr() != original.pos.data_ptr()
    assert graph.edge_index is not None and original.edge_index is not None
    assert graph.edge_index.data_ptr() != original.edge_index.data_ptr()
    assert graph.batch is not None and original.batch is not None
    assert graph.batch.data_ptr() != original.batch.data_ptr()
    assert graph.edge_type is not None and original.edge_type is not None
    assert graph.edge_type.data_ptr() != original.edge_type.data_ptr()
    assert graph.group is not None and original.group is not None
    assert graph.group.data_ptr() != original.group.data_ptr()

    model = ELA("2x0e", width=16, depth=1, cutoff=2.0, edge_types=2)
    model(graph)
    prepared = graph._prepared_graph
    assert prepared is not None
    assert graph._prepared_provenance is not None

    packed = graph._to_packed()
    assert packed._trusted_prepared
    reused = model._prepare_packed(packed)
    assert reused is packed
    assert reused._prepared_graph is prepared


def test_trusted_explicit_reuse_skips_packed_topology_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from equivariant_linear_attention.batch import ELABatch

    graph = ELAGraph(
        torch.randn(32, 2),
        torch.randn(32, 3),
        edge_index=torch.stack(
            (
                torch.arange(32).repeat_interleave(4),
                (torch.arange(32).repeat_interleave(4) + torch.arange(1, 5).repeat(32))
                % 32,
            )
        ),
    ).assume_immutable()
    model = ELA("2x0e", width=16, depth=1, cutoff=2.0)
    model(graph)
    template = graph._packed_template
    assert template is not None

    def forbidden_reconstruction(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("trusted reuse must not reconstruct packed topology")

    monkeypatch.setattr(ELABatch, "from_flat", forbidden_reconstruction)
    assert graph._to_packed() is template
    model(graph)


def test_preexisting_external_alias_cannot_mutate_sealed_topology() -> None:
    position_array = np.random.default_rng(17).normal(size=(4, 3)).astype(np.float32)
    edge_array = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    original = ELAGraph(
        torch.randn(4, 2),
        torch.from_numpy(position_array),
        edge_index=torch.from_numpy(edge_array),
    )
    graph = original.assume_immutable()
    sealed_positions = graph.pos.clone()
    assert graph.edge_index is not None
    sealed_edges = graph.edge_index.clone()

    position_array.fill(99.0)
    edge_array.fill(0)
    torch.testing.assert_close(graph.pos, sealed_positions)
    torch.testing.assert_close(graph.edge_index, sealed_edges)

    model = ELA("2x0e", width=16, depth=1, cutoff=2.0)
    model(graph)
    assert graph._prepared_provenance is not None
    assert graph._packed_template is not None
    assert graph._to_packed() is graph._packed_template


def test_trusted_radius_position_version_refreshes_or_rebuilds() -> None:
    base = ELA("2x0e", width=16, depth=1, cutoff=1.0)
    geometry = replace(base.config.geometry, skin=1.0)
    model = ELA.from_config(replace(base.config, geometry=geometry))
    graph = ELAGraph(
        torch.randn(4, 2),
        torch.tensor(
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.0, 0.0], [1.2, 0.0, 0.0]]
        ),
    ).assume_immutable()
    model(graph)
    first = graph._prepared_graph
    template = graph._packed_template
    initial_provenance = graph._prepared_provenance
    assert first is not None and template is not None and initial_provenance is not None

    graph.pos[0, 1].add_(0.1)
    assert graph._to_packed() is template
    assert graph._prepared_graph is first
    assert graph._prepared_provenance is not initial_provenance

    graph.pos[0, 1].add_(0.6)
    model(graph)
    assert graph._prepared_graph is not None and graph._prepared_graph is not first
    assert graph._packed_template is not None and graph._packed_template is not template


def test_assume_immutable_inside_inference_mode_falls_back_safely() -> None:
    model = ELA("2x0e", width=16, depth=1, cutoff=2.0)
    with torch.inference_mode():
        graph = ELAGraph(
            torch.randn(4, 2),
            torch.randn(4, 3),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        ).assume_immutable()
        model(graph)
        packed = graph._to_packed()

    assert graph._prepared_graph is not None
    assert graph._prepared_provenance is None
    assert graph._packed_template is None
    assert not packed._trusted_prepared


def test_dlpack_alias_mutation_invalidates_public_explicit_cache() -> None:
    graph = ELAGraph(
        torch.randn(3, 2),
        torch.randn(3, 3),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
    )
    model = ELA("2x0e", width=16, depth=1, cutoff=2.0)
    model(graph)
    first = graph._prepared_graph
    assert first is not None

    alias = torch.from_dlpack(graph.edge_index)
    alias.copy_(torch.tensor([[0, 2], [2, 0]]))
    assert graph.edge_index._version == 0
    model(graph)
    second = graph._prepared_graph

    assert second is not None and second is not first
    torch.testing.assert_close(
        second.neighbors.original_edge_index().long(),
        graph.edge_index[[1, 0]],
    )


def test_in_place_edge_and_relation_versions_invalidate_trusted_cache() -> None:
    graph = ELAGraph(
        torch.randn(4, 2),
        torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        edge_type=torch.tensor([0, 1, 0]),
    ).assume_immutable()
    model = ELA("2x0e", width=16, depth=1, cutoff=2.0, edge_types=2)
    model(graph)
    first = graph._prepared_graph
    assert first is not None
    assert graph._prepared_provenance is not None

    graph.edge_index[:, 0].copy_(torch.tensor([3, 0]))
    model(graph)
    second = graph._prepared_graph
    assert second is not None and second is not first

    graph.edge_type[0].fill_(1)
    model(graph)
    third = graph._prepared_graph
    assert third is not None and third is not second
    torch.testing.assert_close(
        third.neighbors.original_relation_id().long(),
        graph.edge_type,
    )


def test_external_edge_and_relation_mutations_use_exact_cache_fallback() -> None:
    edge_array = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    relation_array = np.asarray([0, 1, 0], dtype=np.int64)
    graph = ELAGraph(
        torch.randn(4, 2),
        torch.randn(4, 3),
        edge_index=torch.from_numpy(edge_array),
        edge_type=torch.from_numpy(relation_array),
    )
    model = ELA("2x0e", width=16, depth=1, cutoff=2.0, edge_types=2)
    model(graph)
    first = graph._prepared_graph
    assert first is not None

    # NumPy owns these storages, so these writes do not increment Tensor._version.
    edge_array[:, 0] = (3, 0)
    relation_array[0] = 1
    model(graph)
    second = graph._prepared_graph

    assert second is not None and second is not first
    torch.testing.assert_close(
        second.neighbors.original_edge_index().long(),
        graph.edge_index[[1, 0]],
    )
    torch.testing.assert_close(
        second.neighbors.original_relation_id().long(),
        graph.edge_type,
    )


def test_external_position_mutation_rebuilds_radius_cache() -> None:
    position_array = np.asarray(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    graph = ELAGraph(
        torch.randn(3, 2),
        torch.from_numpy(position_array),
    )
    model = ELA("2x0e", width=16, depth=1, cutoff=0.75)
    model(graph)
    first = graph._prepared_graph
    assert first is not None

    position_array[1, 0] = 10.0
    model(graph)
    second = graph._prepared_graph

    assert second is not None and second is not first
    assert second.num_edges < first.num_edges
