from dataclasses import FrozenInstanceError

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention import moment
from equivariant_attention.neighbors import (
    PackedNeighborGraph,
    _select_index_dtype,
    build_receiver_csr,
    build_reverse_csr,
    pack_neighbor_graph,
    receiver_csr_reduce,
    sender_csr_reduce,
)


def _shuffled_edges() -> torch.Tensor:
    return torch.tensor(
        [
            [2, 0, 1, 0, 2, 0],
            [1, 2, 0, 1, 0, 0],
        ],
        dtype=torch.int64,
    )


def test_pack_neighbor_graph_builds_stable_receiver_csr_and_round_trips() -> None:
    edge_index = _shuffled_edges()

    packed = pack_neighbor_graph(edge_index, num_nodes=3)

    assert isinstance(packed, PackedNeighborGraph)
    assert packed.index_dtype == torch.int32
    assert packed.num_edges == edge_index.shape[1]
    assert torch.equal(
        packed.row_ptr,
        torch.tensor([0, 3, 4, 6], dtype=torch.int32),
    )
    assert torch.equal(
        packed.sender,
        torch.tensor([2, 1, 0, 0, 1, 0], dtype=torch.int32),
    )
    assert torch.equal(
        packed.edge_order,
        torch.tensor([1, 3, 5, 2, 0, 4], dtype=torch.int32),
    )
    assert packed.row_spans == ((0, 3), (3, 4), (4, 6))
    assert torch.equal(
        packed.receiver_index(),
        torch.tensor([0, 0, 0, 1, 2, 2], dtype=torch.int32),
    )
    assert torch.equal(
        packed.packed_edge_index(),
        edge_index[:, packed.edge_order.to(dtype=torch.long)].to(torch.int32),
    )
    assert torch.equal(packed.original_edge_index(), edge_index.to(torch.int32))


def test_reverse_csr_maps_sender_rows_back_to_forward_packed_edges() -> None:
    packed = pack_neighbor_graph(
        _shuffled_edges(),
        num_nodes=3,
        build_reverse=True,
    )

    assert packed.has_reverse
    assert torch.equal(
        packed.reverse_row_ptr,
        torch.tensor([0, 3, 5, 6], dtype=torch.int32),
    )
    assert torch.equal(
        packed.reverse_edge_order,
        torch.tensor([2, 3, 5, 1, 4, 0], dtype=torch.int32),
    )
    assert torch.equal(
        packed.reverse_edge_index(),
        torch.tensor(
            [
                [0, 0, 0, 1, 1, 2],
                [0, 1, 2, 0, 2, 0],
            ],
            dtype=torch.int32,
        ),
    )
    forward = packed.packed_edge_index()
    reverse_order = packed.reverse_edge_order.to(dtype=torch.long)
    assert torch.equal(
        packed.reverse_edge_index(),
        forward.flip(0)[:, reverse_order],
    )


def test_csr_reductions_avoid_receiver_expansion_and_match_scatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed = pack_neighbor_graph(
        _shuffled_edges(),
        num_nodes=3,
        build_reverse=True,
    )
    value = torch.randn(
        packed.num_edges,
        2,
        3,
        generator=torch.Generator().manual_seed(417),
        dtype=torch.float64,
        requires_grad=True,
    )
    receiver = packed.packed_edge_index()[0].to(dtype=torch.long)
    sender = packed.sender.to(dtype=torch.long)
    expected_receiver = value.new_zeros((3, 2, 3)).index_add(0, receiver, value)
    expected_sender = value.new_zeros((3, 2, 3)).index_add(0, sender, value)

    def forbidden(_self: PackedNeighborGraph) -> torch.Tensor:
        raise AssertionError("CSR reduction must not expand a receiver vector")

    monkeypatch.setattr(PackedNeighborGraph, "receiver_index", forbidden)
    actual_receiver = receiver_csr_reduce(packed, value)
    actual_sender = sender_csr_reduce(packed, value)

    assert packed.row_ptr.dtype == torch.int32
    torch.testing.assert_close(actual_receiver, expected_receiver, rtol=0, atol=0)
    torch.testing.assert_close(actual_sender, expected_sender, rtol=0, atol=0)
    expected_gradient = torch.autograd.grad(
        expected_receiver.square().sum() + expected_sender.square().sum(),
        value,
        retain_graph=True,
    )[0]
    actual_gradient = torch.autograd.grad(
        actual_receiver.square().sum() + actual_sender.square().sum(),
        value,
    )[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def test_reverse_csr_sender_reduction_is_edge_permutation_consistent() -> None:
    edge_index = _shuffled_edges()
    edge_value = torch.randn(
        edge_index.shape[1],
        4,
        generator=torch.Generator().manual_seed(418),
        dtype=torch.float64,
    )
    permutation = torch.tensor([5, 2, 0, 4, 1, 3])

    def reduce(
        edges: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        packed = pack_neighbor_graph(edges, num_nodes=3, build_reverse=True)
        packed_value = values.index_select(
            0,
            packed.edge_order.to(dtype=torch.long),
        )
        return sender_csr_reduce(packed, packed_value)

    reference = reduce(edge_index, edge_value)
    candidate = reduce(
        edge_index.index_select(1, permutation),
        edge_value.index_select(0, permutation),
    )

    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)


def test_reverse_csr_rejects_sender_row_mismatch() -> None:
    packed = pack_neighbor_graph(
        _shuffled_edges(),
        num_nodes=3,
        build_reverse=True,
    )
    assert packed.reverse_edge_order is not None
    invalid_order = packed.reverse_edge_order.roll(1)

    with pytest.raises(ValueError, match="reverse CSR rows"):
        PackedNeighborGraph(
            num_nodes=packed.num_nodes,
            row_ptr=packed.row_ptr,
            sender=packed.sender,
            edge_order=packed.edge_order,
            reverse_row_ptr=packed.reverse_row_ptr,
            reverse_edge_order=invalid_order,
        )


def test_empty_edges_keep_all_receiver_rows_and_optional_reverse_rows() -> None:
    packed = pack_neighbor_graph(
        torch.empty((2, 0), dtype=torch.int64),
        num_nodes=3,
        build_reverse=True,
    )

    assert packed.num_edges == 0
    assert torch.equal(packed.row_ptr, torch.zeros(4, dtype=torch.int32))
    assert torch.equal(packed.reverse_row_ptr, torch.zeros(4, dtype=torch.int32))
    assert packed.row_spans == ((0, 0), (0, 0), (0, 0))
    assert packed.packed_edge_index().shape == (2, 0)
    assert packed.original_edge_index().shape == (2, 0)
    assert packed.reverse_edge_index().shape == (2, 0)


def test_to_preserves_index_dtype_and_round_trip_contract() -> None:
    edge_index = _shuffled_edges().to(dtype=torch.int32)
    packed = pack_neighbor_graph(
        edge_index,
        num_nodes=3,
        build_reverse=True,
    )

    moved = packed.to("cpu")

    assert moved is packed
    assert moved.device == torch.device("cpu")
    assert moved.index_dtype == torch.int32
    assert moved.row_ptr.dtype == torch.int32
    assert moved.sender.dtype == torch.int32
    assert moved.edge_order.dtype == torch.int32
    assert moved.reverse_row_ptr is not None
    assert moved.reverse_row_ptr.dtype == torch.int32
    assert moved.reverse_edge_order is not None
    assert moved.reverse_edge_order.dtype == torch.int32
    assert torch.equal(moved.original_edge_index(), edge_index)


def test_cross_device_to_uses_trusted_constructor_without_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed = pack_neighbor_graph(
        _shuffled_edges(),
        num_nodes=3,
        build_reverse=True,
        build_ell=True,
    )

    def forbidden(_self: PackedNeighborGraph) -> None:
        raise AssertionError("validated packed transfer must not revalidate")

    monkeypatch.setattr(PackedNeighborGraph, "__post_init__", forbidden)
    moved = packed.to("meta")

    assert moved.device.type == "meta"
    assert moved.index_dtype == packed.index_dtype
    assert moved.has_reverse
    assert moved.ell_sender is not None
    assert moved.ell_sender.device.type == "meta"
    assert moved.validated
    assert moved.row_spans == packed.row_spans


def test_receiver_and_reverse_csr_builders_are_independent() -> None:
    receiver_only = build_receiver_csr(
        _shuffled_edges(),
        num_nodes=3,
        build_ell=True,
    )
    with_reverse = build_reverse_csr(receiver_only)
    wrapper = pack_neighbor_graph(
        _shuffled_edges(),
        num_nodes=3,
        build_reverse=True,
        build_ell=True,
    )

    assert not receiver_only.has_reverse
    assert with_reverse.has_reverse
    for name in (
        "row_ptr",
        "sender",
        "edge_order",
        "degree",
        "degree_histogram",
        "degree_bucket",
        "ell_sender",
        "ell_mask",
        "reverse_row_ptr",
        "reverse_edge_order",
    ):
        candidate = getattr(with_reverse, name)
        expected = getattr(wrapper, name)
        if candidate is None or expected is None:
            assert candidate is expected
        else:
            torch.testing.assert_close(candidate, expected, rtol=0, atol=0)


def test_packed_degree_metadata_and_ell_round_trip() -> None:
    packed = pack_neighbor_graph(
        _shuffled_edges(),
        num_nodes=3,
        build_ell=True,
    )

    assert torch.equal(
        packed.degree,
        torch.tensor([3, 1, 2], dtype=torch.int32),
    )
    assert packed.max_degree == 3
    assert packed.degree_skew == pytest.approx(1.5)
    assert packed.degree_histogram is not None
    assert int(packed.degree_histogram.sum()) == packed.num_nodes
    assert packed.degree_bucket is not None
    assert packed.ell_sender is not None
    assert packed.ell_mask is not None
    assert packed.ell_sender.shape == (3, 3)
    assert packed.ell_mask.shape == (3, 3)
    for receiver in range(packed.num_nodes):
        start = int(packed.row_ptr[receiver])
        stop = int(packed.row_ptr[receiver + 1])
        degree = stop - start
        assert torch.equal(
            packed.ell_sender[receiver, :degree],
            packed.sender[start:stop],
        )
        assert packed.ell_mask[receiver, :degree].all()
        assert not packed.ell_mask[receiver, degree:].any()


def test_ell_round_trip_keeps_zero_degree_receiver_rows() -> None:
    edge_index = torch.tensor(
        [[1, 1, 3], [0, 2, 1]],
        dtype=torch.long,
    )

    packed = pack_neighbor_graph(
        edge_index,
        num_nodes=4,
        build_ell=True,
    )

    assert torch.equal(
        packed.degree,
        torch.tensor([0, 2, 0, 1], dtype=torch.int32),
    )
    assert packed.ell_sender is not None
    assert packed.ell_mask is not None
    assert not packed.ell_mask[0].any()
    assert not packed.ell_mask[2].any()
    assert torch.equal(
        packed.ell_sender[packed.ell_mask],
        packed.sender,
    )
    assert torch.equal(
        packed.original_edge_index(),
        edge_index.to(dtype=torch.int32),
    )


def test_degree_bucket_boundaries_and_histogram_are_exact() -> None:
    degrees = (0, 1, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129)
    receiver = torch.repeat_interleave(
        torch.arange(len(degrees)),
        torch.tensor(degrees),
    )
    edge_index = torch.stack([receiver, torch.zeros_like(receiver)])

    packed = build_receiver_csr(edge_index, num_nodes=len(degrees))

    assert torch.equal(
        packed.degree_bucket,
        torch.tensor(
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
            dtype=torch.uint8,
        ),
    )
    assert torch.equal(
        packed.degree_histogram,
        torch.tensor([1, 2, 2, 2, 2, 2, 1], dtype=torch.int64),
    )


def test_ell_build_rejects_excessive_padding_before_allocation() -> None:
    receiver = torch.zeros(64, dtype=torch.long)
    edge_index = torch.stack([receiver, torch.zeros_like(receiver)])

    with pytest.raises(ValueError, match="ELL padding ratio"):
        build_receiver_csr(
            edge_index,
            num_nodes=100,
            build_ell=True,
        )

    packed = build_receiver_csr(
        edge_index,
        num_nodes=100,
        build_ell=True,
        ell_max_padding_ratio=128.0,
    )
    assert packed.ell_sender is not None
    assert packed.ell_sender.shape == (100, 64)


def test_public_constructor_rejects_false_execution_metadata() -> None:
    packed = build_receiver_csr(_shuffled_edges(), num_nodes=3)
    assert packed.degree is not None
    assert packed.degree_histogram is not None
    assert packed.degree_bucket is not None
    invalid_bucket = packed.degree_bucket.clone()
    invalid_bucket[0] = 6

    with pytest.raises(ValueError, match="degree_bucket"):
        PackedNeighborGraph(
            num_nodes=packed.num_nodes,
            row_ptr=packed.row_ptr,
            sender=packed.sender,
            edge_order=packed.edge_order,
            degree=packed.degree,
            degree_histogram=packed.degree_histogram,
            degree_bucket=invalid_bucket,
            max_degree=packed.max_degree,
            degree_skew=packed.degree_skew,
        )
    with pytest.raises(ValueError, match="degree_histogram"):
        PackedNeighborGraph(
            num_nodes=packed.num_nodes,
            row_ptr=packed.row_ptr,
            sender=packed.sender,
            edge_order=packed.edge_order,
            degree=packed.degree,
            degree_histogram=packed.degree_histogram.roll(1),
            degree_bucket=packed.degree_bucket,
            max_degree=packed.max_degree,
            degree_skew=packed.degree_skew,
        )
    with pytest.raises(ValueError, match="summary"):
        PackedNeighborGraph(
            num_nodes=packed.num_nodes,
            row_ptr=packed.row_ptr,
            sender=packed.sender,
            edge_order=packed.edge_order,
            max_degree=3,
            degree_skew=1.5,
        )


def test_explicit_int64_storage_and_overflow_selection() -> None:
    packed = pack_neighbor_graph(
        _shuffled_edges(),
        num_nodes=3,
        prefer_int32=False,
    )

    assert packed.index_dtype == torch.int64
    assert packed.row_ptr.dtype == torch.int64
    assert packed.sender.dtype == torch.int64
    assert packed.edge_order.dtype == torch.int64
    assert _select_index_dtype(num_nodes=3, num_edges=6, prefer_int32=True) == (
        torch.int32
    )
    assert _select_index_dtype(
        num_nodes=torch.iinfo(torch.int32).max + 2,
        num_edges=1,
        prefer_int32=True,
    ) == torch.int64
    assert _select_index_dtype(
        num_nodes=1,
        num_edges=torch.iinfo(torch.int32).max + 1,
        prefer_int32=True,
    ) == torch.int64


def test_packed_neighbor_graph_is_frozen() -> None:
    packed = pack_neighbor_graph(_shuffled_edges(), num_nodes=3)

    with pytest.raises(FrozenInstanceError):
        packed.num_nodes = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    ("edge_index", "num_nodes", "message"),
    [
        (torch.tensor([0, 1]), 2, "shape"),
        (torch.tensor([[0.0], [0.0]]), 1, "integer"),
        (torch.tensor([[0], [-1]]), 1, "nonnegative"),
        (torch.tensor([[0], [2]]), 2, "out of range"),
        (torch.empty((2, 0), dtype=torch.int64), -1, "nonnegative"),
    ],
)
def test_pack_neighbor_graph_rejects_invalid_inputs(
    edge_index: torch.Tensor,
    num_nodes: int,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        pack_neighbor_graph(edge_index, num_nodes=num_nodes)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_nodes": True}, "num_nodes"),
        ({"num_nodes": 3, "build_reverse": 1}, "build_reverse"),
        ({"num_nodes": 3, "prefer_int32": 1}, "prefer_int32"),
        ({"num_nodes": 3, "build_ell": 1}, "build_ell"),
        (
            {"num_nodes": 3, "ell_max_padding_ratio": True},
            "ell_max_padding_ratio",
        ),
        ({"num_nodes": 3, "ell_max_elements": 1.5}, "ell_max_elements"),
    ],
)
def test_pack_neighbor_graph_rejects_invalid_controls(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        pack_neighbor_graph(_shuffled_edges(), **kwargs)  # type: ignore[arg-type]


def test_reverse_helpers_require_reverse_csr() -> None:
    packed = pack_neighbor_graph(_shuffled_edges(), num_nodes=3)

    assert not packed.has_reverse
    with pytest.raises(RuntimeError, match="reverse CSR"):
        packed.reverse_edge_index()


def _complete_edges(counts: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    batch = torch.repeat_interleave(
        torch.arange(len(counts)),
        torch.tensor(counts),
    )
    receiver: list[int] = []
    sender: list[int] = []
    for graph_index in range(len(counts)):
        nodes = torch.nonzero(batch == graph_index, as_tuple=False).flatten().tolist()
        for target in nodes:
            for source in reversed(nodes):
                receiver.append(target)
                sender.append(source)
    edge_index = torch.tensor([receiver, sender], dtype=torch.long)
    order = torch.randperm(edge_index.shape[1], generator=torch.Generator().manual_seed(9))
    return batch, edge_index[:, order]


def test_receiver_csr_geometry_uses_int32_offsets_and_matches_index_add_gradients() -> None:
    batch, edge_index = _complete_edges((4, 3))
    generator = torch.Generator().manual_seed(91)
    pos = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    index_geometry = moment._local_geometry(
        pos,
        batch,
        num_graphs=2,
        cutoff=10.0,
        num_rbf=6,
        edge_index=edge_index,
    )
    csr_geometry = moment._local_geometry(
        pos,
        batch,
        num_graphs=2,
        cutoff=10.0,
        num_rbf=6,
        edge_index=edge_index,
        build_receiver_csr=True,
    )
    assert csr_geometry.nonself_row_ptr is not None
    assert csr_geometry.nonself_row_ptr.dtype == torch.int32

    index_value = torch.randn(
        index_geometry.nonself_receiver.numel(),
        3,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    # Associate values with edge identities, not with the two execution orders.
    index_code = (
        index_geometry.nonself_receiver * 7 + index_geometry.nonself_sender
    )
    csr_code = csr_geometry.nonself_receiver * 7 + csr_geometry.nonself_sender
    position = torch.argsort(torch.argsort(index_code))
    csr_position = torch.argsort(torch.argsort(csr_code))
    canonical = index_value[torch.argsort(index_code)]
    csr_value = canonical[csr_position].detach().requires_grad_(True)
    index_sum = moment._local_receiver_sum(
        index_geometry,
        index_geometry.nonself_receiver,
        7,
        index_value,
    )[0]
    csr_sum = moment._local_receiver_sum(
        csr_geometry,
        csr_geometry.nonself_receiver,
        7,
        csr_value,
    )[0]
    torch.testing.assert_close(csr_sum, index_sum, rtol=0, atol=0)
    index_gradient = torch.autograd.grad(index_sum.square().sum(), index_value)[0]
    csr_gradient = torch.autograd.grad(csr_sum.square().sum(), csr_value)[0]
    torch.testing.assert_close(
        csr_gradient[torch.argsort(csr_position)],
        index_gradient[torch.argsort(position)],
        rtol=0,
        atol=0,
    )


def test_packed_geometry_consumes_csr_plans_without_resorting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, edge_index = _complete_edges((4, 3))
    packed = pack_neighbor_graph(edge_index, num_nodes=7, build_reverse=True)
    generator = torch.Generator().manual_seed(918)
    # The small cutoff removes a strict subset, exercising linear plan
    # restriction rather than the trivial all-edge reuse case.
    pos = 0.8 * torch.randn(7, 3, generator=generator, dtype=torch.float64)
    reference = moment._local_geometry(
        pos,
        batch,
        num_graphs=2,
        cutoff=1.0,
        num_rbf=5,
        edge_index=edge_index,
        build_receiver_csr=True,
        build_reverse_csr=True,
    )

    def forbidden(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("packed geometry must not rebuild CSR by sorting")

    monkeypatch.setattr(torch, "argsort", forbidden)
    monkeypatch.setattr(moment, "_receiver_csr_row_ptr", forbidden)
    candidate = moment._local_geometry(
        pos,
        batch,
        num_graphs=2,
        cutoff=1.0,
        num_rbf=5,
        packed_neighbors=packed,
        build_receiver_csr=True,
        build_reverse_csr=True,
    )

    for name in (
        "receiver",
        "sender",
        "displacement",
        "squared_distance",
        "rbf",
        "nonself_receiver",
        "nonself_sender",
        "nonself_displacement",
        "nonself_squared_distance",
        "nonself_rbf",
        "nonself_cutoff",
        "nonself_tensor_features",
        "row_ptr",
        "reverse_order",
        "reverse_row_ptr",
        "nonself_row_ptr",
        "nonself_reverse_order",
        "nonself_reverse_row_ptr",
    ):
        candidate_value = getattr(candidate, name)
        reference_value = getattr(reference, name)
        assert candidate_value is not None
        assert reference_value is not None
        torch.testing.assert_close(
            candidate_value,
            reference_value,
            rtol=0,
            atol=0,
        )
    assert candidate.row_ptr is not None
    assert candidate.row_ptr.dtype == packed.row_ptr.dtype
    assert candidate.reverse_row_ptr is not None
    assert packed.reverse_row_ptr is not None
    assert candidate.reverse_row_ptr.dtype == packed.reverse_row_ptr.dtype


def test_receiver_only_packed_geometry_does_not_build_reverse_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, edge_index = _complete_edges((4, 3))
    packed = pack_neighbor_graph(edge_index, num_nodes=7, build_reverse=False)
    pos = torch.randn(
        7,
        3,
        generator=torch.Generator().manual_seed(919),
        dtype=torch.float64,
    )

    def forbidden(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("receiver-only geometry must not sort reverse rows")

    monkeypatch.setattr(torch, "argsort", forbidden)
    geometry = moment._local_geometry(
        pos,
        batch,
        num_graphs=2,
        cutoff=10.0,
        num_rbf=5,
        packed_neighbors=packed,
        build_receiver_csr=True,
        build_reverse_csr=False,
    )

    assert geometry.row_ptr is not None
    assert geometry.nonself_row_ptr is not None
    assert geometry.reverse_order is None
    assert geometry.reverse_row_ptr is None
    assert geometry.nonself_reverse_order is None
    assert geometry.nonself_reverse_row_ptr is None


def test_packed_geometry_never_silently_drops_requested_reverse_plan() -> None:
    batch, edge_index = _complete_edges((4, 3))
    packed = pack_neighbor_graph(edge_index, num_nodes=7, build_reverse=False)
    pos = torch.randn(
        7,
        3,
        generator=torch.Generator().manual_seed(920),
        dtype=torch.float64,
    )

    with pytest.raises(ValueError, match="reverse CSR metadata"):
        moment._local_geometry(
            pos,
            batch,
            num_graphs=2,
            cutoff=10.0,
            num_rbf=5,
            packed_neighbors=packed,
            build_receiver_csr=True,
            build_reverse_csr=True,
        )


def test_local_geometry_admission_uses_squared_distance_without_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.999, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    batch = torch.zeros(3, dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1, 2, 0, 0], [0, 1, 2, 1, 2]],
        dtype=torch.long,
    )

    def forbidden(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("cutoff admission must not compute a vector norm")

    monkeypatch.setattr(moment, "_stable_vector_norm", forbidden)
    geometry = moment._local_geometry(
        pos,
        batch,
        num_graphs=1,
        cutoff=1.0,
        num_rbf=4,
        edge_index=edge_index,
    )

    assert torch.equal(
        torch.stack([geometry.receiver, geometry.sender]),
        torch.tensor([[0, 1, 2, 0], [0, 1, 2, 1]]),
    )


def test_local_geometry_squared_admission_avoids_float32_overflow() -> None:
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0e20, 0.0, 0.0]],
        dtype=torch.float32,
    )
    batch = torch.zeros(2, dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1, 0, 1], [0, 1, 1, 0]],
        dtype=torch.long,
    )

    geometry = moment._local_geometry(
        pos,
        batch,
        num_graphs=1,
        cutoff=2.0e20,
        num_rbf=4,
        edge_index=edge_index,
    )

    assert torch.equal(geometry.receiver, edge_index[0])
    assert torch.equal(geometry.sender, edge_index[1])
    torch.testing.assert_close(
        geometry.squared_distance,
        torch.tensor([0.0, 0.0, 0.25, 0.25], dtype=torch.float32),
    )


def test_reverse_csr_request_requires_receiver_csr() -> None:
    batch, edge_index = _complete_edges((2,))
    pos = torch.randn(2, 3)

    with pytest.raises(ValueError, match="reverse CSR requires receiver CSR"):
        moment._local_geometry(
            pos,
            batch,
            num_graphs=1,
            cutoff=10.0,
            num_rbf=4,
            edge_index=edge_index,
            build_receiver_csr=False,
            build_reverse_csr=True,
        )


def test_packed_segment_backend_matches_coo_full_model_values_and_input_gradients() -> None:
    common = dict(
        node_dim=5,
        hidden_irreps="12x0e + 3x1o",
        output_irreps="1x0e + 1x1o + 1x2e",
        num_layers=2,
        num_heads=3,
        local_head_counts=(0, 0),
        local_cutoff=10.0,
        use_key_balancing=False,
        use_sparse_low_rank_local_residual=True,
        local_residual_rank=3,
    )
    torch.manual_seed(92)
    reference = EquivariantAttention(
        EquivariantAttentionConfig(**common)
    ).double()
    torch.manual_seed(92)
    candidate = EquivariantAttention(
        EquivariantAttentionConfig(
            **common,
            local_reduction_backend="segment_csr",
        )
    ).double()
    for module in (reference, candidate):
        for name, parameter in module.named_parameters():
            if "sparse_low_rank_local_residual" in name and name.endswith(
                "_out.weight"
            ):
                torch.nn.init.constant_(parameter, 0.07)
    batch, edge_index = _complete_edges((4, 3))
    packed = pack_neighbor_graph(edge_index, num_nodes=7, build_reverse=True)
    generator = torch.Generator().manual_seed(93)
    feats = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    pos = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    reference_feats = feats.clone().requires_grad_(True)
    candidate_feats = feats.clone().requires_grad_(True)
    reference_pos = pos.clone().requires_grad_(True)
    candidate_pos = pos.clone().requires_grad_(True)

    reference_output = reference(
        reference_feats,
        reference_pos,
        batch=batch,
        edge_index=edge_index,
    )
    candidate_output = candidate(
        candidate_feats,
        candidate_pos,
        batch=batch,
        packed_neighbors=packed,
    )
    for name in reference_output:
        torch.testing.assert_close(
            candidate_output[name],
            reference_output[name],
            rtol=2e-12,
            atol=2e-12,
        )
    reference_loss = sum(value.square().sum() for value in reference_output.values())
    candidate_loss = sum(value.square().sum() for value in candidate_output.values())
    reference_gradients = torch.autograd.grad(
        reference_loss,
        (reference_feats, reference_pos),
    )
    candidate_gradients = torch.autograd.grad(
        candidate_loss,
        (candidate_feats, candidate_pos),
    )
    for candidate_gradient, reference_gradient in zip(
        candidate_gradients,
        reference_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            candidate_gradient,
            reference_gradient,
            rtol=2e-11,
            atol=2e-11,
        )


def test_local_reduction_backend_validation_is_explicit() -> None:
    with pytest.raises(ValueError, match="local_reduction_backend"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=3,
                hidden_irreps="8x0e + 2x1o",
                num_heads=2,
                local_reduction_backend="auto",
            )
        )
