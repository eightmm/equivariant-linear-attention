from dataclasses import FrozenInstanceError

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention import moment
from equivariant_attention.neighbors import (
    PackedNeighborGraph,
    _select_index_dtype,
    pack_neighbor_graph,
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

    assert moved is not packed
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
