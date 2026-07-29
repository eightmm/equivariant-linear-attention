from dataclasses import FrozenInstanceError

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention.annotations import RelationTable
from equivariant_attention.neighbors import (
    PackedNeighborGraph,
    build_reverse_csr,
    pack_neighbor_graph,
)


def _edges_and_relations() -> tuple[torch.Tensor, torch.Tensor]:
    edge_index = torch.tensor(
        [
            [2, 0, 1, 0, 2, 0],
            [1, 2, 0, 1, 0, 0],
        ],
        dtype=torch.int64,
    )
    relation_id = torch.tensor([1, 2, 3, 0, 2, 4], dtype=torch.int64)
    return edge_index, relation_id


def test_relation_table_requires_an_explicit_involution() -> None:
    table = RelationTable(
        names=("same", "forward", "reverse", "context"),
        reverse_id=(0, 2, 1, 3),
    )
    relation_id = torch.tensor([0, 1, 2, 3], dtype=torch.int32)

    assert table.num_relations == 4
    assert torch.equal(
        table.reverse(relation_id),
        torch.tensor([0, 2, 1, 3], dtype=torch.int32),
    )
    with pytest.raises(FrozenInstanceError):
        table.names = ("changed",)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("names", "reverse_id", "message"),
    [
        ((), (), "at least one"),
        (("a", "a"), (0, 1), "unique"),
        (("a", "b"), (0,), "same length"),
        (("a", "b"), (0, 2), "range"),
        (("a", "b", "c"), (1, 2, 0), "involution"),
    ],
)
def test_relation_table_rejects_ambiguous_reverse_semantics(
    names: tuple[str, ...],
    reverse_id: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RelationTable(names=names, reverse_id=reverse_id)


def test_packed_relations_follow_stable_receiver_order_and_round_trip() -> None:
    edge_index, relation_id = _edges_and_relations()

    packed = pack_neighbor_graph(
        edge_index,
        num_nodes=3,
        edge_relation_id=relation_id,
    )

    assert packed.relation_id is not None
    assert packed.relation_id.dtype == torch.int32
    assert torch.equal(
        packed.relation_id,
        relation_id.index_select(
            0,
            packed.edge_order.to(dtype=torch.long),
        ).to(dtype=torch.int32),
    )
    assert torch.equal(
        packed.original_relation_id(),
        relation_id.to(dtype=torch.int32),
    )


def test_reverse_csr_is_a_reduction_view_and_does_not_reverse_relations() -> None:
    edge_index, relation_id = _edges_and_relations()
    table = RelationTable(
        names=("same", "forward", "reverse", "context", "self"),
        reverse_id=(0, 2, 1, 3, 4),
    )
    packed = pack_neighbor_graph(
        edge_index,
        num_nodes=3,
        edge_relation_id=relation_id,
    )

    reversed_plan = build_reverse_csr(packed)

    assert reversed_plan.reverse_edge_order is not None
    expected_view = reversed_plan.relation_id.index_select(
        0,
        reversed_plan.reverse_edge_order.to(dtype=torch.long),
    )
    assert torch.equal(reversed_plan.reverse_relation_view(), expected_view)
    assert not torch.equal(
        reversed_plan.reverse_relation_view(),
        table.reverse(expected_view),
    )


def test_relation_metadata_moves_with_trusted_packed_graph() -> None:
    edge_index, relation_id = _edges_and_relations()
    packed = pack_neighbor_graph(
        edge_index,
        num_nodes=3,
        edge_relation_id=relation_id,
        build_reverse=True,
    )

    moved = packed.to("meta")

    assert moved.relation_id is not None
    assert moved.relation_id.device.type == "meta"
    assert moved.validated


@pytest.mark.parametrize(
    "relation_id",
    [
        torch.tensor([0, 1]),
        torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        torch.tensor([0, 1, 2, 3, 4, -1]),
    ],
)
def test_packed_relation_metadata_is_validated(
    relation_id: torch.Tensor,
) -> None:
    edge_index, _ = _edges_and_relations()

    with pytest.raises((TypeError, ValueError), match="relation"):
        pack_neighbor_graph(
            edge_index,
            num_nodes=3,
            edge_relation_id=relation_id,
        )


def test_public_constructor_rejects_misaligned_relation_metadata() -> None:
    edge_index, relation_id = _edges_and_relations()
    packed = pack_neighbor_graph(
        edge_index,
        num_nodes=3,
        edge_relation_id=relation_id,
    )
    assert packed.relation_id is not None

    with pytest.raises(ValueError, match="relation_id"):
        PackedNeighborGraph(
            num_nodes=packed.num_nodes,
            row_ptr=packed.row_ptr,
            sender=packed.sender,
            edge_order=packed.edge_order,
            relation_id=packed.relation_id[:-1],
        )


def _typed_relation_model() -> EquivariantAttention:
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            hidden_irreps="8x0e + 2x1o",
            output_irreps="1x0e + 1x1o",
            num_layers=2,
            num_heads=2,
            local_head_counts=(0, 0),
            local_cutoff=3.0,
            use_sparse_low_rank_local_residual=True,
            local_residual_rank=2,
            num_edge_relations=2,
            relation_cutoffs=(1.0, 3.0),
        )
    ).double()
    for name, parameter in model.named_parameters():
        if (
            "sparse_low_rank_local_residual" in name
            and name.endswith("_out.weight")
        ):
            torch.nn.init.constant_(parameter, 0.1)
    return model


def test_typed_relation_cutoff_uses_one_sparse_list_and_matches_packed_path() -> None:
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 0, 2],
            [0, 1, 0, 1, 1, 2, 2, 0],
        ],
        dtype=torch.long,
    )
    relation_id = torch.tensor([0, 0, 1, 0, 1, 0, 1, 1])
    packed = pack_neighbor_graph(
        edge_index,
        num_nodes=3,
        edge_relation_id=relation_id,
    )
    generator = torch.Generator().manual_seed(20260802)
    features = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [2.0, 0.2, 0.0]],
        dtype=torch.float64,
    )
    batch = torch.zeros(3, dtype=torch.long)
    torch.manual_seed(20260803)
    coo_model = _typed_relation_model()
    torch.manual_seed(20260803)
    packed_model = _typed_relation_model()
    coo_features = features.clone().requires_grad_(True)
    packed_features = features.clone().requires_grad_(True)
    coo_positions = positions.clone().requires_grad_(True)
    packed_positions = positions.clone().requires_grad_(True)

    expected = coo_model(
        coo_features,
        coo_positions,
        batch=batch,
        edge_index=edge_index,
        edge_relation_id=relation_id,
    )
    actual = packed_model(
        packed_features,
        packed_positions,
        batch=batch,
        packed_neighbors=packed,
    )

    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    expected_gradients = torch.autograd.grad(
        sum(value.square().sum() for value in expected.values()),
        (coo_features, coo_positions),
    )
    actual_gradients = torch.autograd.grad(
        sum(value.square().sum() for value in actual.values()),
        (packed_features, packed_positions),
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=1e-12,
            atol=1e-12,
        )


def test_relation_specific_cutoffs_change_only_invariant_sparse_weights() -> None:
    edge_index = torch.tensor(
        [[0, 0, 1, 1], [0, 1, 0, 1]],
        dtype=torch.long,
    )
    generator = torch.Generator().manual_seed(20260804)
    features = torch.randn(2, 4, generator=generator, dtype=torch.float64)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        dtype=torch.float64,
    )
    batch = torch.zeros(2, dtype=torch.long)
    torch.manual_seed(20260805)
    model = _typed_relation_model()

    short = model(
        features,
        positions,
        batch=batch,
        edge_index=edge_index,
        edge_relation_id=torch.zeros(4, dtype=torch.long),
    )
    long = model(
        features,
        positions,
        batch=batch,
        edge_index=edge_index,
        edge_relation_id=torch.ones(4, dtype=torch.long),
    )

    assert not torch.allclose(short["node_scalars"], long["node_scalars"])
