from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention.nn.pair_state import (
    BiomolecularPairContext,
    DensePairState,
    build_dense_pair_layout,
)


def test_dense_layout_isolates_interaction_groups_and_preserves_order() -> None:
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    group = torch.tensor([1, 0, 1, 0, 0, 2], dtype=torch.long)
    layout = build_dense_pair_layout(batch, group, max_pair_tokens=2)

    torch.testing.assert_close(layout.lengths, torch.tensor([1, 2, 2, 1]))
    torch.testing.assert_close(
        layout.packed_batch,
        torch.tensor([1, 0, 1, 2, 2, 3]),
    )
    torch.testing.assert_close(
        layout.packed_slot,
        torch.tensor([0, 0, 1, 0, 1, 0]),
    )
    torch.testing.assert_close(
        layout.node_mask,
        torch.tensor(
            [
                [True, False],
                [True, True],
                [True, True],
                [True, False],
            ]
        ),
    )
    assert torch.equal(
        layout.pair_mask,
        layout.node_mask[:, :, None] & layout.node_mask[:, None, :],
    )

    component_of_node = layout.packed_batch
    for left in range(batch.numel()):
        for right in range(batch.numel()):
            same_component = bool(component_of_node[left] == component_of_node[right])
            expected = bool(batch[left] == batch[right] and group[left] == group[right])
            assert same_component == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_packed_dense_node_round_trip_is_exact_and_differentiable(
    dtype: torch.dtype,
) -> None:
    layout = build_dense_pair_layout(
        torch.tensor([0, 1, 0, 1, 1]),
        max_pair_tokens=3,
    )
    packed = torch.arange(15, dtype=dtype).reshape(5, 3).requires_grad_()
    dense = layout.unpack_node_tensor(packed)
    restored = layout.gather_nodes(dense)

    assert torch.equal(restored, packed)
    assert torch.count_nonzero(dense[0, 2]) == 0
    restored.square().sum().backward()
    torch.testing.assert_close(packed.grad, 2.0 * packed.detach())


def test_pair_state_mask_transpose_and_with_z_are_directed_and_exact() -> None:
    layout = build_dense_pair_layout(
        torch.tensor([0, 0, 1]),
        max_pair_tokens=2,
    )
    z = torch.arange(16, dtype=torch.float64).reshape(2, 2, 2, 2)
    state = layout.with_z(z)
    assert isinstance(state, DensePairState)
    assert state.with_z(z + 1.0).z.data_ptr() != state.z.data_ptr()
    torch.testing.assert_close(state.masked_z()[1, 0, 0], z[1, 0, 0])
    torch.testing.assert_close(
        state.masked_z()[1, 1],
        torch.zeros(2, 2, dtype=z.dtype),
    )

    transposed = state.transpose()
    torch.testing.assert_close(transposed.z, z.transpose(1, 2))
    torch.testing.assert_close(transposed.transpose().z, z)
    assert not torch.equal(state.z[0, 0, 1], state.z[0, 1, 0])

    packed = torch.randn(3, 4, dtype=torch.float64)
    assert torch.equal(
        state.gather_nodes(state.unpack_node_tensor(packed)),
        packed,
    )


def test_max_pair_tokens_guard_fires_before_dense_allocation() -> None:
    with pytest.raises(ValueError, match="exceeds max_pair_tokens=3"):
        build_dense_pair_layout(
            torch.zeros(4, dtype=torch.long),
            max_pair_tokens=3,
        )
    with pytest.raises(ValueError, match="positive"):
        build_dense_pair_layout(torch.zeros(1, dtype=torch.long), max_pair_tokens=0)


def test_empty_pair_layout_round_trips_without_special_state() -> None:
    layout = build_dense_pair_layout(
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        max_pair_tokens=1,
    )
    assert layout.node_mask.shape == (0, 0)
    assert layout.pair_mask.shape == (0, 0, 0)
    dense = layout.unpack_node_tensor(torch.empty(0, 3))
    assert dense.shape == (0, 0, 3)
    assert layout.gather_nodes(dense).shape == (0, 3)


def test_biomolecular_context_keeps_cross_chain_ordered_pairs() -> None:
    nodes = 3
    layout = build_dense_pair_layout(
        torch.zeros(nodes, dtype=torch.long),
        max_pair_tokens=nodes,
    )
    pair_features = torch.zeros(nodes, nodes, 1, dtype=torch.float64)
    for left in range(nodes):
        for right in range(nodes):
            pair_features[left, right, 0] = 10 * left + right
    context = BiomolecularPairContext(
        token_index=torch.arange(nodes),
        chain_id=torch.tensor([0, 1, 1]),
        entity_id=torch.tensor([0, 1, 1]),
        molecule_type=torch.tensor([0, 1, 2]),
        residue_type=torch.tensor([5, 6, 7]),
        bond_index=torch.tensor([[1], [2]]),
        bond_type=torch.tensor([3]),
        pair_features=pair_features,
    ).validate(num_nodes=nodes, device=torch.device("cpu"), dtype=torch.float64)
    dense = context.dense_pair_features(layout)
    assert dense is not None
    assert bool(layout.pair_mask.all())
    torch.testing.assert_close(dense[0, 0, 1], torch.tensor([1.0], dtype=torch.float64))
    torch.testing.assert_close(
        dense[0, 1, 0],
        torch.tensor([10.0], dtype=torch.float64),
    )


def test_biomolecular_context_to_and_collate_preserve_index_contracts() -> None:
    first = BiomolecularPairContext(
        token_index=torch.tensor([0, 1]),
        chain_id=torch.tensor([0, 0]),
        bond_index=torch.tensor([[0], [1]]),
        bond_type=torch.tensor([3]),
        pair_features=torch.arange(8, dtype=torch.float32).reshape(2, 2, 2),
    )
    second = BiomolecularPairContext(
        token_index=torch.tensor([0, 1, 2]),
        chain_id=torch.tensor([1, 1, 2]),
        bond_index=torch.tensor([[0, 1], [1, 2]]),
        bond_type=torch.tensor([4, 5]),
        pair_features=torch.arange(18, dtype=torch.float32).reshape(3, 3, 2),
    )
    collated = BiomolecularPairContext.collate((first, second))
    torch.testing.assert_close(collated.token_index, torch.tensor([0, 1, 0, 1, 2]))
    torch.testing.assert_close(
        collated.bond_index,
        torch.tensor([[0, 2, 3], [1, 3, 4]]),
    )
    assert collated.pair_features is not None
    torch.testing.assert_close(collated.pair_features[:2, :2], first.pair_features)
    torch.testing.assert_close(collated.pair_features[2:, 2:], second.pair_features)
    assert torch.count_nonzero(collated.pair_features[:2, 2:]) == 0

    moved = collated.to(dtype=torch.float64)
    assert moved.pair_features is not None
    assert moved.pair_features.dtype == torch.float64
    assert moved.token_index is not None and moved.token_index.dtype == torch.long
    assert moved.bond_index is not None and moved.bond_index.dtype == torch.long


def test_biomolecular_context_rejects_inconsistent_metadata() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        BiomolecularPairContext(
            token_index=torch.arange(2),
            bond_index=torch.tensor([[0], [1]]),
        )
    with pytest.raises(ValueError, match="out-of-range"):
        BiomolecularPairContext(
            bond_index=torch.tensor([[0], [3]]),
            bond_type=torch.tensor([1]),
        ).validate(num_nodes=3, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="pair_features"):
        BiomolecularPairContext(pair_features=torch.randn(2, 3, 1))
    invalid = torch.zeros(3, 3, 2)
    invalid[0, 1, 0] = torch.nan
    with pytest.raises(ValueError, match="only finite"):
        BiomolecularPairContext(pair_features=invalid)
