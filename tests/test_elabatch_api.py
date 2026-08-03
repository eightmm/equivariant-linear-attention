from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention import ELA, ELABatch, OrderContext, RefinementRequest


def _complete_edges(nodes: int) -> torch.Tensor:
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    sender = torch.arange(nodes).repeat(nodes)
    return torch.stack([receiver, sender])


def test_single_graph_batch_is_the_only_model_input() -> None:
    torch.manual_seed(3)
    model = ELA(
        input_irreps="4x0e",
        output_irreps="1x0e",
        width=16,
        depth=1,
        cutoff=10.0,
    ).double()
    batch = ELABatch(
        node_irreps=torch.randn(5, 4, dtype=torch.float64),
        positions=torch.randn(5, 3, dtype=torch.float64),
    )
    output = model(batch)
    assert output["node_irreps"].shape == (5, 1)
    assert output["graph_irreps"].shape == (1, 1)
    assert output["node"] is output["node_irreps"]
    assert output["graph"] is output["graph_irreps"]
    assert output["graph_mean"] is output["graph_irreps"]
    assert output["pos"] is output["positions"]
    assert output["delta"] is output["coordinate_delta"]
    torch.testing.assert_close(
        output["graph_sum"],
        output["node_irreps"].sum(dim=0, keepdim=True),
    )
    with pytest.raises(TypeError, match="one ELABatch"):
        model(batch.node_irreps)  # type: ignore[arg-type]


def test_ptr_is_the_canonical_graph_membership() -> None:
    batch = ELABatch(
        node_irreps=torch.randn(7, 3),
        positions=torch.randn(7, 3),
        ptr=torch.tensor([0, 2, 7]),
    )
    assert batch.num_graphs == 2
    assert batch.num_nodes == 7
    torch.testing.assert_close(
        batch.batch,
        torch.tensor([0, 0, 1, 1, 1, 1, 1]),
    )


def test_empty_graph_is_rejected_at_the_public_container_boundary() -> None:
    with pytest.raises(ValueError, match="at least one node"):
        ELABatch(
            node_irreps=torch.empty(0, 3),
            positions=torch.empty(0, 3),
        )


def test_from_flat_accepts_graph_major_batch_vector() -> None:
    result = ELABatch.from_flat(
        torch.randn(6, 2),
        torch.randn(6, 3),
        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
    )
    torch.testing.assert_close(result.ptr, torch.tensor([0, 2, 6]))
    with pytest.raises(ValueError, match="graph-major"):
        ELABatch.from_flat(
            torch.randn(4, 2),
            torch.randn(4, 3),
            batch=torch.tensor([0, 1, 0, 1]),
        )


def test_edges_may_not_cross_ptr_boundaries() -> None:
    with pytest.raises(ValueError, match="cross graph boundaries"):
        ELABatch(
            node_irreps=torch.randn(4, 2),
            positions=torch.randn(4, 3),
            ptr=torch.tensor([0, 2, 4]),
            edge_index=torch.tensor([[0], [3]]),
        )


def test_padded_constructor_packs_and_restores_nodes() -> None:
    torch.manual_seed(5)
    node = torch.randn(2, 4, 3, dtype=torch.float64)
    pos = torch.randn(2, 4, 3, dtype=torch.float64)
    mask = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )
    edges = [
        _complete_edges(2),
        _complete_edges(3),
    ]
    batch = ELABatch.from_padded(
        node,
        pos,
        mask,
        edge_index=edges,
    )
    assert batch.node_irreps.shape == (5, 3)
    torch.testing.assert_close(batch.ptr, torch.tensor([0, 2, 5]))
    packed = torch.arange(5, dtype=torch.float64).unsqueeze(-1)
    restored = batch.restore_nodes(packed)
    assert restored.shape == (2, 4, 1)
    torch.testing.assert_close(restored[mask], packed)
    torch.testing.assert_close(restored[~mask], restored.new_zeros(3, 1))


def test_collate_offsets_edges_and_keeps_training_metadata() -> None:
    samples = [
        {
            "x": torch.randn(2, 3),
            "pos": torch.randn(2, 3),
            "edge_index": _complete_edges(2),
            "y": torch.tensor([1.0]),
            "id": "a",
        },
        {
            "x": torch.randn(3, 3),
            "pos": torch.randn(3, 3),
            "edge_index": _complete_edges(3),
            "y": torch.tensor([2.0]),
            "id": "b",
        },
    ]
    batch = ELABatch.collate(samples)
    assert batch.num_graphs == 2
    assert batch.num_nodes == 5
    assert batch.num_edges == 13
    assert batch.sample_ids == ("a", "b")
    assert batch.target is not None and batch.target.shape == (2, 1)
    assert batch.edge_index is not None
    receiver, sender = batch.edge_index
    assert torch.equal(batch.batch[receiver], batch.batch[sender])


def test_prepare_caches_graph_and_hot_path_matches_forward() -> None:
    torch.manual_seed(7)
    model = ELA(
        input_irreps="4x0e",
        output_irreps="2x0e",
        width=16,
        depth=1,
        cutoff=10.0,
    ).double()
    batch = ELABatch(
        node_irreps=torch.randn(6, 4, dtype=torch.float64),
        positions=torch.randn(6, 3, dtype=torch.float64),
        edge_index=_complete_edges(6),
    )
    prepared = model.prepare(batch)
    assert prepared.is_prepared
    assert model.prepare(prepared) is prepared
    validated = model(batch)
    hot = model.forward_prepared(prepared)
    torch.testing.assert_close(hot["node_irreps"], validated["node_irreps"])
    torch.testing.assert_close(hot["graph_irreps"], validated["graph_irreps"])
    torch.testing.assert_close(hot["graph_sum"], validated["graph_sum"])


def test_optional_context_is_carried_by_batch() -> None:
    torch.manual_seed(11)
    model = ELA(
        input_irreps="4x0e",
        output_irreps="1x0e",
        width=16,
        depth=1,
        cutoff=10.0,
        condition_dim=3,
        order_dim=1,
        coordinate_refinement=True,
    ).double()
    nodes = 5
    batch = ELABatch(
        node_irreps=torch.randn(nodes, 4, dtype=torch.float64),
        positions=torch.randn(nodes, 3, dtype=torch.float64),
        edge_index=_complete_edges(nodes),
        condition=torch.randn(1, 3, dtype=torch.float64),
        order=OrderContext.sequence(torch.arange(nodes)),
        refinement=RefinementRequest(steps=1, max_step=0.1),
    )
    output = model(batch)
    assert output["positions"].shape == batch.positions.shape
    assert output["coordinate_delta"].shape == batch.positions.shape
    torch.testing.assert_close(output["positions"], batch.positions)


def test_batch_device_transfer_preserves_geometry_precision() -> None:
    batch = ELABatch(
        node_irreps=torch.randn(3, 4),
        positions=torch.randn(3, 3),
        target=torch.randn(1, 1),
    )
    moved = batch.to("cpu", dtype=torch.bfloat16)
    assert moved.node_irreps.dtype == torch.bfloat16
    assert moved.target is not None and moved.target.dtype == torch.bfloat16
    assert moved.positions.dtype == torch.float32
