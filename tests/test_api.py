from __future__ import annotations

import inspect

import pytest
import torch

import equivariant_linear_attention as ela
from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.advanced import ELAConfig, ELAFeatures


def test_public_surface_and_contract() -> None:
    assert set(ela.__all__) == {"ELA", "ELAGraph"}
    model = ELA("4x0e", "2x0e + 1x1o", width=32, depth=2)
    description = model.describe()
    assert description["public_contract"] == "ELAGraph -> ELA -> ELAGraph"
    assert description["explicit_edges"] is False
    assert description["relative_moment_order"] == 4
    assert description["transient_irreps"] == ("3o", "4e")
    assert "edge-free" in inspect.getdoc(ELA).lower()
    assert "input_irreps='4x0e'" in repr(model)


def test_forward_graph_outputs_and_sum_mean() -> None:
    torch.manual_seed(11)
    model = ELA("3x0e", "2x0e", width=32, depth=2).double()
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    graph = ELAGraph(
        torch.randn(5, 3, dtype=torch.float64),
        torch.randn(5, 3, dtype=torch.float64),
        batch=batch,
    )
    output = model(graph)
    assert isinstance(output, ELAGraph)
    assert output.x.shape == (5, 2)
    assert output.graph_x is not None and output.graph_x.shape == (2, 2)
    assert output.graph_sum is not None and output.graph_sum.shape == (2, 2)
    expected_sum = torch.stack((output.x[:3].sum(0), output.x[3:].sum(0)))
    expected_mean = torch.stack((output.x[:3].mean(0), output.x[3:].mean(0)))
    torch.testing.assert_close(output.graph_sum, expected_sum)
    torch.testing.assert_close(output.graph_x, expected_mean)
    torch.testing.assert_close(output.pos, graph.pos)
    assert output.delta is not None
    torch.testing.assert_close(output.delta, torch.zeros_like(graph.pos))


def test_graph_rejects_edge_fields_by_construction() -> None:
    with pytest.raises(TypeError, match="edge_index"):
        ELAGraph(  # type: ignore[call-arg]
            x=torch.randn(3, 2),
            pos=torch.randn(3, 3),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
        )
    with pytest.raises(TypeError, match="cutoff"):
        ELA("2x0e", cutoff=5.0)  # type: ignore[call-arg]


def test_collate_condition_order_and_masks() -> None:
    first = ELAGraph(
        x=torch.randn(3, 2),
        pos=torch.randn(3, 3),
        condition=torch.randn(1, 4),
        order=torch.randn(3, 1),
        update_mask=torch.tensor([True, False, True]),
        y=torch.randn(1, 1),
        ids=("a",),
    )
    second = ELAGraph(
        x=torch.randn(2, 2),
        pos=torch.randn(2, 3),
        condition=torch.randn(4),
        order=torch.randn(2, 1),
        update_mask=torch.tensor([False, True]),
        y=torch.randn(1, 1),
        ids=("b",),
    )
    graph = ELAGraph.collate((first, second))
    assert graph.x.shape == (5, 2)
    torch.testing.assert_close(graph.batch, torch.tensor([0, 0, 0, 1, 1]))
    assert graph.condition is not None and graph.condition.shape == (2, 4)
    assert graph.order is not None and graph.order.shape == (5, 1)
    assert graph.update_mask is not None and graph.update_mask.sum() == 3
    assert graph.y is not None and graph.y.shape == (2, 1)
    assert graph.ids == ("a", "b")

    model = ELA(
        "2x0e",
        "1x0e",
        width=32,
        depth=1,
        condition_dim=4,
        order_dim=1,
    )
    output = model(graph)
    assert output.x.shape == (5, 1)


def test_config_roundtrip() -> None:
    config = ELAConfig(
        input_irreps="3x0e",
        output_irreps="1x0e",
        width=48,
        depth=3,
        features=ELAFeatures(condition_dim=2, order_dim=1),
        update_positions=True,
        max_coordinate_step=0.1,
    )
    model = ELA.from_config(config)
    assert model.config == config
    assert model.updates_positions
