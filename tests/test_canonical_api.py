from __future__ import annotations

import torch

from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.advanced import ELAConfig, ELAFeatures, SparseGeometry


def _complete_edges(nodes: int) -> torch.Tensor:
    sender = torch.arange(nodes).repeat(nodes)
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    return torch.stack([sender, receiver])


def test_minimal_config_derives_internal_execution_options() -> None:
    config = ELAConfig(
        input_irreps="4x0e + 1x1o",
        output_irreps="1x0e",
        width=64,
        depth=3,
        geometry=SparseGeometry(cutoff=6.0),
    )
    assert config.num_heads == 4
    assert config.local_rank == 4
    advanced = config.to_advanced_config()
    assert advanced.hidden_dim == 64
    assert advanced.num_layers == 3
    assert advanced.condition_dim == 0
    assert advanced.coordinate_updates is False


def test_optional_features_allocate_one_stack_and_coordinate_head() -> None:
    config = ELAConfig(
        input_irreps="4x0e",
        width=64,
        features=ELAFeatures(condition_dim=8, order_dim=1),
        coordinate_updates=1,
    )
    model = ELA.from_config(config)
    assert all(
        layer.condition_dim == config.features.total_condition_dim(64)
        for layer in model.layers
    )
    assert model.coordinate_head is not None
    assert model.updates_positions


def test_canonical_contract_has_no_alternative_spatial_architecture() -> None:
    contract = ELAConfig(input_irreps="4x0e").canonical_contract()
    assert contract["spatial_policy"] == (
        "exact_global_linear_attention_plus_exact_sparse_short_range"
    )
    assert contract["implicit_spatial"] == "not_in_canonical_architecture"
    assert contract["attention_residuals"] == "not_in_canonical_architecture"
    assert contract["message_fusion"] == "fixed_exact_global_plus_local_sum"
    assert contract["public_contract"] == "ELAGraph -> ELA -> ELAGraph"


def test_minimal_public_forward_backward_returns_one_graph_type() -> None:
    torch.manual_seed(13)
    model = ELA(
        input_irreps="4x0e",
        output_irreps="1x0e + 1x1o",
        width=32,
        depth=2,
        cutoff=10.0,
    ).double()
    nodes = 6
    features = torch.randn(nodes, 4, dtype=torch.float64, requires_grad=True)
    positions = torch.randn(nodes, 3, dtype=torch.float64, requires_grad=True)
    graph = ELAGraph(
        x=features,
        pos=positions,
        edge_index=_complete_edges(nodes),
    )
    output = model(graph)
    output.x.square().mean().backward()

    assert isinstance(output, ELAGraph)
    assert output.x.shape == (nodes, 4)
    assert output.graph_x is not None and output.graph_x.shape == (1, 4)
    assert output.graph_sum is not None and output.graph_sum.shape == (1, 4)
    assert output.delta is not None
    assert torch.isfinite(output.x).all()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert all(not hasattr(layer, "branch_fusion") for layer in model.layers)
    assert not any("branch_fusion" in key for key in model.state_dict())
