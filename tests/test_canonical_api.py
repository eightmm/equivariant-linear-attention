from __future__ import annotations

import torch

from equivariant_linear_attention import (
    ELA,
    ELABatch,
    ELAConfig,
    ELAFeatures,
    SparseGeometry,
)


def _complete_edges(nodes: int) -> torch.Tensor:
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    sender = torch.arange(nodes).repeat(nodes)
    return torch.stack([receiver, sender])


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


def test_optional_features_allocate_one_layer_stack() -> None:
    config = ELAConfig(
        input_irreps="4x0e",
        width=64,
        features=ELAFeatures(
            condition_dim=8,
            order_dim=1,
            coordinate_refinement=True,
        ),
    )
    model = ELA(config)
    assert all(
        layer.condition_dim == config.features.total_condition_dim(64)
        for layer in model.layers
    )
    assert model.coordinate_head is not None


def test_canonical_contract_has_no_alternative_spatial_architecture() -> None:
    contract = ELAConfig(input_irreps="4x0e").canonical_contract()
    assert contract["spatial_policy"] == (
        "exact_global_linear_attention_plus_exact_sparse_short_range"
    )
    assert contract["implicit_spatial"] == "not_in_canonical_architecture"
    assert contract["attention_residuals"] == "not_in_canonical_architecture"


def test_minimal_model_forward_backward() -> None:
    torch.manual_seed(13)
    model = ELA(
        input_irreps="4x0e",
        output_irreps="1x0e + 1x1o",
        width=32,
        depth=2,
        cutoff=10.0,
        num_rbf=8,
    ).double()
    nodes = 6
    features = torch.randn(nodes, 4, dtype=torch.float64, requires_grad=True)
    positions = torch.randn(nodes, 3, dtype=torch.float64, requires_grad=True)
    batch = ELABatch(
        node_irreps=features,
        positions=positions,
        edge_index=_complete_edges(nodes),
    )
    output = model(batch)
    output["node_irreps"].square().mean().backward()

    assert torch.isfinite(output["node_irreps"]).all()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert all(hasattr(layer, "branch_fusion") for layer in model.layers)
