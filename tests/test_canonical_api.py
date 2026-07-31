from __future__ import annotations

import torch

from equivariant_attention.canonical import ELA, ELAConfig, SparseGeometry
from equivariant_attention.equivariant_linear_attention import (
    EquivariantLinearAttention,
)
from equivariant_attention.unified import prepare_3d_graph


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
    assert advanced.residual_dropout == 0.0
    assert advanced.drop_path_rate == 0.0


def test_canonical_contract_has_no_implicit_or_schedule_option() -> None:
    contract = ELAConfig(input_irreps="4x0e").canonical_contract()
    assert contract["spatial_policy"] == (
        "exact_global_linear_attention_plus_exact_sparse_short_range"
    )
    assert contract["implicit_spatial"] == "experimental_not_canonical"
    assert "implicit_every" not in contract["public_options"]
    assert "attention_residual_blocks" not in contract["public_options"]


def test_minimal_model_forward_backward() -> None:
    torch.manual_seed(13)
    config = ELAConfig(
        input_irreps="4x0e",
        output_irreps="1x0e + 1x1o",
        width=32,
        depth=2,
        geometry=SparseGeometry(cutoff=10.0, num_rbf=8),
    )
    model = ELA(config).double()
    nodes = 6
    features = torch.randn(nodes, 4, dtype=torch.float64, requires_grad=True)
    positions = torch.randn(nodes, 3, dtype=torch.float64, requires_grad=True)
    graph = prepare_3d_graph(
        torch.zeros(nodes, dtype=torch.long),
        _complete_edges(nodes),
    )
    output = model(features, positions, graph)
    loss = output["node_irreps"].square().mean()
    loss.backward()

    assert torch.isfinite(output["node_irreps"]).all()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert all(hasattr(layer, "branch_fusion") for layer in model.layers)


def test_zero_initialized_router_reproduces_advanced_model_with_shared_weights() -> None:
    torch.manual_seed(17)
    minimal = ELAConfig(
        input_irreps="4x0e",
        output_irreps="1x0e + 1x1o",
        width=32,
        depth=2,
        geometry=SparseGeometry(cutoff=10.0, num_rbf=8),
    )
    control = EquivariantLinearAttention(minimal.to_advanced_config()).double()
    candidate = ELA(minimal).double()
    receipt = candidate.load_state_dict(control.state_dict(), strict=False)
    assert not receipt.unexpected_keys
    assert receipt.missing_keys
    assert all("branch_fusion" in key for key in receipt.missing_keys)

    nodes = 6
    features = torch.randn(nodes, 4, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    graph = prepare_3d_graph(
        torch.zeros(nodes, dtype=torch.long),
        _complete_edges(nodes),
    )
    control.eval()
    candidate.eval()
    with torch.inference_mode():
        expected = control(features, positions, graph)
        actual = candidate(features, positions, graph)
    torch.testing.assert_close(
        actual["node_irreps"],
        expected["node_irreps"],
        atol=2e-10,
        rtol=2e-10,
    )
    torch.testing.assert_close(
        actual["graph_irreps"],
        expected["graph_irreps"],
        atol=2e-10,
        rtol=2e-10,
    )
