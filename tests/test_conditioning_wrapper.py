from __future__ import annotations

import pytest
import torch

from equivariant_attention.canonical import ELA, ELAConfig, SparseGeometry
from equivariant_attention.conditioning import (
    ConditionedELA,
    InvariantConditioningConfig,
)
from equivariant_attention.unified import prepare_3d_graph


def _complete_edges(nodes: int) -> torch.Tensor:
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    sender = torch.arange(nodes).repeat(nodes)
    return torch.stack([receiver, sender])


def _config() -> ELAConfig:
    return ELAConfig(
        input_irreps="4x0e",
        output_irreps="1x0e + 1x1o",
        width=32,
        depth=2,
        geometry=SparseGeometry(cutoff=10.0, num_rbf=8),
    )


def test_zero_initialized_conditioner_reproduces_unconditioned_ela() -> None:
    torch.manual_seed(23)
    base = ELA(_config()).double()
    conditioned = ConditionedELA(
        _config(),
        InvariantConditioningConfig(condition_dim=5),
    ).double()
    receipt = conditioned.load_state_dict(base.state_dict(), strict=False)
    assert not receipt.unexpected_keys
    assert receipt.missing_keys
    assert all("conditioner" in key for key in receipt.missing_keys)

    nodes = 6
    features = torch.randn(nodes, 4, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    condition = torch.randn(1, 5, dtype=torch.float64)
    graph = prepare_3d_graph(
        torch.zeros(nodes, dtype=torch.long),
        _complete_edges(nodes),
    )
    base.eval()
    conditioned.eval()
    with torch.inference_mode():
        expected = base(features, positions, graph)
        actual = conditioned(
            features,
            positions,
            graph,
            condition=condition,
        )
    torch.testing.assert_close(
        actual["node_irreps"],
        expected["node_irreps"],
        atol=2e-10,
        rtol=2e-10,
    )


def test_conditioner_projection_receives_first_step_gradient() -> None:
    torch.manual_seed(29)
    model = ConditionedELA(
        _config(),
        InvariantConditioningConfig(condition_dim=5),
    ).double()
    nodes = 6
    features = torch.randn(nodes, 4, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    condition = torch.randn(1, 5, dtype=torch.float64)
    graph = prepare_3d_graph(
        torch.zeros(nodes, dtype=torch.long),
        _complete_edges(nodes),
    )
    output = model(features, positions, graph, condition=condition)
    output["node_irreps"].square().mean().backward()

    final = model.layers[0].conditioner.projection[-1]
    assert final.weight.grad is not None
    assert torch.isfinite(final.weight.grad).all()
    assert torch.count_nonzero(final.weight.grad) > 0


@pytest.mark.parametrize(
    ("condition", "error", "message"),
    [
        ([0.0] * 5, TypeError, "tensor"),
        (torch.ones(1, 5, dtype=torch.long), TypeError, "floating-point"),
        (
            torch.full((1, 5), float("nan"), dtype=torch.float64),
            ValueError,
            "finite",
        ),
    ],
)
def test_conditioner_rejects_invalid_condition_values(
    condition: object,
    error: type[Exception],
    message: str,
) -> None:
    model = ConditionedELA(
        _config(),
        InvariantConditioningConfig(condition_dim=5),
    ).double()
    nodes = 4
    features = torch.randn(nodes, 4, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    graph = prepare_3d_graph(
        torch.zeros(nodes, dtype=torch.long),
        _complete_edges(nodes),
    )

    with pytest.raises(error, match=message):
        model(features, positions, graph, condition=condition)
