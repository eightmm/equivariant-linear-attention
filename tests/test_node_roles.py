from __future__ import annotations

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention.benchmarking import GraphSample, collate_graphs
from equivariant_attention.training import predict_graph_scalar


def _model(*, num_node_roles: int) -> EquivariantAttention:
    return EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            hidden_irreps="16x0e + 2x1o",
            output_irreps="1x0e",
            num_layers=2,
            num_heads=2,
            num_node_roles=num_node_roles,
        )
    ).double()


def test_node_roles_are_functional_invariant_scalar_inputs() -> None:
    torch.manual_seed(20260826)
    model = _model(num_node_roles=3)
    features = torch.randn(4, 4, dtype=torch.float64)
    positions = torch.randn(4, 3, dtype=torch.float64)
    batch = torch.zeros(4, dtype=torch.long)
    first_roles = torch.tensor([0, 1, 2, 1])
    second_roles = torch.tensor([2, 1, 0, 1])

    first = model(
        features,
        positions,
        batch=batch,
        node_role_id=first_roles,
    )
    second = model(
        features,
        positions,
        batch=batch,
        node_role_id=second_roles,
    )
    translation = torch.tensor([2.0, -1.0, 3.0], dtype=torch.float64)
    translated = model(
        features,
        positions + translation,
        batch=batch,
        node_role_id=first_roles,
    )

    assert not torch.allclose(first["graph_scalars"], second["graph_scalars"])
    torch.testing.assert_close(
        translated["graph_scalars"],
        first["graph_scalars"],
        rtol=1e-12,
        atol=1e-12,
    )


def test_node_role_contract_rejects_missing_disabled_and_out_of_range_ids() -> None:
    features = torch.randn(3, 4, dtype=torch.float64)
    positions = torch.randn(3, 3, dtype=torch.float64)
    batch = torch.zeros(3, dtype=torch.long)
    enabled = _model(num_node_roles=2)
    disabled = _model(num_node_roles=0)

    with pytest.raises(ValueError, match="requires node_role_id"):
        enabled(features, positions, batch=batch)
    with pytest.raises(ValueError, match="num_node_roles"):
        disabled(
            features,
            positions,
            batch=batch,
            node_role_id=torch.zeros(3, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="lie in"):
        enabled(
            features,
            positions,
            batch=batch,
            node_role_id=torch.tensor([0, 1, 2]),
        )


def test_role_annotations_flow_through_graph_batch_prediction() -> None:
    sample = GraphSample(
        node_feats=torch.randn(3, 4, dtype=torch.float64),
        pos=torch.randn(3, 3, dtype=torch.float64),
        target=torch.tensor([0.0], dtype=torch.float64),
        sample_id="generic-role-sample",
        node_role_id=torch.tensor([0, 1, 0]),
    )
    batch = collate_graphs([sample])
    model = _model(num_node_roles=2)

    observed = predict_graph_scalar(model, batch)
    expected = model(
        batch.node_feats,
        batch.pos,
        batch=batch.batch,
        graph_layout=batch.graph_layout,
        node_role_id=batch.node_role_id,
    )["graph_scalars"]

    torch.testing.assert_close(observed, expected)
