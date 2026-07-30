from __future__ import annotations

import torch

from equivariant_attention import (
    GraphSample,
    UnifiedRegressionModel,
    collate_graphs,
    train_regression_step,
)


def _sample(sample_id: str, offset: float) -> GraphSample:
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.1, 1.0, 0.3]]
    ) + offset
    edge = torch.cartesian_prod(torch.arange(3), torch.arange(3)).T
    return GraphSample(
        node_feats=torch.randn(3, 4),
        pos=pos,
        target=torch.tensor([offset]),
        sample_id=sample_id,
        edge_index=edge,
        readout_mask=torch.tensor([True, False, True]),
    )


def test_unified_regression_runs_shared_graph_batch_contract() -> None:
    torch.manual_seed(732)
    batch = collate_graphs([_sample("a", 0.0), _sample("b", 0.5)])
    model = UnifiedRegressionModel(
        node_dim=4,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        local_rank=2,
        local_cutoff=3.0,
    )
    output = model(
        batch.node_feats,
        batch.pos,
        batch=batch.batch,
        edge_index=batch.edge_index,
        edge_index_is_validated=True,
        readout_mask=batch.readout_mask,
    )

    assert output["graph_scalars"].shape == (2, 1)
    assert torch.isfinite(output["graph_scalars"]).all()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_regression_step(model, batch, optimizer)
    assert torch.isfinite(torch.tensor(loss))
