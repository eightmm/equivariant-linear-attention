from __future__ import annotations

import pytest
import torch

from equivariant_attention import SpatialOperatorRegressionModel


def _inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    node_feats = torch.randn(6, 4)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
        ]
    )
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    local = torch.cartesian_prod(torch.arange(3), torch.arange(3)).T
    edge_index = torch.cat([local, local + 3], dim=1)
    readout_mask = torch.tensor([True, False, True, False, True, True])
    return node_feats, pos, batch, edge_index, readout_mask


def _model(arm: str) -> SpatialOperatorRegressionModel:
    return SpatialOperatorRegressionModel(
        arm=arm,
        node_dim=4,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        local_rank=4,
        local_cutoff=2.5,
        implicit_chunk_size=4,
    )


@pytest.mark.parametrize("arm", ["explicit", "implicit", "hybrid"])
def test_spatial_regression_arms_share_schema_and_run_ligand_mask_readout(
    arm: str,
) -> None:
    torch.manual_seed(741)
    node_feats, pos, batch, edge_index, readout_mask = _inputs()
    model = _model(arm)
    output = model(
        node_feats,
        pos,
        batch=batch,
        edge_index=(None if arm == "implicit" else edge_index),
        readout_mask=readout_mask,
    )

    assert output["graph_scalars"].shape == (2, 1)
    assert torch.isfinite(output["graph_scalars"]).all()
    expected = torch.stack(
        [
            output["node_irreps"][:3][readout_mask[:3]].mean(dim=0),
            output["node_irreps"][3:][readout_mask[3:]].mean(dim=0),
        ]
    )
    torch.testing.assert_close(output["graph_scalars"], expected)
    assert sum(parameter.numel() for parameter in model.parameters()) == 27_267


def test_spatial_regression_arms_have_identical_state_schema() -> None:
    schemas = {
        arm: tuple(_model(arm).state_dict())
        for arm in ("explicit", "implicit", "hybrid")
    }

    assert schemas["explicit"] == schemas["implicit"] == schemas["hybrid"]


@pytest.mark.parametrize("arm", ["explicit", "hybrid"])
def test_sparse_spatial_arms_require_edges(arm: str) -> None:
    node_feats, pos, batch, _, _ = _inputs()

    with pytest.raises(ValueError, match="requires sparse edge_index"):
        _model(arm)(node_feats, pos, batch=batch)
