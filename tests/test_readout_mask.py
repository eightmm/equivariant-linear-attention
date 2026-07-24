from __future__ import annotations

import pytest
import torch

from equivariant_attention._egnn_baseline import _StaticEGNNBaseline
from equivariant_attention.benchmarking import GraphSample, collate_graphs
from equivariant_attention.moment import (
    EquivariantAttention,
    EquivariantAttentionConfig,
)
import equivariant_attention.moment as moment


def _samples() -> tuple[GraphSample, GraphSample]:
    first = GraphSample(
        node_feats=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=torch.float64,
        ),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.3, 1.1, -0.2]],
            dtype=torch.float64,
        ),
        target=torch.tensor([1.0], dtype=torch.float64),
        sample_id="first",
        readout_mask=torch.tensor([False, True, True]),
    )
    second = GraphSample(
        node_feats=torch.tensor(
            [[0.5, 0.2], [0.1, 0.8]],
            dtype=torch.float64,
        ),
        pos=torch.tensor(
            [[-0.4, 0.0, 0.1], [0.7, -0.3, 0.2]],
            dtype=torch.float64,
        ),
        target=torch.tensor([2.0], dtype=torch.float64),
        sample_id="second",
        readout_mask=torch.tensor([True, False]),
    )
    return first, second


def _attention(*, readout_mode: str = "mean") -> EquivariantAttention:
    return EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=2,
            hidden_irreps="12x0e + 3x1o",
            output_irreps="2x0e + 1x1o + 1x2e",
            num_layers=2,
            num_heads=3,
            use_key_balancing=False,
            readout_mode=readout_mode,
        )
    ).double()


def test_collate_and_to_preserve_readout_mask() -> None:
    batch = collate_graphs(_samples())

    assert torch.equal(
        batch.readout_mask,
        torch.tensor([False, True, True, True, False]),
    )
    moved = batch.to("cpu", dtype=torch.float32)
    assert moved.readout_mask is not None
    assert moved.readout_mask.dtype == torch.bool
    assert torch.equal(moved.readout_mask, batch.readout_mask)


def test_collate_rejects_mixed_or_empty_readout_masks() -> None:
    first, second = _samples()
    without = GraphSample(
        node_feats=second.node_feats,
        pos=second.pos,
        target=second.target,
        sample_id=second.sample_id,
    )
    with pytest.raises(ValueError, match="readout_mask"):
        collate_graphs((first, without))

    empty = GraphSample(
        node_feats=second.node_feats,
        pos=second.pos,
        target=second.target,
        sample_id=second.sample_id,
        readout_mask=torch.zeros(2, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="at least one"):
        collate_graphs((first, empty))


def test_attention_masked_pool_matches_selected_node_mean() -> None:
    batch = collate_graphs(_samples())
    model = _attention()

    output = model(
        batch.node_feats,
        batch.pos,
        batch=batch.batch,
        readout_mask=batch.readout_mask,
    )

    assert batch.readout_mask is not None
    for graph_index in range(2):
        selected = batch.readout_mask & (batch.batch == graph_index)
        assert torch.allclose(
            output["graph_scalars"][graph_index],
            output["node_scalars"][selected].mean(dim=0),
        )
        assert torch.allclose(
            output["graph_vectors"][graph_index],
            output["node_vectors"][selected].mean(dim=0),
        )
        assert torch.allclose(
            output["graph_tensors"][graph_index],
            output["node_tensors"][selected].mean(dim=0),
        )


def test_readout_mask_is_permutation_consistent_and_default_is_unchanged() -> None:
    batch = collate_graphs(_samples())
    model = _attention()
    assert batch.readout_mask is not None
    reference = model(
        batch.node_feats,
        batch.pos,
        batch=batch.batch,
        readout_mask=batch.readout_mask,
    )
    default = model(batch.node_feats, batch.pos, batch=batch.batch)
    explicit_all = model(
        batch.node_feats,
        batch.pos,
        batch=batch.batch,
        readout_mask=torch.ones_like(batch.readout_mask),
    )
    permutation = torch.tensor([2, 0, 1, 4, 3])
    moved = model(
        batch.node_feats[permutation],
        batch.pos[permutation],
        batch=batch.batch[permutation],
        readout_mask=batch.readout_mask[permutation],
    )

    for key in ("graph_scalars", "graph_vectors", "graph_tensors"):
        assert torch.equal(default[key], explicit_all[key])
        assert torch.allclose(reference[key], moved[key], atol=1e-10, rtol=1e-9)


def test_readout_mask_validation_and_private_egnn_pooling() -> None:
    batch = collate_graphs(_samples())
    model = _StaticEGNNBaseline(
        node_dim=2,
        hidden_dim=8,
        num_layers=2,
    ).double()
    assert batch.readout_mask is not None
    output = model(
        batch.node_feats,
        batch.pos,
        batch=batch.batch,
        readout_mask=batch.readout_mask,
    )
    assert output["graph_scalars"].shape == (2, 1)

    with pytest.raises(TypeError, match="boolean"):
        model(
            batch.node_feats,
            batch.pos,
            batch=batch.batch,
            readout_mask=batch.readout_mask.long(),
        )
    invalid = batch.readout_mask.clone()
    invalid[3] = False
    with pytest.raises(ValueError, match="every graph"):
        model(
            batch.node_feats,
            batch.pos,
            batch=batch.batch,
            readout_mask=invalid,
        )


def test_interaction_readout_is_zero_init_compatible_and_o3_permutation_safe() -> None:
    batch = collate_graphs(_samples())
    assert batch.readout_mask is not None
    torch.manual_seed(1701)
    mean_model = _attention()
    torch.manual_seed(1701)
    interaction_model = _attention(readout_mode="interaction")

    mean_state = mean_model.state_dict()
    interaction_state = interaction_model.state_dict()
    for name in mean_state.keys() & interaction_state.keys():
        assert torch.equal(mean_state[name], interaction_state[name]), name

    mean_output = mean_model(
        batch.node_feats,
        batch.pos,
        batch=batch.batch,
        readout_mask=batch.readout_mask,
    )
    interaction_output = interaction_model(
        batch.node_feats,
        batch.pos,
        batch=batch.batch,
        readout_mask=batch.readout_mask,
    )
    assert torch.equal(
        mean_output["graph_scalars"],
        interaction_output["graph_scalars"],
    )

    with torch.no_grad():
        interaction_model.interaction_readout.output.weight.normal_()
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    orthogonal[:, 0].neg_()
    translation = torch.randn(1, 3, dtype=torch.float64)
    permutation = torch.tensor([2, 0, 1, 4, 3])

    reference = interaction_model(
        batch.node_feats,
        batch.pos,
        batch=batch.batch,
        readout_mask=batch.readout_mask,
    )["graph_scalars"]
    moved = interaction_model(
        batch.node_feats,
        batch.pos @ orthogonal.T + translation,
        batch=batch.batch,
        readout_mask=batch.readout_mask,
    )["graph_scalars"]
    permuted = interaction_model(
        batch.node_feats[permutation],
        batch.pos[permutation],
        batch=batch.batch[permutation],
        readout_mask=batch.readout_mask[permutation],
    )["graph_scalars"]

    assert torch.allclose(moved, reference, atol=1e-9, rtol=1e-9)
    assert torch.allclose(permuted, reference, atol=1e-9, rtol=1e-9)


def test_interaction_readout_requires_ligand_and_pocket_roles() -> None:
    batch = collate_graphs(_samples())
    model = _attention(readout_mode="interaction")

    with pytest.raises(ValueError, match="readout_mask"):
        model(batch.node_feats, batch.pos, batch=batch.batch)
    with pytest.raises(ValueError, match="pocket"):
        model(
            batch.node_feats,
            batch.pos,
            batch=batch.batch,
            readout_mask=torch.ones_like(batch.batch, dtype=torch.bool),
        )


def test_interaction_readout_supports_direct_bfloat16_forward_and_backward() -> None:
    batch = collate_graphs(_samples()).to("cpu", dtype=torch.bfloat16)
    assert batch.readout_mask is not None
    module = moment._InteractionReadout(
        scalars=12,
        output_scalars=2,
        num_rbf=8,
        cutoff=2.5,
        eps=1e-8,
    ).to(dtype=torch.bfloat16)
    with torch.no_grad():
        module.output.weight.normal_()
    scalars = torch.randn(
        batch.node_feats.shape[0],
        12,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    graph_counts = torch.bincount(batch.batch, minlength=2)

    output = module(
        scalars,
        batch.pos.float(),
        batch.batch,
        batch.readout_mask,
        num_graphs=2,
        graph_counts=graph_counts,
        edge_index=None,
        edge_index_is_validated=False,
    )
    output.float().square().sum().backward()

    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert scalars.grad is not None
    assert torch.isfinite(scalars.grad).all()


def test_parity_even_triple_features_preserve_global_reflection() -> None:
    torch.manual_seed(1703)
    polar_moments = torch.randn(4, 6, 3, dtype=torch.float64)
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.linalg.det(orthogonal) > 0:
        orthogonal[:, 0].neg_()

    reference = moment._parity_even_triple_features(polar_moments)
    reflected = moment._parity_even_triple_features(polar_moments @ orthogonal.T)

    assert torch.count_nonzero(reference)
    assert torch.allclose(reflected, reference, atol=1e-12, rtol=1e-12)


def test_readout_mode_validation_and_sum_pooling() -> None:
    with pytest.raises(TypeError, match="readout_mode"):
        _attention(readout_mode=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="readout_mode"):
        _attention(readout_mode="unknown")

    batch = collate_graphs(_samples())
    assert batch.readout_mask is not None
    output = _attention(readout_mode="sum")(
        batch.node_feats,
        batch.pos,
        batch=batch.batch,
        readout_mask=batch.readout_mask,
    )
    for graph_index in range(2):
        selected = batch.readout_mask & (batch.batch == graph_index)
        assert torch.allclose(
            output["graph_scalars"][graph_index],
            output["node_scalars"][selected].sum(dim=0),
        )
