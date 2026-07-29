from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention.neighbors import PackedNeighborGraph, pack_neighbor_graph


def _config(*, backend: str) -> EquivariantAttentionConfig:
    return EquivariantAttentionConfig(
        node_dim=5,
        hidden_irreps="12x0e + 3x1o",
        output_irreps="1x0e + 1x1o + 1x2e",
        num_layers=2,
        num_heads=3,
        local_head_counts=(0, 0),
        local_cutoff=4.0,
        use_sparse_low_rank_local_residual=True,
        local_residual_rank=3,
        sparse_residual_backend=backend,
        sparse_residual_stream_chunk_size=2,
        distance_band_cutoffs=(1.0, 2.5),
        num_edge_relations=2,
        relation_cutoffs=(2.0, 4.0),
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = (4, 3)
    batch = torch.repeat_interleave(
        torch.arange(len(counts)),
        torch.tensor(counts),
    )
    receiver: list[int] = []
    sender: list[int] = []
    for graph in range(len(counts)):
        nodes = torch.nonzero(batch == graph, as_tuple=False).flatten().tolist()
        for target in nodes:
            for source in reversed(nodes):
                receiver.append(target)
                sender.append(source)
    edge_index = torch.tensor([receiver, sender], dtype=torch.long)
    order = torch.randperm(
        edge_index.shape[1],
        generator=torch.Generator().manual_seed(20260809),
    )
    edge_index = edge_index[:, order]
    relation_id = (
        torch.arange(edge_index.shape[1], dtype=torch.long) % 2
    )
    generator = torch.Generator().manual_seed(20260810)
    features = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    positions = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    return features, positions, batch, edge_index, relation_id


def _activate_sparse_outputs(model: EquivariantAttention) -> None:
    for name, parameter in model.named_parameters():
        if (
            "sparse_low_rank_local_residual" in name
            and name.endswith("_out.weight")
        ):
            torch.nn.init.constant_(parameter, 0.07)


@pytest.mark.parametrize("normalization", ["positive", "softmax"])
@pytest.mark.parametrize("backend", ["segment_csr", "ell", "streamed_csr"])
def test_live_streamed_packed_sparse_path_matches_materialized_without_receiver_index(
    normalization: str,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, positions, batch, edge_index, relation_id = _inputs()
    torch.manual_seed(20260811)
    reference = EquivariantAttention(
        replace(
            _config(backend="materialized"),
            sparse_residual_normalization=normalization,
        )
    ).double()
    torch.manual_seed(20260811)
    candidate_config = replace(
        _config(backend=backend),
        sparse_residual_normalization=normalization,
    )
    candidate = EquivariantAttention(candidate_config).double()
    _activate_sparse_outputs(reference)
    _activate_sparse_outputs(candidate)
    packed = pack_neighbor_graph(
        edge_index,
        num_nodes=features.shape[0],
        edge_relation_id=relation_id,
        build_ell=True,
        ell_max_padding_ratio=16.0,
    )
    reference_features = features.clone().requires_grad_(True)
    candidate_features = features.clone().requires_grad_(True)
    reference_positions = positions.clone().requires_grad_(True)
    candidate_positions = positions.clone().requires_grad_(True)

    expected = reference(
        reference_features,
        reference_positions,
        batch=batch,
        edge_index=edge_index,
        edge_relation_id=relation_id,
    )

    def forbidden(_self: PackedNeighborGraph) -> torch.Tensor:
        raise AssertionError("streamed packed path must not expand receiver_index")

    monkeypatch.setattr(PackedNeighborGraph, "receiver_index", forbidden)
    actual = candidate(
        candidate_features,
        candidate_positions,
        batch=batch,
        packed_neighbors=packed,
    )

    for name in expected:
        torch.testing.assert_close(
            actual[name],
            expected[name],
            rtol=3e-11,
            atol=3e-11,
        )
    expected_parameters = tuple(
        parameter for parameter in reference.parameters() if parameter.requires_grad
    )
    actual_parameters = tuple(
        parameter for parameter in candidate.parameters() if parameter.requires_grad
    )
    expected_gradients = torch.autograd.grad(
        sum(value.square().sum() for value in expected.values()),
        (reference_features, reference_positions, *expected_parameters),
        allow_unused=True,
    )
    actual_gradients = torch.autograd.grad(
        sum(value.square().sum() for value in actual.values()),
        (candidate_features, candidate_positions, *actual_parameters),
        allow_unused=True,
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        if actual_gradient is None or expected_gradient is None:
            assert actual_gradient is expected_gradient
        else:
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                rtol=2e-10,
                atol=2e-10,
            )


def test_streamed_backend_requires_prepacked_receiver_plan() -> None:
    features, positions, batch, edge_index, relation_id = _inputs()
    model = EquivariantAttention(_config(backend="streamed_csr")).double()

    with pytest.raises(ValueError, match="packed_neighbors"):
        model(
            features,
            positions,
            batch=batch,
            edge_index=edge_index,
            edge_relation_id=relation_id,
        )


def test_streamed_bfloat16_projections_keep_geometry_and_reductions_fp32() -> None:
    features, positions, batch, edge_index, relation_id = _inputs()
    packed = pack_neighbor_graph(
        edge_index,
        num_nodes=features.shape[0],
        edge_relation_id=relation_id,
    )
    model = EquivariantAttention(
        _config(backend="streamed_csr")
    ).to(dtype=torch.bfloat16)
    _activate_sparse_outputs(model)
    candidate_features = features.to(dtype=torch.bfloat16).requires_grad_(True)
    candidate_positions = positions.float().requires_grad_(True)

    output = model(
        candidate_features,
        candidate_positions,
        batch=batch,
        packed_neighbors=packed,
    )
    loss = sum(value.float().square().sum() for value in output.values())
    feature_gradient, position_gradient = torch.autograd.grad(
        loss,
        (candidate_features, candidate_positions),
    )

    assert output["node_scalars"].dtype == torch.bfloat16
    assert output["node_vectors"].dtype == torch.bfloat16
    assert feature_gradient.dtype == torch.bfloat16
    assert position_gradient.dtype == torch.float32
    assert torch.isfinite(feature_gradient).all()
    assert torch.isfinite(position_gradient).all()
