from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention import ELA, ELABatch
import equivariant_linear_attention.kernels.local as optimized_local
from equivariant_linear_attention.kernels.triton import csr_sum_many


def _complete_edges(nodes: int) -> torch.Tensor:
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    sender = torch.arange(nodes).repeat(nodes)
    return torch.stack([receiver, sender])


def test_grouped_local_path_matches_reference_values_and_input_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(311)
    model = ELA(
        input_irreps="4x0e",
        output_irreps="2x0e",
        width=16,
        depth=1,
        cutoff=10.0,
        num_rbf=6,
    ).double()
    nodes = 5
    features = torch.randn(nodes, 4, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    edges = _complete_edges(nodes)

    reference_features = features.clone().requires_grad_(True)
    reference_positions = positions.clone().requires_grad_(True)
    reference_batch = model.prepare(
        ELABatch(reference_features, reference_positions, edge_index=edges)
    )
    reference = model.forward_prepared(reference_batch)["node_irreps"]
    reference_gradients = torch.autograd.grad(
        reference.square().sum(),
        (reference_features, reference_positions),
    )

    monkeypatch.setattr(
        optimized_local,
        "active_backend",
        lambda *_args, **_kwargs: "triton",
    )
    monkeypatch.setattr(
        optimized_local,
        "_trusted_csr_sum_many",
        lambda values, row_ptr, *, policy, receiver=None: csr_sum_many(
            values,
            row_ptr,
        ),
    )
    candidate_features = features.clone().requires_grad_(True)
    candidate_positions = positions.clone().requires_grad_(True)
    candidate_batch = model.prepare(
        ELABatch(candidate_features, candidate_positions, edge_index=edges)
    )
    candidate = model.forward_prepared(candidate_batch)["node_irreps"]
    candidate_gradients = torch.autograd.grad(
        candidate.square().sum(),
        (candidate_features, candidate_positions),
    )

    torch.testing.assert_close(candidate, reference, atol=2e-12, rtol=2e-12)
    for actual, expected in zip(
        candidate_gradients,
        reference_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
