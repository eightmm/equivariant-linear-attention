from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention import ELA
from equivariant_linear_attention.batch import ELABatch
import equivariant_linear_attention.kernels.local as optimized_local
from equivariant_linear_attention.kernels.triton import csr_sum_many
from equivariant_linear_attention.nn.parity import _st_from_vector


def _complete_edges(nodes: int) -> torch.Tensor:
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    sender = torch.arange(nodes).repeat(nodes)
    return torch.stack([receiver, sender])


def _weighted_pair_reference(
    weight: torch.Tensor,
    radial_gate: torch.Tensor,
    source0: torch.Tensor,
    source1: torch.Tensor,
    sender: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    gate_lanes: tuple[int, int],
    policy: str,
    receiver: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del policy, receiver
    output0, output1 = csr_sum_many(
        (
            weight.unsqueeze(-1)
            * radial_gate[..., gate_lanes[0], None]
            * source0[sender].reshape(sender.shape[0], source0.shape[1], -1),
            weight.unsqueeze(-1)
            * radial_gate[..., gate_lanes[1], None]
            * source1[sender].reshape(sender.shape[0], source1.shape[1], -1),
        ),
        row_ptr,
    )
    rows = row_ptr.numel() - 1
    return (
        output0.reshape(rows, *source0.shape[1:]),
        output1.reshape(rows, *source1.shape[1:]),
    )


def _tensor_pair_reference(
    weight: torch.Tensor,
    radial_gate: torch.Tensor,
    even_source: torch.Tensor,
    odd_source: torch.Tensor,
    direction: torch.Tensor,
    sender: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    gate_lanes: tuple[int, int],
    policy: str,
    receiver: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del policy, receiver
    return csr_sum_many(
        (
            weight.unsqueeze(-1)
            * radial_gate[..., gate_lanes[0], None]
            * (even_source[sender] + _st_from_vector(direction).unsqueeze(1)),
            weight.unsqueeze(-1)
            * radial_gate[..., gate_lanes[1], None]
            * odd_source[sender],
        ),
        row_ptr,
    )


def _direction_triple_reference(
    weight: torch.Tensor,
    radial_gate: torch.Tensor,
    direction_gate: torch.Tensor,
    direction: torch.Tensor,
    sender: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    gate_lanes: tuple[int, int, int],
    policy: str,
    receiver: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del policy, receiver
    return csr_sum_many(
        tuple(
            weight.unsqueeze(-1)
            * radial_gate[..., lane, None]
            * direction_gate[sender, moment].unsqueeze(-1)
            * direction.unsqueeze(1)
            for moment, lane in enumerate(gate_lanes)
        ),
        row_ptr,
    )  # type: ignore[return-value]


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
    ).double()
    nodes = 5
    features = torch.randn(nodes, 4, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    edges = _complete_edges(nodes)

    reference_features = features.clone().requires_grad_(True)
    reference_positions = positions.clone().requires_grad_(True)
    reference_batch = model._prepare_packed(
        ELABatch(reference_features, reference_positions, edge_index=edges)
    )
    reference = model._forward_prepared(reference_batch)["node_irreps"]
    reference_gradients = torch.autograd.grad(
        reference.square().sum(),
        (reference_features, reference_positions),
    )

    observed_csr_payload_shapes: list[tuple[tuple[int, ...], ...]] = []

    def compact_csr_sum_many(
        values: tuple[torch.Tensor, ...],
        row_ptr: torch.Tensor,
        *,
        policy: str,
        receiver: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        del policy, receiver
        shapes = tuple(tuple(value.shape) for value in values)
        observed_csr_payload_shapes.append(shapes)
        assert all(value.ndim == 2 for value in values)
        return csr_sum_many(values, row_ptr)

    monkeypatch.setattr(
        optimized_local,
        "active_backend",
        lambda *_args, **_kwargs: "triton",
    )
    monkeypatch.setattr(
        optimized_local,
        "_trusted_csr_sum_many",
        compact_csr_sum_many,
    )
    monkeypatch.setattr(
        optimized_local,
        "_trusted_weighted_gather_reduce_pair",
        _weighted_pair_reference,
    )
    monkeypatch.setattr(
        optimized_local,
        "_trusted_local_tensor_reduce_pair",
        _tensor_pair_reference,
    )
    monkeypatch.setattr(
        optimized_local,
        "_trusted_direction_reduce_triple",
        _direction_triple_reference,
    )
    candidate_features = features.clone().requires_grad_(True)
    candidate_positions = positions.clone().requires_grad_(True)
    candidate_batch = model._prepare_packed(
        ELABatch(candidate_features, candidate_positions, edge_index=edges)
    )
    candidate = model._forward_prepared(candidate_batch)["node_irreps"]
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
    assert len(observed_csr_payload_shapes) == 1
    assert observed_csr_payload_shapes[0] == ((25, model.layers[0].local_rank),) * 2


def _install_reference_launchers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the fused dispatch while replacing each launch with torch math."""

    monkeypatch.setattr(
        optimized_local,
        "active_backend",
        lambda *_args, **_kwargs: "triton",
    )
    monkeypatch.setattr(
        optimized_local,
        "_trusted_csr_sum_many",
        lambda values, row_ptr, **_kwargs: csr_sum_many(values, row_ptr),
    )
    monkeypatch.setattr(
        optimized_local,
        "_trusted_weighted_gather_reduce_pair",
        _weighted_pair_reference,
    )
    monkeypatch.setattr(
        optimized_local,
        "_trusted_local_tensor_reduce_pair",
        _tensor_pair_reference,
    )
    monkeypatch.setattr(
        optimized_local,
        "_trusted_direction_reduce_triple",
        _direction_triple_reference,
    )


@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("edge_types", [0, 2])
def test_fused_dispatch_tracks_the_reference_across_widths_and_relations(
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    edge_types: int,
) -> None:
    """Pin kernels/local.py to nn/core.py's canonical equations.

    ``dispatch_local_message`` restates the score, gate and value equations of
    ``_torch_local_message`` rather than sharing them, so the two can drift
    silently. The original guard covered one width with one topology and no
    relations; ``local_rank`` changes with width, which is also what selects
    the fused kernels' multi-feature-block path, so the widths are covered
    explicitly here.
    """

    torch.manual_seed(900 + width + edge_types)
    model = ELA(
        input_irreps="4x0e",
        output_irreps="2x0e",
        width=width,
        depth=1,
        cutoff=10.0,
        edge_types=edge_types,
    ).double()
    nodes = 6
    features = torch.randn(nodes, 4, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    edges = _complete_edges(nodes)
    relation = (
        torch.randint(0, edge_types, (edges.shape[1],)) if edge_types else None
    )

    def run() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        node = features.clone().requires_grad_(True)
        pos = positions.clone().requires_grad_(True)
        batch = model._prepare_packed(
            ELABatch(node, pos, edge_index=edges, edge_relation_id=relation)
        )
        value = model._forward_prepared(batch)["node_irreps"]
        return value, torch.autograd.grad(value.square().sum(), (node, pos))

    reference, reference_grads = run()
    _install_reference_launchers(monkeypatch)
    candidate, candidate_grads = run()

    torch.testing.assert_close(candidate, reference, atol=2e-12, rtol=2e-12)
    for actual, expected in zip(candidate_grads, reference_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)


def test_fused_dispatch_tracks_the_reference_on_a_sparse_radius_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The original guard used one complete graph; ragged rows differ."""

    torch.manual_seed(917)
    model = ELA(
        input_irreps="4x0e",
        output_irreps="2x0e",
        width=64,
        depth=1,
        cutoff=2.5,
    ).double()
    nodes = 24
    features = torch.randn(nodes, 4, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64) * 2.0

    def run() -> torch.Tensor:
        node = features.clone().requires_grad_(True)
        pos = positions.clone().requires_grad_(True)
        batch = model._prepare_packed(ELABatch(node, pos))
        return model._forward_prepared(batch)["node_irreps"]

    reference = run()
    _install_reference_launchers(monkeypatch)
    torch.testing.assert_close(run(), reference, atol=2e-12, rtol=2e-12)
