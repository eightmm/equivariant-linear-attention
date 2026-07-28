"""Frozen deterministic-topology contract for cached sparse candidate lists.

The ATOM3D-LBA packets recorded a cross-run topology identity defect: two
processes with identical samples and identical topology code produced
32,303,245 and 32,303,244 directed edges. These tests freeze the repaired
contract so no multi-seed claim consumes a drifting candidate list again.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from equivariant_attention.benchmarking import GraphSample
from equivariant_attention.pdbbind import (
    segment_balanced_knn_edge_index,
    topology_sha256,
)


_CUTOFF = 6.0
_INTRA_K = 16
_CROSS_K = 16


def _reference_geometry(
    node_count: int = 400, *, offset: float = 0.0, seed: int = 7
) -> tuple[torch.Tensor, torch.Tensor]:
    """Float32 coordinates whose spread places many pairs near the cutoff."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.rand((node_count, 3), dtype=torch.float64, generator=generator) * 14.0
    pos = (base + offset).to(dtype=torch.float32)
    segment_mask = torch.zeros(node_count, dtype=torch.bool)
    segment_mask[: node_count // 2] = True
    return pos, segment_mask


def _build(
    pos: torch.Tensor,
    segment_mask: torch.Tensor,
    *,
    intra_k: int = _INTRA_K,
    cross_k: int = _CROSS_K,
    cutoff: float = _CUTOFF,
) -> torch.Tensor:
    return segment_balanced_knn_edge_index(
        pos,
        segment_mask,
        intra_k=intra_k,
        cross_k=cross_k,
        cutoff=cutoff,
    )


def _canonical_codes(edge_index: torch.Tensor, node_count: int) -> torch.Tensor:
    flat = edge_index[0].to(torch.int64) * node_count + edge_index[1].to(torch.int64)
    return torch.sort(flat).values


def _exact_squared_distance(pos: torch.Tensor) -> torch.Tensor:
    geometry = pos.to(dtype=torch.float64)
    delta = geometry.unsqueeze(1) - geometry.unsqueeze(0)
    return delta.square().sum(dim=-1)


def test_topology_is_invariant_to_rigid_translation() -> None:
    """A translated complex must produce exactly the same candidate list.

    A matrix-multiplication Euclidean distance loses accuracy as the
    coordinate magnitude grows, so an untranslated and a translated copy of one
    complex disagreed about boundary pairs.
    """
    node_count = 400
    pos, segment_mask = _reference_geometry(node_count)
    reference = _canonical_codes(_build(pos, segment_mask), node_count)

    for offset in (30.0, 100.0, 300.0):
        shifted, shifted_mask = _reference_geometry(node_count, offset=offset)
        observed = _canonical_codes(_build(shifted, shifted_mask), node_count)
        assert observed.numel() == reference.numel(), f"offset {offset} changed degree"
        assert torch.equal(observed, reference), f"offset {offset} changed topology"


def test_topology_matches_exact_squared_cutoff_when_untruncated() -> None:
    """Without a neighbor budget the retained set is the exact cutoff graph."""
    node_count = 400
    pos, segment_mask = _reference_geometry(node_count, offset=300.0)
    edge_index = _build(pos, segment_mask, intra_k=node_count, cross_k=node_count)

    squared = _exact_squared_distance(pos)
    expected = squared < _CUTOFF * _CUTOFF
    expected.fill_diagonal_(True)
    observed = torch.zeros_like(expected)
    observed[edge_index[0], edge_index[1]] = True
    assert torch.equal(observed, expected)


def test_topology_never_retains_a_pair_outside_the_cutoff() -> None:
    node_count = 400
    pos, segment_mask = _reference_geometry(node_count, offset=300.0)
    receiver, sender = _build(pos, segment_mask)

    nonself = receiver != sender
    squared = _exact_squared_distance(pos)[receiver[nonself], sender[nonself]]
    assert bool((squared < _CUTOFF * _CUTOFF).all())


def test_topology_candidates_survive_the_model_cutoff_filter() -> None:
    """Supplied candidates must not be silently discarded by the layer."""
    node_count = 400
    pos, segment_mask = _reference_geometry(node_count, offset=300.0)
    receiver, sender = _build(pos, segment_mask)

    displacement = pos[sender] - pos[receiver]
    magnitude = displacement.abs().amax(dim=-1, keepdim=True)
    safe = magnitude.clamp_min(torch.finfo(pos.dtype).tiny)
    scaled = displacement / safe
    tiny = torch.finfo(pos.dtype).eps
    distance = (magnitude * scaled.square().sum(dim=-1, keepdim=True).clamp_min(tiny).sqrt()).squeeze(-1)
    assert bool((distance < _CUTOFF).all())


def test_topology_is_permutation_equivariant_for_large_graphs() -> None:
    node_count = 400
    pos, segment_mask = _reference_geometry(node_count, offset=100.0)
    reference = _canonical_codes(_build(pos, segment_mask), node_count)

    generator = torch.Generator().manual_seed(11)
    permutation = torch.randperm(node_count, generator=generator)
    permuted = _build(pos[permutation], segment_mask[permutation])
    restored = _canonical_codes(permutation[permuted], node_count)
    assert restored.numel() == reference.numel()
    assert torch.equal(restored, reference)


def test_topology_is_invariant_to_intra_process_thread_count() -> None:
    node_count = 400
    pos, segment_mask = _reference_geometry(node_count, offset=300.0)
    previous = torch.get_num_threads()
    results: dict[int, torch.Tensor] = {}
    try:
        for threads in (1, 2, 4):
            torch.set_num_threads(threads)
            results[threads] = _build(pos, segment_mask)
    finally:
        torch.set_num_threads(previous)
    assert torch.equal(results[1], results[2])
    assert torch.equal(results[1], results[4])


def test_topology_retains_every_tie_at_the_neighbor_boundary() -> None:
    """Exact ties at the kth boundary are kept, so degree may exceed the budget."""
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    segment_mask = torch.zeros(5, dtype=torch.bool)
    receiver, sender = _build(pos, segment_mask, intra_k=1, cross_k=0, cutoff=2.0)
    neighbors = sender[(receiver == 0) & (sender != 0)]
    assert sorted(int(value) for value in neighbors) == [1, 2, 3, 4]


def test_zero_neighbor_budget_keeps_only_self_edges() -> None:
    pos, segment_mask = _reference_geometry(64)
    receiver, sender = _build(pos, segment_mask, intra_k=0, cross_k=0)
    assert torch.equal(receiver, torch.arange(64))
    assert torch.equal(sender, torch.arange(64))


def test_cross_segment_budget_is_independent_of_intra_segment_budget() -> None:
    node_count = 200
    pos, segment_mask = _reference_geometry(node_count, offset=100.0)
    receiver, sender = _build(pos, segment_mask, intra_k=0, cross_k=4)
    for node in range(node_count):
        neighbors = sender[(receiver == node) & (sender != node)]
        assert bool((segment_mask[neighbors] != segment_mask[node]).all())


def test_topology_hash_is_order_and_content_sensitive() -> None:
    pos, segment_mask = _reference_geometry(64)
    edge_index = _build(pos, segment_mask)
    sample = GraphSample(
        node_feats=torch.zeros((64, 1)),
        pos=pos,
        target=torch.zeros(1),
        sample_id="topology-contract:0",
        edge_index=edge_index,
        readout_mask=segment_mask,
    )
    digest = topology_sha256([sample])
    assert digest == topology_sha256([sample])

    mutated = GraphSample(
        node_feats=sample.node_feats,
        pos=sample.pos,
        target=sample.target,
        sample_id=sample.sample_id,
        edge_index=edge_index[:, :-1],
        readout_mask=sample.readout_mask,
    )
    assert topology_sha256([mutated]) != digest

    renamed = GraphSample(
        node_feats=sample.node_feats,
        pos=sample.pos,
        target=sample.target,
        sample_id="topology-contract:1",
        edge_index=edge_index,
        readout_mask=sample.readout_mask,
    )
    assert topology_sha256([renamed]) != digest


def test_topology_hash_requires_precomputed_edges() -> None:
    sample = GraphSample(
        node_feats=torch.zeros((2, 1)),
        pos=torch.zeros((2, 3)),
        target=torch.zeros(1),
        sample_id="topology-contract:missing",
    )
    with pytest.raises(ValueError, match="precomputed edges"):
        topology_sha256([sample])


_SUBPROCESS_PROGRAM = """
import torch

from equivariant_attention.benchmarking import GraphSample
from equivariant_attention.pdbbind import (
    segment_balanced_knn_edge_index,
    topology_sha256,
)

generator = torch.Generator().manual_seed(7)
base = torch.rand((400, 3), dtype=torch.float64, generator=generator) * 14.0
pos = (base + 300.0).to(dtype=torch.float32)
segment_mask = torch.zeros(400, dtype=torch.bool)
segment_mask[:200] = True
edge_index = segment_balanced_knn_edge_index(
    pos, segment_mask, intra_k=16, cross_k=16, cutoff=6.0
)
sample = GraphSample(
    node_feats=torch.zeros((400, 1)),
    pos=pos,
    target=torch.zeros(1),
    sample_id="topology-contract:subprocess",
    edge_index=edge_index,
)
print(f"{edge_index.shape[1]} {topology_sha256([sample])}")
"""


def test_topology_hash_is_stable_across_fresh_processes() -> None:
    """Fresh processes with different thread budgets must agree bit for bit."""
    observed: list[str] = []
    for threads in ("1", "4"):
        environment = dict(os.environ)
        environment["OMP_NUM_THREADS"] = threads
        environment["MKL_NUM_THREADS"] = threads
        completed = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_PROGRAM],
            capture_output=True,
            check=True,
            env=environment,
            text=True,
        )
        observed.append(completed.stdout.strip())
    assert observed[0] == observed[1], observed
