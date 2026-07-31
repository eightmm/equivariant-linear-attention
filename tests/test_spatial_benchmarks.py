from __future__ import annotations

import torch

from equivariant_attention.spatial_benchmarks import (
    _pair_target,
    make_synthetic_spatial_batch,
    synthetic_batch_sha256,
)


def test_synthetic_batch_is_deterministic_and_graph_isolated() -> None:
    first = make_synthetic_spatial_batch(
        task="mixed",
        num_graphs=3,
        nodes_per_graph=7,
        seed=5,
        dtype=torch.float64,
    )
    second = make_synthetic_spatial_batch(
        task="mixed",
        num_graphs=3,
        nodes_per_graph=7,
        seed=5,
        dtype=torch.float64,
    )
    assert synthetic_batch_sha256(first) == synthetic_batch_sha256(second)
    receiver, sender = first.edge_index
    assert torch.equal(first.batch[receiver], first.batch[sender])


def test_all_synthetic_targets_are_rotation_and_translation_invariant() -> None:
    torch.manual_seed(7)
    nodes = 9
    scalar = torch.randn(nodes, 4, dtype=torch.float64)
    polar = torch.randn(nodes, 3, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.linalg.det(orthogonal) < 0:
        orthogonal[:, 0] = -orthogonal[:, 0]
    translation = torch.tensor([1.2, -0.4, 0.8], dtype=torch.float64)

    for task in ("local_directional", "smooth_gaussian", "mixed"):
        reference = _pair_target(
            scalar,
            polar,
            positions,
            task=task,
            cutoff=1.75,
            gaussian_scale=2.5,
        )
        transformed = _pair_target(
            scalar,
            polar @ orthogonal.T,
            positions @ orthogonal.T + translation,
            task=task,
            cutoff=1.75,
            gaussian_scale=2.5,
        )
        torch.testing.assert_close(
            transformed,
            reference,
            atol=2e-10,
            rtol=2e-10,
        )


def test_local_target_ignores_pairs_beyond_cutoff() -> None:
    scalar = torch.tensor(
        [[1.0, 0.5], [2.0, -1.0], [100.0, 100.0]],
        dtype=torch.float64,
    )
    polar = torch.tensor(
        [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [100.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    with_far_node = _pair_target(
        scalar,
        polar,
        positions,
        task="local_directional",
        cutoff=1.0,
        gaussian_scale=2.5,
    )
    without_far_node = _pair_target(
        scalar[:2],
        polar[:2],
        positions[:2],
        task="local_directional",
        cutoff=1.0,
        gaussian_scale=2.5,
    )
    # The implementation normalizes by graph size; undo that normalization to
    # compare the underlying local pair sum.
    torch.testing.assert_close(
        3.0 * with_far_node,
        2.0 * without_far_node,
        atol=1e-12,
        rtol=1e-12,
    )
