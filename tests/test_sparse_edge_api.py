import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
import equivariant_attention.moment as moment


def _local_model(*, num_layers: int = 2) -> EquivariantAttention:
    return EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            hidden_irreps="8x0e + 2x1o",
            output_irreps="2x0e + 1x1o + 1x2e",
            num_layers=num_layers,
            num_heads=2,
            local_head_counts=(2,) * num_layers,
            use_key_balancing=False,
        )
    ).double()


def _complete_edge_index(batch: torch.Tensor) -> torch.Tensor:
    receiver, sender = moment._batched_complete_graph_edges(
        batch, torch.bincount(batch)
    )
    return torch.stack([receiver, sender])


def _probed_loss(
    outputs: dict[str, torch.Tensor],
    probes: dict[str, torch.Tensor],
) -> torch.Tensor:
    return sum((outputs[name] * probes[name]).sum() for name in sorted(outputs))


def test_precomputed_complete_edges_match_fallback_forward_and_gradients() -> None:
    torch.manual_seed(1201)
    model = _local_model()
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    edge_index = _complete_edge_index(batch)
    node_feats = torch.randn(7, 4, dtype=torch.float64)
    pos = 0.25 * torch.randn(7, 3, dtype=torch.float64)

    fallback_feats = node_feats.detach().requires_grad_()
    fallback_pos = pos.detach().requires_grad_()
    sparse_feats = node_feats.detach().requires_grad_()
    sparse_pos = pos.detach().requires_grad_()
    fallback = model(fallback_feats, fallback_pos, batch=batch)
    sparse = model(
        sparse_feats,
        sparse_pos,
        batch=batch,
        edge_index=edge_index,
    )
    probes = {name: torch.randn_like(value) for name, value in fallback.items()}

    for name in fallback:
        assert torch.allclose(fallback[name], sparse[name], atol=1e-11, rtol=1e-10)
    fallback_gradients = torch.autograd.grad(
        _probed_loss(fallback, probes), (fallback_feats, fallback_pos)
    )
    sparse_gradients = torch.autograd.grad(
        _probed_loss(sparse, probes), (sparse_feats, sparse_pos)
    )
    for expected, actual in zip(fallback_gradients, sparse_gradients, strict=True):
        assert torch.allclose(expected, actual, atol=1e-10, rtol=1e-9)


def test_precomputed_edges_bypass_discovery_and_apply_existing_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1203)
    model = _local_model(num_layers=1)
    node_feats = torch.randn(3, 4, dtype=torch.float64)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    self_edges = torch.arange(3).repeat(2, 1)
    candidates = torch.tensor(
        [[0, 1, 2, 0, 1, 2], [0, 1, 2, 1, 2, 0]], dtype=torch.long
    )

    def fail_discovery(*args: object, **kwargs: object) -> tuple[torch.Tensor, ...]:
        raise AssertionError("complete-pair discovery must be bypassed")

    monkeypatch.setattr(moment, "_batched_complete_graph_edges", fail_discovery)
    expected = model(node_feats, pos, edge_index=self_edges)
    actual = model(node_feats, pos, edge_index=candidates)

    for name in expected:
        assert torch.equal(expected[name], actual[name])


@pytest.mark.parametrize(
    ("edge_index", "message"),
    [
        (torch.tensor([0, 1, 2]), "shape"),
        (torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]), "integer"),
        (torch.tensor([[0, 1, 2], [0, 1, 3]]), "range"),
        (torch.tensor([[0, 1, 2], [0, 1, -1]]), "nonnegative"),
        (
            torch.tensor([[0, 1, 2, 0], [0, 1, 2, 2]]),
            "same graph",
        ),
        (
            torch.tensor([[0, 0, 1, 2], [0, 0, 1, 2]]),
            "duplicate",
        ),
        (torch.tensor([[0, 2], [0, 2]]), "self edge"),
    ],
)
def test_precomputed_edge_validation(
    edge_index: torch.Tensor,
    message: str,
) -> None:
    model = _local_model(num_layers=1)
    node_feats = torch.randn(3, 4, dtype=torch.float64)
    pos = torch.randn(3, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 1])

    with pytest.raises((TypeError, ValueError), match=message):
        model(node_feats, pos, batch=batch, edge_index=edge_index)


def test_precomputed_edges_must_share_the_model_device() -> None:
    model = _local_model(num_layers=1)
    edge_index = torch.empty((2, 0), dtype=torch.long, device="meta")

    with pytest.raises(ValueError, match="same device"):
        model(
            torch.randn(2, 4, dtype=torch.float64),
            torch.randn(2, 3, dtype=torch.float64),
            edge_index=edge_index,
        )


def test_precomputed_edges_are_rejected_without_local_heads() -> None:
    model = EquivariantAttention(
        EquivariantAttentionConfig(node_dim=4, num_layers=1, num_heads=2)
    ).double()

    with pytest.raises(ValueError, match="local heads"):
        model(
            torch.randn(2, 4, dtype=torch.float64),
            torch.randn(2, 3, dtype=torch.float64),
            edge_index=torch.tensor([[0, 1], [0, 1]]),
        )


def test_sparse_edges_preserve_o3_translation_permutation_and_batch_isolation() -> None:
    torch.manual_seed(1207)
    model = _local_model(num_layers=1)
    node_feats = torch.randn(7, 4, dtype=torch.float64)
    pos = 0.4 * torch.randn(7, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6, 1, 2, 0, 4, 5, 6, 3],
        ]
    )
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    orthogonal[:, 0].neg_()
    translation = torch.randn(1, 3, dtype=torch.float64)
    permutation = torch.tensor([2, 0, 1, 6, 4, 3, 5])
    inverse = torch.argsort(permutation)
    permuted_edges = inverse[edge_index]

    reference = model(node_feats, pos, batch=batch, edge_index=edge_index)
    moved = model(
        node_feats,
        pos @ orthogonal.T + translation,
        batch=batch,
        edge_index=edge_index,
    )
    permuted = model(
        node_feats[permutation],
        pos[permutation],
        batch=batch[permutation],
        edge_index=permuted_edges,
    )
    changed_feats = node_feats.clone()
    changed_feats[3:] += 2.0
    changed = model(changed_feats, pos, batch=batch, edge_index=edge_index)

    assert torch.allclose(moved["node_scalars"], reference["node_scalars"], atol=1e-9)
    assert torch.allclose(
        moved["node_vectors"],
        torch.einsum("nca,ba->ncb", reference["node_vectors"], orthogonal),
        atol=1e-9,
    )
    expected_tensor = torch.einsum(
        "ab,nkbc,dc->nkad", orthogonal, reference["node_tensors"], orthogonal
    )
    assert torch.allclose(moved["node_tensors"], expected_tensor, atol=1e-9)
    for name in ("node_scalars", "node_vectors", "node_tensors"):
        assert torch.allclose(permuted[name][inverse], reference[name], atol=1e-9)
        assert torch.allclose(changed[name][:3], reference[name][:3], atol=1e-12)
