from __future__ import annotations

import pytest
import torch
from torch import nn

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention.benchmarking import GraphSample, collate_graphs
from equivariant_attention.training import (
    build_regression_model,
    predict_graph_scalar,
    train_regression_step,
)
import equivariant_attention.moment as moment


def _edge_conditioned_model(
    *,
    num_layers: int = 1,
    normalize_by_sqrt_degree: bool | None = None,
) -> EquivariantAttention:
    normalization_kwargs = (
        {}
        if normalize_by_sqrt_degree is None
        else {
            "normalize_edge_conditioned_local_by_sqrt_degree": (
                normalize_by_sqrt_degree
            )
        }
    )
    return EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            hidden_irreps="8x0e + 2x1o",
            output_irreps="2x0e + 1x1o + 1x2e",
            num_layers=num_layers,
            num_heads=2,
            local_head_counts=(2,) * num_layers,
            use_key_balancing=False,
            use_edge_conditioned_local_transport=True,
            **normalization_kwargs,
        )
    ).double()


def _sparse_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_feats = torch.randn(7, 4, dtype=torch.float64)
    pos = 0.35 * torch.randn(7, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6, 1, 2, 0, 4, 5, 6, 3],
        ]
    )
    return node_feats, pos, batch, edge_index


def test_edge_conditioned_sum_matches_explicit_dense_reference() -> None:
    torch.manual_seed(1211)
    transport = moment._EdgeConditionedLocalTransport(
        scalars=8,
        vectors=2,
        num_heads=2,
        num_rbf=3,
    ).double()
    scalars = torch.randn(5, 8, dtype=torch.float64)
    vectors = torch.randn(5, 2, 3, dtype=torch.float64)
    pos = 0.3 * torch.randn(5, 3, dtype=torch.float64)
    batch = torch.zeros(5, dtype=torch.long)
    edge_index = torch.cartesian_prod(torch.arange(5), torch.arange(5)).T
    geometry = moment._local_geometry(
        pos,
        batch,
        num_graphs=1,
        cutoff=2.5,
        num_rbf=3,
        edge_index=edge_index,
    )

    actual = transport(scalars, vectors, geometry, num_nodes=5)
    receiver, sender, displacement, squared_distance, rbf = geometry
    nonself = receiver != sender
    receiver = receiver[nonself]
    sender = sender[nonself]
    displacement = displacement[nonself]
    squared_distance = squared_distance[nonself]
    rbf = rbf[nonself]
    edge_output = transport.edge_mlp(
        torch.cat([scalars[receiver], scalars[sender], rbf], dim=-1)
    )
    scalar_edge, sender_gate, relative_gate, tensor_gate = torch.split(
        edge_output, [8, 2, 2, 2], dim=-1
    )
    cutoff = moment._cosine_of_squared_distance_cutoff(squared_distance)
    expected = [torch.zeros_like(value) for value in actual]
    for edge in range(receiver.numel()):
        target = int(receiver[edge])
        weight = cutoff[edge]
        expected[0][target] += weight * scalar_edge[edge].reshape(2, 4)
        expected[1][target] += (
            weight * torch.tanh(sender_gate[edge]).unsqueeze(-1) * vectors[sender[edge]]
        )
        expected[2][target] += (
            weight
            * torch.tanh(relative_gate[edge]).unsqueeze(-1)
            * displacement[edge].unsqueeze(0)
        )
        expected[3][target] += (
            weight
            * torch.tanh(tensor_gate[edge]).unsqueeze(-1)
            * moment._symmetric_traceless_features(displacement[edge]).unsqueeze(0)
        )

    for reference, observed in zip(expected, actual, strict=True):
        assert torch.allclose(observed, reference, atol=1e-12, rtol=1e-11)


def test_edge_conditioned_soft_mass_matches_explicit_receiver_reference() -> None:
    torch.manual_seed(1212)
    baseline = moment._EdgeConditionedLocalTransport(
        scalars=8,
        vectors=2,
        num_heads=2,
        num_rbf=3,
        normalize_by_sqrt_degree=False,
    ).double()
    normalized = moment._EdgeConditionedLocalTransport(
        scalars=8,
        vectors=2,
        num_heads=2,
        num_rbf=3,
        normalize_by_sqrt_degree=True,
    ).double()
    normalized.load_state_dict(baseline.state_dict())
    scalars = torch.randn(4, 8, dtype=torch.float64)
    vectors = torch.randn(4, 2, 3, dtype=torch.float64)
    pos = 0.2 * torch.randn(4, 3, dtype=torch.float64)
    batch = torch.zeros(4, dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 0, 1],
            [0, 1, 2, 3, 1, 2, 2],
        ]
    )
    geometry = moment._local_geometry(
        pos,
        batch,
        num_graphs=1,
        cutoff=2.5,
        num_rbf=3,
        edge_index=edge_index,
    )

    raw_outputs = baseline(scalars, vectors, geometry, num_nodes=4)
    actual = normalized(scalars, vectors, geometry, num_nodes=4)
    receiver, sender, _displacement, squared_distance, _rbf = geometry
    nonself = receiver != sender
    receiver = receiver[nonself]
    cutoff = moment._cosine_of_squared_distance_cutoff(squared_distance[nonself])
    cutoff_mass = cutoff.new_zeros(4).index_add(
        0,
        receiver,
        cutoff,
    )

    for raw, observed in zip(raw_outputs, actual, strict=True):
        divisor = (
            (1.0 + cutoff_mass)
            .sqrt()
            .reshape(4, *((1,) * (raw.ndim - 1)))
        )
        assert torch.allclose(observed, raw / divisor, atol=1e-12, rtol=1e-11)
        assert torch.isfinite(observed).all()


def test_edge_conditioned_disabled_option_is_exactly_backward_compatible() -> None:
    torch.manual_seed(1214)
    default_model = _edge_conditioned_model()
    torch.manual_seed(1214)
    explicit_disabled = _edge_conditioned_model(normalize_by_sqrt_degree=False)
    node_feats, pos, batch, edge_index = _sparse_batch()

    default_output = default_model(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index,
    )
    disabled_output = explicit_disabled(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index,
    )

    assert default_model.state_dict().keys() == explicit_disabled.state_dict().keys()
    for name, value in default_model.state_dict().items():
        assert torch.equal(value, explicit_disabled.state_dict()[name])
    for name in default_output:
        assert torch.equal(default_output[name], disabled_output[name])


@pytest.mark.parametrize("normalize_by_sqrt_degree", [False, True])
def test_edge_conditioned_branches_have_finite_nonzero_gradients(
    normalize_by_sqrt_degree: bool,
) -> None:
    torch.manual_seed(1213)
    transport = moment._EdgeConditionedLocalTransport(
        scalars=8,
        vectors=2,
        num_heads=2,
        num_rbf=3,
        normalize_by_sqrt_degree=normalize_by_sqrt_degree,
    ).double()
    scalars = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    vectors = torch.randn(4, 2, 3, dtype=torch.float64, requires_grad=True)
    pos = (0.2 * torch.randn(4, 3, dtype=torch.float64)).requires_grad_()
    batch = torch.zeros(4, dtype=torch.long)
    edge_index = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
    geometry = moment._local_geometry(
        pos,
        batch,
        num_graphs=1,
        cutoff=2.5,
        num_rbf=3,
        edge_index=edge_index,
    )
    outputs = transport(scalars, vectors, geometry, num_nodes=4)
    probes = [torch.randn_like(output) for output in outputs]

    sum((output * probe).sum() for output, probe in zip(outputs, probes)).backward()

    output_gradient = transport.edge_mlp[-1].weight.grad
    assert output_gradient is not None
    for branch in torch.split(output_gradient, [8, 2, 2, 2], dim=0):
        assert torch.isfinite(branch).all()
        assert torch.count_nonzero(branch) > 0
    for value in (scalars.grad, vectors.grad, pos.grad):
        assert value is not None and torch.isfinite(value).all()
        assert torch.count_nonzero(value) > 0


@pytest.mark.parametrize("normalize_by_sqrt_degree", [False, True])
def test_edge_conditioned_public_path_preserves_o3_permutation_and_edge_order(
    normalize_by_sqrt_degree: bool,
) -> None:
    torch.manual_seed(1217)
    model = _edge_conditioned_model(normalize_by_sqrt_degree=normalize_by_sqrt_degree)
    node_feats, pos, batch, edge_index = _sparse_batch()
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    orthogonal[:, 0].neg_()
    translation = torch.randn(1, 3, dtype=torch.float64)
    permutation = torch.tensor([4, 0, 6, 2, 3, 1, 5])
    inverse = torch.argsort(permutation)

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
        edge_index=inverse[edge_index],
    )
    reordered = model(
        node_feats,
        pos,
        batch=batch,
        edge_index=edge_index[:, torch.arange(edge_index.shape[1] - 1, -1, -1)],
    )

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
        assert torch.allclose(reordered[name], reference[name], atol=1e-12)


@pytest.mark.parametrize("normalize_by_sqrt_degree", [False, True])
def test_edge_conditioned_batch_preserves_graph_relabel_frames_and_isolation(
    normalize_by_sqrt_degree: bool,
) -> None:
    torch.manual_seed(1219)
    model = _edge_conditioned_model(normalize_by_sqrt_degree=normalize_by_sqrt_degree)
    node_feats, pos, batch, edge_index = _sparse_batch()
    first_orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    second_orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    first_orthogonal[:, 0].neg_()
    orthogonal = torch.stack([first_orthogonal, second_orthogonal])
    translation = torch.randn(2, 3, dtype=torch.float64)
    node_orthogonal = orthogonal[batch]
    moved_pos = torch.einsum("na,nba->nb", pos, node_orthogonal) + translation[batch]

    reference = model(node_feats, pos, batch=batch, edge_index=edge_index)
    moved = model(node_feats, moved_pos, batch=batch, edge_index=edge_index)
    relabeled = model(
        node_feats,
        pos,
        batch=1 - batch,
        edge_index=edge_index,
    )
    changed_feats = node_feats.clone()
    changed_feats[batch == 0] += torch.randn_like(changed_feats[batch == 0])
    changed = model(changed_feats, pos, batch=batch, edge_index=edge_index)

    assert torch.allclose(moved["node_scalars"], reference["node_scalars"], atol=1e-9)
    assert torch.allclose(
        moved["node_vectors"],
        torch.einsum(
            "nca,nba->ncb",
            reference["node_vectors"],
            node_orthogonal,
        ),
        atol=1e-9,
    )
    assert torch.allclose(
        moved["node_tensors"],
        torch.einsum(
            "nab,nkbc,ndc->nkad",
            node_orthogonal,
            reference["node_tensors"],
            node_orthogonal,
        ),
        atol=1e-9,
    )
    assert torch.allclose(
        moved["graph_scalars"],
        reference["graph_scalars"],
        atol=1e-9,
    )
    assert torch.allclose(
        moved["graph_vectors"],
        torch.einsum(
            "gca,gba->gcb",
            reference["graph_vectors"],
            orthogonal,
        ),
        atol=1e-9,
    )
    assert torch.allclose(
        moved["graph_tensors"],
        torch.einsum(
            "gab,gkbc,gdc->gkad",
            orthogonal,
            reference["graph_tensors"],
            orthogonal,
        ),
        atol=1e-9,
    )

    graph_reorder = torch.tensor([1, 0])
    for name in ("node_scalars", "node_vectors", "node_tensors"):
        assert torch.allclose(relabeled[name], reference[name], atol=1e-9)
        assert torch.allclose(
            changed[name][batch == 1],
            reference[name][batch == 1],
            atol=1e-9,
        )
    for name in ("graph_scalars", "graph_vectors", "graph_tensors"):
        assert torch.allclose(
            relabeled[name][graph_reorder],
            reference[name],
            atol=1e-9,
        )
        assert torch.allclose(changed[name][1], reference[name][1], atol=1e-9)


class _EdgeProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.received_edge_index: torch.Tensor | None = None
        self.received_edge_index_is_validated = False

    def forward(
        self,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None = None,
        *,
        edge_index: torch.Tensor | None = None,
        edge_index_is_validated: bool = False,
    ) -> dict[str, torch.Tensor]:
        del pos
        self.received_edge_index = edge_index
        self.received_edge_index_is_validated = edge_index_is_validated
        assert batch is not None
        graph_count = int(batch.max().item()) + 1
        return {"graph_scalars": node_feats.new_zeros((graph_count, 1))}


def test_prediction_forwards_exact_sparse_edges() -> None:
    node_feats, pos, batch, edge_index = _sparse_batch()
    first_edges = edge_index[:, edge_index[0] < 3]
    second_edges = edge_index[:, edge_index[0] >= 3] - 3
    graph_batch = collate_graphs(
        [
            GraphSample(
                node_feats=node_feats[:3],
                pos=pos[:3],
                target=torch.zeros(1, dtype=torch.float64),
                sample_id="a",
                edge_index=first_edges,
            ),
            GraphSample(
                node_feats=node_feats[3:],
                pos=pos[3:],
                target=torch.zeros(1, dtype=torch.float64),
                sample_id="b",
                edge_index=second_edges,
            ),
        ]
    )
    probe = _EdgeProbe()

    prediction = predict_graph_scalar(probe, graph_batch)

    assert prediction.shape == (2, 1)
    assert probe.received_edge_index is graph_batch.edge_index
    assert probe.received_edge_index_is_validated


def test_training_with_supplied_edges_never_calls_complete_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1223)
    node_feats, pos, _batch, edge_index = _sparse_batch()
    first_edges = edge_index[:, edge_index[0] < 3]
    second_edges = edge_index[:, edge_index[0] >= 3] - 3
    samples = [
        GraphSample(
            node_feats=node_feats[:3].float(),
            pos=pos[:3].float(),
            target=torch.tensor([0.25]),
            sample_id="a",
            edge_index=first_edges,
        ),
        GraphSample(
            node_feats=node_feats[3:].float(),
            pos=pos[3:].float(),
            target=torch.tensor([-0.5]),
            sample_id="b",
            edge_index=second_edges,
        ),
    ]
    graph_batch = collate_graphs(samples)
    model = build_regression_model(
        node_dim=4,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        local_head_counts=(2,),
        use_key_balancing=False,
        use_edge_conditioned_local_transport=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def fail_discovery(*args: object, **kwargs: object) -> tuple[torch.Tensor, ...]:
        raise AssertionError("complete-pair discovery must be bypassed")

    monkeypatch.setattr(moment, "_batched_complete_graph_edges", fail_discovery)
    monkeypatch.setattr(moment, "_validated_local_edge_index", fail_discovery)

    loss = train_regression_step(model, graph_batch, optimizer)

    assert torch.isfinite(torch.tensor(loss))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"local_head_counts": (0,)},
        {
            "local_head_counts": (2,),
            "use_pairwise_local_content": True,
        },
        {
            "local_head_counts": (2,),
            "learn_local_radial_gate": True,
        },
    ],
)
def test_edge_conditioned_config_rejects_inactive_or_ambiguous_routes(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="edge-conditioned"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=4,
                hidden_irreps="8x0e + 2x1o",
                num_layers=1,
                num_heads=2,
                use_edge_conditioned_local_transport=True,
                **kwargs,
            )
        )


def test_edge_conditioned_degree_normalization_requires_edge_transport() -> None:
    with pytest.raises(ValueError, match="degree normalization"):
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=4,
                hidden_irreps="8x0e + 2x1o",
                num_layers=1,
                num_heads=2,
                local_head_counts=(2,),
                normalize_edge_conditioned_local_by_sqrt_degree=True,
            )
        )
