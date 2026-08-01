from __future__ import annotations

import torch

from equivariant_attention import (
    ELA,
    ELAConfig,
    ELAContext,
    ELAFeatures,
    OrderContext,
    RefinementRequest,
    SparseGeometry,
)


def _complete_edges(nodes: int) -> torch.Tensor:
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    sender = torch.arange(nodes).repeat(nodes)
    return torch.stack([receiver, sender])


def _fixture(
    *,
    nodes: int = 6,
    features: ELAFeatures = ELAFeatures(),
) -> tuple[ELA, torch.Tensor, torch.Tensor, object]:
    config = ELAConfig(
        input_irreps="4x0e",
        output_irreps="1x0e + 1x1o",
        width=16,
        depth=2,
        geometry=SparseGeometry(cutoff=10.0, num_rbf=8),
        features=features,
    )
    model = ELA(config).double()
    node_irreps = torch.randn(nodes, 4, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    batch = torch.zeros(nodes, dtype=torch.long)
    graph = config.geometry.prepare(batch, _complete_edges(nodes))
    return model, node_irreps, positions, graph


def test_optional_context_is_neutral_at_initialization() -> None:
    torch.manual_seed(3)
    model, node_irreps, positions, graph = _fixture(
        features=ELAFeatures(condition_dim=5, order_dim=1),
    )
    model.eval()
    rank = torch.arange(node_irreps.shape[0], dtype=torch.float64)
    context = ELAContext(
        condition=torch.randn(1, 5, dtype=torch.float64),
        order=OrderContext.sequence(rank),
    )

    with torch.inference_mode():
        reference = model(node_irreps, positions, graph)["node_irreps"]
        conditioned = model(
            node_irreps,
            positions,
            graph,
            context=context,
        )["node_irreps"]
    torch.testing.assert_close(conditioned, reference, atol=0.0, rtol=0.0)


def test_order_encoder_ignores_disabled_node_labels() -> None:
    torch.manual_seed(5)
    model, node_irreps, _, graph = _fixture(
        features=ELAFeatures(order_dim=1),
    )
    assert model.order_encoder is not None
    enabled = torch.tensor([True, True, True, False, False, False])
    first_rank = torch.arange(node_irreps.shape[0], dtype=torch.float64)
    second_rank = first_rank.clone()
    second_rank[~enabled] = torch.tensor([100.0, -50.0, 999.0])

    first = model.order_encoder(
        OrderContext.sequence(first_rank, enabled=enabled),
        graph,
    )
    second = model.order_encoder(
        OrderContext.sequence(second_rank, enabled=enabled),
        graph,
    )
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        first[~enabled, :-1],
        torch.zeros_like(first[~enabled, :-1]),
        atol=0.0,
        rtol=0.0,
    )


def test_activated_order_conditioning_preserves_node_permutation_equivariance() -> None:
    torch.manual_seed(7)
    model, node_irreps, positions, graph = _fixture(
        features=ELAFeatures(order_dim=1),
    )
    model.eval()
    for layer in model.layers:
        assert layer.conditioner is not None
        with torch.no_grad():
            output = layer.conditioner.projection[-1]
            output.weight.normal_(mean=0.0, std=0.05)
            output.bias.normal_(mean=0.0, std=0.05)

    nodes = node_irreps.shape[0]
    rank = torch.arange(nodes, dtype=torch.float64)
    permutation = torch.randperm(nodes)
    old_to_new = torch.empty_like(permutation)
    old_to_new[permutation] = torch.arange(nodes)
    original_edges = _complete_edges(nodes)
    permuted_edges = old_to_new[original_edges]
    permuted_graph = model.config.geometry.prepare(
        torch.zeros(nodes, dtype=torch.long),
        permuted_edges,
    )

    with torch.inference_mode():
        reference = model(
            node_irreps,
            positions,
            graph,
            context=ELAContext(order=OrderContext.sequence(rank)),
        )["node_irreps"]
        actual = model(
            node_irreps[permutation],
            positions[permutation],
            permuted_graph,
            context=ELAContext(
                order=OrderContext.sequence(rank[permutation])
            ),
        )["node_irreps"]
    torch.testing.assert_close(
        actual,
        reference[permutation],
        atol=4e-8,
        rtol=4e-8,
    )


def test_refinement_is_identity_at_initialization_and_respects_mask_and_bound() -> None:
    torch.manual_seed(11)
    model, node_irreps, positions, graph = _fixture(
        features=ELAFeatures(coordinate_refinement=True),
    )
    selected = torch.tensor([True, True, True, False, False, False])
    request = RefinementRequest(
        steps=2,
        max_step=0.1,
        centering="selected",
        update_mask=selected,
    )

    identity = model(
        node_irreps,
        positions,
        graph,
        context=ELAContext(refinement=request),
    )
    torch.testing.assert_close(
        identity["positions"],
        positions,
        atol=0.0,
        rtol=0.0,
    )

    assert model.coordinate_head is not None
    with torch.no_grad():
        model.coordinate_head.base_weight.fill_(0.2)
    updated = model(
        node_irreps,
        positions,
        graph,
        context=ELAContext(refinement=request),
    )
    delta = updated["coordinate_delta"]
    assert torch.isfinite(delta).all()
    torch.testing.assert_close(
        delta[~selected],
        torch.zeros_like(delta[~selected]),
        atol=0.0,
        rtol=0.0,
    )
    assert torch.linalg.vector_norm(delta, dim=-1).max() <= 0.2 + 1e-12
    torch.testing.assert_close(
        delta[selected].mean(dim=0),
        torch.zeros(3, dtype=delta.dtype),
        atol=2e-9,
        rtol=2e-9,
    )


def test_context_fields_fail_closed_when_not_enabled() -> None:
    model, node_irreps, positions, graph = _fixture()
    rank = torch.arange(node_irreps.shape[0], dtype=torch.float64)

    try:
        model(
            node_irreps,
            positions,
            graph,
            context=ELAContext(order=OrderContext.sequence(rank)),
        )
    except ValueError as error:
        assert "order_dim" in str(error)
    else:
        raise AssertionError("disabled order conditioning must fail closed")
