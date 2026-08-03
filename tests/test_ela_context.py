from __future__ import annotations

import pytest
import torch

from equivariant_attention import ELA, ELABatch, OrderContext, RefinementRequest


def _complete_edges(nodes: int) -> torch.Tensor:
    receiver = torch.arange(nodes).repeat_interleave(nodes)
    sender = torch.arange(nodes).repeat(nodes)
    return torch.stack([receiver, sender])


def _fixture(
    *,
    nodes: int = 6,
    condition_dim: int = 0,
    order_dim: int = 0,
    coordinate_refinement: bool = False,
) -> tuple[ELA, torch.Tensor, torch.Tensor, torch.Tensor]:
    model = ELA(
        input_irreps="4x0e",
        output_irreps="1x0e + 1x1o",
        width=16,
        depth=2,
        cutoff=10.0,
        num_rbf=8,
        condition_dim=condition_dim,
        order_dim=order_dim,
        coordinate_refinement=coordinate_refinement,
    ).double()
    node_irreps = torch.randn(nodes, 4, dtype=torch.float64)
    positions = torch.randn(nodes, 3, dtype=torch.float64)
    return model, node_irreps, positions, _complete_edges(nodes)


def _batch(
    node: torch.Tensor,
    pos: torch.Tensor,
    edges: torch.Tensor,
    **kwargs: object,
) -> ELABatch:
    return ELABatch(node, pos, edge_index=edges, **kwargs)


def test_optional_context_is_neutral_at_initialization() -> None:
    torch.manual_seed(3)
    model, node, pos, edges = _fixture(condition_dim=5, order_dim=1)
    model.eval()
    rank = torch.arange(node.shape[0], dtype=torch.float64)
    with torch.inference_mode():
        reference = model(_batch(node, pos, edges))["node_irreps"]
        conditioned = model(
            _batch(
                node,
                pos,
                edges,
                condition=torch.randn(1, 5, dtype=torch.float64),
                order=OrderContext.sequence(rank),
            )
        )["node_irreps"]
    torch.testing.assert_close(conditioned, reference, atol=0.0, rtol=0.0)


def test_context_free_batch_bypasses_trained_conditioner() -> None:
    torch.manual_seed(4)
    model, node, pos, edges = _fixture(condition_dim=5)
    model.eval()
    with torch.inference_mode():
        reference = model(_batch(node, pos, edges))["node_irreps"]
    for layer in model.layers:
        assert layer.conditioner is not None
        with torch.no_grad():
            output = layer.conditioner.projection[-1]
            output.weight.normal_(mean=0.0, std=0.1)
            output.bias.normal_(mean=0.0, std=0.1)
    with torch.inference_mode():
        bypassed = model(_batch(node, pos, edges))["node_irreps"]
        active = model(
            _batch(
                node,
                pos,
                edges,
                condition=torch.randn(1, 5, dtype=torch.float64),
            )
        )["node_irreps"]
    torch.testing.assert_close(bypassed, reference, atol=0.0, rtol=0.0)
    assert not torch.allclose(active, reference)


def test_conditioner_projection_receives_first_step_gradient() -> None:
    torch.manual_seed(5)
    model, node, pos, edges = _fixture(condition_dim=5)
    output = model(
        _batch(
            node,
            pos,
            edges,
            condition=torch.randn(1, 5, dtype=torch.float64),
        )
    )
    output["node_irreps"].square().mean().backward()
    conditioner = model.layers[0].conditioner
    assert conditioner is not None
    final = conditioner.projection[-1]
    assert final.weight.grad is not None
    assert torch.isfinite(final.weight.grad).all()
    assert torch.count_nonzero(final.weight.grad) > 0


def test_order_encoder_ignores_disabled_node_labels() -> None:
    torch.manual_seed(7)
    model, node, pos, edges = _fixture(order_dim=1)
    prepared = model.prepare(_batch(node, pos, edges))
    assert model.order_encoder is not None
    assert prepared._prepared_graph is not None
    enabled = torch.tensor([True, True, True, False, False, False])
    first_rank = torch.arange(node.shape[0], dtype=torch.float64)
    second_rank = first_rank.clone()
    second_rank[~enabled] = first_rank.new_tensor([100.0, -50.0, 999.0])
    first = model.order_encoder(
        OrderContext.sequence(first_rank, enabled=enabled),
        prepared._prepared_graph,
    )
    second = model.order_encoder(
        OrderContext.sequence(second_rank, enabled=enabled),
        prepared._prepared_graph,
    )
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first[~enabled], torch.zeros_like(first[~enabled]))


def test_activated_order_conditioning_preserves_permutation_equivariance() -> None:
    torch.manual_seed(9)
    model, node, pos, edges = _fixture(order_dim=1)
    model.eval()
    for layer in model.layers:
        assert layer.conditioner is not None
        with torch.no_grad():
            output = layer.conditioner.projection[-1]
            output.weight.normal_(mean=0.0, std=0.05)
            output.bias.normal_(mean=0.0, std=0.05)

    nodes = node.shape[0]
    rank = torch.arange(nodes, dtype=torch.float64)
    permutation = torch.randperm(nodes)
    old_to_new = torch.empty_like(permutation)
    old_to_new[permutation] = torch.arange(nodes)
    with torch.inference_mode():
        reference = model(
            _batch(node, pos, edges, order=OrderContext.sequence(rank))
        )["node_irreps"]
        actual = model(
            _batch(
                node[permutation],
                pos[permutation],
                old_to_new[edges],
                order=OrderContext.sequence(rank[permutation]),
            )
        )["node_irreps"]
    torch.testing.assert_close(
        actual,
        reference[permutation],
        atol=4e-8,
        rtol=4e-8,
    )


def test_refinement_identity_mask_and_bound() -> None:
    torch.manual_seed(11)
    model, node, pos, edges = _fixture(coordinate_refinement=True)
    selected = torch.tensor([True, True, True, False, False, False])
    request = RefinementRequest(
        steps=2,
        max_step=0.1,
        centering="selected",
        update_mask=selected,
    )
    identity = model(_batch(node, pos, edges, refinement=request))
    torch.testing.assert_close(identity["positions"], pos)

    assert model.coordinate_head is not None
    with torch.no_grad():
        model.coordinate_head.base_weight.fill_(0.2)
    updated = model(_batch(node, pos, edges, refinement=request))
    delta = updated["coordinate_delta"]
    assert torch.isfinite(delta).all()
    torch.testing.assert_close(delta[~selected], torch.zeros_like(delta[~selected]))
    assert torch.linalg.vector_norm(delta, dim=-1).max() <= 0.2 + 1e-12
    torch.testing.assert_close(
        delta[selected].mean(dim=0),
        torch.zeros(3, dtype=delta.dtype),
        atol=2e-9,
        rtol=2e-9,
    )


def test_context_fields_fail_closed_when_not_enabled() -> None:
    model, node, pos, edges = _fixture()
    rank = torch.arange(node.shape[0], dtype=torch.float64)
    with pytest.raises(ValueError, match="order_dim"):
        model(_batch(node, pos, edges, order=OrderContext.sequence(rank)))
    with pytest.raises(ValueError, match="condition_dim"):
        model(
            _batch(
                node,
                pos,
                edges,
                condition=torch.randn(1, 3, dtype=torch.float64),
            )
        )
    with pytest.raises(ValueError, match="coordinate_refinement"):
        model(
            _batch(
                node,
                pos,
                edges,
                refinement=RefinementRequest(),
            )
        )
