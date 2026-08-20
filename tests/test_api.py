from __future__ import annotations

import pytest
import torch

import equivariant_linear_attention as ela
from equivariant_linear_attention import (
    BiomolecularPairContext,
    ELAGraph,
    TriELA,
    TriELAConfig,
    TriELAOutput,
)


def _tiny_model(
    input_irreps: str = "3x0e",
    output_irreps: str = "2x0e",
    **overrides: object,
) -> TriELA:
    arguments: dict[str, object] = {
        "width": 16,
        "pair_width": 8,
        "triangle_hidden": 8,
        "num_stages": 1,
        "pair_blocks_per_stage": 1,
        "local_blocks_per_stage": 1,
        "pair_transition_factor": 2,
        "pair_dropout": 0.0,
        "local_points": 3,
        "max_pair_tokens": 8,
        "distance_rbf_bins": 4,
        "distogram_bins": 6,
    }
    arguments.update(overrides)
    return TriELA(input_irreps, output_irreps, **arguments)  # type: ignore[arg-type]


def test_public_surface_and_single_canonical_contract() -> None:
    assert set(ela.__all__) == {
        "BiomolecularPairContext",
        "DensePairState",
        "ELAGraph",
        "TriELA",
        "TriELAConfig",
        "TriELAOutput",
    }
    model = _tiny_model("4x0e", "2x0e + 1x1o")
    description = model.describe()
    assert description["model"] == "TriELA"
    assert description["public_contract"] == (
        "ELAGraph + optional BiomolecularPairContext -> TriELA -> ELAGraph"
    )
    assert description["pair_backend"] == "dense_exact"
    assert description["explicit_dense_pair"] is True
    assert description["legacy_path"] is False
    assert description["fallback_backend"] is False
    assert description["triangle_block"] == "outgoing_then_incoming_then_swiglu"
    assert "TriELA" in repr(model)


def test_tiny_full_model_forward_auxiliary_shapes_and_backward() -> None:
    generator = torch.Generator().manual_seed(11)
    model = _tiny_model().double().train()
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    x = torch.randn(5, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    pos = torch.randn(
        5, 3, generator=generator, dtype=torch.float64, requires_grad=True
    )
    output = model.forward_with_aux(ELAGraph(x, pos, batch=batch))

    assert isinstance(output, TriELAOutput)
    assert output.graph.x.shape == (5, 2)
    assert output.graph.graph_x is not None and output.graph.graph_x.shape == (2, 2)
    assert output.graph.graph_sum is not None and output.graph.graph_sum.shape == (2, 2)
    assert output.pair_state.z.shape == (2, 3, 3, 8)
    assert output.distogram_logits.shape == (2, 3, 3, 6)
    expected_sum = torch.stack((output.graph.x[:3].sum(0), output.graph.x[3:].sum(0)))
    expected_mean = torch.stack(
        (output.graph.x[:3].mean(0), output.graph.x[3:].mean(0))
    )
    torch.testing.assert_close(output.graph.graph_sum, expected_sum)
    torch.testing.assert_close(output.graph.graph_x, expected_mean)

    loss = output.graph.x.square().mean() + output.distogram_logits.square().mean()
    loss.backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())
    assert pos.grad is not None and bool(torch.isfinite(pos.grad).all())
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    injection = model.stages[0].pair_injection[0]
    global_context = model.stages[0].global_blocks[0].context_projection
    assert injection.even_residual.weight.grad is not None
    assert global_context.weight.grad is not None
    assert float(injection.even_residual.weight.grad.abs().sum()) > 0.0
    assert float(global_context.weight.grad.abs().sum()) > 0.0


@pytest.mark.parametrize("update_positions", [False, True])
def test_coordinate_rms_diagnostic_has_finite_zero_motion_gradient(
    update_positions: bool,
) -> None:
    generator = torch.Generator().manual_seed(111 + int(update_positions))
    model = _tiny_model(update_positions=update_positions).double()
    x = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    pos = torch.randn(
        4,
        3,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    output = model.forward_with_aux(ELAGraph(x, pos))
    coordinate_rms = output.diagnostics["coordinate_rms"]
    torch.testing.assert_close(coordinate_rms, torch.zeros_like(coordinate_rms))
    (output.graph.x.square().mean() + coordinate_rms.sum()).backward()
    assert pos.grad is not None and bool(torch.isfinite(pos.grad).all())
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


def test_coordinate_rms_diagnostic_matches_nonzero_coordinate_delta() -> None:
    generator = torch.Generator().manual_seed(114)
    model = _tiny_model(update_positions=True).double()
    updates = model.stages[0].coordinate_updates
    assert updates is not None
    with torch.no_grad():
        updates[0].vector.base_weight.fill_(0.2)
    pos = torch.randn(
        5,
        3,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    output = model.forward_with_aux(
        ELAGraph(
            torch.randn(5, 3, generator=generator, dtype=torch.float64),
            pos,
        )
    )
    assert output.graph.delta is not None
    expected = torch.sqrt(output.graph.delta.square().mean()).reshape(1)
    assert float(expected.detach()) > 0.0
    torch.testing.assert_close(output.diagnostics["coordinate_rms"], expected)
    output.diagnostics["coordinate_rms"].sum().backward()
    assert pos.grad is not None and bool(torch.isfinite(pos.grad).all())


@pytest.mark.parametrize("update_positions", [False, True])
def test_tiny_cpu_bfloat16_autocast_forward_backward_is_finite(
    update_positions: bool,
) -> None:
    generator = torch.Generator().manual_seed(12 + int(update_positions))
    model = _tiny_model(update_positions=update_positions).train()
    x = torch.randn(4, 3, generator=generator, requires_grad=True)
    pos = torch.randn(4, 3, generator=generator, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model.forward_with_aux(ELAGraph(x, pos))
        loss = output.graph.x.square().mean() + output.distogram_logits.square().mean()
    loss.backward()
    assert output.graph.x.dtype == x.dtype
    assert output.graph.pos.dtype == pos.dtype
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())
    assert pos.grad is not None and bool(torch.isfinite(pos.grad).all())
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


def test_forward_returns_graph_and_auxiliary_path_keeps_ordered_pair_state() -> None:
    torch.manual_seed(13)
    model = _tiny_model(pair_feature_dim=2).eval()
    graph = ELAGraph(torch.randn(3, 3), torch.randn(3, 3))
    feature = torch.randn(3, 3, 2)
    context = BiomolecularPairContext(
        token_index=torch.arange(3),
        chain_id=torch.tensor([0, 0, 1]),
        pair_features=feature,
    )
    public = model(graph, context)
    auxiliary = model.forward_with_aux(graph, context)
    assert isinstance(public, ELAGraph)
    torch.testing.assert_close(public.x, auxiliary.graph.x)
    assert not torch.equal(
        auxiliary.pair_state.z[0, 0, 1],
        auxiliary.pair_state.z[0, 1, 0],
    )


def test_graph_validation_and_removed_architecture_arguments_fail_loudly() -> None:
    with pytest.raises(TypeError, match="edge_index"):
        ELAGraph(  # type: ignore[call-arg]
            x=torch.randn(3, 2),
            pos=torch.randn(3, 3),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
        )
    with pytest.raises(TypeError, match="cutoff"):
        TriELA("2x0e", cutoff=5.0)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="share one floating-point dtype"):
        ELAGraph(
            x=torch.randn(3, 2, dtype=torch.float32),
            pos=torch.randn(3, 3, dtype=torch.float64),
        )


def test_graph_rejects_in_place_batch_mutation_after_validation() -> None:
    batch = torch.tensor([0, 0, 1, 1])
    graph = ELAGraph(torch.randn(4, 2), torch.randn(4, 3), batch=batch)
    assert graph.num_graphs == 2
    batch.zero_()
    with pytest.raises(RuntimeError, match="batch was mutated in place"):
        _ = graph.num_graphs
    with pytest.raises(RuntimeError, match="batch was mutated in place"):
        _tiny_model("2x0e")(graph)

    group = torch.tensor([0, 0, 1, 1])
    grouped = ELAGraph(
        torch.randn(4, 2),
        torch.randn(4, 3),
        group=group,
    )
    group.zero_()
    with pytest.raises(RuntimeError, match="group was mutated in place"):
        _tiny_model("2x0e")(grouped)


def test_collate_condition_order_masks_and_tiny_conditioned_forward() -> None:
    first = ELAGraph(
        x=torch.randn(3, 2),
        pos=torch.randn(3, 3),
        condition=torch.randn(1, 4),
        order=torch.randn(3, 1),
        update_mask=torch.tensor([True, False, True]),
        y=torch.randn(1, 1),
        ids=("a",),
    )
    second = ELAGraph(
        x=torch.randn(2, 2),
        pos=torch.randn(2, 3),
        condition=torch.randn(4),
        order=torch.randn(2, 1),
        update_mask=torch.tensor([False, True]),
        y=torch.randn(1, 1),
        ids=("b",),
    )
    graph = ELAGraph.collate((first, second))
    torch.testing.assert_close(graph.batch, torch.tensor([0, 0, 0, 1, 1]))
    assert graph.condition is not None and graph.condition.shape == (2, 4)
    assert graph.order is not None and graph.order.shape == (5, 1)
    assert graph.update_mask is not None and graph.update_mask.sum() == 3
    assert graph.y is not None and graph.y.shape == (2, 1)
    assert graph.ids == ("a", "b")
    output = _tiny_model(
        "2x0e",
        "1x0e",
        condition_dim=4,
        order_dim=1,
    )(graph)
    assert output.x.shape == (5, 1)


def test_config_roundtrip_and_pair_token_guard() -> None:
    config = TriELAConfig(
        input_irreps="3x0e",
        output_irreps="1x0e",
        width=16,
        pair_width=8,
        triangle_hidden=8,
        num_stages=1,
        pair_blocks_per_stage=1,
        local_blocks_per_stage=1,
        pair_transition_factor=2,
        pair_dropout=0.0,
        local_points=3,
        max_pair_tokens=3,
        distance_rbf_bins=4,
        distogram_bins=6,
    )
    model = TriELA.from_config(config)
    assert model.config == config
    assert model.describe()["max_pair_tokens"] == 3
    graph = ELAGraph(torch.randn(4, 3), torch.randn(4, 3))
    with pytest.raises(ValueError, match="exceeds max_pair_tokens=3"):
        model(graph)

    with pytest.raises(ValueError, match="positive dimension"):
        _tiny_model("0")
    with pytest.raises(ValueError, match="positive dimension"):
        _tiny_model(output_irreps="0")
    with pytest.raises(ValueError, match="pair_width must be at least 2"):
        _tiny_model(pair_width=1)
    with pytest.raises(ValueError, match="triangle_hidden must be at least 2"):
        _tiny_model(triangle_hidden=1)
    with pytest.raises(ValueError, match="distogram_bins must be at least 2"):
        _tiny_model(distogram_bins=1)
