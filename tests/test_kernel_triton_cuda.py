from __future__ import annotations

import pytest
import torch

import equivariant_linear_attention.kernels.local as optimized_local
from equivariant_linear_attention import ELA
from equivariant_linear_attention.batch import ELABatch
from equivariant_linear_attention.kernels.triton import (
    _trusted_csr_sum_many,
    _trusted_weighted_gather_reduce_pair,
    active_backend,
    csr_sum,
    csr_sum_many,
    kernel_backend,
    triton_available,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_available(),
    reason="CUDA Triton runtime is unavailable",
)


def _complete_edges(nodes: int, device: torch.device) -> torch.Tensor:
    receiver = torch.arange(nodes, device=device).repeat_interleave(nodes)
    sender = torch.arange(nodes, device=device).repeat(nodes)
    return torch.stack([receiver, sender])


def test_triton_csr_sum_fp32_matches_torch_and_backward() -> None:
    device = torch.device("cuda")
    row_ptr = torch.tensor([0, 3, 3, 7, 9], device=device, dtype=torch.int32)
    value_torch = torch.randn(9, 5, device=device, requires_grad=True)
    value_triton = value_torch.detach().clone().requires_grad_(True)

    with kernel_backend("torch"):
        expected = csr_sum(value_torch, row_ptr)
        expected.square().mean().backward()

    with kernel_backend("triton"):
        assert active_backend(value_triton, row_ptr) == "triton"
        actual = csr_sum(value_triton, row_ptr)
        actual.square().mean().backward()

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        value_triton.grad,
        value_torch.grad,
        atol=2e-5,
        rtol=2e-5,
    )


def test_auto_stays_on_reference_until_a_regime_is_promoted() -> None:
    device = torch.device("cuda")
    row_ptr = torch.tensor([0, 256, 512], device=device, dtype=torch.int32)
    value = torch.randn(512, 8, device=device)
    with kernel_backend("auto"):
        assert active_backend(value, row_ptr) == "torch"


def test_triton_csr_sum_bfloat16_matches_fp32_accumulation() -> None:
    device = torch.device("cuda")
    row_ptr = torch.tensor([0, 17, 33, 64], device=device, dtype=torch.int32)
    value = torch.randn(
        64,
        23,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    with kernel_backend("torch"):
        expected = csr_sum(value.detach(), row_ptr)
    with kernel_backend("triton"):
        output = csr_sum(value, row_ptr)
    output.float().square().mean().backward()
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    torch.testing.assert_close(output, expected, atol=0.0625, rtol=0.01)


def test_triton_csr_sum_float16_and_int64_match_torch_backward() -> None:
    device = torch.device("cuda")
    row_ptr = torch.tensor(
        [0, 0, 3, 68, 73],
        device=device,
        dtype=torch.int64,
    )
    reference_value = torch.randn(
        73,
        11,
        device=device,
        dtype=torch.float16,
        requires_grad=True,
    )
    candidate_value = reference_value.detach().clone().requires_grad_(True)

    with kernel_backend("torch"):
        reference = csr_sum(reference_value, row_ptr)
        reference.float().square().mean().backward()
    with kernel_backend("triton"):
        candidate = csr_sum(candidate_value, row_ptr)
        candidate.float().square().mean().backward()

    assert candidate.dtype == torch.float16
    torch.testing.assert_close(candidate, reference, atol=0.015625, rtol=0.01)
    torch.testing.assert_close(
        candidate_value.grad,
        reference_value.grad,
        atol=0.001,
        rtol=0.01,
    )


def test_triton_handles_noncontiguous_and_skewed_csr() -> None:
    device = torch.device("cuda")
    degree = torch.tensor([0, 1, 5, 129, 3], device=device, dtype=torch.int32)
    compact = torch.cat(
        [torch.zeros(1, device=device, dtype=torch.int32), degree.cumsum(0)]
    )
    storage = torch.empty(
        compact.numel() * 2 - 1,
        device=device,
        dtype=torch.int32,
    )
    storage[::2] = compact
    storage[1::2] = -1
    row_ptr = storage[::2]
    assert not row_ptr.is_contiguous()
    value = torch.randn(int(compact[-1].item()), 7, device=device)
    expected = torch.segment_reduce(value, reduce="sum", offsets=row_ptr)
    with kernel_backend("triton"):
        actual = csr_sum(value, row_ptr)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


def test_triton_csr_sum_supports_double_backward() -> None:
    device = torch.device("cuda")
    row_ptr = torch.tensor([0, 2, 2, 7], device=device, dtype=torch.int32)
    value = torch.randn(7, 4, device=device, requires_grad=True)
    with kernel_backend("triton"):
        output = csr_sum(value, row_ptr)
        first = torch.autograd.grad(
            output.square().sum(),
            value,
            create_graph=True,
        )[0]
        second = torch.autograd.grad(first.square().sum(), value)[0]
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()


def test_triton_pair_and_triple_payloads_match_torch_gradients() -> None:
    device = torch.device("cuda")
    row_ptr = torch.tensor([0, 3, 3, 8, 11], device=device, dtype=torch.int32)
    base = (
        torch.randn(11, 6, device=device),
        torch.randn(11, 2, device=device),
        torch.randn(11, 3, device=device),
    )
    reference_values = tuple(value.clone().requires_grad_(True) for value in base)
    candidate_values = tuple(value.clone().requires_grad_(True) for value in base)

    with kernel_backend("torch"):
        reference_pair = csr_sum_many(reference_values[:2], row_ptr)
        reference_triple = csr_sum_many(reference_values, row_ptr)
        reference_loss = sum(
            value.square().mean() for value in (*reference_pair, *reference_triple)
        )
        reference_gradients = torch.autograd.grad(
            reference_loss,
            reference_values,
            create_graph=True,
        )

    with kernel_backend("triton"):
        actual_pair = csr_sum_many(candidate_values[:2], row_ptr)
        actual_triple = csr_sum_many(candidate_values, row_ptr)
        actual_loss = sum(
            value.square().mean() for value in (*actual_pair, *actual_triple)
        )
        actual_gradients = torch.autograd.grad(
            actual_loss,
            candidate_values,
            create_graph=True,
        )

    for actual, expected in zip(
        (*actual_pair, *actual_triple),
        (*reference_pair, *reference_triple),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    for actual, expected in zip(
        actual_gradients,
        reference_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

    second = torch.autograd.grad(
        sum(value.square().sum() for value in actual_gradients),
        candidate_values,
    )
    assert all(torch.isfinite(value).all() for value in second)


def test_triton_prepared_receiver_groups_unused_gradients_and_gradgrad() -> None:
    device = torch.device("cuda")
    degree = torch.tensor([0, 2, 5, 1], device=device, dtype=torch.int32)
    row_ptr = torch.cat(
        [torch.zeros(1, device=device, dtype=torch.int32), degree.cumsum(0)]
    )
    receiver = torch.arange(4, device=device).repeat_interleave(degree.long())
    values = tuple(
        torch.randn(receiver.shape[0], width, device=device, requires_grad=True)
        for width in (5, 3, 7)
    )

    outputs = _trusted_csr_sum_many(
        values,
        row_ptr,
        policy="triton",
        receiver=receiver,
    )
    for output, value in zip(outputs, values, strict=True):
        expected = torch.segment_reduce(value, reduce="sum", offsets=row_ptr)
        torch.testing.assert_close(output, expected, atol=3e-5, rtol=3e-5)

    gradients = torch.autograd.grad(
        outputs[0].square().sum() + outputs[2].square().sum(),
        values,
        create_graph=True,
        allow_unused=True,
    )
    assert gradients[1] is None
    assert gradients[0] is not None and gradients[2] is not None
    torch.testing.assert_close(
        gradients[0],
        (2.0 * outputs[0]).index_select(0, receiver),
    )
    torch.testing.assert_close(
        gradients[2],
        (2.0 * outputs[2]).index_select(0, receiver),
    )
    second = torch.autograd.grad(
        gradients[0].square().sum() + gradients[2].square().sum(),
        (values[0], values[2]),
    )
    assert all(torch.isfinite(value).all() for value in second)


@pytest.mark.parametrize("widths", [(5, 3), (5, 3, 7)])
def test_triton_prepared_receiver_fast_backward_matches_direct_gather(
    widths: tuple[int, ...],
) -> None:
    device = torch.device("cuda")
    degree = torch.tensor([0, 2, 5, 1], device=device, dtype=torch.int64)
    row_ptr = torch.cat(
        [torch.zeros(1, device=device, dtype=torch.int64), degree.cumsum(0)]
    )
    receiver = torch.arange(4, device=device, dtype=torch.int64).repeat_interleave(
        degree
    )
    values = tuple(
        torch.randn(receiver.shape[0], width, device=device, requires_grad=True)
        for width in widths
    )
    cotangents = tuple(
        torch.randn(4, width, device=device) for width in widths
    )

    outputs = _trusted_csr_sum_many(
        values,
        row_ptr,
        policy="triton",
        receiver=receiver,
    )
    gradients = torch.autograd.grad(
        outputs,
        values,
        grad_outputs=cotangents,
        create_graph=False,
    )

    for gradient, cotangent in zip(gradients, cotangents, strict=True):
        torch.testing.assert_close(
            gradient,
            cotangent.index_select(0, receiver),
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.parametrize("components", [(8, 1), (3, 3)])
def test_triton_weighted_gather_reduce_pair_matches_edge_reference(
    components: tuple[int, int],
) -> None:
    device = torch.device("cuda")
    degree = torch.tensor([0, 2, 5, 1], device=device, dtype=torch.int64)
    row_ptr = torch.cat(
        [torch.zeros(1, device=device, dtype=torch.int64), degree.cumsum(0)]
    )
    sender = torch.tensor([3, 0, 1, 2, 3, 0, 2, 1], device=device)
    rank = 4
    weight = torch.rand(sender.shape[0], rank, device=device)
    radial_gate = torch.rand(sender.shape[0], rank, 9, device=device)
    sources = tuple(
        torch.randn(4, rank, component, device=device) if component > 1 else
        torch.randn(4, rank, device=device)
        for component in components
    )

    expected = tuple(
        torch.segment_reduce(
            weight.unsqueeze(-1)
            * radial_gate[..., lane, None]
            * source[sender].reshape(sender.shape[0], rank, component),
            reduce="sum",
            offsets=row_ptr,
        ).reshape(4, *source.shape[1:])
        for lane, source, component in zip((2, 3), sources, components, strict=True)
    )
    with torch.inference_mode():
        actual = _trusted_weighted_gather_reduce_pair(
            weight,
            radial_gate,
            sources[0],
            sources[1],
            sender,
            row_ptr,
            gate_lanes=(2, 3),
            policy="triton",
        )

    for candidate, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, reference, atol=3e-5, rtol=3e-5)


def test_full_triton_inference_uses_weighted_gather_reduce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = optimized_local._trusted_weighted_gather_reduce_pair
    observed_lanes: list[tuple[int, int]] = []

    def observe(*args: object, gate_lanes: tuple[int, int], **kwargs: object):
        observed_lanes.append(gate_lanes)
        return original(*args, gate_lanes=gate_lanes, **kwargs)

    monkeypatch.setattr(
        optimized_local,
        "_trusted_weighted_gather_reduce_pair",
        observe,
    )
    device = torch.device("cuda")
    model = ELA(
        input_irreps="4x0e",
        output_irreps="2x0e",
        width=16,
        depth=1,
        cutoff=10.0,
    ).to(device)
    features = torch.randn(8, 4, device=device)
    positions = torch.randn(8, 3, device=device)
    prepared = model._prepare_packed(
        ELABatch(features, positions, edge_index=_complete_edges(8, device))
    )

    with torch.inference_mode(), kernel_backend("triton"):
        model._forward_prepared(prepared)

    assert observed_lanes == [(0, 1), (2, 3)]


def _activate_local_outputs(model: ELA) -> None:
    with torch.no_grad():
        for layer in model.layers:
            for name in (
                "local_scalar_out",
                "local_odd_out",
                "local_polar_out",
                "local_axial_out",
                "local_even_tensor_out",
                "local_odd_tensor_out",
                "local_mass_out",
            ):
                module = getattr(layer, name)
                if hasattr(module, "weight"):
                    module.weight.normal_(mean=0.0, std=0.05)


def test_full_ela_triton_matches_torch_output_and_gradients() -> None:
    torch.manual_seed(29)
    device = torch.device("cuda")
    reference_model = ELA(
        input_irreps="4x0e",
        output_irreps="2x0e",
        width=32,
        depth=2,
        cutoff=10.0,
    ).to(device=device, dtype=torch.float32)
    _activate_local_outputs(reference_model)
    triton_model = ELA(
        input_irreps="4x0e",
        output_irreps="2x0e",
        width=32,
        depth=2,
        cutoff=10.0,
    ).to(device=device, dtype=torch.float32)
    triton_model.load_state_dict(reference_model.state_dict(), strict=True)

    nodes = 7
    features = torch.randn(nodes, 4, device=device)
    positions = torch.randn(nodes, 3, device=device)
    base = ELABatch(
        node_irreps=features,
        positions=positions,
        edge_index=_complete_edges(nodes, device),
    )
    prepared = reference_model._prepare_packed(base)

    reference_x = features.detach().clone().requires_grad_(True)
    reference_pos = positions.detach().clone().requires_grad_(True)
    reference_batch = ELABatch(
        node_irreps=reference_x,
        positions=reference_pos,
        edge_index=base.edge_index,
        _prepared_graph=prepared._prepared_graph,
    )
    with kernel_backend("torch"):
        reference_output = reference_model._forward_prepared(reference_batch)[
            "node_irreps"
        ]
        reference_output.square().mean().backward()

    triton_x = features.detach().clone().requires_grad_(True)
    triton_pos = positions.detach().clone().requires_grad_(True)
    triton_batch = ELABatch(
        node_irreps=triton_x,
        positions=triton_pos,
        edge_index=base.edge_index,
        _prepared_graph=prepared._prepared_graph,
    )
    with kernel_backend("triton"):
        triton_output = triton_model._forward_prepared(triton_batch)["node_irreps"]
        triton_output.square().mean().backward()

    torch.testing.assert_close(
        triton_output,
        reference_output,
        atol=4e-4,
        rtol=4e-4,
    )
    torch.testing.assert_close(
        triton_x.grad,
        reference_x.grad,
        atol=7e-4,
        rtol=7e-4,
    )
    torch.testing.assert_close(
        triton_pos.grad,
        reference_pos.grad,
        atol=1e-3,
        rtol=1e-3,
    )
    reference_grad = reference_model.layers[0].local_scalar_out.weight.grad
    triton_grad = triton_model.layers[0].local_scalar_out.weight.grad
    assert reference_grad is not None and triton_grad is not None
    torch.testing.assert_close(
        triton_grad,
        reference_grad,
        atol=7e-4,
        rtol=7e-4,
    )


def test_native_bfloat16_full_ela_matches_torch() -> None:
    torch.manual_seed(31)
    device = torch.device("cuda")
    reference_model = ELA(
        input_irreps="4x0e",
        output_irreps="2x0e",
        width=16,
        depth=1,
        cutoff=10.0,
    ).to(device=device, dtype=torch.bfloat16)
    _activate_local_outputs(reference_model)
    triton_model = ELA(
        input_irreps="4x0e",
        output_irreps="2x0e",
        width=16,
        depth=1,
        cutoff=10.0,
    ).to(device=device, dtype=torch.bfloat16)
    triton_model.load_state_dict(reference_model.state_dict(), strict=True)

    nodes = 24
    features = torch.randn(
        nodes,
        4,
        device=device,
        dtype=torch.bfloat16,
    )
    positions = torch.randn(
        nodes,
        3,
        device=device,
        dtype=torch.bfloat16,
    )
    edges = _complete_edges(nodes, device)
    prepared = reference_model._prepare_packed(ELABatch(features, positions, edge_index=edges))

    reference_x = features.clone().requires_grad_(True)
    reference_pos = positions.clone().requires_grad_(True)
    reference_batch = ELABatch(
        reference_x,
        reference_pos,
        edge_index=edges,
        _prepared_graph=prepared._prepared_graph,
    )
    with kernel_backend("torch"):
        reference = reference_model._forward_prepared(reference_batch)["node_irreps"]
        reference.float().square().mean().backward()

    triton_x = features.clone().requires_grad_(True)
    triton_pos = positions.clone().requires_grad_(True)
    triton_batch = ELABatch(
        triton_x,
        triton_pos,
        edge_index=edges,
        _prepared_graph=prepared._prepared_graph,
    )
    with kernel_backend("triton"):
        actual = triton_model._forward_prepared(triton_batch)["node_irreps"]
        actual.float().square().mean().backward()

    torch.testing.assert_close(actual, reference, atol=0.04, rtol=0.04)
    torch.testing.assert_close(
        triton_x.grad,
        reference_x.grad,
        atol=0.05,
        rtol=0.05,
    )
    torch.testing.assert_close(
        triton_pos.grad,
        reference_pos.grad,
        atol=0.08,
        rtol=0.08,
    )
