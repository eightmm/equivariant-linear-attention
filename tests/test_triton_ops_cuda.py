from __future__ import annotations

import pytest
import torch

from equivariant_attention import ELA, ELABatch
from equivariant_attention.triton_ops import csr_sum, triton_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_available(),
    reason="CUDA Triton runtime is unavailable",
)


def _complete_edges(nodes: int, device: torch.device) -> torch.Tensor:
    receiver = torch.arange(nodes, device=device).repeat_interleave(nodes)
    sender = torch.arange(nodes, device=device).repeat(nodes)
    return torch.stack([receiver, sender])


def test_triton_csr_sum_fp32_matches_torch_and_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    row_ptr = torch.tensor([0, 3, 3, 7, 9], device=device, dtype=torch.int32)
    value_torch = torch.randn(9, 5, device=device, requires_grad=True)
    value_triton = value_torch.detach().clone().requires_grad_(True)

    monkeypatch.setenv("ELA_KERNEL_BACKEND", "torch")
    expected = csr_sum(value_torch, row_ptr)
    expected.square().mean().backward()

    monkeypatch.setenv("ELA_KERNEL_BACKEND", "triton")
    actual = csr_sum(value_triton, row_ptr)
    actual.square().mean().backward()

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        value_triton.grad,
        value_torch.grad,
        atol=2e-5,
        rtol=2e-5,
    )


def test_triton_csr_sum_bfloat16_is_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    row_ptr = torch.tensor([0, 17, 33, 64], device=device, dtype=torch.int32)
    value = torch.randn(
        64,
        23,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    monkeypatch.setenv("ELA_KERNEL_BACKEND", "triton")
    output = csr_sum(value, row_ptr)
    output.float().square().mean().backward()
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert value.grad is not None and torch.isfinite(value.grad).all()


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


def test_full_ela_triton_matches_torch_output_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    prepared = reference_model.prepare(base)

    reference_x = features.detach().clone().requires_grad_(True)
    reference_pos = positions.detach().clone().requires_grad_(True)
    reference_batch = ELABatch(
        node_irreps=reference_x,
        positions=reference_pos,
        edge_index=base.edge_index,
        _prepared_graph=prepared._prepared_graph,
    )
    monkeypatch.setenv("ELA_KERNEL_BACKEND", "torch")
    reference_output = reference_model(reference_batch)["node_irreps"]
    reference_output.square().mean().backward()

    triton_x = features.detach().clone().requires_grad_(True)
    triton_pos = positions.detach().clone().requires_grad_(True)
    triton_batch = ELABatch(
        node_irreps=triton_x,
        positions=triton_pos,
        edge_index=base.edge_index,
        _prepared_graph=prepared._prepared_graph,
    )
    monkeypatch.setenv("ELA_KERNEL_BACKEND", "triton")
    triton_output = triton_model(triton_batch)["node_irreps"]
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
