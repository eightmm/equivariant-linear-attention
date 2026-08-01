from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from equivariant_attention import ELA, ELABatch


def _fixed_degree_edges(
    nodes: int,
    degree: int,
    device: torch.device,
) -> torch.Tensor:
    receiver = torch.arange(nodes, device=device).repeat_interleave(degree)
    offsets = torch.arange(1, degree + 1, device=device).repeat(nodes)
    sender = (receiver + offsets) % nodes
    return torch.stack([receiver, sender])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("precision", "autocast_dtype"),
    [
        ("float32", None),
        ("bfloat16", torch.bfloat16),
    ],
)
def test_canonical_ela_cuda_forward_backward(
    precision: str,
    autocast_dtype: torch.dtype | None,
) -> None:
    torch.manual_seed(47)
    device = torch.device("cuda")
    model = ELA(
        input_irreps="8x0e",
        output_irreps="1x0e + 1x1o",
        width=32,
        depth=2,
        cutoff=6.0,
        num_rbf=8,
    ).to(device=device, dtype=torch.float32)
    nodes = 64
    features = torch.randn(
        nodes,
        8,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    positions = torch.randn(
        nodes,
        3,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    batch = model.prepare(
        ELABatch(
            node_irreps=features,
            positions=positions,
            edge_index=_fixed_degree_edges(nodes, 8, device),
        )
    )

    precision_context = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )
    with precision_context:
        output = model.forward_prepared(batch)["node_irreps"]
        loss = output.float().square().mean()
    loss.backward()

    assert torch.isfinite(output).all()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    router = model.layers[0].branch_fusion.router[-1]
    assert router.weight.grad is not None
    assert torch.isfinite(router.weight.grad).all()
    assert torch.count_nonzero(router.weight.grad) > 0
    expected_dtype = torch.float32 if precision == "float32" else torch.bfloat16
    assert output.dtype == expected_dtype
