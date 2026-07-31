from __future__ import annotations

import torch

from equivariant_attention.branch_fusion import RMSAwareBranchFusion
from equivariant_attention.parity_se3 import _ParityState


def test_zero_initialized_router_and_balance_path_wake_up() -> None:
    torch.manual_seed(37)
    nodes = 5
    width = 12
    heads = 3
    state = _ParityState(
        even_scalar=torch.randn(nodes, width, dtype=torch.float64),
        odd_scalar=torch.randn(nodes, heads, dtype=torch.float64),
        polar_vector=torch.randn(nodes, heads, 3, dtype=torch.float64),
        axial_vector=torch.randn(nodes, heads, 3, dtype=torch.float64),
        even_tensor=torch.randn(nodes, heads, 5, dtype=torch.float64),
        odd_tensor=torch.randn(nodes, heads, 5, dtype=torch.float64),
    )
    global_message = (
        torch.randn(nodes, heads, 4, dtype=torch.float64),
        torch.randn(nodes, heads, dtype=torch.float64),
        torch.randn(nodes, heads, 3, dtype=torch.float64),
        torch.randn(nodes, heads, 3, dtype=torch.float64),
        torch.randn(nodes, heads, 5, dtype=torch.float64),
        torch.randn(nodes, heads, 5, dtype=torch.float64),
    )
    local_message = (
        torch.randn(nodes, heads, 4, dtype=torch.float64),
        torch.randn(nodes, heads, dtype=torch.float64),
        torch.randn(nodes, heads, 3, dtype=torch.float64),
        torch.randn(nodes, heads, 3, dtype=torch.float64),
        torch.randn(nodes, heads, 5, dtype=torch.float64),
        torch.randn(nodes, heads, 5, dtype=torch.float64),
        torch.randn(nodes, heads, dtype=torch.float64),
    )
    fusion = RMSAwareBranchFusion(scalar_width=width).double()
    routed_global, routed_local = fusion(
        state,
        global_message,
        local_message,
    )
    loss = sum(value.square().mean() for value in routed_global)
    loss = loss + routed_local[6].square().mean()
    loss.backward()

    assert fusion.balance_strength.grad is not None
    assert torch.isfinite(fusion.balance_strength.grad).all()
    assert torch.count_nonzero(fusion.balance_strength.grad) > 0
    final = fusion.router[-1]
    assert final.weight.grad is not None
    assert torch.isfinite(final.weight.grad).all()
    assert torch.count_nonzero(final.weight.grad) > 0
