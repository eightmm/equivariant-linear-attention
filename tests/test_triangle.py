from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention.nn.triangle import (
    GatedTriangleMultiplication,
    PairTransition,
    TrianglePairBlock,
)


def _wake_projection(linear: torch.nn.Linear, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        linear.weight.copy_(
            torch.randn(
                linear.weight.shape,
                generator=generator,
                dtype=linear.weight.dtype,
            )
            / max(1, linear.weight.shape[1]) ** 0.5
        )
        linear.bias.copy_(
            torch.randn(
                linear.bias.shape,
                generator=generator,
                dtype=linear.bias.dtype,
            )
            * 0.1
        )


def _triangle_oracle(
    module: GatedTriangleMultiplication,
    z: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    z_norm = module.norm(z)
    mask = pair_mask.unsqueeze(-1).to(dtype=z.dtype)
    a = module.linear_a(z_norm) * torch.sigmoid(module.gate_a(z_norm))
    b = module.linear_b(z_norm) * torch.sigmoid(module.gate_b(z_norm))
    a = a * mask
    b = b * mask
    if module.direction == "outgoing":
        contracted = torch.einsum("bikc,bjkc->bijc", a, b)
    else:
        contracted = torch.einsum("bkjc,bkic->bijc", a, b)
    update = module.output_projection(module.center_norm(contracted))
    return update * torch.sigmoid(module.output_gate(z_norm)) * mask


@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_triangle_multiplication_matches_exact_directed_oracle(
    direction: str,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(301)
    module = GatedTriangleMultiplication(
        4,
        5,
        direction,  # type: ignore[arg-type]
    ).to(dtype=dtype)
    _wake_projection(module.output_projection, 302)
    z = torch.randn(2, 3, 3, 4, dtype=dtype)
    pair_mask = torch.tensor(
        [
            [[True, True, True], [True, True, True], [True, True, True]],
            [[True, True, False], [True, True, False], [False, False, False]],
        ]
    )
    actual = module(z, pair_mask)
    expected = _triangle_oracle(module, z, pair_mask)
    tolerance = 2e-6 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert torch.count_nonzero(actual[1, 2]) == 0
    assert torch.count_nonzero(actual[1, :, 2]) == 0


def test_outgoing_and_incoming_are_independent_and_directional() -> None:
    torch.manual_seed(303)
    outgoing = GatedTriangleMultiplication(3, 4, "outgoing").double()
    incoming = GatedTriangleMultiplication(3, 4, "incoming").double()
    incoming.load_state_dict(outgoing.state_dict())
    _wake_projection(outgoing.output_projection, 304)
    incoming.output_projection.load_state_dict(outgoing.output_projection.state_dict())
    z = torch.randn(1, 3, 3, 3, dtype=torch.float64)
    mask = torch.ones(1, 3, 3, dtype=torch.bool)

    outgoing_value = outgoing(z, mask)
    incoming_value = incoming(z, mask)
    assert not torch.allclose(outgoing_value, incoming_value)

    block = TrianglePairBlock(3, 4, dropout=0.0)
    assert block.outgoing is not block.incoming
    assert (
        block.outgoing.linear_a.weight.data_ptr()
        != block.incoming.linear_a.weight.data_ptr()
    )


@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_triangle_gradcheck_and_masked_values_cannot_leak(direction: str) -> None:
    torch.manual_seed(305)
    module = GatedTriangleMultiplication(
        2,
        3,
        direction,  # type: ignore[arg-type]
        eps=1e-6,
    ).double()
    _wake_projection(module.output_projection, 306)
    mask = torch.tensor(
        [[[True, True, False], [True, True, False], [False, False, False]]]
    )
    z = torch.randn(1, 3, 3, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: module(value, mask),
        (z,),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
    )

    changed = z.detach().clone()
    changed[~mask] = 1e6
    torch.testing.assert_close(
        module(z.detach(), mask),
        module(changed, mask),
        atol=1e-12,
        rtol=1e-12,
    )


def test_zero_initialization_is_exact_for_each_pair_update() -> None:
    z = torch.randn(2, 3, 3, 4)
    mask = torch.tensor(
        [
            [[True, True, True], [True, True, True], [True, True, True]],
            [[True, False, False], [False, False, False], [False, False, False]],
        ]
    )
    outgoing = GatedTriangleMultiplication(4, 5, "outgoing")
    incoming = GatedTriangleMultiplication(4, 5, "incoming")
    transition = PairTransition(4)
    assert torch.count_nonzero(outgoing(z, mask)) == 0
    assert torch.count_nonzero(incoming(z, mask)) == 0
    assert torch.count_nonzero(transition(z, mask)) == 0

    block = TrianglePairBlock(4, 5, dropout=0.5).train()
    expected = z * mask.unsqueeze(-1)
    torch.testing.assert_close(block(z, mask), expected, atol=0.0, rtol=0.0)


def test_pair_transition_chunking_is_numerically_and_gradient_exact() -> None:
    torch.manual_seed(307)
    transition = PairTransition(3, expansion=3).double()
    _wake_projection(transition.out_projection, 308)
    mask = torch.tensor(
        [[[True, True, True], [True, True, False], [True, False, False]]]
    )
    full_input = torch.randn(1, 3, 3, 3, dtype=torch.float64, requires_grad=True)
    chunk_input = full_input.detach().clone().requires_grad_()
    full = transition(full_input, mask)
    chunked = transition(chunk_input, mask, chunk_size=2)
    torch.testing.assert_close(full, chunked, atol=3e-15, rtol=3e-15)
    full.square().sum().backward()
    chunked.square().sum().backward()
    torch.testing.assert_close(
        full_input.grad,
        chunk_input.grad,
        atol=2e-12,
        rtol=2e-12,
    )
    assert torch.count_nonzero(chunked[~mask]) == 0


def test_triangle_pair_block_uses_one_fixed_order_and_remasks_every_update() -> None:
    torch.manual_seed(309)
    block = TrianglePairBlock(3, 4, transition_factor=2, dropout=0.0).double()
    _wake_projection(block.outgoing.output_projection, 310)
    _wake_projection(block.incoming.output_projection, 311)
    _wake_projection(block.transition.out_projection, 312)
    order: list[str] = []
    handles = [
        block.outgoing.register_forward_hook(lambda *_: order.append("outgoing")),
        block.incoming.register_forward_hook(lambda *_: order.append("incoming")),
        block.transition.register_forward_hook(lambda *_: order.append("transition")),
    ]
    z = torch.randn(1, 3, 3, 3, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor(
        [[[True, True, False], [True, True, False], [False, False, False]]]
    )
    output = block(z, mask)
    for handle in handles:
        handle.remove()
    assert order == ["outgoing", "incoming", "transition"]
    assert not hasattr(block, "row_attention")
    assert not hasattr(block, "column_attention")
    assert torch.count_nonzero(output[~mask]) == 0
    output.square().sum().backward()
    assert z.grad is not None and bool(torch.isfinite(z.grad).all())
