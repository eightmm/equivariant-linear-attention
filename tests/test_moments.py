from __future__ import annotations

import itertools

import torch

from equivariant_linear_attention.nn.geometry import (
    AdaptiveMomentBank,
    GeometryContext,
    explicit_centered_moments,
)
from equivariant_linear_attention.nn.ops import stf3, stf4


def _outer_power(value: torch.Tensor, order: int) -> torch.Tensor:
    output = value
    for _ in range(order - 1):
        output = output.unsqueeze(-1) * value.reshape(*((1,) * (output.ndim - 1)), 3)
    return output


def test_exact_relative_moments_match_pair_oracle_through_order_four() -> None:
    generator = torch.Generator().manual_seed(101)
    position = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    weight = torch.rand(7, 4, generator=generator, dtype=torch.float64) + 0.1
    index = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    _, *moments = explicit_centered_moments(position, weight, index, 2)

    for order, actual in enumerate(moments, start=1):
        expected = torch.zeros_like(actual)
        for receiver in range(position.shape[0]):
            denominator = weight[index == index[receiver]].sum(dim=0)
            for sender in range(position.shape[0]):
                if index[sender] != index[receiver]:
                    continue
                displacement = position[sender] - position[receiver]
                expected[receiver] += weight[sender].reshape(
                    weight.shape[1], *((1,) * order)
                ) * _outer_power(displacement, order).unsqueeze(0)
            expected[receiver] /= denominator.reshape(weight.shape[1], *((1,) * order))
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


def test_stf3_and_stf4_are_trace_free() -> None:
    generator = torch.Generator().manual_seed(103)
    rank3 = torch.randn(4, 3, 3, 3, generator=generator, dtype=torch.float64)
    rank3 = (
        sum(
            rank3.permute(0, *(axis + 1 for axis in permutation))
            for permutation in itertools.permutations(range(3))
        )
        / 6.0
    )
    projected3 = stf3(rank3)
    torch.testing.assert_close(
        torch.einsum("...aac->...c", projected3),
        torch.zeros(4, 3, dtype=torch.float64),
        atol=2e-12,
        rtol=0.0,
    )

    rank4 = torch.randn(4, 3, 3, 3, 3, generator=generator, dtype=torch.float64)
    rank4 = (
        sum(
            rank4.permute(0, *(axis + 1 for axis in permutation))
            for permutation in itertools.permutations(range(4))
        )
        / 24.0
    )
    projected4 = stf4(rank4)
    torch.testing.assert_close(
        torch.einsum("...aakl->...kl", projected4),
        torch.zeros(4, 3, 3, dtype=torch.float64),
        atol=3e-12,
        rtol=0.0,
    )


def test_adaptive_moment_bank_is_translation_invariant() -> None:
    generator = torch.Generator().manual_seed(107)
    position = torch.randn(8, 3, generator=generator, dtype=torch.float64)
    index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    scalar = torch.randn(8, 16, generator=generator, dtype=torch.float64)
    bank = AdaptiveMomentBank(scalar_width=16, rank=4, eps=1e-10).double()
    geometry = GeometryContext.build(position, index, num_segments=2, eps=1e-10)
    shifted = GeometryContext.build(
        position + torch.tensor([5.0, -3.0, 2.0], dtype=torch.float64),
        index,
        num_segments=2,
        eps=1e-10,
    )
    reference = bank(scalar, geometry)
    moved = bank(scalar, shifted)
    for left, right in zip(
        reference.__dict__.values(), moved.__dict__.values(), strict=True
    ):
        torch.testing.assert_close(left, right, atol=2e-11, rtol=2e-11)
