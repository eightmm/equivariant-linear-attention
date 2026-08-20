from __future__ import annotations

import itertools
from dataclasses import replace

import torch
from conftest import orthogonal

from equivariant_linear_attention.nn.closure import EquivariantClosure
from equivariant_linear_attention.nn.geometry import (
    AdaptiveMomentBank,
    GeometryContext,
    compact_centered_moments,
    compact_third_trace,
    explicit_centered_moments,
    radial_invariants,
)
from equivariant_linear_attention.nn.ops import compact_stf3, stf3, stf4
from equivariant_linear_attention.nn.relation import RelationMessage
from equivariant_linear_attention.nn.state import ParityState


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


def test_compact_third_moment_decomposes_exactly_into_3o_and_1o() -> None:
    generator = torch.Generator().manual_seed(105)
    moment = torch.randn(5, 4, 10, generator=generator, dtype=torch.float64)
    trace = compact_third_trace(moment)
    tx, ty, tz = trace.unbind(dim=-1)
    a, c, d, e, f, b, g = compact_stf3(moment).unbind(dim=-1)
    reconstructed = torch.stack(
        (
            a + 3.0 * tx / 5.0,
            c + ty / 5.0,
            d + tz / 5.0,
            e + tx / 5.0,
            f,
            -a - e + tx / 5.0,
            b + 3.0 * ty / 5.0,
            g + tz / 5.0,
            -c - b + ty / 5.0,
            -d - g + 3.0 * tz / 5.0,
        ),
        dim=-1,
    )
    torch.testing.assert_close(reconstructed, moment, atol=2e-12, rtol=2e-12)


def test_compact_third_trace_matches_full_cartesian_contraction() -> None:
    generator = torch.Generator().manual_seed(104)
    position = torch.randn(8, 3, generator=generator, dtype=torch.float64)
    weight = torch.rand(8, 3, generator=generator, dtype=torch.float64) + 0.1
    index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    _, compact = compact_centered_moments(position, weight, index, 2)
    _, _, _, full_third, _ = explicit_centered_moments(position, weight, index, 2)
    actual = compact_third_trace(compact[..., 10:20])
    expected = torch.einsum("nrabb->nra", full_third)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)


def test_adaptive_bank_exposes_bounded_equivariant_third_trace() -> None:
    generator = torch.Generator().manual_seed(106)
    nodes, rank = 9, 4
    position = torch.randn(nodes, 3, generator=generator, dtype=torch.float64)
    scalar = torch.randn(nodes, 16, generator=generator, dtype=torch.float64)
    index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1])
    bank = AdaptiveMomentBank(scalar_width=16, rank=rank, eps=1e-10).double()
    geometry = GeometryContext.build(position, index, num_segments=2, eps=1e-10)
    actual = bank(scalar, geometry)
    assert actual.third_trace is not None

    radial = radial_invariants(geometry, scalar.dtype)
    temperature = torch.nn.functional.softplus(bank.raw_temperature) + 0.25
    log_weight = (
        bank.content_weight(scalar) + bank.radial_weight(radial)
    ) / temperature
    weight = torch.nn.functional.softplus(log_weight) + bank.eps
    _, centered = compact_centered_moments(
        geometry.normalized,
        weight,
        geometry.index,
        geometry.num_segments,
    )
    third = centered[..., 10:20]
    raw_trace = compact_third_trace(third)
    expected = raw_trace / torch.sqrt(
        1.0 + raw_trace.square().sum(dim=-1, keepdim=True) + bank.eps
    )
    torch.testing.assert_close(actual.third_trace, expected, atol=3e-11, rtol=3e-11)

    transform = orthogonal(reflection=True, seed=106)
    moved_geometry = GeometryContext.build(
        position @ transform.T + torch.tensor([2.0, -1.0, 4.0]),
        index,
        num_segments=2,
        eps=1e-10,
    )
    moved = bank(scalar, moved_geometry)
    assert moved.third_trace is not None
    torch.testing.assert_close(
        moved.third_trace,
        actual.third_trace @ transform.T,
        atol=3e-10,
        rtol=3e-10,
    )


def test_closure_has_an_independent_third_trace_to_polar_path() -> None:
    generator = torch.Generator().manual_seed(108)
    nodes, width, heads, rank = 6, 16, 2, 4
    index = torch.tensor([0, 0, 0, 1, 1, 1])
    geometry = GeometryContext.build(
        torch.randn(nodes, 3, generator=generator, dtype=torch.float64),
        index,
        num_segments=2,
        eps=1e-10,
    )
    moments = AdaptiveMomentBank(
        scalar_width=width,
        rank=rank,
        eps=1e-10,
    ).double()(
        torch.randn(nodes, width, generator=generator, dtype=torch.float64), geometry
    )
    assert moments.third_trace is not None
    state = ParityState(
        torch.zeros(nodes, width, dtype=torch.float64),
        torch.zeros(nodes, heads, dtype=torch.float64),
        torch.zeros(nodes, heads, 3, dtype=torch.float64),
        torch.zeros(nodes, heads, 3, dtype=torch.float64),
        torch.zeros(nodes, heads, 5, dtype=torch.float64),
        torch.zeros(nodes, heads, 5, dtype=torch.float64),
    )
    message = RelationMessage(
        torch.zeros(nodes, heads, width // heads, dtype=torch.float64),
        state.odd_scalar,
        state.polar_vector,
        state.axial_vector,
        state.even_tensor,
        state.odd_tensor,
    )
    closure = EquivariantClosure(
        scalar_width=width,
        num_heads=heads,
        head_dim=width // heads,
        moment_rank=rank,
        eps=1e-10,
    ).double()
    assert torch.count_nonzero(closure.moment_third_polar.weight) == 0
    initialized = closure(state, message, moments)
    without_trace = closure(
        state,
        message,
        replace(moments, third_trace=torch.zeros_like(moments.third_trace)),
    )
    for actual, expected in zip(
        initialized.as_tuple(), without_trace.as_tuple(), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    initialized.polar_vector.square().sum().backward()
    trace_gradient = closure.moment_third_polar.weight.grad
    assert trace_gradient is not None and bool(torch.isfinite(trace_gradient).all())
    assert float(trace_gradient.detach().abs().sum()) > 0.0

    with torch.no_grad():
        for parameter in closure.parameters():
            parameter.zero_()
        closure.moment_third_polar.weight[0, 0] = 1.0
        closure.moment_third_polar.weight[1, 1] = 1.0

    output = closure(state, message, moments)
    projected = closure.moment_third_polar(moments.third_trace)
    expected = projected / torch.sqrt(
        1.0 + projected.square().sum(dim=-1, keepdim=True) + closure.eps
    )
    torch.testing.assert_close(output.polar_vector, expected, atol=2e-12, rtol=2e-12)
    assert float(output.polar_vector.detach().abs().sum()) > 0.0


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
