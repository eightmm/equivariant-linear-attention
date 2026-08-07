from __future__ import annotations

import math

import torch

from equivariant_linear_attention.nn.geometry import (
    AdaptiveMomentBank,
    GeometryContext,
    compact_centered_moments,
    explicit_centered_moments,
)
from equivariant_linear_attention.nn.ops import (
    SYMMETRIC_DEGREE_SLICES,
    compact_stf3,
    compact_stf4,
    matrix_to_st,
    st_to_matrix,
    stf3,
    stf3_contract_st,
    stf3_contract_vector,
    stf4,
    stf4_contract_st,
    stf4_contract_st_vector,
    symmetric_monomials,
)
from equivariant_linear_attention.nn.relation import (
    MercerFeatures,
    RelationMessage,
    SelfAdjointRelation,
)
from equivariant_linear_attention.nn.state import ParityState


def _state(
    nodes: int,
    width: int,
    heads: int,
    generator: torch.Generator,
) -> ParityState:
    return ParityState(
        torch.randn(nodes, width, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, 3, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, 3, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, 5, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, 5, generator=generator, dtype=torch.float64),
    )


def test_compact_symmetric_basis_preserves_polynomial_kernel() -> None:
    generator = torch.Generator().manual_seed(401)
    left = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    right = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    left_feature = symmetric_monomials(left, orthonormal=True)
    right_feature = symmetric_monomials(right, orthonormal=True)
    dot = (left * right).sum(dim=-1)

    assert left_feature.shape[-1] == 35
    for degree, block in enumerate(SYMMETRIC_DEGREE_SLICES):
        actual = (left_feature[:, block] * right_feature[:, block]).sum(dim=-1)
        torch.testing.assert_close(
            actual,
            dot.pow(degree),
            atol=3e-12,
            rtol=3e-12,
        )


def test_compact_centered_moments_match_full_cartesian_oracle() -> None:
    generator = torch.Generator().manual_seed(403)
    position = torch.randn(8, 3, generator=generator, dtype=torch.float64)
    weight = torch.rand(8, 5, generator=generator, dtype=torch.float64) + 0.2
    index = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    full_mass, *full = explicit_centered_moments(position, weight, index, 2)
    compact_mass, compact = compact_centered_moments(position, weight, index, 2)
    compact_one = compact[..., SYMMETRIC_DEGREE_SLICES[1]]
    compact_two = compact[..., SYMMETRIC_DEGREE_SLICES[2]]
    compact_three = compact[..., SYMMETRIC_DEGREE_SLICES[3]]
    compact_four = compact[..., SYMMETRIC_DEGREE_SLICES[4]]

    torch.testing.assert_close(compact_mass, full_mass, atol=2e-12, rtol=2e-12)
    # k=1 is already Cartesian.
    torch.testing.assert_close(compact_one, full[0], atol=2e-12, rtol=2e-12)

    # The compact degree-two block is [xx, xy, xz, yy, yz, zz].
    expected_two = torch.stack(
        (
            full[1][..., 0, 0],
            full[1][..., 0, 1],
            full[1][..., 0, 2],
            full[1][..., 1, 1],
            full[1][..., 1, 2],
            full[1][..., 2, 2],
        ),
        dim=-1,
    )
    torch.testing.assert_close(compact_two, expected_two, atol=3e-12, rtol=3e-12)

    # Recover the compact degree-three and degree-four monomials directly from
    # the full symmetric tensors in the same exponent order as the basis.
    expected_three = torch.stack(
        (
            full[2][..., 0, 0, 0],
            full[2][..., 0, 0, 1],
            full[2][..., 0, 0, 2],
            full[2][..., 0, 1, 1],
            full[2][..., 0, 1, 2],
            full[2][..., 0, 2, 2],
            full[2][..., 1, 1, 1],
            full[2][..., 1, 1, 2],
            full[2][..., 1, 2, 2],
            full[2][..., 2, 2, 2],
        ),
        dim=-1,
    )
    expected_four = torch.stack(
        (
            full[3][..., 0, 0, 0, 0],
            full[3][..., 0, 0, 0, 1],
            full[3][..., 0, 0, 0, 2],
            full[3][..., 0, 0, 1, 1],
            full[3][..., 0, 0, 1, 2],
            full[3][..., 0, 0, 2, 2],
            full[3][..., 0, 1, 1, 1],
            full[3][..., 0, 1, 1, 2],
            full[3][..., 0, 1, 2, 2],
            full[3][..., 0, 2, 2, 2],
            full[3][..., 1, 1, 1, 1],
            full[3][..., 1, 1, 1, 2],
            full[3][..., 1, 1, 2, 2],
            full[3][..., 1, 2, 2, 2],
            full[3][..., 2, 2, 2, 2],
        ),
        dim=-1,
    )
    torch.testing.assert_close(compact_three, expected_three, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(compact_four, expected_four, atol=2e-10, rtol=2e-10)


def test_compact_transient_contractions_equal_full_cartesian_contractions() -> None:
    generator = torch.Generator().manual_seed(405)
    nodes, rank = 4, 3
    vector = torch.randn(nodes, rank, 3, generator=generator, dtype=torch.float64)
    tensor = torch.randn(nodes, rank, 5, generator=generator, dtype=torch.float64)
    tensor_matrix = st_to_matrix(tensor)

    third_seed = torch.randn(nodes, rank, 3, generator=generator, dtype=torch.float64)
    full_third = stf3(
        torch.einsum("nra,nrb,nrc->nrabc", third_seed, third_seed, third_seed)
    )
    compact_third = compact_stf3(
        torch.stack(
            (
                full_third[..., 0, 0, 0],
                full_third[..., 0, 0, 1],
                full_third[..., 0, 0, 2],
                full_third[..., 0, 1, 1],
                full_third[..., 0, 1, 2],
                full_third[..., 0, 2, 2],
                full_third[..., 1, 1, 1],
                full_third[..., 1, 1, 2],
                full_third[..., 1, 2, 2],
                full_third[..., 2, 2, 2],
            ),
            dim=-1,
        )
    )
    torch.testing.assert_close(
        stf3_contract_st(compact_third, tensor),
        torch.einsum("nrabc,nrbc->nra", full_third, tensor_matrix),
        atol=3e-11,
        rtol=3e-11,
    )
    torch.testing.assert_close(
        stf3_contract_vector(compact_third, vector),
        matrix_to_st(torch.einsum("nrabc,nrc->nrab", full_third, vector)),
        atol=3e-11,
        rtol=3e-11,
    )

    fourth_seed = torch.randn(nodes, rank, 3, generator=generator, dtype=torch.float64)
    full_fourth = stf4(
        torch.einsum(
            "nra,nrb,nrc,nrd->nrabcd",
            fourth_seed,
            fourth_seed,
            fourth_seed,
            fourth_seed,
        )
    )
    full_fourth_compact = torch.stack(
        (
            full_fourth[..., 0, 0, 0, 0],
            full_fourth[..., 0, 0, 0, 1],
            full_fourth[..., 0, 0, 0, 2],
            full_fourth[..., 0, 0, 1, 1],
            full_fourth[..., 0, 0, 1, 2],
            full_fourth[..., 0, 0, 2, 2],
            full_fourth[..., 0, 1, 1, 1],
            full_fourth[..., 0, 1, 1, 2],
            full_fourth[..., 0, 1, 2, 2],
            full_fourth[..., 0, 2, 2, 2],
            full_fourth[..., 1, 1, 1, 1],
            full_fourth[..., 1, 1, 1, 2],
            full_fourth[..., 1, 1, 2, 2],
            full_fourth[..., 1, 2, 2, 2],
            full_fourth[..., 2, 2, 2, 2],
        ),
        dim=-1,
    )
    compact_fourth = compact_stf4(full_fourth_compact)
    torch.testing.assert_close(
        stf4_contract_st(compact_fourth, tensor),
        matrix_to_st(torch.einsum("nrabcd,nrcd->nrab", full_fourth, tensor_matrix)),
        atol=4e-10,
        rtol=4e-10,
    )
    torch.testing.assert_close(
        stf4_contract_st_vector(compact_fourth, tensor, vector),
        torch.einsum("nrabcd,nrbc,nrd->nra", full_fourth, tensor_matrix, vector),
        atol=4e-10,
        rtol=4e-10,
    )


def test_relation_uses_one_packed_value_and_compact_mercer_basis() -> None:
    generator = torch.Generator().manual_seed(407)
    nodes, width, heads = 7, 16, 2
    index = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    geometry = GeometryContext.build(
        torch.randn(nodes, 3, generator=generator, dtype=torch.float64),
        index,
        num_segments=2,
        eps=1e-10,
    )
    state = _state(nodes, width, heads, generator)
    relation = SelfAdjointRelation(
        scalar_width=width,
        num_heads=heads,
        feature_width=5,
        num_charts=3,
        eps=1e-10,
    ).double()
    value, content = relation.project(state)
    factors = relation.build(state, geometry, content_feature=content)
    result = relation.apply(factors, value)

    assert value.data.shape == (nodes, heads, width // heads + 17)
    assert result.data.shape == value.data.shape
    assert factors.mercer_feature.shape[-1] == 35
    torch.testing.assert_close(
        RelationMessage.from_packed(value.data, head_dim=value.head_dim).data,
        value.data,
    )

    expected = torch.zeros_like(result.data)
    for graph in range(2):
        selected = index == graph
        for head in range(heads):
            feature = factors.feature[selected, head]
            dense = feature @ feature.T
            torch.testing.assert_close(dense, dense.T, atol=2e-12, rtol=0.0)
            assert float(torch.linalg.eigvalsh(dense).min().detach()) >= -3e-10
            expected[selected, head] = dense @ value.data[selected, head]
    torch.testing.assert_close(result.data, expected, atol=4e-11, rtol=4e-11)


def test_adaptive_bank_stores_only_irreducible_transient_components() -> None:
    generator = torch.Generator().manual_seed(409)
    nodes = 8
    index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    geometry = GeometryContext.build(
        torch.randn(nodes, 3, generator=generator, dtype=torch.float64),
        index,
        num_segments=2,
        eps=1e-10,
    )
    bank = AdaptiveMomentBank(scalar_width=16, rank=4, eps=1e-10).double()
    moment = bank(
        torch.randn(nodes, 16, generator=generator, dtype=torch.float64),
        geometry,
    )
    assert geometry.monomials.shape == (nodes, 35)
    assert moment.third_tensor.shape == (nodes, 4, 7)
    assert moment.fourth_rank4.shape == (nodes, 4, 9)
    assert all(torch.isfinite(value).all() for value in moment.__dict__.values())


def test_mercer_feature_dimension_formula() -> None:
    assert sum(math.comb(order + 2, 2) for order in range(5)) == 35
    feature = MercerFeatures(num_heads=3).double()(
        torch.randn(5, 3, dtype=torch.float64),
        symmetric_monomials(torch.randn(5, 3, dtype=torch.float64)),
    )
    assert feature.shape == (5, 3, 35)
