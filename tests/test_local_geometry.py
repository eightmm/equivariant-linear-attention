from __future__ import annotations

import pytest
import torch
from conftest import orthogonal

from equivariant_linear_attention.nn.geometry import GeometryContext, MomentFeatures
from equivariant_linear_attention.nn.local_geometry import (
    LocalMercerMomentBank,
    LocalMomentFusion,
)
from equivariant_linear_attention.nn.ops import (
    bounded_scalar,
    bounded_st,
    matrix_to_st,
    st_to_matrix,
    unit_ball,
)


def test_degree_two_mercer_kernel_is_local_and_rotation_invariant() -> None:
    bank = LocalMercerMomentBank(scalar_width=8, rank=4, eps=1e-10).double()
    position = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [1.6, 0.0, 0.0]],
        dtype=torch.float64,
    )
    feature = bank.feature_map(position)
    kernel = torch.einsum("nrp,mrp->rnm", feature, feature)
    assert bool((kernel[:, 0, 1] > kernel[:, 0, 2]).all().item())

    transform = orthogonal(reflection=True, seed=401)
    moved_feature = bank.feature_map(position @ transform.T)
    moved_kernel = torch.einsum("nrp,mrp->rnm", moved_feature, moved_feature)
    torch.testing.assert_close(moved_kernel, kernel, atol=2e-10, rtol=2e-10)


def test_segmented_mercer_reduction_matches_explicit_local_oracle() -> None:
    generator = torch.Generator().manual_seed(405)
    bank = LocalMercerMomentBank(scalar_width=6, rank=4, eps=1e-10).double()
    position = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    scalar = torch.randn(7, 6, generator=generator, dtype=torch.float64)
    index = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    geometry = GeometryContext.build(position, index, num_segments=2, eps=1e-10)
    output = bank(scalar, geometry)

    coordinate = geometry.normalized
    feature = bank.feature_map(coordinate, geometry.monomials)
    kernel = torch.einsum("nrp,mrp->rnm", feature, feature)
    radius2 = coordinate.square().sum(dim=-1)
    radial = torch.stack(
        (radius2, torch.log1p(radius2), radius2 / (1.0 + radius2)),
        dim=-1,
    )
    temperature = torch.nn.functional.softplus(bank.raw_temperature) + 0.25
    source_weight = torch.nn.functional.softplus(
        (bank.content_weight(scalar) + bank.radial_weight(radial)) / temperature
    ) + bank.eps

    explicit_mass = torch.zeros_like(output.mass)
    explicit_m1 = torch.zeros_like(output.polar)
    explicit_m2 = torch.zeros(
        position.shape[0],
        bank.rank,
        3,
        3,
        dtype=torch.float64,
    )
    for receiver in range(position.shape[0]):
        same = index == index[receiver]
        delta = coordinate[same] - coordinate[receiver]
        local_weight = (
            source_weight[same]
            * kernel[:, receiver, same].transpose(0, 1)
        )
        denominator = local_weight.sum(dim=0)
        explicit_mass[receiver] = torch.log1p(denominator)
        explicit_m1[receiver] = torch.einsum(
            "jr,ja->ra", local_weight, delta
        ) / denominator[:, None]
        explicit_m2[receiver] = torch.einsum(
            "jr,ja,jb->rab", local_weight, delta, delta
        ) / denominator[:, None, None]

    explicit_scalar = explicit_m2.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
    explicit_tensor = matrix_to_st(explicit_m2)
    torch.testing.assert_close(output.mass, explicit_mass, atol=5e-10, rtol=5e-10)
    torch.testing.assert_close(
        output.polar,
        unit_ball(explicit_m1, bank.eps),
        atol=5e-10,
        rtol=5e-10,
    )
    torch.testing.assert_close(
        output.second_scalar,
        bounded_scalar(explicit_scalar, bank.eps),
        atol=5e-10,
        rtol=5e-10,
    )
    torch.testing.assert_close(
        output.even_tensor,
        bounded_st(explicit_tensor, bank.eps),
        atol=5e-10,
        rtol=5e-10,
    )


@pytest.mark.parametrize("reflection", [False, True])
def test_local_moments_obey_o3_and_translation(reflection: bool) -> None:
    generator = torch.Generator().manual_seed(409)
    bank = LocalMercerMomentBank(scalar_width=8, rank=4, eps=1e-10).double()
    position = torch.randn(11, 3, generator=generator, dtype=torch.float64)
    scalar = torch.randn(11, 8, generator=generator, dtype=torch.float64)
    index = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    geometry = GeometryContext.build(position, index, num_segments=2, eps=1e-10)
    reference = bank(scalar, geometry)

    transform = orthogonal(reflection=reflection, seed=419)
    determinant = torch.linalg.det(transform)
    translation = torch.tensor([2.0, -3.0, 0.5], dtype=torch.float64)
    moved_geometry = GeometryContext.build(
        position @ transform.T + translation,
        index,
        num_segments=2,
        eps=1e-10,
    )
    moved = bank(scalar, moved_geometry)

    for original, transformed in (
        (reference.mass, moved.mass),
        (reference.second_scalar, moved.second_scalar),
        (reference.pair_scalar, moved.pair_scalar),
    ):
        torch.testing.assert_close(transformed, original, atol=5e-10, rtol=5e-10)
    torch.testing.assert_close(
        moved.odd_scalar,
        determinant * reference.odd_scalar,
        atol=5e-10,
        rtol=5e-10,
    )
    torch.testing.assert_close(
        moved.polar,
        reference.polar @ transform.T,
        atol=5e-10,
        rtol=5e-10,
    )
    torch.testing.assert_close(
        moved.axial,
        determinant * (reference.axial @ transform.T),
        atol=5e-10,
        rtol=5e-10,
    )
    for original, transformed in (
        (reference.even_tensor, moved.even_tensor),
        (reference.pair_tensor, moved.pair_tensor),
    ):
        expected = torch.einsum(
            "ia,...ab,jb->...ij",
            transform,
            st_to_matrix(original),
            transform,
        )
        torch.testing.assert_close(
            st_to_matrix(transformed),
            expected,
            atol=5e-10,
            rtol=5e-10,
        )
    expected_odd = determinant * torch.einsum(
        "ia,...ab,jb->...ij",
        transform,
        st_to_matrix(reference.odd_tensor),
        transform,
    )
    torch.testing.assert_close(
        st_to_matrix(moved.odd_tensor),
        expected_odd,
        atol=5e-10,
        rtol=5e-10,
    )


def test_zero_local_fusion_preserves_global_moments() -> None:
    generator = torch.Generator().manual_seed(421)
    nodes, rank = 5, 4

    def sample(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    global_moments = MomentFeatures(
        mass=sample(nodes, rank),
        polar=sample(nodes, rank, 3),
        second_scalar=sample(nodes, rank),
        even_tensor=sample(nodes, rank, 5),
        axial=sample(nodes, rank, 3),
        odd_scalar=sample(nodes, rank),
        odd_tensor=sample(nodes, rank, 5),
        third_tensor=sample(nodes, rank, 7),
        fourth_scalar=sample(nodes, rank),
        fourth_tensor=sample(nodes, rank, 5),
        fourth_rank4=sample(nodes, rank, 9),
    )
    bank = LocalMercerMomentBank(scalar_width=8, rank=rank, eps=1e-10).double()
    position = sample(nodes, 3)
    index = torch.zeros(nodes, dtype=torch.long)
    local_moments = bank(
        sample(nodes, 8),
        GeometryContext.build(position, index, num_segments=1, eps=1e-10),
    )
    fused = LocalMomentFusion(rank=rank).double()(global_moments, local_moments)
    for name in global_moments.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(fused, name),
            getattr(global_moments, name),
        )


def test_local_moment_path_has_finite_gradients() -> None:
    generator = torch.Generator().manual_seed(431)
    bank = LocalMercerMomentBank(scalar_width=8, rank=4, eps=1e-8)
    scalar = torch.randn(9, 8, generator=generator, requires_grad=True)
    position = torch.randn(9, 3, generator=generator, requires_grad=True)
    index = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])
    geometry = GeometryContext.build(position, index, num_segments=2, eps=1e-8)
    local = bank(scalar, geometry)
    loss = (
        local.mass.mean()
        + local.polar.square().mean()
        + local.even_tensor.square().mean()
        + local.odd_scalar.square().mean()
    )
    loss.backward()
    assert scalar.grad is not None and bool(torch.isfinite(scalar.grad).all().item())
    assert position.grad is not None and bool(torch.isfinite(position.grad).all().item())
    assert bank.raw_gamma.grad is not None
    assert bool(torch.isfinite(bank.raw_gamma.grad).all().item())
