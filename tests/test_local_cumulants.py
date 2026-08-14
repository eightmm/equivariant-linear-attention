from __future__ import annotations

import pytest
import torch
from conftest import orthogonal

from equivariant_linear_attention.nn.geometry import GeometryContext
from equivariant_linear_attention.nn.local_moments import LocalCumulantBank
from equivariant_linear_attention.nn.local_support import build_local_support
from equivariant_linear_attention.nn.ops import matrix_to_st, st_to_matrix


def _geometry(position: torch.Tensor, index: torch.Tensor) -> GeometryContext:
    return GeometryContext.build(
        position,
        index,
        num_segments=int(index.max().item()) + 1,
        eps=1e-10,
    )


def _rotate_st(value: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    matrix = st_to_matrix(value)
    moved = torch.einsum("ia,...ab,jb->...ij", transform, matrix, transform)
    return matrix_to_st(moved)


@pytest.mark.parametrize("reflection", [False, True])
def test_local_cumulants_obey_o3_and_parity(reflection: bool) -> None:
    generator = torch.Generator().manual_seed(509)
    position = torch.randn(12, 3, generator=generator, dtype=torch.float64)
    scalar = torch.randn(12, 8, generator=generator, dtype=torch.float64)
    index = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    bank = LocalCumulantBank(scalar_width=8, rank=4, eps=1e-10).double()
    reference = bank(
        scalar,
        build_local_support(
            _geometry(position, index), max_points=8, chunk_size=4, eps=1e-10
        ),
    )

    transform = orthogonal(reflection=reflection, seed=521)
    determinant = torch.linalg.det(transform)
    moved = bank(
        scalar,
        build_local_support(
            _geometry(
                position @ transform.T + torch.tensor([2.0, -1.0, 3.0]), index
            ),
            max_points=8,
            chunk_size=4,
            eps=1e-10,
        ),
    )
    for original, transformed in (
        (reference.mass, moved.mass),
        (reference.second_scalar, moved.second_scalar),
        (reference.fourth_scalar, moved.fourth_scalar),
    ):
        torch.testing.assert_close(transformed, original, atol=6e-10, rtol=6e-10)
    torch.testing.assert_close(
        moved.odd_scalar,
        determinant * reference.odd_scalar,
        atol=6e-10,
        rtol=6e-10,
    )
    torch.testing.assert_close(
        moved.polar, reference.polar @ transform.T, atol=6e-10, rtol=6e-10
    )
    torch.testing.assert_close(
        moved.axial,
        determinant * (reference.axial @ transform.T),
        atol=6e-10,
        rtol=6e-10,
    )
    torch.testing.assert_close(
        moved.even_tensor,
        _rotate_st(reference.even_tensor, transform),
        atol=7e-10,
        rtol=7e-10,
    )
    torch.testing.assert_close(
        moved.odd_tensor,
        determinant * _rotate_st(reference.odd_tensor, transform),
        atol=7e-10,
        rtol=7e-10,
    )
