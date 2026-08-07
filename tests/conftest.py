from __future__ import annotations

import torch

from equivariant_linear_attention.irreps import pack_irreps, split_irreps
from equivariant_linear_attention.nn.ops import matrix_to_st, st_to_matrix


def orthogonal(*, reflection: bool, seed: int = 123) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + int(reflection))
    value, _ = torch.linalg.qr(
        torch.randn(3, 3, generator=generator, dtype=torch.float64)
    )
    if bool((torch.linalg.det(value) < 0).item()) != reflection:
        value[:, 0].neg_()
    return value


def transform_irreps(
    value: torch.Tensor, layout, transform: torch.Tensor
) -> torch.Tensor:
    determinant = torch.linalg.det(transform)
    blocks = split_irreps(layout, value)
    output: dict[str, torch.Tensor] = {}
    for name, block in blocks.items():
        if name == "0e":
            output[name] = block
        elif name == "0o":
            output[name] = determinant * block
        elif name == "1o":
            output[name] = block @ transform.T
        elif name == "1e":
            output[name] = determinant * (block @ transform.T)
        elif name in {"2e", "2o"}:
            matrix = st_to_matrix(block)
            moved = torch.einsum(
                "ia,...ab,jb->...ij",
                transform,
                matrix,
                transform,
            )
            tensor = matrix_to_st(moved)
            output[name] = tensor if name == "2e" else determinant * tensor
        else:  # pragma: no cover - persistent model contract
            raise ValueError(name)
    return pack_irreps(layout, output)
