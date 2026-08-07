from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention.advanced import (
    IrrepLayout,
    TensorProductPlan,
    matrix_to_st5,
    pack_irreps,
    split_irreps,
    st5_mse,
    st5_to_matrix,
)


def test_layout_pack_split_and_canonicalization() -> None:
    layout = IrrepLayout.parse("1x1o + 2x0e + 1x1o + 1x2e")
    assert str(layout) == "2x0e + 2x1o + 1x2e"
    blocks = {
        "0e": torch.randn(4, 2, 1),
        "1o": torch.randn(4, 2, 3),
        "2e": torch.randn(4, 1, 5),
    }
    packed = pack_irreps(layout, blocks)
    restored = split_irreps(layout, packed)
    for name, value in blocks.items():
        torch.testing.assert_close(restored[name], value)


def test_tensor_product_selection_rules() -> None:
    plan = TensorProductPlan.compile("1x1o", "1x1o")
    assert set(plan.signatures) == {
        ("1o", "1o", "0e"),
        ("1o", "1o", "1e"),
        ("1o", "1o", "2e"),
    }
    with pytest.raises(ValueError, match="parity"):
        TensorProductPlan.compile("1x1o", "1x1o", output="1x1o")


def test_st5_roundtrip_and_invariant_loss() -> None:
    matrix = torch.randn(6, 3, 3, dtype=torch.float64)
    compact = matrix_to_st5(matrix)
    restored = st5_to_matrix(compact)
    torch.testing.assert_close(restored.diagonal(dim1=-2, dim2=-1).sum(-1), torch.zeros(6, dtype=torch.float64))
    assert float(st5_mse(compact, compact)) == 0.0
