from __future__ import annotations

import pytest

from equivariant_attention.irreps import (
    CartesianIrreps,
    Irrep,
    IrrepBlock,
    IrrepLayout,
    TensorProductPath,
    TensorProductPlan,
)


def test_irrep_layout_parses_canonicalizes_and_assigns_flat_slices() -> None:
    layout = IrrepLayout.parse(
        "2x3o + 1x0e + 2x1e + 3x3o + 4x2o + 1x1e"
    )

    assert layout.blocks == (
        IrrepBlock(1, Irrep(0, "e")),
        IrrepBlock(3, Irrep(1, "e")),
        IrrepBlock(4, Irrep(2, "o")),
        IrrepBlock(5, Irrep(3, "o")),
    )
    assert str(layout) == "1x0e + 3x1e + 4x2o + 5x3o"
    assert layout.dim == 1 + 3 * 3 + 4 * 5 + 5 * 7
    assert layout.slices == {
        Irrep(0, "e"): slice(0, 1),
        Irrep(1, "e"): slice(1, 10),
        Irrep(2, "o"): slice(10, 30),
        Irrep(3, "o"): slice(30, 65),
    }
    assert layout.slice_for("2o") == slice(10, 30)
    assert layout.multiplicity("3o") == 5
    assert layout.max_degree == 3


def test_irrep_and_layout_validate_arbitrary_degree_terms() -> None:
    assert Irrep.parse("12o") == Irrep(12, "o")
    assert Irrep(12, "o").dim == 25
    assert IrrepLayout.parse("0").blocks == ()
    assert IrrepLayout.parse("0").dim == 0

    with pytest.raises(ValueError, match="nonnegative integer"):
        Irrep(-1, "e")
    with pytest.raises(ValueError, match="parity"):
        Irrep(1, "bad")
    with pytest.raises(ValueError, match="positive integer"):
        IrrepBlock(0, Irrep(0, "e"))
    with pytest.raises(ValueError, match="unsupported irreps term"):
        IrrepLayout.parse("2x1")
    with pytest.raises(ValueError, match="must not be empty"):
        IrrepLayout.parse("")


def test_o3_tensor_product_plan_obeys_triangle_and_parity_rules() -> None:
    plan = TensorProductPlan.compile(
        "1x2e",
        "1x1o",
        output="1x0e + 1x1o + 1x2e + 1x2o + 1x3o + 1x4o",
        symmetry_group="O3",
    )

    assert plan.signatures == (
        ("2e", "1o", "1o"),
        ("2e", "1o", "2o"),
        ("2e", "1o", "3o"),
    )
    assert all(not path.parity_mixed for path in plan.paths)


def test_se3_tensor_product_plan_allows_explicit_parity_mixing() -> None:
    o3 = TensorProductPlan.compile(
        "1x2e",
        "1x2e",
        output="1x1e + 1x1o",
        symmetry_group="O3",
    )
    se3 = TensorProductPlan.compile(
        "1x2e",
        "1x2e",
        output="1x1e + 1x1o",
        symmetry_group="SE3",
    )

    assert o3.signatures == (("2e", "2e", "1e"),)
    assert se3.signatures == (
        ("2e", "2e", "1e"),
        ("2e", "2e", "1o"),
    )
    assert not se3.paths[0].parity_mixed
    assert se3.paths[1].parity_mixed


def test_plan_without_output_enumerates_each_o3_coupling_once() -> None:
    plan = TensorProductPlan.compile("1x1o", "1x1o", symmetry_group="O3")

    assert plan.output == IrrepLayout.parse("1x0e + 1x1e + 1x2e")
    assert plan.signatures == (
        ("1o", "1o", "0e"),
        ("1o", "1o", "1e"),
        ("1o", "1o", "2e"),
    )


def test_plan_binds_only_explicitly_supported_executors() -> None:
    plan = TensorProductPlan.compile(
        "1x2e + 1x1o",
        "1x1o + 1x0e",
        output="1x1o + 1x2e",
        symmetry_group="O3",
    )
    supported = {
        ("2e", "1o", "1o"): "tensor_direction",
        ("2e", "0e", "2e"): "tensor_passthrough",
        ("1o", "1o", "2e"): "vector_direction",
        ("1o", "0e", "1o"): "vector_passthrough",
    }

    with pytest.raises(ValueError, match="unsupported tensor-product paths"):
        plan.bind_executors(
            {
                ("2e", "1o", "1o"): "tensor_direction",
            }
        )

    bound = plan.bind_executors(supported)
    assert tuple(path.executor for path in bound.paths) == (
        "vector_passthrough",
        "vector_direction",
        "tensor_passthrough",
        "tensor_direction",
    )
    assert all(path.executor is None for path in plan.paths)


def test_plan_and_path_reject_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="symmetry_group"):
        TensorProductPlan.compile("1x1o", "1x1o", symmetry_group="SO3")
    with pytest.raises(ValueError, match="selection rule"):
        TensorProductPath(Irrep(0, "e"), Irrep(0, "e"), Irrep(1, "e"))
    with pytest.raises(ValueError, match="non-empty string"):
        TensorProductPath(
            Irrep(0, "e"),
            Irrep(0, "e"),
            Irrep(0, "e"),
            executor="",
        )


def test_cartesian_irreps_keeps_its_narrow_legacy_contract() -> None:
    parsed = CartesianIrreps.parse("2x2e + 3x0e + 4x1o + 1x2e")

    assert parsed == CartesianIrreps(scalars=3, vectors=4, tensors=3)
    assert str(parsed) == "3x0e + 4x1o + 3x2e"
    assert parsed.dim == 30
    assert parsed.storage_dim == 42
    with pytest.raises(ValueError, match="supports only"):
        CartesianIrreps.parse("1x1e")
    with pytest.raises(ValueError, match="unsupported irreps term"):
        CartesianIrreps.parse("1x3o")
