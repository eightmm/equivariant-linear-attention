from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch.func import functional_call

from equivariant_attention.irreps import Irrep, IrrepLayout, TensorProductPlan
from equivariant_attention.reference_irreps import (
    IrrepLinear,
    IrrepRMSNorm,
    ReferenceTensorProductPath,
    ScalarGatedIrreps,
    tensor_product_path,
    transform_irreps,
)
from equivariant_attention.spherical import (
    real_clebsch_gordan,
    real_spherical_harmonics,
)


def _orthogonal(size: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(size, size, dtype=torch.float64, generator=generator)
    q, r = torch.linalg.qr(matrix)
    signs = torch.where(
        r.diagonal() >= 0,
        torch.ones(size, dtype=torch.float64),
        -torch.ones(size, dtype=torch.float64),
    )
    return q * signs


def _spherical_representation(degree: int, rotation: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1400 + degree)
    calibration = torch.randn(
        max(48, 6 * (2 * degree + 1)),
        3,
        dtype=torch.float64,
        generator=generator,
    )
    source = real_spherical_harmonics(degree, calibration)
    rotated = real_spherical_harmonics(degree, calibration @ rotation.mT)
    return torch.linalg.lstsq(source, rotated).solution.mT


def _rotation() -> torch.Tensor:
    matrix = _orthogonal(3, seed=713)
    if torch.linalg.det(matrix) < 0:
        matrix = matrix.clone()
        matrix[:, 0].neg_()
    return matrix


def _manual_linear_block(
    module: IrrepLinear,
    value: torch.Tensor,
    output_irrep: str,
) -> torch.Tensor:
    parsed_output = Irrep.parse(output_irrep)
    output_multiplicity = module.output_layout.multiplicity(parsed_output)
    result = value.new_zeros(
        *value.shape[:-1],
        output_multiplicity,
        parsed_output.dim,
    )
    for path in module.mixing_paths:
        if path.output != parsed_output:
            continue
        source = value[..., module.input_layout.slice_for(path.input)]
        source = source.reshape(
            *value.shape[:-1],
            module.input_layout.multiplicity(path.input),
            path.input.dim,
        )
        result = result + torch.einsum(
            "oi,...im->...om",
            module.weight_for(path.input, path.output),
            source,
        )
    return result


def test_irrep_linear_obeys_o3_and_se3_parity_mixing_rules() -> None:
    input_layout = IrrepLayout.parse("2x1e + 3x1o + 2x3e")
    output_layout = IrrepLayout.parse("1x1e + 2x1o + 1x3e + 2x3o")

    o3 = IrrepLinear(
        input_layout,
        output_layout,
        symmetry_group="O3",
        dtype=torch.float64,
    )
    assert {
        (str(path.input), str(path.output)) for path in o3.mixing_paths
    } == {("1e", "1e"), ("1o", "1o"), ("3e", "3e")}

    se3 = IrrepLinear(
        input_layout,
        output_layout,
        symmetry_group="SE3",
        dtype=torch.float64,
    )
    assert {
        (str(path.input), str(path.output)) for path in se3.mixing_paths
    } == {
        ("1e", "1e"),
        ("1e", "1o"),
        ("1o", "1e"),
        ("1o", "1o"),
        ("3e", "3e"),
        ("3e", "3o"),
    }
    assert se3.parity_is_provenance_only
    assert not o3.parity_is_provenance_only


@pytest.mark.parametrize("symmetry_group", ["O3", "SE3"])
def test_irrep_linear_matches_copy_mixing_einsum(
    symmetry_group: str,
) -> None:
    input_layout = IrrepLayout.parse("2x1e + 3x1o + 2x3e")
    output_layout = IrrepLayout.parse("1x1e + 2x1o + 1x3e + 2x3o")
    module = IrrepLinear(
        input_layout,
        output_layout,
        symmetry_group=symmetry_group,
        dtype=torch.float64,
    )
    with torch.no_grad():
        for index, parameter in enumerate(module.parameters(), start=1):
            parameter.copy_(
                torch.arange(
                    1,
                    parameter.numel() + 1,
                    dtype=torch.float64,
                ).reshape_as(parameter)
                / (10 * index)
            )

    value = torch.arange(
        2 * 3 * input_layout.dim,
        dtype=torch.float64,
    ).reshape(2, 3, input_layout.dim) / 50
    observed = module(value)
    expected = torch.cat(
        [
            _manual_linear_block(module, value, str(block.irrep)).flatten(-2)
            for block in output_layout.blocks
        ],
        dim=-1,
    )
    torch.testing.assert_close(observed, expected)


def test_linear_norm_and_scalar_gates_commute_with_arbitrary_irrep_actions() -> None:
    layout = IrrepLayout.parse("2x0e + 3x3o + 2x4e")
    matrices = {
        block.irrep: _orthogonal(block.irrep.dim, seed=900 + block.irrep.degree)
        for block in layout.blocks
    }
    generator = torch.Generator().manual_seed(73)
    value = torch.randn(5, layout.dim, dtype=torch.float64, generator=generator)

    linear = IrrepLinear(layout, layout, dtype=torch.float64)
    norm = IrrepRMSNorm(layout, eps=1e-6, dtype=torch.float64)
    gates = ScalarGatedIrreps(layout, activation="silu")
    gate_value = torch.randn(
        5,
        gates.num_gates,
        dtype=torch.float64,
        generator=generator,
    )

    transformed = transform_irreps(value, layout, matrices)
    torch.testing.assert_close(
        linear(transformed),
        transform_irreps(linear(value), layout, matrices),
        atol=2e-12,
        rtol=2e-12,
    )
    torch.testing.assert_close(
        norm(transformed),
        transform_irreps(norm(value), layout, matrices),
        atol=2e-12,
        rtol=2e-12,
    )
    torch.testing.assert_close(
        gates(transformed, gate_value),
        transform_irreps(gates(value, gate_value), layout, matrices),
        atol=2e-12,
        rtol=2e-12,
    )


def test_irrep_rms_norm_is_independent_for_every_copy() -> None:
    layout = IrrepLayout.parse("2x0e + 3x3o + 2x4e")
    module = IrrepRMSNorm(layout, eps=0.0, dtype=torch.float64)
    with torch.no_grad():
        for block_index, block in enumerate(layout.blocks, start=1):
            module.weight_for(block.irrep).copy_(
                torch.arange(
                    block_index,
                    block_index + block.multiplicity,
                    dtype=torch.float64,
                )
            )
    generator = torch.Generator().manual_seed(147)
    value = torch.randn(4, layout.dim, dtype=torch.float64, generator=generator)
    observed = module(value)

    for block in layout.blocks:
        block_value = observed[..., layout.slice_for(block.irrep)].reshape(
            4,
            block.multiplicity,
            block.irrep.dim,
        )
        expected_rms = module.weight_for(block.irrep).abs().expand(4, -1)
        torch.testing.assert_close(
            block_value.square().mean(dim=-1).sqrt(),
            expected_rms,
        )


def test_scalar_gates_leave_scalars_unchanged_and_gate_each_nonscalar_copy() -> None:
    layout = IrrepLayout.parse("2x0o + 2x3e + 1x4o")
    module = ScalarGatedIrreps(layout, activation="sigmoid")
    assert module.num_gates == 3
    value = torch.arange(2 * layout.dim, dtype=torch.float64).reshape(
        2, layout.dim
    )
    raw_gates = torch.tensor(
        [[-2.0, 0.0, 2.0], [1.0, -1.0, 0.5]],
        dtype=torch.float64,
    )
    observed = module(value, raw_gates)

    scalar_slice = layout.slice_for("0o")
    torch.testing.assert_close(observed[..., scalar_slice], value[..., scalar_slice])
    gate_offset = 0
    for block in layout.blocks:
        if block.irrep.degree == 0:
            continue
        source = value[..., layout.slice_for(block.irrep)].reshape(
            2,
            block.multiplicity,
            block.irrep.dim,
        )
        expected = source * raw_gates[
            ..., gate_offset : gate_offset + block.multiplicity
        ].sigmoid().unsqueeze(-1)
        actual = observed[..., layout.slice_for(block.irrep)].reshape_as(expected)
        torch.testing.assert_close(actual, expected)
        gate_offset += block.multiplicity

    with pytest.raises(ValueError, match="even scalar"):
        ScalarGatedIrreps(layout, symmetry_group="O3", gate_parity="o")
    assert ScalarGatedIrreps(
        layout,
        symmetry_group="SE3",
        gate_parity="o",
    ).parity_is_provenance_only


def test_tensor_product_path_matches_manual_cg_and_multiplicity_einsums() -> None:
    path = ReferenceTensorProductPath.parse("3o", "1e", "4o")
    generator = torch.Generator().manual_seed(888)
    left = torch.randn(2, 2, 7, dtype=torch.float64, generator=generator)
    right = torch.randn(2, 3, 3, dtype=torch.float64, generator=generator)
    weights = torch.randn(4, 2, 3, dtype=torch.float64, generator=generator)
    coefficients = real_clebsch_gordan(
        3,
        1,
        4,
        dtype=torch.float64,
        device="cpu",
    )
    pairwise = torch.einsum(
        "Mab,...ua,...vb->...uvM",
        coefficients,
        left,
        right,
    )

    torch.testing.assert_close(
        tensor_product_path(left, right, path),
        pairwise,
    )
    torch.testing.assert_close(
        tensor_product_path(left, right, path, weights=weights),
        torch.einsum("ouv,...uvM->...oM", weights, pairwise),
    )


def test_reference_path_binds_plan_selection_but_keeps_weights_external() -> None:
    plan = TensorProductPlan.compile(
        "2x3o",
        "3x1e",
        output="4x4o",
        symmetry_group="O3",
    )
    path = ReferenceTensorProductPath.from_plan(plan, 0)
    assert path == ReferenceTensorProductPath.parse("3o", "1e", "4o")

    left = torch.randn(2, 2, 7, dtype=torch.float64)
    right = torch.randn(2, 3, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="weights"):
        tensor_product_path(
            left,
            right,
            path,
            weights=torch.randn(4, 3, 2, dtype=torch.float64),
        )


def test_tensor_product_path_has_explicit_o3_and_se3_parity_contract() -> None:
    with pytest.raises(ValueError, match="parity"):
        ReferenceTensorProductPath.parse("3o", "1e", "4e", symmetry_group="O3")

    path = ReferenceTensorProductPath.parse(
        "3o",
        "1e",
        "4e",
        symmetry_group="SE3",
    )
    assert path.parity_mixed
    assert path.parity_is_provenance_only


def test_high_degree_tensor_product_is_rotation_equivariant() -> None:
    rotation = _rotation()
    representations = {
        degree: _spherical_representation(degree, rotation)
        for degree in (1, 3, 4)
    }
    for representation in representations.values():
        torch.testing.assert_close(
            representation @ representation.mT,
            torch.eye(representation.shape[0], dtype=torch.float64),
            atol=3e-12,
            rtol=3e-12,
        )

    generator = torch.Generator().manual_seed(209)
    left = torch.randn(3, 2, 7, dtype=torch.float64, generator=generator)
    right = torch.randn(3, 3, 3, dtype=torch.float64, generator=generator)
    weights = torch.randn(4, 2, 3, dtype=torch.float64, generator=generator)
    path = ReferenceTensorProductPath.parse("3o", "1e", "4o")

    rotated_left = torch.einsum("ab,...ub->...ua", representations[3], left)
    rotated_right = torch.einsum("ab,...ub->...ua", representations[1], right)
    observed = tensor_product_path(
        rotated_left,
        rotated_right,
        path,
        weights=weights,
    )
    expected = torch.einsum(
        "ab,...ub->...ua",
        representations[4],
        tensor_product_path(left, right, path, weights=weights),
    )
    torch.testing.assert_close(observed, expected, atol=4e-12, rtol=4e-12)

    inverted = tensor_product_path(-left, right, path, weights=weights)
    torch.testing.assert_close(
        inverted,
        -tensor_product_path(left, right, path, weights=weights),
    )


def test_transform_irreps_matches_manual_block_actions_with_multiplicity() -> None:
    layout = IrrepLayout.parse("2x0e + 3x3o + 2x4e")
    matrices = {
        block.irrep: _orthogonal(block.irrep.dim, seed=120 + block.irrep.degree)
        for block in layout.blocks
    }
    value = torch.arange(
        2 * 3 * layout.dim,
        dtype=torch.float64,
    ).reshape(2, 3, layout.dim)
    expected_blocks = []
    for block in layout.blocks:
        source = value[..., layout.slice_for(block.irrep)].reshape(
            2,
            3,
            block.multiplicity,
            block.irrep.dim,
        )
        expected_blocks.append(
            torch.einsum("ij,...cj->...ci", matrices[block.irrep], source).flatten(-2)
        )
    torch.testing.assert_close(
        transform_irreps(value, layout, matrices),
        torch.cat(expected_blocks, dim=-1),
    )


def test_fp64_modules_and_tensor_product_have_parameter_gradgrad() -> None:
    linear_layout = IrrepLayout.parse("2x1o + 1x3e")
    linear = IrrepLinear(
        linear_layout,
        linear_layout,
        dtype=torch.float64,
    )
    linear_names = tuple(dict(linear.named_parameters()))
    linear_parameters = tuple(linear.parameters())
    linear_input = torch.randn(
        2,
        linear_layout.dim,
        dtype=torch.float64,
        requires_grad=True,
    )

    def evaluate_linear(
        value: torch.Tensor,
        *parameters: torch.Tensor,
    ) -> torch.Tensor:
        return functional_call(
            linear,
            dict(zip(linear_names, parameters, strict=True)),
            (value,),
        )

    assert torch.autograd.gradgradcheck(
        evaluate_linear,
        (linear_input, *linear_parameters),
        fast_mode=True,
    )

    norm = IrrepRMSNorm(
        IrrepLayout.parse("2x3o"),
        eps=1e-4,
        dtype=torch.float64,
    )
    norm_names = tuple(dict(norm.named_parameters()))
    norm_parameters = tuple(norm.parameters())
    norm_input = torch.randn(2, 14, dtype=torch.float64, requires_grad=True)

    def evaluate_norm(
        value: torch.Tensor,
        *parameters: torch.Tensor,
    ) -> torch.Tensor:
        return functional_call(
            norm,
            dict(zip(norm_names, parameters, strict=True)),
            (value,),
        )

    assert torch.autograd.gradgradcheck(
        evaluate_norm,
        (norm_input, *norm_parameters),
        fast_mode=True,
    )

    path = ReferenceTensorProductPath.parse("1o", "1o", "2e")
    left = torch.randn(2, 1, 3, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(2, 1, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(
        lambda x, y, w: tensor_product_path(x, y, path, weights=w),
        (left, right, weights),
        fast_mode=True,
    )


def test_bfloat16_uses_a_finite_float32_reference_compute_path() -> None:
    layout = IrrepLayout.parse("2x0e + 2x3o + 1x4e")
    value = torch.randn(3, layout.dim, dtype=torch.bfloat16, requires_grad=True)
    matrices = {
        block.irrep: _orthogonal(
            block.irrep.dim,
            seed=500 + block.irrep.degree,
        ).to(torch.bfloat16)
        for block in layout.blocks
    }
    linear = IrrepLinear(layout, layout, dtype=torch.bfloat16)
    norm = IrrepRMSNorm(layout, dtype=torch.bfloat16)
    gates = ScalarGatedIrreps(layout)
    raw_gates = torch.randn(3, gates.num_gates, dtype=torch.bfloat16)

    outputs = (
        linear(value),
        norm(value),
        gates(value, raw_gates),
        transform_irreps(value, layout, matrices),
    )
    path = ReferenceTensorProductPath.parse("3o", "1o", "4e")
    left = torch.randn(3, 2, 7, dtype=torch.bfloat16, requires_grad=True)
    right = torch.randn(3, 1, 3, dtype=torch.bfloat16, requires_grad=True)
    weights = torch.randn(2, 2, 1, dtype=torch.bfloat16, requires_grad=True)
    outputs += (tensor_product_path(left, right, path, weights=weights),)

    assert all(output.dtype == torch.bfloat16 for output in outputs)
    assert all(torch.isfinite(output).all() for output in outputs)
    sum(output.float().square().mean() for output in outputs).backward()
    gradients = [
        value.grad,
        left.grad,
        right.grad,
        weights.grad,
        *(parameter.grad for parameter in linear.parameters()),
        *(parameter.grad for parameter in norm.parameters()),
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: transform_irreps(
                torch.zeros(2, 7),
                IrrepLayout.parse("1x3e"),
                {},
            ),
            "missing",
        ),
        (
                lambda: tensor_product_path(
                    torch.zeros(2, 1, 4),
                torch.zeros(2, 1, 3),
                ReferenceTensorProductPath.parse("2e", "1o", "3o"),
            ),
            "left",
        ),
        (
            lambda: ScalarGatedIrreps(IrrepLayout.parse("1x1o"))(
                torch.zeros(2, 3),
                torch.zeros(2, 2),
            ),
            "gates",
        ),
    ],
)
def test_reference_irrep_operations_reject_ambiguous_shapes(
    factory: Callable[[], object],
    match: str,
) -> None:
    with pytest.raises((KeyError, ValueError), match=match):
        factory()


def test_reference_module_parameter_count_tracks_only_allowed_copy_paths() -> None:
    module = IrrepLinear(
        "2x1e + 3x1o + 2x4e",
        "3x1e + 2x1o + 5x4e + 1x4o",
        symmetry_group="O3",
    )
    assert sum(parameter.numel() for parameter in module.parameters()) == (
        3 * 2 + 2 * 3 + 5 * 2
    )
    assert all(parameter.ndim == 2 for parameter in module.parameters())
