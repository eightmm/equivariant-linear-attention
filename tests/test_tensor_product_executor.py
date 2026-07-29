from __future__ import annotations

import json

import pytest
import torch

from equivariant_attention.irreps import TensorProductPlan
from equivariant_attention.spherical import (
    cartesian_to_real_l1,
    matrix_to_real_l2,
    real_clebsch_gordan,
)
from equivariant_attention.tensor_product_executor import (
    ExecutableTensorProductPlan,
    ReferenceTensorProduct,
    compile_executable_tensor_product,
)


def _independent_cg_einsum(
    module: ReferenceTensorProduct,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Assemble a plan without calling either executor implementation."""

    output_blocks = []
    for block in module.plan.output.blocks:
        result = left.new_zeros(
            *torch.broadcast_shapes(left.shape[:-1], right.shape[:-1]),
            block.multiplicity,
            block.irrep.dim,
        )
        for instruction in module.plan.instructions:
            if instruction.output != block.irrep:
                continue
            left_block = left[
                ..., module.plan.left.slice_for(instruction.left)
            ].reshape(
                *left.shape[:-1],
                instruction.left_multiplicity,
                instruction.left.dim,
            )
            right_block = right[
                ..., module.plan.right.slice_for(instruction.right)
            ].reshape(
                *right.shape[:-1],
                instruction.right_multiplicity,
                instruction.right.dim,
            )
            coefficients = real_clebsch_gordan(
                instruction.left.degree,
                instruction.right.degree,
                instruction.output.degree,
                dtype=left.dtype,
                device=left.device,
            )
            pairwise = torch.einsum(
                "Mab,...ua,...vb->...uvM",
                coefficients,
                left_block,
                right_block,
            )
            weights = module.path_weights[
                f"path_{instruction.order:04d}"
            ]
            result = result + torch.einsum(
                "ouv,...uvM->...oM",
                weights,
                pairwise,
            )
        output_blocks.append(result.flatten(-2))
    if not output_blocks:
        return left.new_empty(
            *torch.broadcast_shapes(left.shape[:-1], right.shape[:-1]),
            0,
        )
    return torch.cat(output_blocks, dim=-1)


def test_executable_plan_records_complete_deterministic_instruction_contract() -> (
    None
):
    selection = TensorProductPlan.compile(
        "2x0e + 3x1o + 1x2e",
        "1x0e + 2x1o",
        output="4x0e + 2x1o + 3x2e + 1x3o",
        symmetry_group="O3",
    )

    plan = compile_executable_tensor_product(
        selection,
        dtype=torch.float64,
        device="cpu",
    )
    restored = ExecutableTensorProductPlan.from_json(plan.to_json())

    assert restored == plan
    assert plan.to_json() == json.dumps(
        json.loads(plan.to_json()),
        sort_keys=True,
        separators=(",", ":"),
    )
    assert [instruction.order for instruction in plan.instructions] == list(
        range(len(plan.instructions))
    )
    assert len({instruction.signature for instruction in plan.instructions}) == len(
        plan.instructions
    )
    for instruction in plan.instructions:
        assert instruction.connection_mode == "uvw"
        assert instruction.shared_weights
        assert instruction.cg_convention == "real_tesseral_condon_shortley"
        assert instruction.normalization == "component"
        assert instruction.coefficient_dtype == "float64"
        assert instruction.coefficient_device == "cpu"
        assert instruction.left_multiplicity > 0
        assert instruction.right_multiplicity > 0
        assert instruction.output_multiplicity > 0


def test_multi_path_reference_executor_matches_sum_of_its_bound_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = TensorProductPlan.compile(
        "2x0e + 2x1o",
        "2x0e + 1x1o",
        output="3x0e + 2x1o + 2x2e",
        symmetry_group="O3",
    )
    plan = compile_executable_tensor_product(
        selection,
        dtype=torch.float64,
        device="cpu",
    )
    torch.manual_seed(20260827)
    module = ReferenceTensorProduct(plan)
    left = torch.randn(
        4,
        plan.left.dim,
        dtype=torch.float64,
        requires_grad=True,
    )
    right = torch.randn(
        4,
        plan.right.dim,
        dtype=torch.float64,
        requires_grad=True,
    )

    observed = module(left, right)
    expected = _independent_cg_einsum(module, left, right)
    monkeypatch.setattr(
        module,
        "_execute",
        lambda *_: pytest.fail(
            "pathwise_reference delegated to the forward executor"
        ),
    )
    pathwise = module.pathwise_reference(left, right)
    observed_gradients = torch.autograd.grad(
        observed.square().sum(),
        (left, right, *module.parameters()),
        retain_graph=True,
    )
    expected_gradients = torch.autograd.grad(
        expected.square().sum(),
        (left, right, *module.parameters()),
    )

    torch.testing.assert_close(observed, pathwise, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)
    for observed_gradient, expected_gradient in zip(
        observed_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            observed_gradient,
            expected_gradient,
            rtol=1e-12,
            atol=1e-12,
        )


def test_reference_executor_tracks_dtype_device_moves_and_state_roundtrip() -> None:
    selection = TensorProductPlan.compile(
        "2x0e + 1x1o",
        "1x0e + 1x1o",
        output="2x0e + 1x1o + 1x2e",
        symmetry_group="O3",
    )
    source_plan = compile_executable_tensor_product(
        selection,
        dtype=torch.float64,
        device="cpu",
    )
    torch.manual_seed(20260829)
    module = ReferenceTensorProduct(source_plan).float()
    left = torch.randn(3, source_plan.left.dim, dtype=torch.float32)
    right = torch.randn(3, source_plan.right.dim, dtype=torch.float32)
    expected = module(left, right)

    assert module.plan.coefficient_dtype == "float32"
    assert module.plan.coefficient_device == "cpu"
    assert all(
        instruction.coefficient_dtype == "float32"
        and instruction.coefficient_device == "cpu"
        for instruction in module.plan.instructions
    )

    restored = ReferenceTensorProduct(source_plan).float()
    restored.load_state_dict(module.state_dict(), strict=True)
    torch.testing.assert_close(restored(left, right), expected, rtol=0, atol=0)

    if torch.cuda.is_available():
        moved = restored.to(device="cuda")
        observed = moved(left.cuda(), right.cuda()).cpu()
        assert moved.plan.coefficient_device == "cuda:0"
        assert all(
            instruction.coefficient_device == "cuda:0"
            for instruction in moved.plan.instructions
        )
        torch.testing.assert_close(observed, expected, rtol=2e-5, atol=2e-5)


def test_se3_parity_mixed_plan_executes_but_o3_rejects_the_same_paths() -> None:
    selection = TensorProductPlan.compile(
        "2x1o",
        "1x1o",
        output="1x0o + 2x1o + 1x2o",
        symmetry_group="SE3",
    )
    plan = compile_executable_tensor_product(
        selection,
        dtype=torch.float64,
        device="cpu",
    )
    module = ReferenceTensorProduct(plan)
    torch.manual_seed(20260729)
    left = torch.randn(5, plan.left.dim, dtype=torch.float64)
    right = torch.randn(5, plan.right.dim, dtype=torch.float64)

    observed = module(left, right)
    expected = _independent_cg_einsum(module, left, right)

    assert len(plan.instructions) == 3
    assert all(
        instruction.output.parity != (
            "e"
            if instruction.left.parity == instruction.right.parity
            else "o"
        )
        for instruction in plan.instructions
    )
    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)

    o3_payload = plan.to_dict()
    o3_payload["symmetry_group"] = "O3"
    with pytest.raises(ValueError, match="parity selection rule"):
        ExecutableTensorProductPlan.from_json(json.dumps(o3_payload))


def test_plan_json_rejects_duplicate_keys_at_every_nesting_level() -> None:
    selection = TensorProductPlan.compile(
        "1x1o",
        "1x1o",
        output="1x0e",
        symmetry_group="O3",
    )
    serialized = compile_executable_tensor_product(
        selection,
        dtype=torch.float64,
        device="cpu",
    ).to_json()
    duplicate_root = serialized.replace(
        '"schema_version":1',
        '"schema_version":1,"schema_version":1',
        1,
    )
    duplicate_instruction = serialized.replace(
        '"order":0',
        '"order":0,"order":0',
        1,
    )

    for payload in (duplicate_root, duplicate_instruction):
        with pytest.raises(ValueError, match="duplicate JSON key"):
            ExecutableTensorProductPlan.from_json(payload)


def test_plan_json_rejects_unknown_fields_and_noninteger_schema_versions() -> None:
    selection = TensorProductPlan.compile(
        "1x1o",
        "1x1o",
        output="1x0e",
        symmetry_group="O3",
    )
    plan = compile_executable_tensor_product(
        selection,
        dtype=torch.float64,
        device="cpu",
    )

    unknown_plan_field = plan.to_dict()
    unknown_plan_field["future_option"] = False
    with pytest.raises(ValueError, match="fields do not match schema"):
        ExecutableTensorProductPlan.from_json(json.dumps(unknown_plan_field))

    unknown_instruction_field = plan.to_dict()
    unknown_instruction_field["instructions"][0]["future_option"] = False
    with pytest.raises(ValueError, match="instruction fields"):
        ExecutableTensorProductPlan.from_json(
            json.dumps(unknown_instruction_field)
        )

    for invalid_version in (True, "1", 2):
        invalid_version_payload = plan.to_dict()
        invalid_version_payload["schema_version"] = invalid_version
        with pytest.raises(
            ValueError,
            match="unsupported executable tensor-product schema_version",
        ):
            ExecutableTensorProductPlan.from_json(
                json.dumps(invalid_version_payload)
            )


def test_generic_low_degree_paths_match_cartesian_forward_and_gradients() -> None:
    selection = TensorProductPlan.compile(
        "1x1o",
        "1x1o",
        output="1x0e + 1x2e",
        symmetry_group="O3",
    )
    plan = compile_executable_tensor_product(
        selection,
        dtype=torch.float64,
        device="cpu",
    )
    module = ReferenceTensorProduct(plan)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(1.0)
    x = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    generic = module(
        cartesian_to_real_l1(x).reshape(6, -1),
        cartesian_to_real_l1(y).reshape(6, -1),
    )

    scalar = (x * y).sum(dim=-1, keepdim=True) / (3.0**0.5)
    outer = x[:, :, None] * y[:, None, :]
    symmetric_traceless = 0.5 * (outer + outer.mT) - (
        torch.eye(3, dtype=torch.float64)[None]
        * torch.diagonal(outer, dim1=-2, dim2=-1).sum(dim=-1)[:, None, None]
        / 3.0
    )
    cartesian = torch.cat(
        [scalar, matrix_to_real_l2(symmetric_traceless)],
        dim=-1,
    )
    generic_gradients = torch.autograd.grad(
        generic.square().sum(),
        (x, y),
        retain_graph=True,
    )
    cartesian_gradients = torch.autograd.grad(
        cartesian.square().sum(),
        (x, y),
    )

    torch.testing.assert_close(generic, cartesian, rtol=2e-12, atol=2e-12)
    for generic_gradient, cartesian_gradient in zip(
        generic_gradients,
        cartesian_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            generic_gradient,
            cartesian_gradient,
            rtol=2e-12,
            atol=2e-12,
        )
