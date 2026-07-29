"""Executable, serializable reference tensor-product instruction plans."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import json
from math import sqrt

import torch
from torch import nn

from .irreps import Irrep, IrrepLayout, TensorProductPlan
from .reference_irreps import (
    ReferenceTensorProductPath,
    tensor_product_path,
)


_SCHEMA = "equivariant_attention.executable_tensor_product"
_SCHEMA_VERSION = 1
_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class TensorProductInstruction:
    """One complete numerical path and multiplicity connection contract."""

    left: Irrep
    right: Irrep
    output: Irrep
    left_multiplicity: int
    right_multiplicity: int
    output_multiplicity: int
    left_offset: int
    right_offset: int
    output_offset: int
    order: int
    connection_mode: str = "uvw"
    shared_weights: bool = True
    cg_convention: str = "real_tesseral_condon_shortley"
    normalization: str = "component"
    coefficient_dtype: str = "float64"
    coefficient_device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("left", "right", "output"):
            value = getattr(self, name)
            if not isinstance(value, Irrep):
                raise TypeError(f"{name} must be an Irrep")
        # An instruction does not own the plan's symmetry contract.  Validate
        # only the SO(3) angular-momentum rule here; the enclosing plan applies
        # the additional O(3) parity rule when appropriate.
        ReferenceTensorProductPath(
            self.left,
            self.right,
            self.output,
            "SE3",
        )
        for name in (
            "left_multiplicity",
            "right_multiplicity",
            "output_multiplicity",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("left_offset", "right_offset", "output_offset", "order"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.connection_mode != "uvw":
            raise ValueError("reference connection_mode must be 'uvw'")
        if not isinstance(self.shared_weights, bool) or not self.shared_weights:
            raise ValueError("reference instructions require shared_weights=True")
        if self.cg_convention != "real_tesseral_condon_shortley":
            raise ValueError("unsupported Clebsch--Gordan convention")
        if self.normalization != "component":
            raise ValueError("reference normalization must be 'component'")
        if self.coefficient_dtype not in _DTYPES:
            raise ValueError("unsupported coefficient dtype")
        try:
            torch.device(self.coefficient_device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError("invalid coefficient device") from exc

    @property
    def signature(self) -> tuple[str, str, str]:
        return str(self.left), str(self.right), str(self.output)

    def to_dict(self) -> dict[str, object]:
        return {
            "left": str(self.left),
            "right": str(self.right),
            "output": str(self.output),
            "left_multiplicity": self.left_multiplicity,
            "right_multiplicity": self.right_multiplicity,
            "output_multiplicity": self.output_multiplicity,
            "left_offset": self.left_offset,
            "right_offset": self.right_offset,
            "output_offset": self.output_offset,
            "order": self.order,
            "connection_mode": self.connection_mode,
            "shared_weights": self.shared_weights,
            "cg_convention": self.cg_convention,
            "normalization": self.normalization,
            "coefficient_dtype": self.coefficient_dtype,
            "coefficient_device": self.coefficient_device,
        }

    @classmethod
    def from_dict(cls, value: object) -> TensorProductInstruction:
        if not isinstance(value, dict):
            raise TypeError("tensor-product instruction must be an object")
        expected = {
            "left",
            "right",
            "output",
            "left_multiplicity",
            "right_multiplicity",
            "output_multiplicity",
            "left_offset",
            "right_offset",
            "output_offset",
            "order",
            "connection_mode",
            "shared_weights",
            "cg_convention",
            "normalization",
            "coefficient_dtype",
            "coefficient_device",
        }
        if set(value) != expected:
            raise ValueError("tensor-product instruction fields do not match schema")
        payload = dict(value)
        payload["left"] = Irrep.parse(payload["left"])
        payload["right"] = Irrep.parse(payload["right"])
        payload["output"] = Irrep.parse(payload["output"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ExecutableTensorProductPlan:
    """Frozen numerical plan with explicit offsets, order, dtype, and device."""

    left: IrrepLayout
    right: IrrepLayout
    output: IrrepLayout
    symmetry_group: str
    coefficient_dtype: str
    coefficient_device: str
    instructions: tuple[TensorProductInstruction, ...]

    def __post_init__(self) -> None:
        if not all(
            isinstance(layout, IrrepLayout)
            for layout in (self.left, self.right, self.output)
        ):
            raise TypeError("executable plan layouts must be IrrepLayout values")
        if self.symmetry_group not in {"O3", "SE3"}:
            raise ValueError("symmetry_group must be 'O3' or 'SE3'")
        if self.coefficient_dtype not in _DTYPES:
            raise ValueError("unsupported coefficient dtype")
        normalized_device = str(torch.device(self.coefficient_device))
        object.__setattr__(self, "coefficient_device", normalized_device)
        if not isinstance(self.instructions, tuple):
            raise TypeError("instructions must be a tuple")
        if any(
            not isinstance(instruction, TensorProductInstruction)
            for instruction in self.instructions
        ):
            raise TypeError(
                "instructions must contain TensorProductInstruction values"
            )
        if tuple(item.order for item in self.instructions) != tuple(
            range(len(self.instructions))
        ):
            raise ValueError("instruction order must be contiguous and deterministic")
        signatures = tuple(item.signature for item in self.instructions)
        if len(set(signatures)) != len(signatures):
            raise ValueError("duplicate tensor-product instructions are forbidden")
        for instruction in self.instructions:
            if instruction.coefficient_dtype != self.coefficient_dtype:
                raise ValueError("instruction dtype does not match plan dtype")
            if instruction.coefficient_device != self.coefficient_device:
                raise ValueError("instruction device does not match plan device")
            if self.symmetry_group == "O3":
                ReferenceTensorProductPath(
                    instruction.left,
                    instruction.right,
                    instruction.output,
                    "O3",
                )
            for layout, irrep, multiplicity, offset in (
                (
                    self.left,
                    instruction.left,
                    instruction.left_multiplicity,
                    instruction.left_offset,
                ),
                (
                    self.right,
                    instruction.right,
                    instruction.right_multiplicity,
                    instruction.right_offset,
                ),
                (
                    self.output,
                    instruction.output,
                    instruction.output_multiplicity,
                    instruction.output_offset,
                ),
            ):
                if layout.multiplicity(irrep) != multiplicity:
                    raise ValueError(
                        "instruction multiplicity does not match its layout"
                    )
                if layout.slice_for(irrep).start != offset:
                    raise ValueError("instruction offset does not match its layout")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "left": str(self.left),
            "right": str(self.right),
            "output": str(self.output),
            "symmetry_group": self.symmetry_group,
            "coefficient_dtype": self.coefficient_dtype,
            "coefficient_device": self.coefficient_device,
            "instructions": [
                instruction.to_dict() for instruction in self.instructions
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(
        cls,
        payload: str | bytes | bytearray,
    ) -> ExecutableTensorProductPlan:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        if not isinstance(value, dict):
            raise TypeError("executable tensor-product plan must be an object")
        expected = {
            "schema",
            "schema_version",
            "left",
            "right",
            "output",
            "symmetry_group",
            "coefficient_dtype",
            "coefficient_device",
            "instructions",
        }
        if set(value) != expected:
            raise ValueError("executable tensor-product fields do not match schema")
        if value.pop("schema") != _SCHEMA:
            raise ValueError("unsupported executable tensor-product schema")
        schema_version = value.pop("schema_version")
        if (
            type(schema_version) is not int
            or schema_version != _SCHEMA_VERSION
        ):
            raise ValueError("unsupported executable tensor-product schema_version")
        instructions = value.pop("instructions")
        if not isinstance(instructions, list):
            raise TypeError("instructions must be a JSON array")
        return cls(
            left=IrrepLayout.parse(value.pop("left")),
            right=IrrepLayout.parse(value.pop("right")),
            output=IrrepLayout.parse(value.pop("output")),
            instructions=tuple(
                TensorProductInstruction.from_dict(item)
                for item in instructions
            ),
            **value,
        )


def compile_executable_tensor_product(
    selection: TensorProductPlan,
    *,
    dtype: torch.dtype,
    device: torch.device | str | int,
) -> ExecutableTensorProductPlan:
    """Compile all reachable selection paths into deterministic instructions."""

    if not isinstance(selection, TensorProductPlan):
        raise TypeError("selection must be a TensorProductPlan")
    dtype_name = str(dtype).removeprefix("torch.")
    if dtype_name not in _DTYPES:
        raise TypeError("dtype must be a supported real floating dtype")
    device_name = str(torch.device(device))
    instructions = tuple(
        TensorProductInstruction(
            left=path.left,
            right=path.right,
            output=path.output,
            left_multiplicity=selection.left.multiplicity(path.left),
            right_multiplicity=selection.right.multiplicity(path.right),
            output_multiplicity=selection.output.multiplicity(path.output),
            left_offset=selection.left.slice_for(path.left).start,
            right_offset=selection.right.slice_for(path.right).start,
            output_offset=selection.output.slice_for(path.output).start,
            order=order,
            coefficient_dtype=dtype_name,
            coefficient_device=device_name,
        )
        for order, path in enumerate(selection.paths)
    )
    return ExecutableTensorProductPlan(
        left=selection.left,
        right=selection.right,
        output=selection.output,
        symmetry_group=selection.symmetry_group,
        coefficient_dtype=dtype_name,
        coefficient_device=device_name,
        instructions=instructions,
    )


class ReferenceTensorProduct(nn.Module):
    """Learned multi-path executor for an :class:`ExecutableTensorProductPlan`."""

    def __init__(self, plan: ExecutableTensorProductPlan) -> None:
        super().__init__()
        if not isinstance(plan, ExecutableTensorProductPlan):
            raise TypeError("plan must be an ExecutableTensorProductPlan")
        self.plan = plan
        dtype = getattr(torch, plan.coefficient_dtype)
        device = torch.device(plan.coefficient_device)
        self.register_buffer(
            "_execution_anchor",
            torch.empty(0, dtype=dtype, device=device),
            persistent=False,
        )
        self.path_weights = nn.ParameterDict(
            {
                self._key(instruction.order): nn.Parameter(
                    torch.empty(
                        instruction.output_multiplicity,
                        instruction.left_multiplicity,
                        instruction.right_multiplicity,
                        dtype=dtype,
                        device=device,
                    )
                )
                for instruction in plan.instructions
            }
        )
        self.reset_parameters()

    def _apply(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        recurse: bool = True,
    ) -> ReferenceTensorProduct:
        """Keep the immutable execution receipt aligned with module migration."""

        result = super()._apply(fn, recurse=recurse)
        dtype_name = str(self._execution_anchor.dtype).removeprefix("torch.")
        if dtype_name not in _DTYPES:
            raise TypeError(
                "ReferenceTensorProduct supports only real floating dtypes"
            )
        device_name = str(self._execution_anchor.device)
        instructions = tuple(
            replace(
                instruction,
                coefficient_dtype=dtype_name,
                coefficient_device=device_name,
            )
            for instruction in self.plan.instructions
        )
        self.plan = replace(
            self.plan,
            coefficient_dtype=dtype_name,
            coefficient_device=device_name,
            instructions=instructions,
        )
        return result

    @staticmethod
    def _key(order: int) -> str:
        return f"path_{order:04d}"

    def reset_parameters(self) -> None:
        for instruction in self.plan.instructions:
            weight = self.path_weights[self._key(instruction.order)]
            bound = 1.0 / sqrt(
                instruction.left_multiplicity
                * instruction.right_multiplicity
            )
            nn.init.uniform_(weight, -bound, bound)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        return self._execute(left, right)

    def pathwise_reference(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate via an instruction-first reference independent of ``forward``."""

        self._validate_inputs(left, right)
        leading_shape = torch.broadcast_shapes(
            left.shape[:-1],
            right.shape[:-1],
        )
        contributions: dict[Irrep, list[torch.Tensor]] = {
            block.irrep: [] for block in self.plan.output.blocks
        }
        for instruction in self.plan.instructions:
            left_block = left[
                ..., self.plan.left.slice_for(instruction.left)
            ].reshape(
                *left.shape[:-1],
                instruction.left_multiplicity,
                instruction.left.dim,
            )
            right_block = right[
                ..., self.plan.right.slice_for(instruction.right)
            ].reshape(
                *right.shape[:-1],
                instruction.right_multiplicity,
                instruction.right.dim,
            )
            path = ReferenceTensorProductPath(
                instruction.left,
                instruction.right,
                instruction.output,
                self.plan.symmetry_group,
            )
            contributions[instruction.output].append(
                tensor_product_path(
                    left_block,
                    right_block,
                    path,
                    weights=self.path_weights[
                        self._key(instruction.order)
                    ],
                )
            )

        output_blocks = []
        for block in self.plan.output.blocks:
            block_contributions = contributions[block.irrep]
            if block_contributions:
                result = block_contributions[0]
                for contribution in block_contributions[1:]:
                    result = result + contribution
            else:
                result = left.new_zeros(
                    *leading_shape,
                    block.multiplicity,
                    block.irrep.dim,
                )
            output_blocks.append(result.flatten(-2))
        if not output_blocks:
            return left.new_empty(*leading_shape, 0)
        return torch.cat(output_blocks, dim=-1)

    def _execute(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(left, right)
        leading_shape = torch.broadcast_shapes(
            left.shape[:-1],
            right.shape[:-1],
        )
        output_blocks = []
        for block in self.plan.output.blocks:
            result = left.new_zeros(
                *leading_shape,
                block.multiplicity,
                block.irrep.dim,
            )
            for instruction in self.plan.instructions:
                if instruction.output != block.irrep:
                    continue
                left_block = left[..., self.plan.left.slice_for(
                    instruction.left
                )].reshape(
                    *left.shape[:-1],
                    instruction.left_multiplicity,
                    instruction.left.dim,
                )
                right_block = right[..., self.plan.right.slice_for(
                    instruction.right
                )].reshape(
                    *right.shape[:-1],
                    instruction.right_multiplicity,
                    instruction.right.dim,
                )
                path = ReferenceTensorProductPath(
                    instruction.left,
                    instruction.right,
                    instruction.output,
                    self.plan.symmetry_group,
                )
                result = result + tensor_product_path(
                    left_block,
                    right_block,
                    path,
                    weights=self.path_weights[self._key(instruction.order)],
                )
            output_blocks.append(result.flatten(-2))
        if not output_blocks:
            return left.new_empty(*leading_shape, 0)
        return torch.cat(output_blocks, dim=-1)

    def _validate_inputs(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> None:
        dtype = getattr(torch, self.plan.coefficient_dtype)
        device = torch.device(self.plan.coefficient_device)
        for name, value, layout in (
            ("left", left, self.plan.left),
            ("right", right, self.plan.right),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.ndim < 1 or value.shape[-1] != layout.dim:
                raise ValueError(
                    f"{name} must have final dimension {layout.dim}"
                )
            if value.dtype != dtype:
                raise TypeError(
                    f"{name} must use the plan dtype {self.plan.coefficient_dtype}"
                )
            if value.device != device:
                raise ValueError(
                    f"{name} must use the plan device {self.plan.coefficient_device}"
                )
