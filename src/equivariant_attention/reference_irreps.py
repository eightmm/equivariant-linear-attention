r"""Pure-PyTorch reference operations for flattened real irrep layouts.

This module is a numerical reference, not the optimized model executor.  An
``IrrepLayout`` stores every block as ``(..., multiplicity, 2 * l + 1)`` inside
one flattened final dimension.  Operations below discover no representations
from data: layouts and allowed paths are fixed at construction.

``TensorProductPlan`` deliberately contains selection rules only.  It does not
specify multiplicity connection modes, normalization, or learned weights.
Consequently :func:`tensor_product_path` takes an honest local path descriptor
and an optional external weight of shape ``(m_out, m_left, m_right)`` rather
than pretending those missing instructions are part of the plan.

For ``O3``, parity labels are enforced.  For ``SE3`` they are retained as
provenance but proper rotations may mix equal-degree ``e`` and ``o`` blocks.
Half and bfloat16 inputs are evaluated in float32; float64 inputs and
parameters retain float64 computation.  The implementation uses ordinary
PyTorch operations throughout and is safe for higher-order autograd.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .irreps import (
    Irrep,
    IrrepLayout,
    TensorProductPath,
    TensorProductPlan,
)
from .spherical import real_clebsch_gordan


_REAL_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}

__all__ = [
    "IrrepLinear",
    "IrrepLinearPath",
    "IrrepRMSNorm",
    "ReferenceTensorProductPath",
    "ScalarGatedIrreps",
    "tensor_product_path",
    "transform_irreps",
]


def _as_layout(value: str | IrrepLayout, *, name: str) -> IrrepLayout:
    try:
        return IrrepLayout.parse(value)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{name}: {exc}") from exc


def _validate_symmetry_group(symmetry_group: str) -> str:
    if symmetry_group not in {"O3", "SE3"}:
        raise ValueError("symmetry_group must be 'O3' or 'SE3'")
    return symmetry_group


def _validate_real_tensor(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a PyTorch tensor")
    if value.dtype not in _REAL_DTYPES:
        raise TypeError(f"{name} must have a real floating PyTorch dtype")


def _validate_flattened(
    value: torch.Tensor,
    layout: IrrepLayout,
    *,
    name: str,
) -> None:
    _validate_real_tensor(value, name=name)
    if value.ndim < 1 or value.shape[-1] != layout.dim:
        raise ValueError(
            f"{name} must have final dimension {layout.dim} for layout {layout}"
        )


def _promoted_dtype(*values: torch.Tensor) -> torch.dtype:
    if not values:
        return torch.get_default_dtype()
    result = values[0].dtype
    for value in values[1:]:
        result = torch.promote_types(result, value.dtype)
    return result


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def _ensure_same_device(*values: torch.Tensor) -> torch.device:
    if not values:
        return torch.device("cpu")
    device = values[0].device
    if any(value.device != device for value in values[1:]):
        raise ValueError("all tensors must be on the same device")
    return device


def _reshape_block(
    value: torch.Tensor,
    layout: IrrepLayout,
    irrep: Irrep,
) -> torch.Tensor:
    return value[..., layout.slice_for(irrep)].reshape(
        *value.shape[:-1],
        layout.multiplicity(irrep),
        irrep.dim,
    )


def _flatten_blocks(
    blocks: list[torch.Tensor],
    *,
    leading_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not blocks:
        return torch.empty(*leading_shape, 0, dtype=dtype, device=device)
    return torch.cat([block.flatten(-2) for block in blocks], dim=-1)


@dataclass(frozen=True)
class IrrepLinearPath:
    """One equal-degree copy-mixing path."""

    input: Irrep
    output: Irrep

    def __post_init__(self) -> None:
        if not isinstance(self.input, Irrep) or not isinstance(self.output, Irrep):
            raise TypeError("linear path entries must be Irrep instances")
        if self.input.degree != self.output.degree:
            raise ValueError("linear paths may only connect equal angular degrees")


def _linear_path_key(input_irrep: Irrep, output_irrep: Irrep) -> str:
    return f"{input_irrep}_to_{output_irrep}"


class IrrepLinear(nn.Module):
    """Shared-component linear mixing between copies of equal-degree irreps.

    Under ``O3`` both degree and parity must agree.  Under ``SE3`` only degree
    must agree; parity labels remain visible as provenance and the corresponding
    cross-parity paths are explicit in :attr:`mixing_paths`.
    """

    def __init__(
        self,
        input_layout: str | IrrepLayout,
        output_layout: str | IrrepLayout | None = None,
        *,
        symmetry_group: str = "O3",
        device: torch.device | str | int | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.input_layout = _as_layout(input_layout, name="input_layout")
        self.output_layout = _as_layout(
            input_layout if output_layout is None else output_layout,
            name="output_layout",
        )
        self.symmetry_group = _validate_symmetry_group(symmetry_group)
        self.mixing_paths = tuple(
            IrrepLinearPath(input_block.irrep, output_block.irrep)
            for output_block in self.output_layout.blocks
            for input_block in self.input_layout.blocks
            if input_block.irrep.degree == output_block.irrep.degree
            and (
                self.symmetry_group == "SE3"
                or input_block.irrep.parity == output_block.irrep.parity
            )
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.path_weights = nn.ParameterDict(
            {
                _linear_path_key(path.input, path.output): nn.Parameter(
                    torch.empty(
                        self.output_layout.multiplicity(path.output),
                        self.input_layout.multiplicity(path.input),
                        **factory_kwargs,
                    )
                )
                for path in self.mixing_paths
            }
        )
        self.reset_parameters()

    @property
    def parity_is_provenance_only(self) -> bool:
        return self.symmetry_group == "SE3"

    def reset_parameters(self) -> None:
        output_fan_in = {
            block.irrep: sum(
                self.input_layout.multiplicity(path.input)
                for path in self.mixing_paths
                if path.output == block.irrep
            )
            for block in self.output_layout.blocks
        }
        for path in self.mixing_paths:
            parameter = self.path_weights[
                _linear_path_key(path.input, path.output)
            ]
            bound = 1.0 / sqrt(max(1, output_fan_in[path.output]))
            nn.init.uniform_(parameter, -bound, bound)

    def weight_for(
        self,
        input_irrep: str | Irrep,
        output_irrep: str | Irrep,
    ) -> nn.Parameter:
        parsed_input = Irrep.parse(input_irrep)
        parsed_output = Irrep.parse(output_irrep)
        key = _linear_path_key(parsed_input, parsed_output)
        if key not in self.path_weights:
            raise KeyError(
                f"no {self.symmetry_group} linear path from "
                f"{parsed_input} to {parsed_output}"
            )
        return self.path_weights[key]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        _validate_flattened(value, self.input_layout, name="value")
        parameters = tuple(self.path_weights.values())
        _ensure_same_device(value, *parameters)
        output_dtype = _promoted_dtype(value, *parameters)
        computation_dtype = _compute_dtype(output_dtype)
        source_value = value.to(dtype=computation_dtype)

        output_blocks = []
        for output_block in self.output_layout.blocks:
            result = torch.zeros(
                *value.shape[:-1],
                output_block.multiplicity,
                output_block.irrep.dim,
                dtype=computation_dtype,
                device=value.device,
            )
            for path in self.mixing_paths:
                if path.output != output_block.irrep:
                    continue
                source = _reshape_block(
                    source_value,
                    self.input_layout,
                    path.input,
                )
                weight = self.weight_for(path.input, path.output).to(
                    dtype=computation_dtype
                )
                result = result + torch.einsum(
                    "oi,...im->...om",
                    weight,
                    source,
                )
            output_blocks.append(result)
        return _flatten_blocks(
            output_blocks,
            leading_shape=value.shape[:-1],
            dtype=computation_dtype,
            device=value.device,
        ).to(dtype=output_dtype)


def _irrep_key(irrep: Irrep) -> str:
    return str(irrep)


class IrrepRMSNorm(nn.Module):
    """Invariant RMS normalization performed independently for every copy."""

    def __init__(
        self,
        layout: str | IrrepLayout,
        *,
        eps: float = 1e-8,
        affine: bool = True,
        device: torch.device | str | int | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.layout = _as_layout(layout, name="layout")
        if not isinstance(eps, (float, int)) or isinstance(eps, bool) or eps < 0:
            raise ValueError("eps must be a nonnegative real number")
        if not isinstance(affine, bool):
            raise TypeError("affine must be a bool")
        self.eps = float(eps)
        self.affine = affine
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weights = nn.ParameterDict(
            {
                _irrep_key(block.irrep): nn.Parameter(
                    torch.ones(block.multiplicity, **factory_kwargs)
                )
                for block in self.layout.blocks
            }
            if affine
            else {}
        )

    def weight_for(self, irrep: str | Irrep) -> nn.Parameter:
        parsed = Irrep.parse(irrep)
        key = _irrep_key(parsed)
        if key not in self.weights:
            if not self.affine:
                raise KeyError("IrrepRMSNorm has affine=False")
            raise KeyError(f"irrep {parsed} is not present in the layout")
        return self.weights[key]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        _validate_flattened(value, self.layout, name="value")
        parameters = tuple(self.weights.values())
        _ensure_same_device(value, *parameters)
        output_dtype = _promoted_dtype(value, *parameters)
        computation_dtype = _compute_dtype(output_dtype)
        source_value = value.to(dtype=computation_dtype)

        output_blocks = []
        for block in self.layout.blocks:
            source = _reshape_block(source_value, self.layout, block.irrep)
            inverse_rms = (
                source.square().mean(dim=-1, keepdim=True) + self.eps
            ).rsqrt()
            result = source * inverse_rms
            if self.affine:
                result = result * self.weight_for(block.irrep).to(
                    dtype=computation_dtype
                ).reshape(
                    *((1,) * (value.ndim - 1)),
                    block.multiplicity,
                    1,
                )
            output_blocks.append(result)
        return _flatten_blocks(
            output_blocks,
            leading_shape=value.shape[:-1],
            dtype=computation_dtype,
            device=value.device,
        ).to(dtype=output_dtype)


class ScalarGatedIrreps(nn.Module):
    """Gate every non-scalar irrep copy by one external scalar."""

    def __init__(
        self,
        layout: str | IrrepLayout,
        *,
        symmetry_group: str = "O3",
        gate_parity: str = "e",
        activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.layout = _as_layout(layout, name="layout")
        self.symmetry_group = _validate_symmetry_group(symmetry_group)
        if gate_parity not in {"e", "o"}:
            raise ValueError("gate_parity must be 'e' or 'o'")
        if self.symmetry_group == "O3" and gate_parity != "e":
            raise ValueError("O3 scalar gates must use an even scalar")
        if activation not in {"identity", "sigmoid", "silu", "tanh"}:
            raise ValueError(
                "activation must be 'identity', 'sigmoid', 'silu', or 'tanh'"
            )
        self.gate_parity = gate_parity
        self.activation = activation
        self.num_gates = sum(
            block.multiplicity
            for block in self.layout.blocks
            if block.irrep.degree > 0
        )

    @property
    def parity_is_provenance_only(self) -> bool:
        return self.symmetry_group == "SE3"

    def _activate(self, value: torch.Tensor) -> torch.Tensor:
        if self.activation == "identity":
            return value
        if self.activation == "sigmoid":
            return value.sigmoid()
        if self.activation == "silu":
            return F.silu(value)
        return value.tanh()

    def forward(
        self,
        value: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        _validate_flattened(value, self.layout, name="value")
        _validate_real_tensor(gates, name="gates")
        if gates.shape != (*value.shape[:-1], self.num_gates):
            raise ValueError(
                "gates must have shape "
                f"{(*value.shape[:-1], self.num_gates)}, got {tuple(gates.shape)}"
            )
        _ensure_same_device(value, gates)
        output_dtype = _promoted_dtype(value, gates)
        computation_dtype = _compute_dtype(output_dtype)
        source_value = value.to(dtype=computation_dtype)
        activated_gates = self._activate(gates.to(dtype=computation_dtype))

        output_blocks = []
        gate_offset = 0
        for block in self.layout.blocks:
            result = _reshape_block(source_value, self.layout, block.irrep)
            if block.irrep.degree > 0:
                block_gates = activated_gates[
                    ..., gate_offset : gate_offset + block.multiplicity
                ]
                result = result * block_gates.unsqueeze(-1)
                gate_offset += block.multiplicity
            output_blocks.append(result)
        return _flatten_blocks(
            output_blocks,
            leading_shape=value.shape[:-1],
            dtype=computation_dtype,
            device=value.device,
        ).to(dtype=output_dtype)


@dataclass(frozen=True)
class ReferenceTensorProductPath:
    """One numerical CG path with an explicit symmetry contract."""

    left: Irrep
    right: Irrep
    output: Irrep
    symmetry_group: str = "O3"

    def __post_init__(self) -> None:
        object.__setattr__(self, "left", Irrep.parse(self.left))
        object.__setattr__(self, "right", Irrep.parse(self.right))
        object.__setattr__(self, "output", Irrep.parse(self.output))
        object.__setattr__(
            self,
            "symmetry_group",
            _validate_symmetry_group(self.symmetry_group),
        )
        if not (
            abs(self.left.degree - self.right.degree)
            <= self.output.degree
            <= self.left.degree + self.right.degree
        ):
            raise ValueError(
                "tensor-product path violates the angular-momentum selection rule"
            )
        if self.symmetry_group == "O3" and self.parity_mixed:
            raise ValueError("O3 tensor-product path violates the parity selection rule")

    @classmethod
    def parse(
        cls,
        left: str | Irrep,
        right: str | Irrep,
        output: str | Irrep,
        *,
        symmetry_group: str = "O3",
    ) -> ReferenceTensorProductPath:
        return cls(
            Irrep.parse(left),
            Irrep.parse(right),
            Irrep.parse(output),
            symmetry_group,
        )

    @classmethod
    def from_plan(
        cls,
        plan: TensorProductPlan,
        path: int | TensorProductPath,
    ) -> ReferenceTensorProductPath:
        """Bind one selection-rule path without inventing connection weights."""

        if not isinstance(plan, TensorProductPlan):
            raise TypeError("plan must be a TensorProductPlan")
        if isinstance(path, bool) or not isinstance(path, (int, TensorProductPath)):
            raise TypeError("path must be an integer index or TensorProductPath")
        if isinstance(path, int):
            try:
                selected = plan.paths[path]
            except IndexError as exc:
                raise IndexError("tensor-product plan path index is out of range") from exc
        else:
            selected = path
            if selected not in plan.paths:
                raise ValueError("tensor-product path is not present in the plan")
        return cls(
            selected.left,
            selected.right,
            selected.output,
            plan.symmetry_group,
        )

    @property
    def natural_parity(self) -> str:
        return "e" if self.left.parity == self.right.parity else "o"

    @property
    def parity_mixed(self) -> bool:
        return self.output.parity != self.natural_parity

    @property
    def parity_is_provenance_only(self) -> bool:
        return self.symmetry_group == "SE3"


def tensor_product_path(
    left: torch.Tensor,
    right: torch.Tensor,
    path: ReferenceTensorProductPath,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate one real-CG path over explicit multiplicity dimensions.

    ``left`` and ``right`` have shapes ``(..., m_left, 2*l_left+1)`` and
    ``(..., m_right, 2*l_right+1)``.  Without ``weights``, the output is
    ``(..., m_left, m_right, 2*L+1)``.  With an external connection tensor of
    shape ``(m_out, m_left, m_right)``, the output is
    ``(..., m_out, 2*L+1)``.
    """

    if not isinstance(path, ReferenceTensorProductPath):
        raise TypeError("path must be a ReferenceTensorProductPath")
    _validate_real_tensor(left, name="left")
    _validate_real_tensor(right, name="right")
    if left.ndim < 2 or left.shape[-1] != path.left.dim:
        raise ValueError(
            f"left must have final irrep dimension {path.left.dim}"
        )
    if right.ndim < 2 or right.shape[-1] != path.right.dim:
        raise ValueError(
            f"right must have final irrep dimension {path.right.dim}"
        )
    if left.shape[-2] <= 0 or right.shape[-2] <= 0:
        raise ValueError("tensor-product multiplicities must be positive")
    try:
        leading_shape = torch.broadcast_shapes(
            left.shape[:-2],
            right.shape[:-2],
        )
    except RuntimeError as exc:
        raise ValueError(
            "left and right leading dimensions are not broadcastable"
        ) from exc

    values = [left, right]
    if weights is not None:
        _validate_real_tensor(weights, name="weights")
        expected_tail = (left.shape[-2], right.shape[-2])
        if weights.ndim != 3 or weights.shape[1:] != expected_tail:
            raise ValueError(
                "weights must have shape "
                f"(m_out, {expected_tail[0]}, {expected_tail[1]})"
            )
        if weights.shape[0] <= 0:
            raise ValueError("output multiplicity must be positive")
        values.append(weights)
    device = _ensure_same_device(*values)
    output_dtype = _promoted_dtype(*values)
    computation_dtype = _compute_dtype(output_dtype)

    expanded_left = left.expand(
        *leading_shape,
        left.shape[-2],
        left.shape[-1],
    ).to(dtype=computation_dtype)
    expanded_right = right.expand(
        *leading_shape,
        right.shape[-2],
        right.shape[-1],
    ).to(dtype=computation_dtype)
    coefficients = real_clebsch_gordan(
        path.left.degree,
        path.right.degree,
        path.output.degree,
        dtype=computation_dtype,
        device=device,
    )
    pairwise = torch.einsum(
        "Mab,...ua,...vb->...uvM",
        coefficients,
        expanded_left,
        expanded_right,
    )
    if weights is None:
        return pairwise.to(dtype=output_dtype)
    result = torch.einsum(
        "ouv,...uvM->...oM",
        weights.to(dtype=computation_dtype),
        pairwise,
    )
    return result.to(dtype=output_dtype)


def transform_irreps(
    value: torch.Tensor,
    layout: str | IrrepLayout,
    matrices: Mapping[str | Irrep, torch.Tensor],
) -> torch.Tensor:
    """Apply one supplied representation matrix to every copy of each irrep."""

    parsed_layout = _as_layout(layout, name="layout")
    _validate_flattened(value, parsed_layout, name="value")
    if not isinstance(matrices, Mapping):
        raise TypeError("matrices must be a mapping")

    normalized: dict[Irrep, torch.Tensor] = {}
    for key, matrix in matrices.items():
        irrep = Irrep.parse(key)
        if irrep in normalized:
            raise ValueError(f"duplicate transformation matrix for irrep {irrep}")
        _validate_real_tensor(matrix, name=f"matrix[{irrep}]")
        if matrix.shape != (irrep.dim, irrep.dim):
            raise ValueError(
                f"matrix[{irrep}] must have shape {(irrep.dim, irrep.dim)}"
            )
        normalized[irrep] = matrix

    expected = {block.irrep for block in parsed_layout.blocks}
    missing = expected - normalized.keys()
    extra = normalized.keys() - expected
    if missing:
        rendered = ", ".join(str(irrep) for irrep in sorted(missing))
        raise KeyError(f"missing transformation matrices for: {rendered}")
    if extra:
        rendered = ", ".join(str(irrep) for irrep in sorted(extra))
        raise KeyError(f"unexpected transformation matrices for: {rendered}")

    matrix_values = tuple(normalized.values())
    device = _ensure_same_device(value, *matrix_values)
    output_dtype = _promoted_dtype(value, *matrix_values)
    computation_dtype = _compute_dtype(output_dtype)
    source_value = value.to(dtype=computation_dtype)
    output_blocks = []
    for block in parsed_layout.blocks:
        source = _reshape_block(source_value, parsed_layout, block.irrep)
        output_blocks.append(
            torch.einsum(
                "ij,...cj->...ci",
                normalized[block.irrep].to(dtype=computation_dtype),
                source,
            )
        )
    return _flatten_blocks(
        output_blocks,
        leading_shape=value.shape[:-1],
        dtype=computation_dtype,
        device=device,
    ).to(dtype=output_dtype)
