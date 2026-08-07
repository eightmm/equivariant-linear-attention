from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

import torch

_IRREP = re.compile(r"^\s*(\d+)([eo])\s*$")
_TERM = re.compile(r"^\s*(\d+)x(\d+)([eo])\s*$")


@dataclass(frozen=True, order=True)
class Irrep:
    """One real O(3) irrep, identified by angular degree and parity."""

    degree: int
    parity: str

    def __post_init__(self) -> None:
        if isinstance(self.degree, bool) or not isinstance(self.degree, int) or self.degree < 0:
            raise ValueError("irrep degree must be a nonnegative integer")
        if self.parity not in {"e", "o"}:
            raise ValueError("irrep parity must be 'e' or 'o'")

    @classmethod
    def parse(cls, spec: str | Irrep) -> Irrep:
        if isinstance(spec, Irrep):
            return spec
        if not isinstance(spec, str):
            raise TypeError("irrep spec must be a string or Irrep")
        match = _IRREP.match(spec)
        if match is None:
            raise ValueError(f"unsupported irrep: {spec.strip()!r}")
        return cls(int(match.group(1)), match.group(2))

    @property
    def dim(self) -> int:
        return 2 * self.degree + 1

    def __str__(self) -> str:
        return f"{self.degree}{self.parity}"


@dataclass(frozen=True)
class IrrepBlock:
    multiplicity: int
    irrep: Irrep

    def __post_init__(self) -> None:
        if (
            isinstance(self.multiplicity, bool)
            or not isinstance(self.multiplicity, int)
            or self.multiplicity <= 0
        ):
            raise ValueError("irrep block multiplicity must be a positive integer")
        if not isinstance(self.irrep, Irrep):
            raise TypeError("irrep block must contain an Irrep")

    @property
    def dim(self) -> int:
        return self.multiplicity * self.irrep.dim

    def __str__(self) -> str:
        return f"{self.multiplicity}x{self.irrep}"


@dataclass(frozen=True)
class IrrepLayout:
    blocks: tuple[IrrepBlock, ...] = ()

    def __post_init__(self) -> None:
        multiplicities: dict[Irrep, int] = {}
        for block in tuple(self.blocks):
            if not isinstance(block, IrrepBlock):
                raise TypeError("irrep layout blocks must be IrrepBlock instances")
            multiplicities[block.irrep] = multiplicities.get(block.irrep, 0) + block.multiplicity
        canonical = tuple(
            IrrepBlock(multiplicities[irrep], irrep)
            for irrep in sorted(multiplicities, key=lambda item: (item.degree, item.parity))
        )
        object.__setattr__(self, "blocks", canonical)

    @classmethod
    def parse(cls, spec: str | IrrepLayout) -> IrrepLayout:
        if isinstance(spec, IrrepLayout):
            return spec
        if not isinstance(spec, str):
            raise TypeError("irreps spec must be a string or IrrepLayout")
        if not spec.strip():
            raise ValueError("irreps spec must not be empty")
        if spec.strip() == "0":
            return cls()
        blocks: list[IrrepBlock] = []
        for raw in spec.split("+"):
            match = _TERM.match(raw)
            if match is None:
                raise ValueError(f"unsupported irreps term: {raw.strip()!r}")
            blocks.append(
                IrrepBlock(
                    int(match.group(1)),
                    Irrep(int(match.group(2)), match.group(3)),
                )
            )
        return cls(tuple(blocks))

    @property
    def dim(self) -> int:
        return sum(block.dim for block in self.blocks)

    @property
    def max_degree(self) -> int:
        return max((block.irrep.degree for block in self.blocks), default=-1)

    @property
    def slices(self) -> dict[Irrep, slice]:
        output: dict[Irrep, slice] = {}
        start = 0
        for block in self.blocks:
            stop = start + block.dim
            output[block.irrep] = slice(start, stop)
            start = stop
        return output

    def slice_for(self, irrep: str | Irrep) -> slice:
        parsed = Irrep.parse(irrep)
        if parsed not in self.slices:
            raise KeyError(f"irrep {parsed} is not present in the layout")
        return self.slices[parsed]

    def multiplicity(self, irrep: str | Irrep) -> int:
        parsed = Irrep.parse(irrep)
        return next((b.multiplicity for b in self.blocks if b.irrep == parsed), 0)

    def __str__(self) -> str:
        return " + ".join(str(block) for block in self.blocks) if self.blocks else "0"


def split_irreps(layout: str | IrrepLayout, value: torch.Tensor) -> dict[str, torch.Tensor]:
    parsed = IrrepLayout.parse(layout)
    if value.shape[-1] != parsed.dim:
        raise ValueError(f"value final dimension must be {parsed.dim}")
    return {
        str(block.irrep): value[..., parsed.slice_for(block.irrep)].reshape(
            *value.shape[:-1], block.multiplicity, block.irrep.dim
        )
        for block in parsed.blocks
    }


def pack_irreps(layout: str | IrrepLayout, blocks: Mapping[str, torch.Tensor]) -> torch.Tensor:
    parsed = IrrepLayout.parse(layout)
    expected = {str(block.irrep) for block in parsed.blocks}
    if set(blocks) != expected:
        raise ValueError(f"blocks must exactly match layout sectors {sorted(expected)}")
    if not parsed.blocks:
        raise ValueError("cannot infer leading shape for an empty layout")
    flattened: list[torch.Tensor] = []
    prefix: tuple[int, ...] | None = None
    for block in parsed.blocks:
        tensor = blocks[str(block.irrep)]
        if tensor.shape[-2:] != (block.multiplicity, block.irrep.dim):
            raise ValueError(
                f"{block.irrep} block must end with shape {(block.multiplicity, block.irrep.dim)}"
            )
        if prefix is None:
            prefix = tensor.shape[:-2]
        elif tensor.shape[:-2] != prefix:
            raise ValueError("all irrep blocks must share leading dimensions")
        flattened.append(tensor.flatten(start_dim=-2))
    return torch.cat(flattened, dim=-1)


def project_symmetric_traceless(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-2:] != (3, 3):
        raise ValueError("value must end with shape (3,3)")
    symmetric = 0.5 * (value + value.transpose(-1, -2))
    trace = symmetric.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
    identity = torch.eye(3, device=value.device, dtype=value.dtype)
    return symmetric - trace[..., None, None] * identity


def matrix_to_st5(value: torch.Tensor) -> torch.Tensor:
    projected = project_symmetric_traceless(value)
    return torch.stack(
        [
            projected[..., 0, 0],
            projected[..., 1, 1],
            projected[..., 0, 1],
            projected[..., 0, 2],
            projected[..., 1, 2],
        ],
        dim=-1,
    )


def st5_to_matrix(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] != 5:
        raise ValueError("value must end with dimension 5")
    xx, yy, xy, xz, yz = value.unbind(dim=-1)
    zz = -xx - yy
    return torch.stack(
        [
            torch.stack([xx, xy, xz], dim=-1),
            torch.stack([xy, yy, yz], dim=-1),
            torch.stack([xz, yz, zz], dim=-1),
        ],
        dim=-2,
    )


def st5_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.shape[-1] != 5:
        raise ValueError("left and right must have equal shapes ending in 5")
    lxx, lyy, lxy, lxz, lyz = left.unbind(dim=-1)
    rxx, ryy, rxy, rxz, ryz = right.unbind(dim=-1)
    return (
        lxx * rxx
        + lyy * ryy
        + (lxx + lyy) * (rxx + ryy)
        + 2.0 * (lxy * rxy + lxz * rxz + lyz * ryz)
    )


def st5_norm(value: torch.Tensor, *, eps: float = 0.0) -> torch.Tensor:
    squared = st5_inner(value, value).clamp_min(0.0) + eps
    if eps > 0.0:
        return torch.sqrt(squared)
    positive = squared > 0.0
    guarded = torch.sqrt(torch.where(positive, squared, torch.ones_like(squared)))
    return torch.where(positive, guarded, torch.zeros_like(guarded))


def st5_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    error = st5_inner(prediction - target, prediction - target) / 5.0
    if reduction == "none":
        return error
    if reduction == "mean":
        return error.mean()
    if reduction == "sum":
        return error.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def _product_parity(left: Irrep, right: Irrep) -> str:
    return "e" if left.parity == right.parity else "o"


@dataclass(frozen=True)
class TensorProductPath:
    left: Irrep
    right: Irrep
    output: Irrep

    def __post_init__(self) -> None:
        if not (
            abs(self.left.degree - self.right.degree)
            <= self.output.degree
            <= self.left.degree + self.right.degree
        ):
            raise ValueError("path violates angular-momentum selection")
        if self.output.parity != _product_parity(self.left, self.right):
            raise ValueError("path violates O(3) parity selection")

    @property
    def signature(self) -> tuple[str, str, str]:
        return str(self.left), str(self.right), str(self.output)


@dataclass(frozen=True)
class TensorProductPlan:
    left: IrrepLayout
    right: IrrepLayout
    output: IrrepLayout
    paths: tuple[TensorProductPath, ...]

    @classmethod
    def compile(
        cls,
        left: str | IrrepLayout,
        right: str | IrrepLayout,
        *,
        output: str | IrrepLayout | None = None,
    ) -> TensorProductPlan:
        left_layout = IrrepLayout.parse(left)
        right_layout = IrrepLayout.parse(right)
        if output is None:
            irreps: set[Irrep] = set()
            for a in left_layout.blocks:
                for b in right_layout.blocks:
                    parity = _product_parity(a.irrep, b.irrep)
                    for degree in range(
                        abs(a.irrep.degree - b.irrep.degree),
                        a.irrep.degree + b.irrep.degree + 1,
                    ):
                        irreps.add(Irrep(degree, parity))
            output_layout = IrrepLayout(tuple(IrrepBlock(1, irrep) for irrep in irreps))
        else:
            output_layout = IrrepLayout.parse(output)
        paths = tuple(
            TensorProductPath(a.irrep, b.irrep, c.irrep)
            for a in left_layout.blocks
            for b in right_layout.blocks
            for c in output_layout.blocks
            if abs(a.irrep.degree - b.irrep.degree)
            <= c.irrep.degree
            <= a.irrep.degree + b.irrep.degree
            and c.irrep.parity == _product_parity(a.irrep, b.irrep)
        )
        if output is not None and output_layout.blocks and not paths:
            raise ValueError("requested output violates angular-momentum or parity selection")
        return cls(left_layout, right_layout, output_layout, paths)

    @property
    def signatures(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(path.signature for path in self.paths)


@dataclass(frozen=True)
class CartesianIrreps:
    scalars: int = 0
    vectors: int = 0
    tensors: int = 0

    @classmethod
    def parse(cls, spec: str | CartesianIrreps) -> CartesianIrreps:
        if isinstance(spec, CartesianIrreps):
            return spec
        layout = IrrepLayout.parse(spec)
        unsupported = [b for b in layout.blocks if str(b.irrep) not in {"0e", "1o", "2e"}]
        if unsupported:
            raise ValueError("CartesianIrreps supports only 0e, 1o, and 2e")
        return cls(layout.multiplicity("0e"), layout.multiplicity("1o"), layout.multiplicity("2e"))

    @property
    def dim(self) -> int:
        return self.scalars + 3 * self.vectors + 5 * self.tensors

    def __str__(self) -> str:
        terms: list[str] = []
        if self.scalars:
            terms.append(f"{self.scalars}x0e")
        if self.vectors:
            terms.append(f"{self.vectors}x1o")
        if self.tensors:
            terms.append(f"{self.tensors}x2e")
        return " + ".join(terms) if terms else "0"
