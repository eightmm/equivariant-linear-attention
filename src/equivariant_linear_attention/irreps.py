from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

import torch


_TERM = re.compile(r"^\s*(\d+)x([012])([eo])\s*$")
_IRREP = re.compile(r"^\s*(\d+)([eo])\s*$")
_GENERAL_TERM = re.compile(r"^\s*(\d+)x(\d+)([eo])\s*$")


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
    def l(self) -> int:  # noqa: E743 - standard angular-momentum notation
        return self.degree

    @property
    def dim(self) -> int:
        return 2 * self.degree + 1

    def __str__(self) -> str:
        return f"{self.degree}{self.parity}"


@dataclass(frozen=True)
class IrrepBlock:
    """A multiplicity of one irrep in a flattened feature layout."""

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
    """Canonical, degree-unbounded collection of O(3) irrep blocks."""

    blocks: tuple[IrrepBlock, ...] = ()

    def __post_init__(self) -> None:
        try:
            blocks = tuple(self.blocks)
        except TypeError as exc:
            raise TypeError("irrep layout blocks must be iterable") from exc
        if any(not isinstance(block, IrrepBlock) for block in blocks):
            raise TypeError("irrep layout blocks must be IrrepBlock instances")

        multiplicities: dict[Irrep, int] = {}
        for block in blocks:
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
            match = _GENERAL_TERM.match(raw)
            if match is None:
                raise ValueError(f"unsupported irreps term: {raw.strip()!r}")
            multiplicity = int(match.group(1))
            if multiplicity <= 0:
                raise ValueError("irrep block multiplicity must be a positive integer")
            blocks.append(
                IrrepBlock(
                    multiplicity,
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
        result: dict[Irrep, slice] = {}
        start = 0
        for block in self.blocks:
            stop = start + block.dim
            result[block.irrep] = slice(start, stop)
            start = stop
        return result

    def slice_for(self, irrep: str | Irrep) -> slice:
        parsed = Irrep.parse(irrep)
        try:
            return self.slices[parsed]
        except KeyError as exc:
            raise KeyError(f"irrep {parsed} is not present in the layout") from exc

    def multiplicity(self, irrep: str | Irrep) -> int:
        parsed = Irrep.parse(irrep)
        return next(
            (
                block.multiplicity
                for block in self.blocks
                if block.irrep == parsed
            ),
            0,
        )

    def __str__(self) -> str:
        return " + ".join(str(block) for block in self.blocks) if self.blocks else "0"


def split_irreps(
    layout: str | IrrepLayout,
    value: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """View one flattened irrep tensor as canonical per-sector blocks."""

    parsed = IrrepLayout.parse(layout)
    if value.shape[-1] != parsed.dim:
        raise ValueError(f"value final dimension must be {parsed.dim}")
    return {
        str(block.irrep): value[..., parsed.slice_for(block.irrep)].reshape(
            *value.shape[:-1],
            block.multiplicity,
            block.irrep.dim,
        )
        for block in parsed.blocks
    }


def pack_irreps(
    layout: str | IrrepLayout,
    blocks: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Pack canonical per-sector blocks into one flattened irrep tensor."""

    parsed = IrrepLayout.parse(layout)
    expected = {str(block.irrep) for block in parsed.blocks}
    if set(blocks) != expected:
        raise ValueError(
            f"blocks must exactly match layout sectors {sorted(expected)}"
        )
    if not parsed.blocks:
        raise ValueError("packing an empty layout requires an explicit shape")
    flattened: list[torch.Tensor] = []
    prefix: tuple[int, ...] | None = None
    for block in parsed.blocks:
        value = blocks[str(block.irrep)]
        expected_tail = (block.multiplicity, block.irrep.dim)
        if value.shape[-2:] != expected_tail:
            raise ValueError(
                f"{block.irrep} block must end with shape {expected_tail}"
            )
        if prefix is None:
            prefix = value.shape[:-2]
        elif value.shape[:-2] != prefix:
            raise ValueError("all irrep blocks must share leading dimensions")
        flattened.append(value.flatten(start_dim=-2))
    return torch.cat(flattened, dim=-1)


def project_symmetric_traceless(value: torch.Tensor) -> torch.Tensor:
    """Orthogonally project a matrix onto the symmetric-traceless subspace."""

    if value.shape[-2:] != (3, 3):
        raise ValueError("value must end with shape (3, 3)")
    symmetric = 0.5 * (value + value.transpose(-1, -2))
    trace_third = torch.diagonal(symmetric, dim1=-2, dim2=-1).sum(-1) / 3.0
    identity = torch.eye(3, device=value.device, dtype=value.dtype)
    return symmetric - trace_third[..., None, None] * identity


def matrix_to_st5(value: torch.Tensor) -> torch.Tensor:
    """Project and compress a 3x3 tensor to ``[xx, yy, xy, xz, yz]``."""

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
    """Expand ``[xx, yy, xy, xz, yz]`` with ``zz=-xx-yy``."""

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
    """Rotation-invariant Frobenius inner product in compact ST5 coordinates."""

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
    """Frobenius norm of a compact symmetric-traceless tensor.

    The zero tensor is a legal input -- a tensor target is frequently zero --
    so the derivative of ``sqrt`` at the origin is masked out rather than
    propagated. With the default ``eps=0.0`` the returned values are exactly
    ``sqrt(<v, v>)``; only the gradient there changes, from ``nan`` to zero.
    """

    if eps < 0.0:
        raise ValueError("eps must be nonnegative")
    squared = st5_inner(value, value).clamp_min(0.0) + eps
    if eps > 0.0:
        return torch.sqrt(squared)
    positive = squared > 0.0
    # sqrt is evaluated on a strictly positive surrogate so its backward never
    # sees zero; ``where`` then discards that branch wherever the norm is zero.
    guarded = torch.sqrt(torch.where(positive, squared, torch.ones_like(squared)))
    return torch.where(positive, guarded, torch.zeros_like(guarded))


def st5_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """O(3)-invariant tensor MSE using the true Frobenius metric."""

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


def _degree_allowed(left: Irrep, right: Irrep, output: Irrep) -> bool:
    return abs(left.degree - right.degree) <= output.degree <= left.degree + right.degree


@dataclass(frozen=True)
class TensorProductPath:
    """One angular-momentum coupling and its optional concrete executor."""

    left: Irrep
    right: Irrep
    output: Irrep
    executor: str | None = None

    def __post_init__(self) -> None:
        if not all(isinstance(irrep, Irrep) for irrep in (self.left, self.right, self.output)):
            raise TypeError("tensor-product path entries must be Irrep instances")
        if not _degree_allowed(self.left, self.right, self.output):
            raise ValueError("tensor-product path violates the angular-momentum selection rule")
        if self.executor is not None and (
            not isinstance(self.executor, str) or not self.executor.strip()
        ):
            raise ValueError("tensor-product executor must be a non-empty string")

    @property
    def natural_parity(self) -> str:
        return _product_parity(self.left, self.right)

    @property
    def parity_mixed(self) -> bool:
        return self.output.parity != self.natural_parity

    @property
    def signature(self) -> tuple[str, str, str]:
        return (str(self.left), str(self.right), str(self.output))


@dataclass(frozen=True)
class TensorProductPlan:
    """Static selection-rule plan, separate from any numerical CG executor."""

    left: IrrepLayout
    right: IrrepLayout
    output: IrrepLayout
    symmetry_group: str
    paths: tuple[TensorProductPath, ...]

    def __post_init__(self) -> None:
        if not all(
            isinstance(layout, IrrepLayout)
            for layout in (self.left, self.right, self.output)
        ):
            raise TypeError("tensor-product plan layouts must be IrrepLayout instances")
        if self.symmetry_group not in {"O3", "SE3"}:
            raise ValueError("symmetry_group must be 'O3' or 'SE3'")

        paths = tuple(self.paths)
        if any(not isinstance(path, TensorProductPath) for path in paths):
            raise TypeError("tensor-product plan paths must be TensorProductPath instances")
        left_irreps = {block.irrep for block in self.left.blocks}
        right_irreps = {block.irrep for block in self.right.blocks}
        output_irreps = {block.irrep for block in self.output.blocks}
        for path in paths:
            if (
                path.left not in left_irreps
                or path.right not in right_irreps
                or path.output not in output_irreps
            ):
                raise ValueError("tensor-product path is outside the plan layouts")
            if self.symmetry_group == "O3" and path.parity_mixed:
                raise ValueError("O3 tensor-product path violates the parity selection rule")
        object.__setattr__(self, "paths", paths)

    @classmethod
    def compile(
        cls,
        left: str | IrrepLayout,
        right: str | IrrepLayout,
        *,
        output: str | IrrepLayout | None = None,
        symmetry_group: str = "O3",
    ) -> TensorProductPlan:
        if symmetry_group not in {"O3", "SE3"}:
            raise ValueError("symmetry_group must be 'O3' or 'SE3'")
        left_layout = IrrepLayout.parse(left)
        right_layout = IrrepLayout.parse(right)

        if output is None:
            inferred: set[Irrep] = set()
            for left_block in left_layout.blocks:
                for right_block in right_layout.blocks:
                    parity = _product_parity(left_block.irrep, right_block.irrep)
                    inferred.update(
                        Irrep(degree, parity)
                        for degree in range(
                            abs(left_block.irrep.degree - right_block.irrep.degree),
                            left_block.irrep.degree + right_block.irrep.degree + 1,
                        )
                    )
            output_layout = IrrepLayout(
                tuple(IrrepBlock(1, irrep) for irrep in inferred)
            )
        else:
            output_layout = IrrepLayout.parse(output)

        paths = tuple(
            TensorProductPath(left_block.irrep, right_block.irrep, output_block.irrep)
            for left_block in left_layout.blocks
            for right_block in right_layout.blocks
            for output_block in output_layout.blocks
            if _degree_allowed(left_block.irrep, right_block.irrep, output_block.irrep)
            and (
                symmetry_group == "SE3"
                or output_block.irrep.parity
                == _product_parity(left_block.irrep, right_block.irrep)
            )
        )
        return cls(
            left=left_layout,
            right=right_layout,
            output=output_layout,
            symmetry_group=symmetry_group,
            paths=paths,
        )

    @property
    def signatures(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(path.signature for path in self.paths)

    def bind_executors(
        self,
        registry: Mapping[tuple[str, str, str], str],
    ) -> TensorProductPlan:
        missing = tuple(path.signature for path in self.paths if path.signature not in registry)
        if missing:
            rendered = ", ".join(" x ".join(signature) for signature in missing)
            raise ValueError(f"unsupported tensor-product paths: {rendered}")

        bound_paths = tuple(
            TensorProductPath(
                path.left,
                path.right,
                path.output,
                executor=registry[path.signature],
            )
            for path in self.paths
        )
        return TensorProductPlan(
            left=self.left,
            right=self.right,
            output=self.output,
            symmetry_group=self.symmetry_group,
            paths=bound_paths,
        )


@dataclass(frozen=True)
class CartesianIrreps:
    """Supported Cartesian O(3) channels: scalar 0e, polar 1o, and tensor 2e."""

    scalars: int = 0
    vectors: int = 0
    tensors: int = 0
    scalar_parity: str = "e"
    vector_parity: str = "o"
    tensor_parity: str = "e"

    def __post_init__(self) -> None:
        counts = (self.scalars, self.vectors, self.tensors)
        if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts):
            raise ValueError("irreps multiplicities must be nonnegative integers")
        parities = (self.scalar_parity, self.vector_parity, self.tensor_parity)
        if any(parity not in {"e", "o"} for parity in parities):
            raise ValueError("irreps parity must be 'e' or 'o'")
        supported = (
            (self.scalars, self.scalar_parity, "e"),
            (self.vectors, self.vector_parity, "o"),
            (self.tensors, self.tensor_parity, "e"),
        )
        if any(count > 0 and parity != expected for count, parity, expected in supported):
            raise ValueError("CartesianIrreps supports only 0e, polar 1o, and 2e channels")

    @classmethod
    def parse(cls, spec: str | CartesianIrreps) -> CartesianIrreps:
        if isinstance(spec, CartesianIrreps):
            return spec
        if not spec.strip():
            raise ValueError("irreps spec must not be empty")

        counts = {0: 0, 1: 0, 2: 0}
        parities = {0: "e", 1: "o", 2: "e"}
        for raw in spec.split("+"):
            match = _TERM.match(raw)
            if match is None:
                msg = f"unsupported irreps term: {raw.strip()!r}"
                raise ValueError(msg)
            count = int(match.group(1))
            degree = int(match.group(2))
            parity = match.group(3)
            if count <= 0:
                raise ValueError("irreps multiplicities must be positive")
            expected_parity = {0: "e", 1: "o", 2: "e"}[degree]
            if parity != expected_parity:
                raise ValueError("CartesianIrreps supports only 0e, polar 1o, and 2e channels")
            counts[degree] += count
            parities[degree] = parity

        return cls(
            scalars=counts[0],
            vectors=counts[1],
            tensors=counts[2],
            scalar_parity=parities[0],
            vector_parity=parities[1],
            tensor_parity=parities[2],
        )

    @property
    def dim(self) -> int:
        return self.scalars + 3 * self.vectors + 5 * self.tensors

    @property
    def storage_dim(self) -> int:
        return self.scalars + 3 * self.vectors + 9 * self.tensors

    def __str__(self) -> str:
        terms = []
        if self.scalars:
            terms.append(f"{self.scalars}x0{self.scalar_parity}")
        if self.vectors:
            terms.append(f"{self.vectors}x1{self.vector_parity}")
        if self.tensors:
            terms.append(f"{self.tensors}x2{self.tensor_parity}")
        return " + ".join(terms) if terms else "0"
