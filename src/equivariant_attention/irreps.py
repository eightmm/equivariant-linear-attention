from __future__ import annotations

from dataclasses import dataclass
import re


_TERM = re.compile(r"^\s*(\d+)x([012])([eo])\s*$")


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
