"""Torch-only sparse-neighbor provider contracts.

The radius builder in this module is an intentionally quadratic reference
implementation.  It defines deterministic candidate semantics for tests and
small graphs; it is not a production cell-list implementation.  Likewise, the
Verlet provider adds a correct cache contract around that reference builder but
does not make the underlying ``O(N^2)`` rebuild production-scalable.

Hard learned top-k selection, periodic boundary conditions, and cell-list
construction are deliberately unavailable until their equivariance,
determinism, and device contracts have dedicated implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import NoReturn, Protocol, runtime_checkable

import torch


_INTEGER_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
)
_UNSUPPORTED_CAPABILITIES = frozenset(
    {
        "cell_list",
        "hard_learned_topk",
        "pbc",
    }
)


@runtime_checkable
class NeighborProvider(Protocol):
    """Minimal candidate-list interface shared by fixed and rebuilding policies."""

    def __call__(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        *,
        cutoff: float,
    ) -> torch.Tensor:
        """Return deterministic receiver/sender candidates with shape ``(2, E)``."""


ExternalNeighborCallable = Callable[
    [torch.Tensor, torch.Tensor],
    torch.Tensor,
]


@dataclass(frozen=True, slots=True)
class NeighborCapabilities:
    """Machine-readable scope receipt for a neighbor provider."""

    provider: str
    complexity: str
    production_ready: bool
    deterministic_selection: bool
    supports_skin: bool = False
    supports_pbc: bool = False
    supports_cell_list: bool = False
    supports_hard_learned_topk: bool = False


@dataclass(frozen=True, slots=True, init=False)
class PrecomputedNeighborProvider:
    """Immutable fixed candidate list.

    The input is cloned and canonicalized at construction, and every result is
    cloned so neither the caller nor the original tensor can mutate the stored
    topology.  ``cutoff`` is validated but intentionally does not change the
    list: the consuming model remains responsible for its smooth, strict cutoff
    filter.
    """

    _edge_index: torch.Tensor
    num_nodes: int

    def __init__(self, edge_index: torch.Tensor, *, num_nodes: int) -> None:
        _validate_num_nodes(num_nodes)
        canonical = _validated_edge_index(
            edge_index,
            num_nodes=num_nodes,
            batch=None,
            device=edge_index.device if isinstance(edge_index, torch.Tensor) else None,
        )
        object.__setattr__(self, "_edge_index", canonical.detach().clone())
        object.__setattr__(self, "num_nodes", num_nodes)

    @property
    def edge_index(self) -> torch.Tensor:
        """Return a defensive copy of the fixed canonical candidates."""
        return self._edge_index.clone()

    def __call__(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        *,
        cutoff: float,
    ) -> torch.Tensor:
        _validate_provider_inputs(pos, batch, cutoff=cutoff)
        if pos.shape[0] != self.num_nodes:
            raise ValueError("precomputed neighbor num_nodes must match pos")
        _validated_edge_index(
            self._edge_index,
            num_nodes=self.num_nodes,
            batch=batch,
            device=pos.device,
        )
        return self._edge_index.clone()


@dataclass(frozen=True, slots=True)
class ReferenceRadiusNeighborProvider:
    """Deterministic ``O(N^2)`` strict-radius reference, including self edges.

    Pairs are admitted exactly when their squared, cutoff-normalized
    displacement is below one and both endpoints belong to the same graph.
    ``nonzero`` over the receiver-by-sender mask produces a stable row-major
    order: ascending receiver, then ascending sender.
    """

    def __call__(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        *,
        cutoff: float,
    ) -> torch.Tensor:
        numeric_cutoff = _validate_provider_inputs(pos, batch, cutoff=cutoff)
        node_count = pos.shape[0]
        if node_count == 0:
            return torch.empty((2, 0), dtype=torch.long, device=pos.device)

        geometry = pos.detach().to(dtype=torch.float64)
        displacement = geometry[:, None, :] - geometry[None, :, :]
        scaled = displacement / numeric_cutoff
        squared_scaled_distance = scaled.square().sum(dim=-1)
        same_graph = batch[:, None] == batch[None, :]
        within = same_graph & (squared_scaled_distance < 1.0)
        receiver, sender = within.nonzero(as_tuple=True)
        return torch.stack([receiver, sender])


@dataclass(frozen=True, slots=True)
class ExternalCallableNeighborProvider:
    """Validate and canonicalize candidates returned by an external callable.

    The callable is still required to be torch-native at this boundary: it must
    accept ``(pos, batch, cutoff=<float>)`` and return one tensor on the input
    device.  The adapter does not claim that the callable's *selection* is
    deterministic; it only freezes output ordering and validates graph
    isolation, uniqueness, and self-edge coverage.
    """

    function: Callable[..., torch.Tensor]

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("external neighbor function must be callable")

    def __call__(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        *,
        cutoff: float,
    ) -> torch.Tensor:
        numeric_cutoff = _validate_provider_inputs(pos, batch, cutoff=cutoff)
        edge_index = self.function(
            pos,
            batch,
            cutoff=numeric_cutoff,
        )
        return _validated_edge_index(
            edge_index,
            num_nodes=pos.shape[0],
            batch=batch,
            device=pos.device,
        )


class VerletRadiusNeighborProvider:
    """Radius-plus-skin candidate cache around the quadratic reference builder.

    A rebuild occurs when there is no cache, cutoff/batch/device/shape changes,
    or any node has moved by more than ``skin / 2`` from the coordinates at the
    last rebuild.  Between rebuilds the exact candidate tensor is fixed.  It was
    built at ``cutoff + skin``, so an edge can cross the model cutoff without
    leaving the candidate list.  :meth:`invalidate` explicitly drops all cache
    state.

    This class is a semantic reference, not a production neighbor search:
    rebuilds still use the deterministic ``O(N^2)`` implementation above.
    """

    def __init__(
        self,
        *,
        skin: float,
        builder: NeighborProvider | None = None,
    ) -> None:
        if (
            isinstance(skin, bool)
            or not isinstance(skin, (int, float))
            or not isfinite(float(skin))
        ):
            raise TypeError("skin must be a finite real number")
        if float(skin) < 0.0:
            raise ValueError("skin must be nonnegative")
        if builder is not None and not isinstance(builder, NeighborProvider):
            raise TypeError("builder must satisfy the NeighborProvider protocol")
        self.skin = float(skin)
        self._builder = builder or ReferenceRadiusNeighborProvider()
        self._cached_edge_index: torch.Tensor | None = None
        self._reference_pos: torch.Tensor | None = None
        self._reference_batch: torch.Tensor | None = None
        self._cached_cutoff: float | None = None
        self._rebuild_count = 0

    @property
    def rebuild_count(self) -> int:
        return self._rebuild_count

    def invalidate(self) -> None:
        """Drop cached topology and rebuild reference coordinates."""
        self._cached_edge_index = None
        self._reference_pos = None
        self._reference_batch = None
        self._cached_cutoff = None

    def needs_rebuild(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        *,
        cutoff: float,
    ) -> bool:
        numeric_cutoff = _validate_provider_inputs(pos, batch, cutoff=cutoff)
        if (
            self._cached_edge_index is None
            or self._reference_pos is None
            or self._reference_batch is None
            or self._cached_cutoff is None
        ):
            return True
        if self._cached_cutoff != numeric_cutoff:
            return True
        if (
            self._reference_pos.shape != pos.shape
            or self._reference_pos.device != pos.device
            or self._reference_batch.shape != batch.shape
            or self._reference_batch.device != batch.device
        ):
            return True
        if not torch.equal(self._reference_batch, batch):
            return True
        displacement = pos.detach().to(dtype=torch.float64) - self._reference_pos
        squared_displacement = displacement.square().sum(dim=-1)
        threshold_square = (0.5 * self.skin) ** 2
        return bool((squared_displacement > threshold_square).any().item())

    def __call__(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        *,
        cutoff: float,
    ) -> torch.Tensor:
        numeric_cutoff = _validate_provider_inputs(pos, batch, cutoff=cutoff)
        if self.needs_rebuild(pos, batch, cutoff=numeric_cutoff):
            candidate_cutoff = numeric_cutoff + self.skin
            if not isfinite(candidate_cutoff):
                raise ValueError("cutoff + skin must be finite")
            candidate = self._builder(
                pos,
                batch,
                cutoff=candidate_cutoff,
            )
            self._cached_edge_index = _validated_edge_index(
                candidate,
                num_nodes=pos.shape[0],
                batch=batch,
                device=pos.device,
            ).detach().clone()
            self._reference_pos = pos.detach().to(dtype=torch.float64).clone()
            self._reference_batch = batch.detach().clone()
            self._cached_cutoff = numeric_cutoff
            self._rebuild_count += 1
        if self._cached_edge_index is None:
            raise RuntimeError("Verlet neighbor cache rebuild did not produce edges")
        return self._cached_edge_index.clone()


def neighbor_provider_capabilities(
    provider: NeighborProvider,
) -> NeighborCapabilities:
    """Return conservative, serializable provider capability metadata."""

    if not isinstance(provider, NeighborProvider):
        raise TypeError("provider must satisfy the NeighborProvider protocol")
    if isinstance(provider, PrecomputedNeighborProvider):
        return NeighborCapabilities(
            provider=type(provider).__name__,
            complexity="precomputed",
            production_ready=True,
            deterministic_selection=True,
        )
    if isinstance(provider, ReferenceRadiusNeighborProvider):
        return NeighborCapabilities(
            provider=type(provider).__name__,
            complexity="quadratic_reference",
            production_ready=False,
            deterministic_selection=True,
        )
    if isinstance(provider, ExternalCallableNeighborProvider):
        return NeighborCapabilities(
            provider=type(provider).__name__,
            complexity="external_unknown",
            production_ready=False,
            deterministic_selection=False,
        )
    if isinstance(provider, VerletRadiusNeighborProvider):
        builder = neighbor_provider_capabilities(provider._builder)
        return NeighborCapabilities(
            provider=type(provider).__name__,
            complexity=f"cached_{builder.complexity}",
            production_ready=False,
            deterministic_selection=builder.deterministic_selection,
            supports_skin=True,
            supports_pbc=builder.supports_pbc,
            supports_cell_list=builder.supports_cell_list,
            supports_hard_learned_topk=(
                builder.supports_hard_learned_topk
            ),
        )
    return NeighborCapabilities(
        provider=type(provider).__name__,
        complexity="unknown",
        production_ready=False,
        deterministic_selection=False,
    )


def unsupported_neighbor_capability(capability: str) -> NoReturn:
    """Reject deliberately unavailable topology capabilities.

    The explicit failure prevents a quadratic reference or nondifferentiable
    heuristic from being silently presented as a production implementation.
    """
    if not isinstance(capability, str):
        raise TypeError("neighbor capability must be a string")
    normalized = capability.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "celllist": "cell_list",
        "hard_learned_top_k": "hard_learned_topk",
        "learned_topk": "hard_learned_topk",
        "periodic": "pbc",
        "periodic_boundary_conditions": "pbc",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _UNSUPPORTED_CAPABILITIES:
        choices = ", ".join(sorted(_UNSUPPORTED_CAPABILITIES))
        raise ValueError(f"unknown neighbor capability; unavailable choices: {choices}")
    raise NotImplementedError(
        f"{normalized} neighbor construction is not implemented and is not "
        "production-ready"
    )


def reject_unsupported_neighbor_capability(capability: str) -> NoReturn:
    """Spelled-out alias for :func:`unsupported_neighbor_capability`."""
    unsupported_neighbor_capability(capability)


def _validate_num_nodes(num_nodes: int) -> None:
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int):
        raise TypeError("num_nodes must be an integer")
    if num_nodes < 0:
        raise ValueError("num_nodes must be nonnegative")


def _validate_provider_inputs(
    pos: torch.Tensor,
    batch: torch.Tensor,
    *,
    cutoff: float,
) -> float:
    if not isinstance(pos, torch.Tensor):
        raise TypeError("pos must be a tensor")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must have shape (N, 3)")
    if not torch.is_floating_point(pos):
        raise TypeError("pos must be floating point")
    if not bool(torch.isfinite(pos).all().item()):
        raise ValueError("pos must be finite")
    if not isinstance(batch, torch.Tensor):
        raise TypeError("batch must be a tensor")
    if batch.shape != (pos.shape[0],):
        raise ValueError("batch must have shape (N,)")
    if batch.dtype not in _INTEGER_DTYPES:
        raise TypeError("batch must use an integer dtype")
    if batch.device != pos.device:
        raise ValueError("batch and pos must use the same device")
    batch_long = batch.to(dtype=torch.long)
    if batch_long.numel():
        if bool((batch_long < 0).any().item()):
            raise ValueError("batch indices must be nonnegative")
        labels = torch.unique(batch_long, sorted=True)
        expected = torch.arange(labels.numel(), device=batch.device)
        if not torch.equal(labels, expected):
            raise ValueError("batch indices must be contiguous and start at zero")
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
        raise TypeError("cutoff must be a real number")
    numeric_cutoff = float(cutoff)
    if not isfinite(numeric_cutoff) or numeric_cutoff <= 0.0:
        raise ValueError("cutoff must be finite and positive")
    return numeric_cutoff


def _validated_edge_index(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    batch: torch.Tensor | None,
    device: torch.device | None,
) -> torch.Tensor:
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError("edge_index must be a tensor")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if edge_index.dtype not in _INTEGER_DTYPES:
        raise TypeError("edge_index must use an integer dtype")
    if device is not None and edge_index.device != device:
        raise ValueError("edge_index and positions must use the same device")
    _validate_num_nodes(num_nodes)
    index = edge_index.to(dtype=torch.long)
    if index.numel():
        if bool((index < 0).any().item()):
            raise ValueError("edge_index values must be nonnegative")
        if bool((index >= num_nodes).any().item()):
            raise ValueError("edge_index values are out of range")
    receiver, sender = index.unbind(dim=0)
    pair_code = receiver * max(num_nodes, 1) + sender
    if torch.unique(pair_code).numel() != pair_code.numel():
        raise ValueError("edge_index must not contain duplicate directed edges")
    if batch is not None and receiver.numel():
        batch_long = batch.to(dtype=torch.long)
        if not torch.equal(batch_long[receiver], batch_long[sender]):
            raise ValueError("edge_index must not connect different graphs")
    has_self = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    self_nodes = receiver[receiver == sender]
    has_self[self_nodes] = True
    if not bool(has_self.all().item()):
        raise ValueError("edge_index must contain a self edge for every node")
    order = torch.argsort(pair_code, stable=True)
    return edge_index[:, order].contiguous()


__all__ = [
    "ExternalCallableNeighborProvider",
    "NeighborCapabilities",
    "NeighborProvider",
    "PrecomputedNeighborProvider",
    "ReferenceRadiusNeighborProvider",
    "VerletRadiusNeighborProvider",
    "neighbor_provider_capabilities",
    "reject_unsupported_neighbor_capability",
    "unsupported_neighbor_capability",
]
