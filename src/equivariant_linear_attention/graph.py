"""Packed token-level 3D data and result container used by TriELA."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from .nn.ops import INTEGER_DTYPES, canonical_batch


def _floating(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating-point dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def _index(name: str, value: torch.Tensor, nodes: int, device: torch.device) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.shape != (nodes,):
        raise ValueError(f"{name} must have shape (N,)")
    if value.dtype not in INTEGER_DTYPES:
        raise TypeError(f"{name} must use an integer dtype")
    if value.device != device:
        raise ValueError(f"{name} and x must share one device")
    if value.numel() and bool((value < 0).any().item()):
        raise ValueError(f"{name} values must be nonnegative")


def _graph_condition(
    value: torch.Tensor, *, num_graphs: int, device: torch.device
) -> torch.Tensor:
    _floating("condition", value)
    if value.device != device:
        raise ValueError("condition and x must share one device")
    if value.ndim == 1:
        return value.unsqueeze(0).expand(num_graphs, -1)
    if value.ndim != 2:
        raise ValueError("condition must have shape (D), (1,D), or (B,D)")
    if value.shape[0] == 1:
        return value.expand(num_graphs, -1)
    if value.shape[0] != num_graphs:
        raise ValueError("condition leading dimension must be one or graph count")
    return value


def _collate_target(values: list[torch.Tensor], counts: list[int]) -> torch.Tensor:
    normalized: list[torch.Tensor] = []
    for value, count in zip(values, counts, strict=True):
        if value.ndim == 0:
            if count != 1:
                raise ValueError("scalar target requires one graph")
            normalized.append(value.reshape(1))
        elif value.shape[0] == count:
            normalized.append(value)
        elif count == 1:
            normalized.append(value.unsqueeze(0))
        else:
            raise ValueError("target leading dimension must equal graph count")
    return torch.cat(normalized, dim=0)


@dataclass(frozen=True, slots=True)
class ELAGraph:
    """Packed 3D node data, interaction isolation, and model outputs."""

    x: torch.Tensor
    pos: torch.Tensor
    batch: torch.Tensor | None = None
    group: torch.Tensor | None = None
    condition: torch.Tensor | None = None
    order: torch.Tensor | None = None
    update_mask: torch.Tensor | None = None
    y: torch.Tensor | None = None
    ids: tuple[Any, ...] | None = None
    graph_x: torch.Tensor | None = None
    graph_sum: torch.Tensor | None = None
    delta: torch.Tensor | None = None
    _num_graphs: int = field(init=False, repr=False, compare=False)
    _batch_version: int = field(init=False, repr=False, compare=False)
    _group_version: int = field(init=False, repr=False, compare=False)
    _condition_matrix: torch.Tensor | None = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _floating("x", self.x)
        _floating("pos", self.pos)
        if self.x.ndim != 2:
            raise ValueError("x must have shape (N,D)")
        if self.pos.shape != (self.x.shape[0], 3):
            raise ValueError("pos must have shape (N,3)")
        if self.x.device != self.pos.device:
            raise ValueError("x and pos must share one device")
        if self.x.dtype != self.pos.dtype:
            raise ValueError("x and pos must share one floating-point dtype")
        nodes = self.x.shape[0]
        _, graphs, _ = canonical_batch(
            self.batch, num_nodes=nodes, device=self.x.device
        )
        # Validation happens when the public container is constructed.  Cache
        # this Python metadata so the numerical forward does not repeat a
        # Tensor.item() extraction that splits torch.compile graphs.
        object.__setattr__(self, "_num_graphs", graphs)
        object.__setattr__(
            self,
            "_batch_version",
            -1 if self.batch is None else self.batch._version,
        )
        object.__setattr__(
            self,
            "_group_version",
            -1 if self.group is None else self.group._version,
        )
        if self.batch is not None:
            _index("batch", self.batch, nodes, self.x.device)
        if self.group is not None:
            _index("group", self.group, nodes, self.x.device)
        normalized_condition = None
        if self.condition is not None:
            normalized_condition = _graph_condition(
                self.condition, num_graphs=graphs, device=self.x.device
            )
        object.__setattr__(self, "_condition_matrix", normalized_condition)
        if self.order is not None:
            _floating("order", self.order)
            if self.order.ndim != 2 or self.order.shape[0] != nodes:
                raise ValueError("order must have shape (N,O)")
            if self.order.device != self.x.device:
                raise ValueError("order and x must share one device")
        if self.update_mask is not None:
            if self.update_mask.dtype != torch.bool or self.update_mask.shape != (
                nodes,
            ):
                raise ValueError("update_mask must be boolean with shape (N,)")
            if self.update_mask.device != self.x.device:
                raise ValueError("update_mask and x must share one device")
        if self.y is not None:
            _floating("y", self.y)
            if self.y.device != self.x.device:
                raise ValueError("y and x must share one device")
        if self.ids is not None and len(self.ids) != graphs:
            raise ValueError("ids length must equal graph count")
        for name, value in (("graph_x", self.graph_x), ("graph_sum", self.graph_sum)):
            if value is None:
                continue
            _floating(name, value)
            if value.ndim != 2 or value.shape[0] != graphs:
                raise ValueError(f"{name} must have shape (B,D)")
        if self.delta is not None:
            _floating("delta", self.delta)
            if self.delta.shape != self.pos.shape:
                raise ValueError("delta must have shape (N,3)")

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_graphs(self) -> int:
        if not torch.compiler.is_compiling():
            if self.batch is not None and self.batch._version != self._batch_version:
                raise RuntimeError(
                    "ELAGraph.batch was mutated in place after validation;"
                    " construct a new ELAGraph instead"
                )
            if self.group is not None and self.group._version != self._group_version:
                raise RuntimeError(
                    "ELAGraph.group was mutated in place after validation;"
                    " construct a new ELAGraph instead"
                )
        return self._num_graphs

    @property
    def batch_index(self) -> torch.Tensor:
        if self.batch is None:
            return torch.zeros(self.num_nodes, device=self.x.device, dtype=torch.long)
        return self.batch.to(dtype=torch.long)

    @property
    def condition_matrix(self) -> torch.Tensor | None:
        return self._condition_matrix

    @torch.compiler.disable
    def with_output(
        self,
        *,
        x: torch.Tensor,
        pos: torch.Tensor,
        graph_x: torch.Tensor,
        graph_sum: torch.Tensor,
        delta: torch.Tensor,
    ) -> ELAGraph:
        """Package validated public outputs outside the compiled tensor core."""

        return replace(
            self, x=x, pos=pos, graph_x=graph_x, graph_sum=graph_sum, delta=delta
        )

    def to(self, *args: Any, **kwargs: Any) -> ELAGraph:
        x = self.x.to(*args, **kwargs)
        device = x.device

        def floating(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.to(*args, **kwargs)

        def indexed(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.to(device=device)

        return replace(
            self,
            x=x,
            pos=self.pos.to(*args, **kwargs),
            batch=indexed(self.batch),
            group=indexed(self.group),
            condition=floating(self.condition),
            order=floating(self.order),
            update_mask=indexed(self.update_mask),
            y=floating(self.y),
            graph_x=floating(self.graph_x),
            graph_sum=floating(self.graph_sum),
            delta=floating(self.delta),
        )

    @classmethod
    def collate(cls, graphs: Iterable[ELAGraph]) -> ELAGraph:
        values = tuple(graphs)
        if not values:
            raise ValueError("cannot collate an empty collection")
        if any(not isinstance(graph, ELAGraph) for graph in values):
            raise TypeError("ELAGraph.collate accepts only ELAGraph")
        if len({graph.x.shape[1] for graph in values}) != 1:
            raise ValueError("all graphs must share x dimension")
        device = values[0].x.device
        if any(graph.x.device != device for graph in values):
            raise ValueError("all graphs must share one device")

        xs: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        batches: list[torch.Tensor] = []
        groups: list[torch.Tensor] = []
        conditions: list[torch.Tensor] = []
        orders: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        ids: list[Any] = []
        counts: list[int] = []
        offset = 0
        has_group = any(g.group is not None for g in values)
        has_condition = any(g.condition is not None for g in values)
        has_order = any(g.order is not None for g in values)
        has_mask = any(g.update_mask is not None for g in values)
        has_target = any(g.y is not None for g in values)
        for flag, attr in (
            (has_condition, "condition"),
            (has_order, "order"),
            (has_target, "y"),
        ):
            if flag and any(getattr(g, attr) is None for g in values):
                raise ValueError(f"{attr} must be present on every collated graph")

        for graph in values:
            local_batch = graph.batch_index
            local_graphs = graph.num_graphs
            counts.append(local_graphs)
            xs.append(graph.x)
            positions.append(graph.pos)
            batches.append(local_batch + offset)
            if has_group:
                groups.append(
                    graph.group.to(dtype=torch.long)
                    if graph.group is not None
                    else torch.zeros_like(local_batch)
                )
            if has_condition:
                assert graph.condition_matrix is not None
                conditions.append(graph.condition_matrix)
            if has_order:
                assert graph.order is not None
                orders.append(graph.order)
            if has_mask:
                masks.append(
                    graph.update_mask
                    if graph.update_mask is not None
                    else torch.ones(graph.num_nodes, dtype=torch.bool, device=device)
                )
            if has_target:
                assert graph.y is not None
                targets.append(graph.y)
            ids.extend(graph.ids if graph.ids is not None else [None] * local_graphs)
            offset += local_graphs

        return cls(
            x=torch.cat(xs),
            pos=torch.cat(positions),
            batch=torch.cat(batches),
            group=torch.cat(groups) if has_group else None,
            condition=torch.cat(conditions) if has_condition else None,
            order=torch.cat(orders) if has_order else None,
            update_mask=torch.cat(masks) if has_mask else None,
            y=_collate_target(targets, counts) if has_target else None,
            ids=tuple(ids),
        )


__all__ = ["ELAGraph"]
