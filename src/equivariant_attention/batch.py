from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from .context import ELAContext, OrderContext, RefinementRequest
from .data import (
    BatchLayout,
    collate_graphs as _collate_mappings,
    pack_edges,
    pack_node_input,
)
from .unified import Prepared3DGraph

_INTEGER_DTYPES = frozenset(
    {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
)


def _pin_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None or value.device.type != "cpu" or not torch.cuda.is_available():
        return value
    return value.pin_memory()


def _move_float(
    value: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype | None,
    non_blocking: bool,
) -> torch.Tensor | None:
    if value is None:
        return None
    return value.to(
        device=device,
        dtype=dtype if value.is_floating_point() else None,
        non_blocking=non_blocking,
    )


def _ptr_from_batch(batch: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if batch.shape != (num_nodes,):
        raise ValueError("batch must have shape (N,)")
    if batch.dtype not in _INTEGER_DTYPES:
        raise TypeError("batch must use an integer dtype")
    batch_long = batch.to(dtype=torch.long)
    if num_nodes == 0:
        return batch_long.new_zeros((1,))
    if bool((batch_long < 0).any().item()):
        raise ValueError("batch values must be nonnegative")
    if bool((batch_long[1:] < batch_long[:-1]).any().item()):
        raise ValueError(
            "ELABatch is graph-major; batch IDs must be nondecreasing. "
            "Use ELABatch.collate or reorder nodes before construction."
        )
    num_graphs = int(batch_long[-1].item()) + 1
    expected = torch.arange(num_graphs, device=batch.device)
    if not torch.equal(torch.unique_consecutive(batch_long), expected):
        raise ValueError("batch graph IDs must be contiguous from zero")
    counts = torch.bincount(batch_long, minlength=num_graphs)
    return torch.cat([counts.new_zeros((1,)), counts.cumsum(0)])


def _batch_from_ptr(ptr: torch.Tensor) -> torch.Tensor:
    counts = (ptr[1:] - ptr[:-1]).to(dtype=torch.long)
    return torch.repeat_interleave(
        torch.arange(ptr.numel() - 1, device=ptr.device, dtype=torch.long),
        counts,
        output_size=int(ptr[-1].item()),
    )


def _flatten_order(
    order: OrderContext | None,
    layout: BatchLayout,
) -> OrderContext | None:
    if order is None:
        return None
    coordinates = layout.flatten_node_tensor(
        order.coordinates,
        name="order.coordinates",
    )
    group = None
    if order.group_index is not None:
        group = layout.flatten_node_tensor(
            order.group_index,
            name="order.group_index",
        )
    enabled = None
    if order.enabled is not None:
        enabled = layout.flatten_node_tensor(
            order.enabled,
            name="order.enabled",
        )
    return OrderContext(
        coordinates=coordinates,
        group_index=group,
        periods=order.periods,
        enabled=enabled,
    )


def _flatten_refinement(
    refinement: RefinementRequest | None,
    layout: BatchLayout,
) -> RefinementRequest | None:
    if refinement is None or refinement.update_mask is None:
        return refinement
    return RefinementRequest(
        steps=refinement.steps,
        max_step=refinement.max_step,
        centering=refinement.centering,
        update_mask=layout.flatten_node_tensor(
            refinement.update_mask,
            name="refinement.update_mask",
        ),
        graph_rebuilder=refinement.graph_rebuilder,
    )


def _move_order(
    order: OrderContext | None,
    device: torch.device,
    *,
    non_blocking: bool,
) -> OrderContext | None:
    return None if order is None else order.to(device, non_blocking=non_blocking)


def _move_refinement(
    refinement: RefinementRequest | None,
    device: torch.device,
    *,
    non_blocking: bool,
) -> RefinementRequest | None:
    if refinement is None:
        return None
    mask = refinement.update_mask
    return RefinementRequest(
        steps=refinement.steps,
        max_step=refinement.max_step,
        centering=refinement.centering,
        update_mask=None
        if mask is None
        else mask.to(device=device, non_blocking=non_blocking),
        graph_rebuilder=refinement.graph_rebuilder,
    )


@dataclass(frozen=True, slots=True)
class ELABatch:
    """The single public graph container consumed by :class:`ELA`.

    Nodes are packed graph-major on one ragged axis. ``ptr`` stores graph
    boundaries and ``edge_index`` is optional receiver/sender COO. Omitting
    edges requests geometric radius candidates when the model prepares the
    batch.
    """

    node_irreps: torch.Tensor
    positions: torch.Tensor
    ptr: torch.Tensor | None = None
    edge_index: torch.Tensor | None = None
    edge_relation_id: torch.Tensor | None = None
    condition: torch.Tensor | None = None
    order: OrderContext | None = None
    refinement: RefinementRequest | None = None
    target: torch.Tensor | None = None
    sample_ids: tuple[Any, ...] | None = None
    _layout: BatchLayout | None = field(default=None, repr=False, compare=False)
    _prepared_graph: Prepared3DGraph | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _batch: torch.Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        node = self.node_irreps
        pos = self.positions
        if not isinstance(node, torch.Tensor) or node.ndim != 2:
            raise ValueError("node_irreps must have shape (N,D)")
        if not isinstance(pos, torch.Tensor) or pos.shape != (node.shape[0], 3):
            raise ValueError("positions must have shape (N,3)")
        if node.device != pos.device:
            raise ValueError("node_irreps and positions must share one device")
        if not node.is_floating_point() or not pos.is_floating_point():
            raise TypeError("node_irreps and positions must be floating point")
        if node.shape[0] == 0:
            raise ValueError("ELABatch requires at least one node")

        ptr = self.ptr
        if ptr is None:
            ptr = torch.tensor(
                [0, node.shape[0]],
                device=node.device,
                dtype=torch.long,
            )
            object.__setattr__(self, "ptr", ptr)
        if not isinstance(ptr, torch.Tensor) or ptr.ndim != 1:
            raise ValueError("ptr must have shape (B+1,)")
        if ptr.device != node.device:
            raise ValueError("ptr and nodes must share one device")
        if ptr.dtype not in {torch.int32, torch.int64}:
            raise TypeError("ptr must use int32 or int64")
        ptr_long = ptr.to(dtype=torch.long)
        if ptr_long.numel() == 0 or int(ptr_long[0].item()) != 0:
            raise ValueError("ptr must start at zero")
        if int(ptr_long[-1].item()) != node.shape[0]:
            raise ValueError("ptr must end at the packed node count")
        if bool((ptr_long[1:] < ptr_long[:-1]).any().item()):
            raise ValueError("ptr must be nondecreasing")
        if ptr_long.numel() > 2 and bool(
            (ptr_long[1:] == ptr_long[:-1]).any().item()
        ):
            raise ValueError("empty graphs are not supported in ELABatch")
        object.__setattr__(self, "_batch", _batch_from_ptr(ptr_long))

        edge = self.edge_index
        if edge is not None:
            if (
                not isinstance(edge, torch.Tensor)
                or edge.ndim != 2
                or edge.shape[0] != 2
            ):
                raise ValueError("edge_index must have shape (2,E)")
            if edge.device != node.device or edge.dtype not in _INTEGER_DTYPES:
                raise TypeError(
                    "edge_index must be an integer tensor on the node device"
                )
            edge_long = edge.to(dtype=torch.long)
            if edge_long.numel():
                if bool((edge_long < 0).any().item()) or int(
                    edge_long.max().item()
                ) >= node.shape[0]:
                    raise ValueError("edge_index contains an out-of-range node")
                if not torch.equal(
                    self._batch[edge_long[0]],
                    self._batch[edge_long[1]],
                ):
                    raise ValueError("edges may not cross graph boundaries")
            object.__setattr__(self, "edge_index", edge_long)
        relation = self.edge_relation_id
        if relation is not None:
            if edge is None:
                raise ValueError("edge_relation_id requires edge_index")
            if relation.shape != (self.edge_index.shape[1],):
                raise ValueError("edge_relation_id must have shape (E,)")
            if relation.device != node.device or relation.dtype not in _INTEGER_DTYPES:
                raise TypeError(
                    "edge_relation_id must be an integer tensor on the node device"
                )
            object.__setattr__(
                self,
                "edge_relation_id",
                relation.to(dtype=torch.long),
            )

        if self.condition is not None and self.condition.device != node.device:
            raise ValueError("condition and nodes must share one device")
        if self.order is not None and self.order.coordinates.shape[0] != node.shape[0]:
            raise ValueError("order must contain one coordinate per packed node")
        if (
            self.refinement is not None
            and self.refinement.update_mask is not None
            and self.refinement.update_mask.shape != (node.shape[0],)
        ):
            raise ValueError("refinement update_mask must have shape (N,)")
        if self.sample_ids is not None and len(self.sample_ids) != self.num_graphs:
            raise ValueError("sample_ids length must equal num_graphs")
        if self._prepared_graph is not None:
            graph = self._prepared_graph
            if graph.num_nodes != node.shape[0] or graph.device != node.device:
                raise ValueError("prepared graph does not match the batch")
            if not torch.equal(graph.batch, self._batch):
                raise ValueError("prepared graph membership does not match ptr")

    @property
    def x(self) -> torch.Tensor:
        return self.node_irreps

    @property
    def pos(self) -> torch.Tensor:
        return self.positions

    @property
    def batch(self) -> torch.Tensor:
        return self._batch

    @property
    def num_nodes(self) -> int:
        return int(self.node_irreps.shape[0])

    @property
    def num_graphs(self) -> int:
        if self.ptr is None:
            raise RuntimeError("ptr normalization failed")
        return int(self.ptr.numel() - 1)

    @property
    def num_edges(self) -> int:
        return 0 if self.edge_index is None else int(self.edge_index.shape[1])

    @property
    def context(self) -> ELAContext | None:
        if self.condition is None and self.order is None and self.refinement is None:
            return None
        return ELAContext(
            condition=self.condition,
            order=self.order,
            refinement=self.refinement,
        )

    @property
    def is_prepared(self) -> bool:
        return self._prepared_graph is not None

    @classmethod
    def from_flat(
        cls,
        node_irreps: torch.Tensor,
        positions: torch.Tensor,
        *,
        batch: torch.Tensor | None = None,
        ptr: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_relation_id: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        order: OrderContext | None = None,
        refinement: RefinementRequest | None = None,
        target: torch.Tensor | None = None,
        sample_ids: tuple[Any, ...] | None = None,
    ) -> ELABatch:
        if batch is not None and ptr is not None:
            raise ValueError("supply batch or ptr, not both")
        resolved_ptr = ptr
        if batch is not None:
            resolved_ptr = _ptr_from_batch(batch, node_irreps.shape[0])
        return cls(
            node_irreps=node_irreps,
            positions=positions,
            ptr=resolved_ptr,
            edge_index=edge_index,
            edge_relation_id=edge_relation_id,
            condition=condition,
            order=order,
            refinement=refinement,
            target=target,
            sample_ids=sample_ids,
        )

    @classmethod
    def from_padded(
        cls,
        node_irreps: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        *,
        edge_index: torch.Tensor | Sequence[torch.Tensor] | None = None,
        edge_mask: torch.Tensor | None = None,
        adjacency: torch.Tensor | None = None,
        edge_relation_id: torch.Tensor | Sequence[torch.Tensor] | None = None,
        condition: torch.Tensor | None = None,
        order: OrderContext | None = None,
        refinement: RefinementRequest | None = None,
        target: torch.Tensor | None = None,
        sample_ids: tuple[Any, ...] | None = None,
    ) -> ELABatch:
        packed = pack_node_input(node_irreps, positions, mask=mask)
        edges, relations = pack_edges(
            packed,
            edge_index=edge_index,
            edge_mask=edge_mask,
            adjacency=adjacency,
            edge_relation_id=edge_relation_id,
        )
        counts = mask.sum(dim=1).to(dtype=torch.long)
        ptr = torch.cat([counts.new_zeros((1,)), counts.cumsum(0)])
        packed_condition = condition
        if (
            condition is not None
            and condition.ndim >= 3
            and condition.shape[:2] == mask.shape
        ):
            packed_condition = packed.layout.flatten_node_tensor(
                condition,
                name="condition",
            )
        return cls(
            node_irreps=packed.node_irreps,
            positions=packed.positions,
            ptr=ptr,
            edge_index=edges,
            edge_relation_id=relations,
            condition=packed_condition,
            order=_flatten_order(order, packed.layout),
            refinement=_flatten_refinement(refinement, packed.layout),
            target=target,
            sample_ids=sample_ids,
            _layout=packed.layout,
        )

    @classmethod
    def from_mapping(cls, sample: Mapping[str, Any]) -> ELABatch:
        if isinstance(sample, ELABatch):
            return sample

        def one(primary: str, aliases: tuple[str, ...] = ()) -> Any:
            present = [name for name in (primary, *aliases) if name in sample]
            if len(present) > 1:
                raise ValueError(
                    f"multiple aliases supplied for {primary}: {present}"
                )
            return None if not present else sample[present[0]]

        node = one("node_irreps", ("x", "node_features"))
        pos = one("positions", ("pos",))
        if not isinstance(node, torch.Tensor) or not isinstance(pos, torch.Tensor):
            raise TypeError(
                "mapping requires tensor node_irreps/x and positions/pos"
            )
        order = sample.get("order")
        if isinstance(order, torch.Tensor):
            order = (
                OrderContext.sequence(
                    order,
                    segment_id=sample.get("order_group"),
                    enabled=sample.get("order_mask"),
                )
                if order.ndim == 1
                else OrderContext.grid(
                    order,
                    segment_id=sample.get("order_group"),
                    periods=sample.get("order_periods"),
                    enabled=sample.get("order_mask"),
                )
            )
        ids = one("sample_ids", ("sample_id", "id", "idx"))
        if ids is not None and not isinstance(ids, tuple):
            ids = (ids,)
        return cls.from_flat(
            node,
            pos,
            batch=sample.get("batch"),
            ptr=sample.get("ptr"),
            edge_index=sample.get("edge_index"),
            edge_relation_id=one("edge_relation_id", ("edge_type",)),
            condition=sample.get("condition"),
            order=order,
            refinement=sample.get("refinement"),
            target=one("target", ("y", "label", "labels")),
            sample_ids=ids,
        )

    @classmethod
    def collate(cls, samples: Sequence[Mapping[str, Any]]) -> ELABatch:
        normalized: list[dict[str, Any]] = []
        for sample in samples:
            item = dict(sample)
            for canonical, aliases in (
                ("target", ("y", "label", "labels")),
                ("edge_relation_id", ("edge_type",)),
                ("sample_id", ("id", "idx")),
            ):
                present = [name for name in (canonical, *aliases) if name in item]
                if len(present) > 1:
                    raise ValueError(
                        f"sample contains multiple aliases for {canonical}: {present}"
                    )
                if present and present[0] != canonical:
                    item[canonical] = item[present[0]]
            normalized.append(item)
        return cls.from_mapping(_collate_mappings(normalized))

    def with_prepared_graph(self, graph: Prepared3DGraph) -> ELABatch:
        return replace(self, _prepared_graph=graph)

    def without_prepared_graph(self) -> ELABatch:
        return replace(self, _prepared_graph=None)

    def restore_nodes(
        self,
        value: torch.Tensor,
        *,
        template: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._layout is None:
            return value
        return self._layout.restore_node_tensor(value, template=template)

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        *,
        geometry_dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> ELABatch:
        target_device = torch.device(device)
        if geometry_dtype is None:
            geometry_dtype = (
                torch.float64 if dtype == torch.float64 else torch.float32
            )
        graph = (
            None
            if self._prepared_graph is None
            else self._prepared_graph.to(target_device)
        )
        layout = None
        if self._layout is not None:
            layout = BatchLayout(
                kind=self._layout.kind,
                batch=self._layout.batch.to(
                    device=target_device,
                    non_blocking=non_blocking,
                ),
                batch_size=self._layout.batch_size,
                max_nodes=self._layout.max_nodes,
                node_mask=None
                if self._layout.node_mask is None
                else self._layout.node_mask.to(
                    device=target_device,
                    non_blocking=non_blocking,
                ),
            )
        return ELABatch(
            node_irreps=self.node_irreps.to(
                device=target_device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            positions=self.positions.to(
                device=target_device,
                dtype=geometry_dtype,
                non_blocking=non_blocking,
            ),
            ptr=self.ptr.to(device=target_device, non_blocking=non_blocking)
            if self.ptr is not None
            else None,
            edge_index=None
            if self.edge_index is None
            else self.edge_index.to(
                device=target_device,
                non_blocking=non_blocking,
            ),
            edge_relation_id=None
            if self.edge_relation_id is None
            else self.edge_relation_id.to(
                device=target_device,
                non_blocking=non_blocking,
            ),
            condition=_move_float(
                self.condition,
                device=target_device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            order=_move_order(
                self.order,
                target_device,
                non_blocking=non_blocking,
            ),
            refinement=_move_refinement(
                self.refinement,
                target_device,
                non_blocking=non_blocking,
            ),
            target=_move_float(
                self.target,
                device=target_device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            sample_ids=self.sample_ids,
            _layout=layout,
            _prepared_graph=graph,
        )

    def pin_memory(self) -> ELABatch:
        order = self.order
        if order is not None:
            order = OrderContext(
                coordinates=_pin_tensor(order.coordinates),
                group_index=_pin_tensor(order.group_index),
                periods=_pin_tensor(order.periods),
                enabled=_pin_tensor(order.enabled),
            )
        refinement = self.refinement
        if refinement is not None and refinement.update_mask is not None:
            refinement = RefinementRequest(
                steps=refinement.steps,
                max_step=refinement.max_step,
                centering=refinement.centering,
                update_mask=_pin_tensor(refinement.update_mask),
                graph_rebuilder=refinement.graph_rebuilder,
            )
        return replace(
            self,
            node_irreps=_pin_tensor(self.node_irreps),
            positions=_pin_tensor(self.positions),
            ptr=_pin_tensor(self.ptr),
            edge_index=_pin_tensor(self.edge_index),
            edge_relation_id=_pin_tensor(self.edge_relation_id),
            condition=_pin_tensor(self.condition),
            order=order,
            refinement=refinement,
            target=_pin_tensor(self.target),
            _prepared_graph=None,
        )


__all__ = ["ELABatch"]
