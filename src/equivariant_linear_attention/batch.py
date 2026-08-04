from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch

from .context import ELAContext, OrderContext
from .geometry.prepared import PreparationSpec, Prepared3DGraph

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
        raise ValueError("batch IDs must be nondecreasing")
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


def _interaction_batch(
    sample_batch: torch.Tensor,
    interaction_group: torch.Tensor | None,
) -> torch.Tensor:
    if interaction_group is None:
        return sample_batch
    if not isinstance(interaction_group, torch.Tensor):
        raise TypeError("interaction_group must be a tensor")
    if interaction_group.shape != sample_batch.shape:
        raise ValueError("interaction_group must have shape (N,)")
    if interaction_group.device != sample_batch.device:
        raise ValueError("interaction_group and nodes must share one device")
    if interaction_group.dtype not in _INTEGER_DTYPES:
        raise TypeError("interaction_group must use an integer dtype")
    value = interaction_group.to(dtype=torch.long)
    if bool((value < 0).any().item()):
        raise ValueError("interaction_group values must be nonnegative")
    pair = torch.stack((sample_batch, value), dim=-1)
    _, inverse = torch.unique(pair, dim=0, sorted=True, return_inverse=True)
    return inverse.to(dtype=torch.long)


def _move_order(
    order: OrderContext | None,
    device: torch.device,
    *,
    non_blocking: bool,
) -> OrderContext | None:
    return None if order is None else order.to(device, non_blocking=non_blocking)


@dataclass(frozen=True, slots=True)
class ELABatch:
    """Private packed receiver-major execution representation.

    The public API is :class:`ELAGraph`. This class remains intentionally small
    and internal so geometry preparation and numerical layers see one stable
    tensor contract.
    """

    node_irreps: torch.Tensor
    positions: torch.Tensor
    ptr: torch.Tensor | None = None
    edge_index: torch.Tensor | None = None
    edge_relation_id: torch.Tensor | None = None
    interaction_group: torch.Tensor | None = None
    condition: torch.Tensor | None = None
    order: OrderContext | None = None
    update_mask: torch.Tensor | None = None
    target: torch.Tensor | None = None
    sample_ids: tuple[Any, ...] | None = None
    _prepared_graph: Prepared3DGraph | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _batch: torch.Tensor = field(init=False, repr=False, compare=False)
    _interaction_batch: torch.Tensor = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        node = self.node_irreps
        pos = self.positions
        if not isinstance(node, torch.Tensor) or node.ndim != 2:
            raise ValueError("node_irreps must have shape (N,D)")
        if not isinstance(pos, torch.Tensor) or pos.shape != (node.shape[0], 3):
            raise ValueError("positions must have shape (N,3)")
        if node.shape[0] == 0:
            raise ValueError("packed graph requires at least one node")
        if node.device != pos.device:
            raise ValueError("node_irreps and positions must share one device")
        if not node.is_floating_point() or not pos.is_floating_point():
            raise TypeError("node_irreps and positions must be floating point")

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
        if ptr_long.numel() < 2 or int(ptr_long[0].item()) != 0:
            raise ValueError("ptr must start at zero and contain a graph")
        if int(ptr_long[-1].item()) != node.shape[0]:
            raise ValueError("ptr must end at the packed node count")
        if bool((ptr_long[1:] <= ptr_long[:-1]).any().item()):
            raise ValueError("empty graphs are not supported")
        batch = _batch_from_ptr(ptr_long)
        object.__setattr__(self, "_batch", batch)
        interaction_batch = _interaction_batch(batch, self.interaction_group)
        object.__setattr__(self, "_interaction_batch", interaction_batch)
        if self.interaction_group is not None:
            object.__setattr__(
                self,
                "interaction_group",
                self.interaction_group.to(dtype=torch.long),
            )

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
            edge = edge.to(dtype=torch.long)
            if edge.numel():
                if bool((edge < 0).any().item()) or int(edge.max().item()) >= node.shape[0]:
                    raise ValueError("edge_index contains an out-of-range node")
                if not torch.equal(
                    interaction_batch[edge[0]],
                    interaction_batch[edge[1]],
                ):
                    raise ValueError(
                        "edges may not cross graph or interaction-group boundaries"
                    )
            object.__setattr__(self, "edge_index", edge)

        relation = self.edge_relation_id
        if relation is not None:
            if edge is None:
                raise ValueError("edge_relation_id requires edge_index")
            if not isinstance(relation, torch.Tensor) or relation.shape != (
                edge.shape[1],
            ):
                raise ValueError("edge_relation_id must have shape (E,)")
            if relation.device != node.device or relation.dtype not in _INTEGER_DTYPES:
                raise TypeError(
                    "edge_relation_id must be an integer tensor on the node device"
                )
            object.__setattr__(self, "edge_relation_id", relation.to(dtype=torch.long))

        if self.condition is not None and self.condition.device != node.device:
            raise ValueError("condition and nodes must share one device")
        if self.order is not None and self.order.coordinates.shape[0] != node.shape[0]:
            raise ValueError("order must contain one coordinate per packed node")
        if self.update_mask is not None:
            if (
                self.update_mask.shape != (node.shape[0],)
                or self.update_mask.dtype != torch.bool
            ):
                raise ValueError("update_mask must be boolean with shape (N,)")
            if self.update_mask.device != node.device:
                raise ValueError("update_mask and nodes must share one device")
        if self.sample_ids is not None and len(self.sample_ids) != self.num_graphs:
            raise ValueError("sample_ids length must equal num_graphs")

        prepared = self._prepared_graph
        if prepared is not None:
            if prepared.num_nodes != node.shape[0] or prepared.device != node.device:
                raise ValueError("prepared graph does not match the packed graph")
            if not torch.equal(prepared.batch, interaction_batch):
                raise ValueError("prepared graph membership does not match")

    @property
    def batch(self) -> torch.Tensor:
        return self._batch

    @property
    def interaction_batch(self) -> torch.Tensor:
        return self._interaction_batch

    @property
    def num_interactions(self) -> int:
        return int(self._interaction_batch[-1].item()) + 1

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
        if self.condition is None and self.order is None:
            return None
        return ELAContext(condition=self.condition, order=self.order)

    @property
    def is_prepared(self) -> bool:
        return self._prepared_graph is not None

    @property
    def preparation_spec(self) -> PreparationSpec | None:
        return None if self._prepared_graph is None else self._prepared_graph.spec

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
        interaction_group: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        order: OrderContext | None = None,
        update_mask: torch.Tensor | None = None,
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
            interaction_group=interaction_group,
            condition=condition,
            order=order,
            update_mask=update_mask,
            target=target,
            sample_ids=sample_ids,
        )

    def with_prepared_graph(self, graph: Prepared3DGraph) -> ELABatch:
        if not isinstance(graph, Prepared3DGraph):
            raise TypeError("graph must be a Prepared3DGraph")
        if not graph.spec.can_reuse_positions(self.positions):
            raise ValueError("prepared radius topology does not match positions")
        return replace(self, _prepared_graph=graph)

    def without_prepared_graph(self) -> ELABatch:
        return replace(self, _prepared_graph=None)

    def with_positions(self, positions: torch.Tensor) -> ELABatch:
        graph = self._prepared_graph
        if graph is not None and not graph.spec.can_reuse_positions(positions):
            graph = None
        return replace(self, positions=positions, _prepared_graph=graph)

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        *,
        geometry_dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> ELABatch:
        target = torch.device(device)
        if geometry_dtype is None:
            geometry_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        moved_positions = self.positions.to(
            device=target,
            dtype=geometry_dtype,
            non_blocking=non_blocking,
        )
        prepared = (
            None
            if self._prepared_graph is None
            else self._prepared_graph.to(target)
        )
        if prepared is not None and not prepared.spec.can_reuse_positions(
            moved_positions
        ):
            prepared = None
        return ELABatch(
            node_irreps=self.node_irreps.to(
                device=target,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            positions=moved_positions,
            ptr=None
            if self.ptr is None
            else self.ptr.to(device=target, non_blocking=non_blocking),
            edge_index=None
            if self.edge_index is None
            else self.edge_index.to(device=target, non_blocking=non_blocking),
            edge_relation_id=None
            if self.edge_relation_id is None
            else self.edge_relation_id.to(device=target, non_blocking=non_blocking),
            interaction_group=None
            if self.interaction_group is None
            else self.interaction_group.to(device=target, non_blocking=non_blocking),
            condition=_move_float(
                self.condition,
                device=target,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            order=_move_order(self.order, target, non_blocking=non_blocking),
            update_mask=None
            if self.update_mask is None
            else self.update_mask.to(device=target, non_blocking=non_blocking),
            target=_move_float(
                self.target,
                device=target,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            sample_ids=self.sample_ids,
            _prepared_graph=prepared,
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
        return replace(
            self,
            node_irreps=_pin_tensor(self.node_irreps),
            positions=_pin_tensor(self.positions),
            ptr=_pin_tensor(self.ptr),
            edge_index=_pin_tensor(self.edge_index),
            edge_relation_id=_pin_tensor(self.edge_relation_id),
            interaction_group=_pin_tensor(self.interaction_group),
            condition=_pin_tensor(self.condition),
            order=order,
            update_mask=_pin_tensor(self.update_mask),
            target=_pin_tensor(self.target),
            _prepared_graph=None,
        )


__all__: list[str] = []
