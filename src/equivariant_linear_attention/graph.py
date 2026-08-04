"""The single public graph container used by :class:`ELA`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from .batch import ELABatch
from .context import OrderContext
from .geometry.prepared import Prepared3DGraph


_INTEGER_DTYPES = frozenset(
    {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
)


def _require_finite(name: str, value: torch.Tensor) -> None:
    finite = torch.isfinite(value).all()
    async_assert = getattr(torch, "_assert_async", None)
    if value.device.type == "cuda" and async_assert is not None:
        async_assert(finite, f"{name} must contain only finite values")
    elif not bool(finite.item()):
        raise ValueError(f"{name} must contain only finite values")


def _require_float_tensor(
    name: str,
    value: object,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if device is not None and value.device != device:
        raise ValueError(f"{name} and x must share one device")
    _require_finite(name, value)
    return value


def _normalize_index(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.shape != shape:
        rendered = ",".join(str(size) for size in shape)
        raise ValueError(f"{name} must have shape ({rendered},)")
    if value.device != device:
        raise ValueError(f"{name} and x must share one device")
    if value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must use an integer dtype")
    result = value.to(dtype=torch.long)
    if result.numel() and bool((result < 0).any().item()):
        raise ValueError(f"{name} values must be nonnegative")
    return result


def _sample_batch(
    batch: torch.Tensor | None,
    *,
    nodes: int,
    device: torch.device,
) -> torch.Tensor:
    if batch is None:
        return torch.zeros(nodes, device=device, dtype=torch.long)
    value = _normalize_index("batch", batch, shape=(nodes,), device=device)
    if bool((value[1:] < value[:-1]).any().item()):
        raise ValueError("batch must be graph-major and nondecreasing")
    graphs = int(value[-1].item()) + 1
    expected = torch.arange(graphs, device=device)
    if not torch.equal(torch.unique_consecutive(value), expected):
        raise ValueError("batch graph IDs must be contiguous from zero")
    return value


def _interaction_batch(
    sample_batch: torch.Tensor,
    group: torch.Tensor | None,
) -> torch.Tensor:
    if group is None:
        return sample_batch
    pair = torch.stack((sample_batch, group), dim=-1)
    _, inverse = torch.unique(pair, dim=0, sorted=True, return_inverse=True)
    return inverse.to(dtype=torch.long)


def _validate_order(
    order: object,
    *,
    nodes: int,
    device: torch.device,
) -> OrderContext:
    if not isinstance(order, OrderContext):
        raise TypeError("order must be an OrderContext")
    coordinates = order.coordinates
    if not isinstance(coordinates, torch.Tensor):
        raise TypeError("order.coordinates must be a tensor")
    if not coordinates.is_floating_point() and coordinates.dtype not in _INTEGER_DTYPES:
        raise TypeError("order.coordinates must use a numeric dtype")
    if coordinates.device != device:
        raise ValueError("order.coordinates and x must share one device")
    if coordinates.is_floating_point():
        _require_finite("order.coordinates", coordinates)
    if coordinates.ndim != 2 or coordinates.shape[0] != nodes:
        raise ValueError("order.coordinates must have shape (N,K)")
    if coordinates.shape[1] == 0:
        raise ValueError("order.coordinates must contain at least one coordinate")

    group_index = order.group_index
    if group_index is not None:
        group_index = _normalize_index(
            "order.group_index",
            group_index,
            shape=(nodes,),
            device=device,
        )

    periods = order.periods
    if periods is not None:
        if not isinstance(periods, torch.Tensor):
            raise TypeError("order.periods must be a tensor")
        if not periods.is_floating_point():
            raise TypeError("periods must be floating point")
        if periods.shape != (coordinates.shape[1],):
            raise ValueError("order.periods must have shape (K,)")
        if periods.device != device:
            raise ValueError("order.periods and x must share one device")
        _require_finite("order.periods", periods)
        if bool((periods < 0).any().item()):
            raise ValueError("order.periods must be nonnegative")

    enabled = order.enabled
    if enabled is not None:
        if not isinstance(enabled, torch.Tensor):
            raise TypeError("order.enabled must be a tensor")
        if enabled.shape != (nodes,) or enabled.dtype != torch.bool:
            raise ValueError("order.enabled must be boolean with shape (N,)")
        if enabled.device != device:
            raise ValueError("order.enabled and x must share one device")

    if (
        coordinates is order.coordinates
        and group_index is order.group_index
        and periods is order.periods
        and enabled is order.enabled
    ):
        return order
    return OrderContext(
        coordinates=coordinates,
        group_index=group_index,
        periods=periods,
        enabled=enabled,
    )


def _pin_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None or value.device.type != "cpu" or not torch.cuda.is_available():
        return value
    return value.pin_memory()


def _move_value(
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


def _presence(name: str, values: Sequence[object | None]) -> bool:
    present = tuple(value is not None for value in values)
    if any(present) and not all(present):
        raise ValueError(f"all samples must provide {name} or all must omit it")
    return all(present)


def _same_tensor_schema(name: str, values: Sequence[torch.Tensor]) -> None:
    first = values[0]
    for value in values[1:]:
        if value.device != first.device:
            raise ValueError(f"all sample {name} tensors must share one device")
        if value.dtype != first.dtype:
            raise TypeError(f"all sample {name} tensors must share one dtype")


@dataclass(frozen=True, slots=True)
class _TensorProvenance:
    identity: int
    version: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def capture(cls, value: torch.Tensor) -> _TensorProvenance | None:
        try:
            version = int(value._version)
        except RuntimeError:
            return None
        return cls(
            identity=id(value),
            version=version,
            shape=tuple(value.shape),
            stride=tuple(value.stride()),
            dtype=value.dtype,
            device=value.device,
        )

    def matches(self, value: torch.Tensor) -> bool:
        current = self.capture(value)
        return current == self


@dataclass(frozen=True, slots=True)
class _PreparedProvenance:
    source: str
    pos: _TensorProvenance | None
    edge_index: _TensorProvenance | None
    batch: _TensorProvenance | None
    edge_type: _TensorProvenance | None
    group: _TensorProvenance | None


def _capture_tensor(value: torch.Tensor | None) -> _TensorProvenance | None:
    return None if value is None else _TensorProvenance.capture(value)


@dataclass(frozen=True, slots=True)
class ELAGraph:
    """One immutable input/output graph for :class:`ELA`.

    Public ``edge_index`` follows source-to-target order: row zero is the
    sender and row one is the receiver. The conversion to the private
    receiver-major representation happens only in :meth:`_to_packed`.
    """

    x: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor | None = None
    batch: torch.Tensor | None = None
    edge_type: torch.Tensor | None = None
    group: torch.Tensor | None = None
    condition: torch.Tensor | None = None
    order: OrderContext | None = None
    update_mask: torch.Tensor | None = None
    y: torch.Tensor | None = None
    ids: tuple[Any, ...] | None = None
    graph_x: torch.Tensor | None = None
    graph_sum: torch.Tensor | None = None
    delta: torch.Tensor | None = None
    _prepared_graph: Prepared3DGraph | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _prepared_provenance: _PreparedProvenance | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _packed_template: ELABatch | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _assume_immutable_storage: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self._assume_immutable_storage, bool):
            raise TypeError("_assume_immutable_storage must be boolean")
        x = _require_float_tensor("x", self.x)
        if x.ndim != 2:
            raise ValueError("x must have shape (N,D)")
        if x.shape[0] == 0:
            raise ValueError("ELAGraph requires at least one node")
        pos = _require_float_tensor("pos", self.pos, device=x.device)
        if pos.shape != (x.shape[0], 3):
            raise ValueError("pos must have shape (N,3)")

        nodes = x.shape[0]
        sample_batch = _sample_batch(
            self.batch,
            nodes=nodes,
            device=x.device,
        )
        if self.batch is not None:
            object.__setattr__(self, "batch", sample_batch)
        groups = self.group
        if groups is not None:
            groups = _normalize_index(
                "group",
                groups,
                shape=(nodes,),
                device=x.device,
            )
            object.__setattr__(self, "group", groups)
        interaction_batch = _interaction_batch(sample_batch, groups)

        edge = self.edge_index
        if edge is not None:
            if not isinstance(edge, torch.Tensor):
                raise TypeError("edge_index must be a tensor")
            if edge.ndim != 2 or edge.shape[0] != 2:
                raise ValueError("edge_index must have shape (2,E)")
            if edge.device != x.device:
                raise ValueError("edge_index and x must share one device")
            if edge.dtype not in _INTEGER_DTYPES:
                raise TypeError("edge_index must use an integer dtype")
            edge = edge.to(dtype=torch.long)
            if edge.numel():
                if bool((edge < 0).any().item()) or int(edge.max().item()) >= nodes:
                    raise ValueError("edge_index contains an out-of-range node")
                source, target = edge
                if not torch.equal(sample_batch[source], sample_batch[target]):
                    raise ValueError("edges may not cross graph boundaries")
                if not torch.equal(
                    interaction_batch[source],
                    interaction_batch[target],
                ):
                    raise ValueError("edges may not cross interaction-group boundaries")
            object.__setattr__(self, "edge_index", edge)

        edge_type = self.edge_type
        if edge_type is not None:
            if edge is None:
                raise ValueError("edge_type requires edge_index")
            edge_type = _normalize_index(
                "edge_type",
                edge_type,
                shape=(edge.shape[1],),
                device=x.device,
            )
            object.__setattr__(self, "edge_type", edge_type)

        condition = self.condition
        if condition is not None:
            if not isinstance(condition, torch.Tensor):
                raise TypeError("condition must be a tensor")
            if not condition.is_floating_point():
                raise TypeError("condition must be floating point")
            if condition.device != x.device:
                raise ValueError("condition and x must share one device")
            if condition.ndim not in {1, 2}:
                raise ValueError("condition must have shape (C,), (G,C), or (N,C)")
            if condition.ndim == 2 and condition.shape[0] not in {
                1,
                nodes,
                self.num_graphs,
            }:
                raise ValueError(
                    "condition leading dimension must be one, the node count, "
                    "or the graph count"
                )
            _require_finite("condition", condition)

        if self.order is not None:
            object.__setattr__(
                self,
                "order",
                _validate_order(self.order, nodes=nodes, device=x.device),
            )

        if self.update_mask is not None:
            mask = self.update_mask
            if not isinstance(mask, torch.Tensor):
                raise TypeError("update_mask must be a tensor")
            if mask.shape != (nodes,) or mask.dtype != torch.bool:
                raise ValueError("update_mask must be boolean with shape (N,)")
            if mask.device != x.device:
                raise ValueError("update_mask and x must share one device")

        if self.y is not None:
            if not isinstance(self.y, torch.Tensor):
                raise TypeError("y must be a tensor")
            if self.y.device != x.device:
                raise ValueError("y and x must share one device")
            if self.y.is_floating_point() or self.y.is_complex():
                _require_finite("y", self.y)
            if self.num_graphs > 1 and (
                self.y.ndim == 0 or self.y.shape[0] != self.num_graphs
            ):
                raise ValueError("packed y must have one leading item per graph")

        if self.ids is not None:
            if not isinstance(self.ids, tuple):
                raise TypeError("ids must be a tuple")
            if len(self.ids) != self.num_graphs:
                raise ValueError("ids length must equal the graph count")

        for name in ("graph_x", "graph_sum"):
            value = getattr(self, name)
            if value is None:
                continue
            value = _require_float_tensor(name, value, device=x.device)
            if value.ndim != 2 or value.shape[0] != self.num_graphs:
                raise ValueError(f"{name} must have shape (B,D)")
        if self.graph_x is not None and self.graph_sum is not None:
            if self.graph_x.shape != self.graph_sum.shape:
                raise ValueError("graph_x and graph_sum must have equal shapes")

        if self.delta is not None:
            delta = _require_float_tensor("delta", self.delta, device=x.device)
            if delta.shape != (nodes, 3):
                raise ValueError("delta must have shape (N,3)")

        prepared = self._prepared_graph
        if prepared is not None:
            if not isinstance(prepared, Prepared3DGraph):
                raise TypeError("private prepared cache must be a Prepared3DGraph")
            if prepared.num_nodes != nodes or prepared.device != x.device:
                raise ValueError("private prepared cache does not match the graph")
            if not torch.equal(prepared.batch, interaction_batch):
                raise ValueError("private prepared cache membership does not match")
            if not prepared.spec.can_reuse_positions(pos):
                raise ValueError("private prepared cache is stale for pos")

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return 0 if self.edge_index is None else int(self.edge_index.shape[1])

    @property
    def num_graphs(self) -> int:
        return 1 if self.batch is None else int(self.batch[-1].item()) + 1

    def _capture_prepared_provenance(
        self,
        source: str,
    ) -> _PreparedProvenance | None:
        if not self._assume_immutable_storage:
            # Public tensors may be exported through DLPack and changed through
            # an alias whose version counter is independent. The safe default
            # therefore takes exact-content validation on every cache reuse.
            return None
        values = (self.edge_index, self.batch, self.edge_type, self.group)
        stamps = tuple(_capture_tensor(value) for value in values)
        if any(
            value is not None and stamp is None
            for value, stamp in zip(values, stamps, strict=True)
        ):
            return None
        pos = None
        if source == "radius":
            pos = _capture_tensor(self.pos)
            if pos is None:
                return None
        return _PreparedProvenance(
            source=source,
            pos=pos,
            edge_index=stamps[0],
            batch=stamps[1],
            edge_type=stamps[2],
            group=stamps[3],
        )

    def assume_immutable(self) -> ELAGraph:
        """Return graph-owned topology storage eligible for trusted cache reuse.

        The returned topology-bearing tensors are fresh clones, so aliases to
        this graph's inputs cannot change them. Callers must not mutate or
        export aliases of the returned ``pos``, ``edge_index``, ``batch``,
        ``edge_type``, or ``group`` tensors. Use the default graph path when
        that lifetime contract cannot be guaranteed; it validates content
        exactly on every reuse.
        """

        owned = replace(
            self,
            pos=self.pos.clone(),
            edge_index=None if self.edge_index is None else self.edge_index.clone(),
            batch=None if self.batch is None else self.batch.clone(),
            edge_type=None if self.edge_type is None else self.edge_type.clone(),
            group=None if self.group is None else self.group.clone(),
        )
        object.__setattr__(owned, "_assume_immutable_storage", True)
        return owned

    def _prepared_provenance_matches(self) -> bool:
        if not self._assume_immutable_storage:
            return False
        prepared = self._prepared_graph
        provenance = self._prepared_provenance
        if prepared is None or provenance is None:
            return False
        if provenance.source != prepared.spec.source:
            return False
        fields = (
            (self.edge_index, provenance.edge_index),
            (self.batch, provenance.batch),
            (self.edge_type, provenance.edge_type),
            (self.group, provenance.group),
        )
        for value, stamp in fields:
            if (value is None) != (stamp is None):
                return False
            if value is not None and (stamp is None or not stamp.matches(value)):
                return False
        if provenance.source == "radius":
            if provenance.pos is None:
                return False
            if not provenance.pos.matches(self.pos):
                # A version change is a cheap invalidation signal, not proof
                # that a Verlet shell expired. Recheck displacement only on
                # that slow path, then refresh the trusted stamp.
                if not prepared.spec.can_reuse_positions(self.pos):
                    return False
                refreshed = self._capture_prepared_provenance("radius")
                if refreshed is None:
                    return False
                object.__setattr__(self, "_prepared_provenance", refreshed)
        return True

    def _trusted_packed_template(self) -> ELABatch | None:
        """Return the O(1) packed reuse carrier admitted by the opt-in contract."""

        if not self._prepared_provenance_matches():
            return None
        prepared = self._prepared_graph
        template = self._packed_template
        if (
            prepared is None
            or template is None
            or template._prepared_graph is not prepared
            or not template._trusted_prepared
            or template.node_irreps is not self.x
            or template.positions is not self.pos
            or template.condition is not self.condition
            or template.order is not self.order
            or template.update_mask is not self.update_mask
            or template.target is not self.y
            or template.sample_ids is not self.ids
            or self.x.ndim != 2
            or self.x.shape[0] != prepared.num_nodes
            or self.pos.shape != (prepared.num_nodes, 3)
            or self.x.device != prepared.device
            or self.pos.device != prepared.device
        ):
            return None
        return template

    def _to_packed(self) -> ELABatch:
        """Convert source/target public topology to private receiver/sender COO."""

        template = self._trusted_packed_template()
        if template is not None:
            return template

        internal_edges = None
        if self.edge_index is not None:
            internal_edges = self.edge_index[[1, 0]]
        packed = ELABatch.from_flat(
            self.x,
            self.pos,
            batch=self.batch,
            edge_index=internal_edges,
            edge_relation_id=self.edge_type,
            interaction_group=self.group,
            condition=self.condition,
            order=self.order,
            update_mask=self.update_mask,
            target=self.y,
            sample_ids=self.ids,
        )
        if self._prepared_provenance_matches():
            if self._prepared_graph is None:
                raise RuntimeError("prepared provenance invariant failed")
            packed = packed._with_prepared_graph_trusted(self._prepared_graph)
        elif self._prepared_graph is not None:
            # Tensors created in inference mode have no version counters, and
            # a changed-then-restored tensor may have safe content despite a
            # newer version. Preserve the validated slow fallback in those
            # cases; ELA performs exact edge/relation checks before reuse.
            prepared = self._prepared_graph
            if torch.equal(prepared.batch, packed.interaction_batch) and (
                prepared.spec.can_reuse_positions(self.pos)
            ):
                packed = packed.with_prepared_graph(prepared)
        return packed

    def _with_prepared(
        self,
        prepared: Prepared3DGraph,
        packed_template: ELABatch | None = None,
    ) -> ELAGraph:
        if not isinstance(prepared, Prepared3DGraph):
            raise TypeError("prepared must be a Prepared3DGraph")
        if (
            packed_template is not None
            and packed_template._prepared_graph is not prepared
        ):
            raise ValueError("packed template must carry the prepared graph")
        candidate = replace(self)
        object.__setattr__(
            candidate,
            "_assume_immutable_storage",
            self._assume_immutable_storage,
        )
        object.__setattr__(candidate, "_prepared_graph", prepared)
        object.__setattr__(
            candidate,
            "_prepared_provenance",
            candidate._capture_prepared_provenance(prepared.spec.source),
        )
        if candidate._prepared_provenance is not None and packed_template is not None:
            object.__setattr__(candidate, "_packed_template", packed_template)
        return candidate

    def _cache_prepared(self, packed: ELABatch) -> None:
        """Attach validated private topology without changing the public graph."""

        prepared = packed._prepared_graph
        if prepared is None:
            raise ValueError("packed cache requires a prepared graph")
        candidate = self._with_prepared(prepared, packed)
        object.__setattr__(self, "_prepared_graph", candidate._prepared_graph)
        object.__setattr__(
            self,
            "_prepared_provenance",
            candidate._prepared_provenance,
        )
        object.__setattr__(self, "_packed_template", candidate._packed_template)

    def _with_output(
        self,
        *,
        x: torch.Tensor,
        graph_x: torch.Tensor,
        graph_sum: torch.Tensor,
        pos: torch.Tensor,
        delta: torch.Tensor,
        prepared: Prepared3DGraph | None,
    ) -> ELAGraph:
        candidate = replace(
            self,
            x=x,
            pos=pos,
            graph_x=graph_x,
            graph_sum=graph_sum,
            delta=delta,
        )
        object.__setattr__(
            candidate,
            "_assume_immutable_storage",
            self._assume_immutable_storage,
        )
        object.__setattr__(candidate, "_prepared_graph", prepared)
        if prepared is not None:
            object.__setattr__(
                candidate,
                "_prepared_provenance",
                candidate._capture_prepared_provenance(prepared.spec.source),
            )
        return candidate

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        *,
        geometry_dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> ELAGraph:
        target = torch.device(device)
        if dtype is not None and not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("dtype must be a floating-point dtype")
        if geometry_dtype is None:
            geometry_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        if not torch.empty((), dtype=geometry_dtype).is_floating_point():
            raise TypeError("geometry_dtype must be a floating-point dtype")
        moved_pos = self.pos.to(
            device=target,
            dtype=geometry_dtype,
            non_blocking=non_blocking,
        )
        cached = self._prepared_graph
        prepared = (
            None
            if cached is None or not self._prepared_provenance_matches()
            else cached.to(target, non_blocking=non_blocking)
        )
        if prepared is not None and not prepared.spec.can_reuse_positions(moved_pos):
            prepared = None
        moved = ELAGraph(
            x=self.x.to(device=target, dtype=dtype, non_blocking=non_blocking),
            pos=moved_pos,
            edge_index=None
            if self.edge_index is None
            else self.edge_index.to(device=target, non_blocking=non_blocking),
            batch=None
            if self.batch is None
            else self.batch.to(device=target, non_blocking=non_blocking),
            edge_type=None
            if self.edge_type is None
            else self.edge_type.to(device=target, non_blocking=non_blocking),
            group=None
            if self.group is None
            else self.group.to(device=target, non_blocking=non_blocking),
            condition=_move_value(
                self.condition,
                device=target,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            order=None
            if self.order is None
            else self.order.to(target, non_blocking=non_blocking),
            update_mask=None
            if self.update_mask is None
            else self.update_mask.to(device=target, non_blocking=non_blocking),
            y=_move_value(
                self.y,
                device=target,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            ids=self.ids,
            graph_x=_move_value(
                self.graph_x,
                device=target,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            graph_sum=_move_value(
                self.graph_sum,
                device=target,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            delta=_move_value(
                self.delta,
                device=target,
                dtype=geometry_dtype,
                non_blocking=non_blocking,
            ),
        )
        object.__setattr__(
            moved,
            "_assume_immutable_storage",
            self._assume_immutable_storage,
        )
        return moved if prepared is None else moved._with_prepared(prepared)

    def pin_memory(self) -> ELAGraph:
        order = self.order
        if order is not None:
            order = OrderContext(
                coordinates=_pin_tensor(order.coordinates),
                group_index=_pin_tensor(order.group_index),
                periods=_pin_tensor(order.periods),
                enabled=_pin_tensor(order.enabled),
            )
        pinned = replace(
            self,
            x=_pin_tensor(self.x),
            pos=_pin_tensor(self.pos),
            edge_index=_pin_tensor(self.edge_index),
            batch=_pin_tensor(self.batch),
            edge_type=_pin_tensor(self.edge_type),
            group=_pin_tensor(self.group),
            condition=_pin_tensor(self.condition),
            order=order,
            update_mask=_pin_tensor(self.update_mask),
            y=_pin_tensor(self.y),
            graph_x=_pin_tensor(self.graph_x),
            graph_sum=_pin_tensor(self.graph_sum),
            delta=_pin_tensor(self.delta),
        )
        object.__setattr__(
            pinned,
            "_assume_immutable_storage",
            self._assume_immutable_storage,
        )
        return pinned

    @classmethod
    def collate(cls, samples: Sequence[ELAGraph]) -> ELAGraph:
        """Pack graph-major dataset samples without padded dummy nodes."""

        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
            raise TypeError("samples must be a sequence of ELAGraph objects")
        if not samples:
            raise ValueError("at least one graph sample is required")
        graphs = tuple(samples)
        if any(not isinstance(graph, ELAGraph) for graph in graphs):
            raise TypeError("every sample must be an ELAGraph")
        if any(graph.num_graphs != 1 for graph in graphs):
            raise ValueError("every collated sample must contain exactly one graph")

        nodes = tuple(graph.x for graph in graphs)
        positions = tuple(graph.pos for graph in graphs)
        _same_tensor_schema("x", nodes)
        _same_tensor_schema("pos", positions)
        width = nodes[0].shape[1]
        if any(node.shape[1] != width for node in nodes):
            raise ValueError("all sample x tensors must share one feature width")
        device = nodes[0].device
        counts = torch.tensor(
            [graph.num_nodes for graph in graphs],
            device=device,
            dtype=torch.long,
        )
        batch = torch.repeat_interleave(
            torch.arange(len(graphs), device=device),
            counts,
            output_size=int(counts.sum().item()),
        )

        edges = tuple(graph.edge_index for graph in graphs)
        edge_index = None
        if _presence("edge_index", edges):
            offsets = counts.cumsum(0) - counts
            edge_index = torch.cat(
                [
                    edge + offset
                    for edge, offset in zip(edges, offsets, strict=True)
                    if edge is not None
                ],
                dim=1,
            )

        edge_types = tuple(graph.edge_type for graph in graphs)
        edge_type = None
        if _presence("edge_type", edge_types):
            if edge_index is None:
                raise ValueError("edge_type requires edge_index in every sample")
            edge_type = torch.cat(
                [value for value in edge_types if value is not None],
                dim=0,
            )

        groups = tuple(graph.group for graph in graphs)
        group = None
        if _presence("group", groups):
            group = torch.cat([value for value in groups if value is not None])

        conditions = tuple(graph.condition for graph in graphs)
        condition = None
        if _presence("condition", conditions):
            values = tuple(value for value in conditions if value is not None)
            _same_tensor_schema("condition", values)
            widths = tuple(
                value.shape[0] if value.ndim == 1 else value.shape[1]
                for value in values
            )
            if len(set(widths)) != 1:
                raise ValueError("all sample conditions must share one width")
            can_graph = tuple(
                value.ndim == 1 or (value.ndim == 2 and value.shape[0] == 1)
                for value in values
            )
            can_node = tuple(
                value.ndim == 2 and value.shape[0] == graph.num_nodes
                for value, graph in zip(values, graphs, strict=True)
            )
            if all(can_graph):
                condition = torch.cat(
                    [value.reshape(1, -1) for value in values],
                    dim=0,
                )
            elif all(can_node):
                condition = torch.cat(values, dim=0)
            else:
                raise ValueError(
                    "conditions must all be graph-level or all be node-level"
                )

        orders = tuple(graph.order for graph in graphs)
        order = None
        if _presence("order", orders):
            values = tuple(value for value in orders if value is not None)
            coordinates = tuple(value.coordinates for value in values)
            _same_tensor_schema("order coordinates", coordinates)
            if len({value.shape[1] for value in coordinates}) != 1:
                raise ValueError("all order coordinates must share one width")
            order_groups = tuple(value.group_index for value in values)
            periods = tuple(value.periods for value in values)
            enabled = tuple(value.enabled for value in values)
            packed_group = None
            if _presence("order.group_index", order_groups):
                packed_group = torch.cat(
                    [value for value in order_groups if value is not None]
                )
            packed_periods = None
            if _presence("order.periods", periods):
                period_values = tuple(value for value in periods if value is not None)
                first = period_values[0]
                if any(
                    value.shape != first.shape
                    or value.dtype != first.dtype
                    or value.device != first.device
                    or not torch.equal(value, first)
                    for value in period_values[1:]
                ):
                    raise ValueError("all sample order.periods must be identical")
                packed_periods = first
            packed_enabled = None
            if _presence("order.enabled", enabled):
                packed_enabled = torch.cat(
                    [value for value in enabled if value is not None]
                )
            order = OrderContext(
                coordinates=torch.cat(coordinates),
                group_index=packed_group,
                periods=packed_periods,
                enabled=packed_enabled,
            )

        masks = tuple(graph.update_mask for graph in graphs)
        update_mask = None
        if _presence("update_mask", masks):
            update_mask = torch.cat([value for value in masks if value is not None])

        targets = tuple(graph.y for graph in graphs)
        y = None
        if _presence("y", targets):
            values = tuple(value for value in targets if value is not None)
            _same_tensor_schema("y", values)
            if any(value.shape != values[0].shape for value in values[1:]):
                raise ValueError("all sample y tensors must have equal shapes")
            y = torch.stack(values)

        identifiers = tuple(graph.ids for graph in graphs)
        ids = None
        if _presence("ids", identifiers):
            ids = tuple(value[0] for value in identifiers if value is not None)

        def pack_graph_field(name: str) -> torch.Tensor | None:
            raw = tuple(getattr(graph, name) for graph in graphs)
            if not _presence(name, raw):
                return None
            values = tuple(value for value in raw if value is not None)
            _same_tensor_schema(name, values)
            if any(value.shape[1:] != values[0].shape[1:] for value in values[1:]):
                raise ValueError(f"all sample {name} tensors must share trailing shape")
            return torch.cat(values, dim=0)

        deltas = tuple(graph.delta for graph in graphs)
        delta = None
        if _presence("delta", deltas):
            values = tuple(value for value in deltas if value is not None)
            _same_tensor_schema("delta", values)
            delta = torch.cat(values)

        return cls(
            x=torch.cat(nodes),
            pos=torch.cat(positions),
            edge_index=edge_index,
            batch=batch,
            edge_type=edge_type,
            group=group,
            condition=condition,
            order=order,
            update_mask=update_mask,
            y=y,
            ids=ids,
            graph_x=pack_graph_field("graph_x"),
            graph_sum=pack_graph_field("graph_sum"),
            delta=delta,
        )


__all__ = ["ELAGraph"]
