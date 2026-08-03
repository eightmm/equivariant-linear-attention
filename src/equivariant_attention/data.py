from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

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
BatchKind = Literal["flat", "padded"]


def _require_integer(name: str, value: torch.Tensor) -> None:
    if value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must use an integer dtype")


def _require_contiguous_batch(batch: torch.Tensor) -> int:
    if batch.ndim != 1:
        raise ValueError("batch must have shape (N,)")
    _require_integer("batch", batch)
    if batch.numel() == 0:
        return 0
    batch_long = batch.to(dtype=torch.long)
    if bool((batch_long < 0).any().item()):
        raise ValueError("batch values must be nonnegative")
    num_graphs = int(batch_long.max().item()) + 1
    observed = torch.unique(batch_long, sorted=True)
    expected = torch.arange(num_graphs, device=batch.device)
    if not torch.equal(observed, expected):
        raise ValueError("batch graph IDs must be contiguous from zero")
    return num_graphs


@dataclass(frozen=True)
class BatchLayout:
    """How a user-facing node batch maps to the packed internal node axis."""

    kind: BatchKind
    batch: torch.Tensor
    batch_size: int
    max_nodes: int | None = None
    node_mask: torch.Tensor | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.batch.numel())

    def flatten_node_tensor(
        self,
        value: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        if self.kind == "flat":
            if value.shape[0] != self.num_nodes:
                raise ValueError(
                    f"{name} leading dimension must equal the packed node count"
                )
            return value
        if self.node_mask is None or self.max_nodes is None:
            raise RuntimeError("padded batch layout is incomplete")
        if value.shape[:2] != self.node_mask.shape:
            raise ValueError(
                f"{name} must start with shape "
                f"({self.batch_size}, {self.max_nodes})"
            )
        return value[self.node_mask]

    def restore_node_tensor(
        self,
        value: torch.Tensor,
        *,
        template: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.kind == "flat":
            return value
        if self.node_mask is None or self.max_nodes is None:
            raise RuntimeError("padded batch layout is incomplete")
        expected = (self.batch_size, self.max_nodes, *value.shape[1:])
        if template is None:
            output = value.new_zeros(expected)
        else:
            if template.shape != expected:
                raise ValueError("restore template shape does not match output")
            output = template.clone()
        output[self.node_mask] = value
        return output


@dataclass(frozen=True)
class PackedNodeInput:
    """Packed node tensors plus padded-to-packed lookup metadata."""

    node_irreps: torch.Tensor
    positions: torch.Tensor
    layout: BatchLayout
    padded_to_flat: torch.Tensor | None = None

    @property
    def batch(self) -> torch.Tensor:
        return self.layout.batch


def pack_node_input(
    node_irreps: torch.Tensor,
    positions: torch.Tensor,
    *,
    batch: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> PackedNodeInput:
    """Normalize flat or padded graph tensors to one packed node axis.

    Supported layouts are:

    - flat: ``node_irreps[N,D]``, ``positions[N,3]``, optional ``batch[N]``;
    - padded: ``node_irreps[B,M,D]``, ``positions[B,M,3]``, optional
      ``mask[B,M]``.
    """

    if not isinstance(node_irreps, torch.Tensor):
        raise TypeError("node_irreps must be a tensor")
    if not isinstance(positions, torch.Tensor):
        raise TypeError("positions must be a tensor")
    if node_irreps.device != positions.device:
        raise ValueError("node_irreps and positions must share one device")
    if node_irreps.ndim == 2:
        if positions.shape != (node_irreps.shape[0], 3):
            raise ValueError("flat positions must have shape (N, 3)")
        if mask is not None:
            if mask.shape != (node_irreps.shape[0],) or mask.dtype != torch.bool:
                raise ValueError("flat mask must be boolean with shape (N,)")
            if not bool(mask.all().item()):
                raise ValueError(
                    "flat inputs are already packed; use padded [B,M,D] input "
                    "to omit nodes with a mask"
                )
        if batch is None:
            packed_batch = torch.zeros(
                node_irreps.shape[0],
                device=node_irreps.device,
                dtype=torch.long,
            )
            batch_size = 1 if node_irreps.shape[0] else 0
        else:
            if batch.device != node_irreps.device:
                raise ValueError("batch and node tensors must share one device")
            batch_size = _require_contiguous_batch(batch)
            packed_batch = batch.to(dtype=torch.long)
        return PackedNodeInput(
            node_irreps=node_irreps,
            positions=positions,
            layout=BatchLayout(
                kind="flat",
                batch=packed_batch,
                batch_size=batch_size,
            ),
        )

    if node_irreps.ndim != 3:
        raise ValueError("node_irreps must have shape (N,D) or (B,M,D)")
    if positions.ndim != 3 or positions.shape[:2] != node_irreps.shape[:2]:
        raise ValueError("padded positions must have shape (B, M, 3)")
    if positions.shape[-1] != 3:
        raise ValueError("positions must end with Cartesian dimension three")
    if batch is not None:
        raise ValueError("batch is inferred for padded inputs; do not supply it")

    batch_size, max_nodes = node_irreps.shape[:2]
    if mask is None:
        node_mask = torch.ones(
            (batch_size, max_nodes),
            device=node_irreps.device,
            dtype=torch.bool,
        )
    else:
        if mask.device != node_irreps.device:
            raise ValueError("mask and node tensors must share one device")
        if mask.shape != (batch_size, max_nodes) or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape (B, M)")
        node_mask = mask
    if bool((node_mask.sum(dim=1) == 0).any().item()):
        raise ValueError("every padded graph must contain at least one valid node")

    graph_id = torch.arange(
        batch_size,
        device=node_irreps.device,
        dtype=torch.long,
    ).unsqueeze(1).expand(batch_size, max_nodes)
    packed_batch = graph_id[node_mask]
    lookup = torch.full(
        (batch_size, max_nodes),
        -1,
        device=node_irreps.device,
        dtype=torch.long,
    )
    lookup[node_mask] = torch.arange(
        int(node_mask.sum().item()),
        device=node_irreps.device,
        dtype=torch.long,
    )
    return PackedNodeInput(
        node_irreps=node_irreps[node_mask],
        positions=positions[node_mask],
        layout=BatchLayout(
            kind="padded",
            batch=packed_batch,
            batch_size=batch_size,
            max_nodes=max_nodes,
            node_mask=node_mask,
        ),
        padded_to_flat=lookup,
    )


def _normalize_edge_tensor(
    edge_index: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    if edge_index.ndim == 2:
        if batch_size != 1 or edge_index.shape[0] != 2:
            raise ValueError(
                "padded edge_index must have shape (B,2,E); shape (2,E) is "
                "accepted only for B=1"
            )
        return edge_index.unsqueeze(0)
    if edge_index.ndim != 3:
        raise ValueError("padded edge_index must have shape (B,2,E) or (B,E,2)")
    if edge_index.shape[0] != batch_size:
        raise ValueError("padded edge_index batch dimension is incorrect")
    if edge_index.shape[1] == 2:
        return edge_index
    if edge_index.shape[2] == 2:
        return edge_index.transpose(1, 2)
    raise ValueError("padded edge_index must contain a dimension of size two")


def pack_edges(
    packed: PackedNodeInput,
    *,
    edge_index: torch.Tensor | Sequence[torch.Tensor] | None = None,
    edge_mask: torch.Tensor | None = None,
    adjacency: torch.Tensor | None = None,
    edge_relation_id: torch.Tensor | Sequence[torch.Tensor] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Map user edge formats into packed receiver/sender COO.

    Padded inputs accept ``edge_index[B,2,E]``, a sequence of ``[2,E_b]``
    tensors, or boolean ``adjacency[B,M,M]``. Negative endpoints are treated as
    padding. ``edge_index[0]`` is the receiver and ``edge_index[1]`` is the
    sender.
    """

    if edge_index is not None and adjacency is not None:
        raise ValueError("supply edge_index or adjacency, not both")
    if packed.layout.kind == "flat":
        if adjacency is not None:
            if adjacency.ndim != 2 or adjacency.shape != (
                packed.layout.num_nodes,
                packed.layout.num_nodes,
            ):
                raise ValueError("flat adjacency must have shape (N,N)")
            if adjacency.dtype != torch.bool:
                raise TypeError("adjacency must use torch.bool")
            receiver, sender = adjacency.nonzero(as_tuple=True)
            return torch.stack([receiver, sender]), None
        if edge_index is None:
            if edge_relation_id is not None:
                raise ValueError("edge_relation_id requires edge_index")
            return None, None
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError("flat edge_index must be a tensor")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("flat edge_index must have shape (2,E)")
        _require_integer("edge_index", edge_index)
        if edge_index.device != packed.node_irreps.device:
            raise ValueError("edge_index and nodes must share one device")
        edge_long = edge_index.to(dtype=torch.long)
        if edge_long.numel():
            if bool((edge_long < 0).any().item()):
                raise ValueError("flat edge_index must be nonnegative")
            if int(edge_long.max().item()) >= packed.layout.num_nodes:
                raise ValueError("flat edge_index contains an out-of-range node")
        relation = None
        if edge_relation_id is not None:
            if not isinstance(edge_relation_id, torch.Tensor):
                raise TypeError("flat edge_relation_id must be a tensor")
            if edge_relation_id.shape != (edge_long.shape[1],):
                raise ValueError("edge_relation_id must have shape (E,)")
            if edge_relation_id.device != edge_long.device:
                raise ValueError("edge_relation_id and edges must share one device")
            _require_integer("edge_relation_id", edge_relation_id)
            relation = edge_relation_id.to(dtype=torch.long)
        return edge_long, relation

    lookup = packed.padded_to_flat
    node_mask = packed.layout.node_mask
    if lookup is None or node_mask is None or packed.layout.max_nodes is None:
        raise RuntimeError("padded input metadata is incomplete")
    batch_size = packed.layout.batch_size
    max_nodes = packed.layout.max_nodes

    if adjacency is not None:
        if adjacency.device != packed.node_irreps.device:
            raise ValueError("adjacency and nodes must share one device")
        if adjacency.dtype != torch.bool:
            raise TypeError("adjacency must use torch.bool")
        if adjacency.ndim == 2 and batch_size == 1:
            adjacency = adjacency.unsqueeze(0)
        if adjacency.shape != (batch_size, max_nodes, max_nodes):
            raise ValueError("padded adjacency must have shape (B,M,M)")
        valid = adjacency & node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        graph_id, receiver, sender = valid.nonzero(as_tuple=True)
        return torch.stack(
            [lookup[graph_id, receiver], lookup[graph_id, sender]]
        ), None

    if edge_index is None:
        if edge_relation_id is not None:
            raise ValueError("edge_relation_id requires edge_index")
        return None, None

    if isinstance(edge_index, torch.Tensor):
        if edge_index.device != packed.node_irreps.device:
            raise ValueError("edge_index and nodes must share one device")
        _require_integer("edge_index", edge_index)
        edges = _normalize_edge_tensor(edge_index, batch_size=batch_size).to(
            dtype=torch.long
        )
        max_edges = edges.shape[2]
        if edge_mask is None:
            valid_edge = (edges >= 0).all(dim=1)
        else:
            if edge_mask.device != edges.device:
                raise ValueError("edge_mask and edges must share one device")
            if edge_mask.shape != (batch_size, max_edges):
                raise ValueError("edge_mask must have shape (B,E)")
            if edge_mask.dtype != torch.bool:
                raise TypeError("edge_mask must use torch.bool")
            valid_edge = edge_mask & (edges >= 0).all(dim=1)
        graph_id = torch.arange(
            batch_size,
            device=edges.device,
            dtype=torch.long,
        ).unsqueeze(1).expand(batch_size, max_edges)
        selected_graph = graph_id[valid_edge]
        receiver = edges[:, 0][valid_edge]
        sender = edges[:, 1][valid_edge]
        if receiver.numel():
            if bool((receiver >= max_nodes).any().item()) or bool(
                (sender >= max_nodes).any().item()
            ):
                raise ValueError("padded edge_index contains an out-of-range node")
            if bool((~node_mask[selected_graph, receiver]).any().item()) or bool(
                (~node_mask[selected_graph, sender]).any().item()
            ):
                raise ValueError("padded edges may not reference masked nodes")
        packed_edges = torch.stack(
            [
                lookup[selected_graph, receiver],
                lookup[selected_graph, sender],
            ]
        )
        relation = None
        if edge_relation_id is not None:
            if not isinstance(edge_relation_id, torch.Tensor):
                raise TypeError(
                    "tensor edge_index requires tensor edge_relation_id"
                )
            if edge_relation_id.shape != (batch_size, max_edges):
                raise ValueError("edge_relation_id must have shape (B,E)")
            if edge_relation_id.device != edges.device:
                raise ValueError("edge_relation_id and edges must share one device")
            _require_integer("edge_relation_id", edge_relation_id)
            relation = edge_relation_id[valid_edge].to(dtype=torch.long)
        return packed_edges, relation

    if not isinstance(edge_index, Sequence) or len(edge_index) != batch_size:
        raise TypeError("edge_index sequence length must equal the batch size")
    if edge_mask is not None:
        raise ValueError("edge_mask is only used with padded tensor edge_index")
    relation_sequence = None
    if edge_relation_id is not None:
        if not isinstance(edge_relation_id, Sequence) or len(edge_relation_id) != batch_size:
            raise TypeError(
                "edge_relation_id sequence length must equal the batch size"
            )
        relation_sequence = edge_relation_id

    parts: list[torch.Tensor] = []
    relation_parts: list[torch.Tensor] = []
    for graph_id, item in enumerate(edge_index):
        if not isinstance(item, torch.Tensor):
            raise TypeError("every edge_index item must be a tensor")
        if item.device != packed.node_irreps.device:
            raise ValueError("edge tensors and nodes must share one device")
        if item.ndim != 2 or item.shape[0] != 2:
            raise ValueError("each edge_index item must have shape (2,E_b)")
        _require_integer("edge_index", item)
        item_long = item.to(dtype=torch.long)
        if item_long.numel():
            if bool((item_long < 0).any().item()):
                raise ValueError("ragged edge_index must be nonnegative")
            if int(item_long.max().item()) >= max_nodes:
                raise ValueError("ragged edge_index contains an out-of-range node")
            if bool((~node_mask[graph_id, item_long]).any().item()):
                raise ValueError("ragged edges may not reference masked nodes")
        parts.append(
            torch.stack(
                [
                    lookup[graph_id, item_long[0]],
                    lookup[graph_id, item_long[1]],
                ]
            )
        )
        if relation_sequence is not None:
            relation_item = relation_sequence[graph_id]
            if not isinstance(relation_item, torch.Tensor):
                raise TypeError("every edge_relation_id item must be a tensor")
            if relation_item.shape != (item_long.shape[1],):
                raise ValueError("ragged relation IDs must match edge counts")
            if relation_item.device != item.device:
                raise ValueError("relation IDs and edges must share one device")
            _require_integer("edge_relation_id", relation_item)
            relation_parts.append(relation_item.to(dtype=torch.long))
    packed_edges = (
        torch.cat(parts, dim=1)
        if parts
        else torch.empty(
            (2, 0),
            device=packed.node_irreps.device,
            dtype=torch.long,
        )
    )
    relation = torch.cat(relation_parts) if relation_parts else None
    return packed_edges, relation


def _sample_value(
    sample: Mapping[str, Any],
    primary: str,
    aliases: tuple[str, ...],
) -> Any:
    if primary in sample:
        return sample[primary]
    present = [alias for alias in aliases if alias in sample]
    if len(present) == 1:
        return sample[present[0]]
    if present:
        raise ValueError(f"sample contains multiple aliases for {primary}")
    raise KeyError(f"sample is missing {primary}")


def collate_graphs(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Dependency-free ``DataLoader.collate_fn`` for ragged graph dictionaries.

    Required per-sample keys are ``node_irreps`` (aliases ``x`` or
    ``node_features``) and ``pos`` (alias ``positions``). Optional edge tensors
    are offset automatically. If every sample omits edges, the returned mapping
    also omits ``edge_index`` and :class:`ELA` builds radius candidates.
    """

    if not samples:
        raise ValueError("at least one graph sample is required")
    nodes: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    edges: list[torch.Tensor | None] = []
    relations: list[torch.Tensor | None] = []
    targets: list[torch.Tensor | None] = []
    sample_ids: list[Any] = []
    conditions: list[torch.Tensor | None] = []
    orders: list[torch.Tensor | None] = []
    order_groups: list[torch.Tensor | None] = []
    order_masks: list[torch.Tensor | None] = []

    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise TypeError("every graph sample must be a mapping")
        node = _sample_value(sample, "node_irreps", ("x", "node_features"))
        pos = _sample_value(sample, "pos", ("positions",))
        if not isinstance(node, torch.Tensor) or node.ndim != 2:
            raise ValueError("sample node_irreps must have shape (N,D)")
        if not isinstance(pos, torch.Tensor) or pos.shape != (node.shape[0], 3):
            raise ValueError("sample pos must have shape (N,3)")
        if node.device != pos.device:
            raise ValueError("sample nodes and positions must share one device")
        nodes.append(node)
        positions.append(pos)
        edges.append(sample.get("edge_index"))
        relations.append(sample.get("edge_relation_id"))
        targets.append(sample.get("target"))
        sample_ids.append(sample.get("sample_id", index))
        conditions.append(sample.get("condition"))
        orders.append(sample.get("order"))
        order_groups.append(sample.get("order_group"))
        order_masks.append(sample.get("order_mask"))

    device = nodes[0].device
    if any(node.device != device or pos.device != device for node, pos in zip(nodes, positions, strict=True)):
        raise ValueError("all samples must already share one device")
    counts = [node.shape[0] for node in nodes]
    batch = torch.repeat_interleave(
        torch.arange(len(samples), device=device, dtype=torch.long),
        torch.tensor(counts, device=device, dtype=torch.long),
    )
    output: dict[str, Any] = {
        "node_irreps": torch.cat(nodes, dim=0),
        "pos": torch.cat(positions, dim=0),
        "batch": batch,
        "sample_ids": tuple(sample_ids),
    }

    edge_presence = [edge is not None for edge in edges]
    if any(edge_presence) and not all(edge_presence):
        raise ValueError("all samples must provide edge_index or all must omit it")
    relation_presence = [relation is not None for relation in relations]
    if any(relation_presence) and not all(relation_presence):
        raise ValueError(
            "all samples must provide edge_relation_id or all must omit it"
        )
    if all(edge_presence):
        edge_parts: list[torch.Tensor] = []
        relation_parts: list[torch.Tensor] = []
        offset = 0
        for node, edge, relation in zip(nodes, edges, relations, strict=True):
            if not isinstance(edge, torch.Tensor) or edge.ndim != 2 or edge.shape[0] != 2:
                raise ValueError("sample edge_index must have shape (2,E)")
            if edge.device != device:
                raise ValueError("all sample edges must share the node device")
            _require_integer("edge_index", edge)
            edge_long = edge.to(dtype=torch.long)
            if edge_long.numel():
                if bool((edge_long < 0).any().item()) or int(edge_long.max().item()) >= node.shape[0]:
                    raise ValueError("sample edge_index contains an invalid node")
            edge_parts.append(edge_long + offset)
            if relation is not None:
                if not isinstance(relation, torch.Tensor) or relation.shape != (edge_long.shape[1],):
                    raise ValueError("sample relation IDs must have shape (E,)")
                _require_integer("edge_relation_id", relation)
                relation_parts.append(relation.to(device=device, dtype=torch.long))
            offset += node.shape[0]
        output["edge_index"] = torch.cat(edge_parts, dim=1)
        if relation_parts:
            output["edge_relation_id"] = torch.cat(relation_parts)
    elif any(relation_presence):
        raise ValueError("edge_relation_id requires edge_index")

    if all(target is not None for target in targets):
        if not all(isinstance(target, torch.Tensor) for target in targets):
            raise TypeError("targets must be tensors")
        output["target"] = torch.stack(
            [target.reshape(-1) for target in targets if isinstance(target, torch.Tensor)]
        )
    elif any(target is not None for target in targets):
        raise ValueError("all samples must provide target or all must omit it")

    if all(condition is not None for condition in conditions):
        if not all(isinstance(condition, torch.Tensor) for condition in conditions):
            raise TypeError("conditions must be tensors")
        condition_tensors = [condition for condition in conditions if isinstance(condition, torch.Tensor)]
        if all(condition.ndim == 1 for condition in condition_tensors):
            output["condition"] = torch.stack(condition_tensors)
        elif all(
            condition.ndim == 2 and condition.shape[0] == count
            for condition, count in zip(condition_tensors, counts, strict=True)
        ):
            output["condition"] = torch.cat(condition_tensors)
        else:
            raise ValueError(
                "conditions must all be graph-level [C] or node-level [N_i,C]"
            )
    elif any(condition is not None for condition in conditions):
        raise ValueError("all samples must provide condition or all must omit it")

    for key, values in (
        ("order", orders),
        ("order_group", order_groups),
        ("order_mask", order_masks),
    ):
        if all(value is not None for value in values):
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise TypeError(f"{key} values must be tensors")
            tensors = [value for value in values if isinstance(value, torch.Tensor)]
            if any(value.shape[0] != count for value, count in zip(tensors, counts, strict=True)):
                raise ValueError(f"{key} must be node-level in every sample")
            output[key] = torch.cat(tensors)
        elif any(value is not None for value in values):
            raise ValueError(f"all samples must provide {key} or all must omit it")
    return output


__all__ = [
    "BatchLayout",
    "PackedNodeInput",
    "collate_graphs",
    "pack_edges",
    "pack_node_input",
]
