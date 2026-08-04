from __future__ import annotations

from itertools import product
from math import isfinite

import torch

from .neighbors import (
    PackedNeighborGraph,
    _build_receiver_csr_from_major,
    build_receiver_csr,
)

_INTEGER_DTYPES = frozenset(
    {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
)
_CELL_OFFSETS = tuple(product((-1, 0, 1), repeat=3))
_INT64_SAFE = (1 << 62) - 1
_BATCHED_DENSE_PAIR_BUDGET = 1 << 20


def _validate_positive_int(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or None")


def _ptr_from_batch(batch: torch.Tensor, nodes: int) -> torch.Tensor:
    if batch.shape != (nodes,):
        raise ValueError("batch must have shape (N,)")
    if batch.dtype not in _INTEGER_DTYPES:
        raise TypeError("batch must use an integer dtype")
    value = batch.to(dtype=torch.long)
    if nodes == 0:
        return value.new_zeros((1,))
    if bool((value < 0).any().item()):
        raise ValueError("batch values must be nonnegative")
    if bool((value[1:] < value[:-1]).any().item()):
        raise ValueError("batch must be graph-major and nondecreasing")
    graphs = int(value[-1].item()) + 1
    if not torch.equal(
        torch.unique_consecutive(value),
        torch.arange(graphs, device=value.device),
    ):
        raise ValueError("batch graph IDs must be contiguous from zero")
    counts = torch.bincount(value, minlength=graphs)
    return torch.cat([counts.new_zeros((1,)), counts.cumsum(0)])


def _normalize_ptr(
    positions: torch.Tensor,
    *,
    ptr: torch.Tensor | None,
    batch: torch.Tensor | None,
) -> torch.Tensor:
    if ptr is not None and batch is not None:
        raise ValueError("supply ptr or batch, not both")
    if ptr is None:
        if batch is None:
            return torch.tensor(
                [0, positions.shape[0]],
                device=positions.device,
                dtype=torch.long,
            )
        return _ptr_from_batch(batch, positions.shape[0])
    if ptr.ndim != 1 or ptr.dtype not in {torch.int32, torch.int64}:
        raise TypeError("ptr must be a one-dimensional int32/int64 tensor")
    if ptr.device != positions.device:
        raise ValueError("ptr and positions must share one device")
    value = ptr.to(dtype=torch.long)
    if value.numel() == 0 or int(value[0].item()) != 0:
        raise ValueError("ptr must start at zero")
    if int(value[-1].item()) != positions.shape[0]:
        raise ValueError("ptr must end at the node count")
    if bool((value[1:] <= value[:-1]).any().item()) and value.numel() > 1:
        raise ValueError("ptr must be strictly increasing")
    return value


def _limit_neighbors(
    receiver: torch.Tensor,
    sender: torch.Tensor,
    distance_squared: torch.Tensor,
    *,
    nodes: int,
    max_neighbors: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the nearest receiver neighbors without breaking exact ties.

    A strict index-based top-k is not permutation equivariant when several
    candidates have the same distance.  ``max_neighbors`` therefore keeps the
    k-th distance shell in full.  The limit is exact for generic coordinates
    and may be exceeded only by candidates tied at its boundary.
    """

    if max_neighbors is None or receiver.numel() == 0:
        return receiver, sender

    # Stable least-to-most-significant sorting gives receiver, distance,
    # sender order.  Sender only makes the returned COO deterministic; it does
    # not decide membership because the complete boundary shell is retained.
    order = torch.argsort(sender, stable=True)
    order = order[
        torch.argsort(distance_squared[order], stable=True)
    ]
    order = order[torch.argsort(receiver[order], stable=True)]
    receiver = receiver[order]
    sender = sender[order]
    distance_squared = distance_squared[order]
    counts = torch.bincount(receiver, minlength=nodes)
    starts = counts.cumsum(0) - counts
    boundary_rank = counts.clamp_max(max_neighbors) - 1
    boundary_index = starts + boundary_rank.clamp_min(0)
    threshold = distance_squared.new_zeros((nodes,))
    nonempty = counts > 0
    threshold[nonempty] = distance_squared[boundary_index[nonempty]]
    keep = distance_squared <= threshold[receiver]
    return receiver[keep], sender[keep]


def _receiver_major_edges(
    receiver: torch.Tensor,
    sender: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stably group a discovered cell-list stream by receiver once."""

    if receiver.numel() < 2:
        return receiver, sender
    order = torch.argsort(receiver, stable=True)
    return receiver[order], sender[order]


def _dense_edges(
    positions: torch.Tensor,
    *,
    cutoff_squared: float,
    include_self: bool,
    max_neighbors: int | None,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    nodes = positions.shape[0]
    receiver_parts: list[torch.Tensor] = []
    sender_parts: list[torch.Tensor] = []
    distance_parts: list[torch.Tensor] = []
    for start in range(0, nodes, chunk_size):
        stop = min(start + chunk_size, nodes)
        distance_squared = (
            positions[start:stop, None, :] - positions[None, :, :]
        ).square().sum(dim=-1)
        valid = distance_squared < cutoff_squared
        if not include_self:
            local = torch.arange(stop - start, device=positions.device)
            valid[local, torch.arange(start, stop, device=positions.device)] = False
        receiver_local, sender = valid.nonzero(as_tuple=True)
        receiver = receiver_local + start
        receiver_parts.append(receiver)
        sender_parts.append(sender)
        distance_parts.append(distance_squared[receiver_local, sender])
    if not receiver_parts:
        empty = torch.empty(0, device=positions.device, dtype=torch.long)
        return empty, empty
    receiver = torch.cat(receiver_parts)
    sender = torch.cat(sender_parts)
    distance_squared = torch.cat(distance_parts)
    return _limit_neighbors(
        receiver,
        sender,
        distance_squared,
        nodes=nodes,
        max_neighbors=max_neighbors,
    )


def _batched_dense_edges(
    positions: torch.Tensor,
    graph_ptr: torch.Tensor,
    *,
    padded_nodes: int,
    cutoff_squared: float,
    include_self: bool,
    max_neighbors: int | None,
    receiver_major: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorize exact dense discovery across many small grouped graphs."""

    counts = graph_ptr[1:] - graph_ptr[:-1]
    starts = graph_ptr[:-1]
    local = torch.arange(padded_nodes, device=positions.device)
    present = local.unsqueeze(0) < counts.unsqueeze(1)
    global_index = starts.unsqueeze(1) + local.unsqueeze(0)
    safe_index = torch.where(present, global_index, torch.zeros_like(global_index))
    work = positions.detach().to(
        dtype=torch.float64 if positions.dtype == torch.float64 else torch.float32
    )
    padded = work[safe_index]
    distance_squared = (
        padded[:, :, None, :] - padded[:, None, :, :]
    ).square().sum(dim=-1)
    valid = (
        present[:, :, None]
        & present[:, None, :]
        & (distance_squared < cutoff_squared)
    )
    if not include_self:
        diagonal = torch.eye(
            padded_nodes,
            device=positions.device,
            dtype=torch.bool,
        ).unsqueeze(0)
        valid = valid & ~diagonal
    graph, receiver_local, sender_local = valid.nonzero(as_tuple=True)
    receiver = starts[graph] + receiver_local
    sender = starts[graph] + sender_local
    if max_neighbors is not None:
        receiver, sender = _limit_neighbors(
            receiver,
            sender,
            distance_squared[graph, receiver_local, sender_local],
            nodes=positions.shape[0],
            max_neighbors=max_neighbors,
        )
    if receiver_major and max_neighbors is None:
        receiver, sender = _receiver_major_edges(receiver, sender)
    return receiver, sender


def _encode_cells(
    shifted: torch.Tensor,
    span_y: int,
    span_z: int,
) -> torch.Tensor:
    return (
        (shifted[:, 0] * span_y + shifted[:, 1]) * span_z
        + shifted[:, 2]
    )


def _cell_edges(
    positions: torch.Tensor,
    *,
    cutoff: float,
    cutoff_squared: float,
    include_self: bool,
    max_neighbors: int | None,
    dense_chunk_size: int,
    receiver_major: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    nodes = positions.shape[0]
    if nodes == 0:
        empty = torch.empty(0, device=positions.device, dtype=torch.long)
        return empty, empty
    work = positions.detach().to(
        dtype=torch.float64 if positions.dtype == torch.float64 else torch.float32
    )
    cell = torch.floor(work / cutoff).to(dtype=torch.long)
    minimum = cell.amin(dim=0)
    maximum = cell.amax(dim=0)
    spans = (maximum - minimum + 1).to(device="cpu")
    span_x, span_y, span_z = (int(value) for value in spans.tolist())
    if span_x * span_y * span_z > _INT64_SAFE:
        return _dense_edges(
            work,
            cutoff_squared=cutoff_squared,
            include_self=include_self,
            max_neighbors=max_neighbors,
            chunk_size=dense_chunk_size,
        )

    shifted = cell - minimum
    key = _encode_cells(shifted, span_y, span_z)
    node_order = torch.argsort(key, stable=True)
    sorted_key = key[node_order]
    unique_key, counts = torch.unique_consecutive(
        sorted_key,
        return_counts=True,
    )
    cell_ptr = torch.cat([counts.new_zeros((1,)), counts.cumsum(0)])
    unique_cell = cell[node_order[cell_ptr[:-1]]]
    cell_ids = torch.arange(unique_key.numel(), device=positions.device)

    receiver_parts: list[torch.Tensor] = []
    sender_parts: list[torch.Tensor] = []
    distance_parts: list[torch.Tensor] = []
    for offset_tuple in _CELL_OFFSETS:
        offset = unique_cell.new_tensor(offset_tuple)
        neighbor_cell = unique_cell + offset
        neighbor_shifted = neighbor_cell - minimum
        in_bounds = (
            (neighbor_shifted[:, 0] >= 0)
            & (neighbor_shifted[:, 0] < span_x)
            & (neighbor_shifted[:, 1] >= 0)
            & (neighbor_shifted[:, 1] < span_y)
            & (neighbor_shifted[:, 2] >= 0)
            & (neighbor_shifted[:, 2] < span_z)
        )
        if not bool(in_bounds.any().item()):
            continue
        source_cell = cell_ids[in_bounds]
        candidate_key = _encode_cells(
            neighbor_shifted[in_bounds],
            span_y,
            span_z,
        )
        destination_cell = torch.searchsorted(unique_key, candidate_key)
        found = destination_cell < unique_key.numel()
        if bool(found.any().item()):
            safe_destination = destination_cell.clamp_max(
                max(0, unique_key.numel() - 1)
            )
            found = found & (unique_key[safe_destination] == candidate_key)
        if not bool(found.any().item()):
            continue
        source_cell = source_cell[found]
        destination_cell = destination_cell[found]
        source_count = counts[source_cell]
        destination_count = counts[destination_cell]
        pair_count = source_count * destination_count
        total_pairs = int(pair_count.sum().item())
        if total_pairs == 0:
            continue
        group = torch.repeat_interleave(
            torch.arange(source_cell.numel(), device=positions.device),
            pair_count,
            output_size=total_pairs,
        )
        group_start = torch.repeat_interleave(
            pair_count.cumsum(0) - pair_count,
            pair_count,
            output_size=total_pairs,
        )
        within = torch.arange(total_pairs, device=positions.device) - group_start
        destination_width = destination_count[group]
        source_local = within // destination_width
        destination_local = within - source_local * destination_width
        receiver = node_order[cell_ptr[source_cell[group]] + source_local]
        sender = node_order[
            cell_ptr[destination_cell[group]] + destination_local
        ]
        distance_squared = (work[sender] - work[receiver]).square().sum(dim=-1)
        valid = distance_squared < cutoff_squared
        if not include_self:
            valid = valid & (receiver != sender)
        if bool(valid.any().item()):
            receiver_parts.append(receiver[valid])
            sender_parts.append(sender[valid])
            distance_parts.append(distance_squared[valid])

    if not receiver_parts:
        empty = torch.empty(0, device=positions.device, dtype=torch.long)
        return empty, empty
    receiver = torch.cat(receiver_parts)
    sender = torch.cat(sender_parts)
    distance_squared = torch.cat(distance_parts)
    receiver, sender = _limit_neighbors(
        receiver,
        sender,
        distance_squared,
        nodes=nodes,
        max_neighbors=max_neighbors,
    )
    if receiver_major and max_neighbors is None:
        receiver, sender = _receiver_major_edges(receiver, sender)
    return receiver, sender


def _batched_cell_edges(
    positions: torch.Tensor,
    batch: torch.Tensor,
    *,
    num_graphs: int,
    cutoff: float,
    cutoff_squared: float,
    include_self: bool,
    max_neighbors: int | None,
    receiver_major: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Exact multi-graph cell list without a Python loop over graphs."""

    nodes = positions.shape[0]
    if nodes == 0:
        empty = torch.empty(0, device=positions.device, dtype=torch.long)
        return empty, empty
    work = positions.detach().to(
        dtype=torch.float64 if positions.dtype == torch.float64 else torch.float32
    )
    cell = torch.floor(work / cutoff).to(dtype=torch.long)
    minimum = torch.full(
        (num_graphs, 3),
        torch.iinfo(torch.long).max,
        device=positions.device,
        dtype=torch.long,
    )
    maximum = torch.full(
        (num_graphs, 3),
        torch.iinfo(torch.long).min,
        device=positions.device,
        dtype=torch.long,
    )
    expanded_batch = batch[:, None].expand(-1, 3)
    minimum.scatter_reduce_(
        0, expanded_batch, cell, reduce="amin", include_self=True
    )
    maximum.scatter_reduce_(
        0, expanded_batch, cell, reduce="amax", include_self=True
    )
    spans_by_graph = maximum - minimum + 1
    spans = spans_by_graph.amax(dim=0).to(device="cpu")
    span_x, span_y, span_z = (int(value) for value in spans.tolist())
    if num_graphs * span_x * span_y * span_z > _INT64_SAFE:
        return None

    shifted = cell - minimum[batch]
    key = (
        ((batch * span_x + shifted[:, 0]) * span_y + shifted[:, 1])
        * span_z
        + shifted[:, 2]
    )
    node_order = torch.argsort(key, stable=True)
    sorted_key = key[node_order]
    unique_key, counts = torch.unique_consecutive(sorted_key, return_counts=True)
    cell_ptr = torch.cat([counts.new_zeros((1,)), counts.cumsum(0)])
    representative = node_order[cell_ptr[:-1]]
    unique_graph = batch[representative]
    unique_shifted = shifted[representative]
    cell_ids = torch.arange(unique_key.numel(), device=positions.device)

    receiver_parts: list[torch.Tensor] = []
    sender_parts: list[torch.Tensor] = []
    distance_parts: list[torch.Tensor] = []
    for offset_tuple in _CELL_OFFSETS:
        offset = unique_shifted.new_tensor(offset_tuple)
        neighbor_shifted = unique_shifted + offset
        graph_spans = spans_by_graph[unique_graph]
        in_bounds = (
            (neighbor_shifted >= 0) & (neighbor_shifted < graph_spans)
        ).all(dim=-1)
        if not bool(in_bounds.any().item()):
            continue
        source_cell = cell_ids[in_bounds]
        candidate_graph = unique_graph[in_bounds]
        candidate_shifted = neighbor_shifted[in_bounds]
        candidate_key = (
            (
                (candidate_graph * span_x + candidate_shifted[:, 0])
                * span_y
                + candidate_shifted[:, 1]
            )
            * span_z
            + candidate_shifted[:, 2]
        )
        destination_cell = torch.searchsorted(unique_key, candidate_key)
        found = destination_cell < unique_key.numel()
        if bool(found.any().item()):
            safe = destination_cell.clamp_max(max(0, unique_key.numel() - 1))
            found = found & (unique_key[safe] == candidate_key)
        if not bool(found.any().item()):
            continue
        source_cell = source_cell[found]
        destination_cell = destination_cell[found]
        source_count = counts[source_cell]
        destination_count = counts[destination_cell]
        pair_count = source_count * destination_count
        total_pairs = int(pair_count.sum().item())
        if total_pairs == 0:
            continue
        group = torch.repeat_interleave(
            torch.arange(source_cell.numel(), device=positions.device),
            pair_count,
            output_size=total_pairs,
        )
        group_start = torch.repeat_interleave(
            pair_count.cumsum(0) - pair_count,
            pair_count,
            output_size=total_pairs,
        )
        within = torch.arange(total_pairs, device=positions.device) - group_start
        destination_width = destination_count[group]
        source_local = within // destination_width
        destination_local = within - source_local * destination_width
        receiver = node_order[cell_ptr[source_cell[group]] + source_local]
        sender = node_order[
            cell_ptr[destination_cell[group]] + destination_local
        ]
        distance_squared = (work[sender] - work[receiver]).square().sum(dim=-1)
        valid = distance_squared < cutoff_squared
        if not include_self:
            valid = valid & (receiver != sender)
        if bool(valid.any().item()):
            receiver_parts.append(receiver[valid])
            sender_parts.append(sender[valid])
            distance_parts.append(distance_squared[valid])

    if not receiver_parts:
        empty = torch.empty(0, device=positions.device, dtype=torch.long)
        return empty, empty
    receiver = torch.cat(receiver_parts)
    sender = torch.cat(sender_parts)
    distance_squared = torch.cat(distance_parts)
    receiver, sender = _limit_neighbors(
        receiver,
        sender,
        distance_squared,
        nodes=nodes,
        max_neighbors=max_neighbors,
    )
    if receiver_major and max_neighbors is None:
        receiver, sender = _receiver_major_edges(receiver, sender)
    return receiver, sender


def _radius_edges(
    positions: torch.Tensor,
    *,
    cutoff: float,
    ptr: torch.Tensor | None = None,
    batch: torch.Tensor | None = None,
    _trusted_graph_counts: torch.Tensor | None = None,
    max_neighbors: int | None = None,
    include_self: bool = False,
    dense_threshold: int = 256,
    chunk_size: int = 1024,
    receiver_major: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build exact radius candidates without PyG.

    Small graphs use a chunked dense reference. Larger graphs use a cell list
    with exact distance filtering. Both paths return directed receiver/sender
    COO and never connect nodes from different graphs. ``max_neighbors`` keeps
    the nearest distance shells; a tie at the boundary is retained in full so
    topology remains permutation equivariant.
    """

    if not isinstance(positions, torch.Tensor):
        raise TypeError("positions must be a tensor")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N,3)")
    if not positions.is_floating_point():
        raise TypeError("positions must use a floating-point dtype")
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
        raise TypeError("cutoff must be a real number")
    cutoff_value = float(cutoff)
    if not isfinite(cutoff_value) or cutoff_value <= 0.0:
        raise ValueError("cutoff must be finite and positive")
    _validate_positive_int("max_neighbors", max_neighbors)
    _validate_positive_int("dense_threshold", dense_threshold)
    _validate_positive_int("chunk_size", chunk_size)

    if _trusted_graph_counts is not None and (ptr is not None or batch is not None):
        raise ValueError("trusted graph counts replace ptr and batch")
    if ptr is not None and batch is not None:
        raise ValueError("supply ptr or batch, not both")
    if batch is not None:
        if batch.shape != (positions.shape[0],):
            raise ValueError("batch must have shape (N,)")
        if batch.dtype not in _INTEGER_DTYPES:
            raise TypeError("batch must use an integer dtype")
        batch_long = batch.to(dtype=torch.long)
        if batch_long.numel() and bool((batch_long < 0).any().item()):
            raise ValueError("batch values must be nonnegative")
        if batch_long.numel():
            unique = torch.unique(batch_long, sorted=True)
            expected = torch.arange(
                int(batch_long.max().item()) + 1,
                device=batch_long.device,
            )
            if not torch.equal(unique, expected):
                raise ValueError("batch graph IDs must be contiguous from zero")
        if bool((batch_long[1:] < batch_long[:-1]).any().item()):
            order = torch.argsort(batch_long, stable=True)
            sorted_receiver, sorted_sender = _radius_edges(
                positions[order],
                cutoff=cutoff,
                batch=batch_long[order],
                max_neighbors=max_neighbors,
                include_self=include_self,
                dense_threshold=dense_threshold,
                chunk_size=chunk_size,
                receiver_major=receiver_major,
            )
            return order[sorted_receiver], order[sorted_sender]
    if _trusted_graph_counts is None:
        graph_ptr = _normalize_ptr(positions, ptr=ptr, batch=batch)
    else:
        # The counts come only from ELABatch's already validated ptr.  Keeping
        # this private lane tensor-only avoids re-reading grouped membership
        # and synchronizing a CUDA boolean merely to rediscover the same fact.
        graph_ptr = torch.cat(
            (
                _trusted_graph_counts.new_zeros((1,)),
                _trusted_graph_counts.cumsum(0),
            )
        ).to(dtype=torch.long)

    cutoff_squared = cutoff_value * cutoff_value
    num_graphs = graph_ptr.numel() - 1
    if num_graphs > 1:
        counts = graph_ptr[1:] - graph_ptr[:-1]
        padded_nodes = int(counts.max().item())
        padded_pair_count = num_graphs * padded_nodes * padded_nodes
        if (
            padded_nodes <= dense_threshold
            and padded_pair_count <= _BATCHED_DENSE_PAIR_BUDGET
        ):
            return _batched_dense_edges(
                positions,
                graph_ptr,
                padded_nodes=padded_nodes,
                cutoff_squared=cutoff_squared,
                include_self=include_self,
                max_neighbors=max_neighbors,
                receiver_major=receiver_major,
            )
        batch_index = torch.repeat_interleave(
            torch.arange(num_graphs, device=positions.device),
            counts,
            output_size=positions.shape[0],
        )
        batched = _batched_cell_edges(
            positions,
            batch_index,
            num_graphs=num_graphs,
            cutoff=cutoff_value,
            cutoff_squared=cutoff_squared,
            include_self=include_self,
            max_neighbors=max_neighbors,
            receiver_major=receiver_major,
        )
        if batched is not None:
            return batched

    receivers: list[torch.Tensor] = []
    senders: list[torch.Tensor] = []
    boundaries = [
        int(value) for value in graph_ptr.detach().to("cpu").tolist()
    ]
    geometry_dtype = (
        torch.float64 if positions.dtype == torch.float64 else torch.float32
    )
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        # Radius membership must not depend on whether this graph happens to
        # cross ``dense_threshold``. Half-precision squared distances otherwise
        # disagree with the cell path, which requires float32 for robust IDs.
        graph_positions = positions[start:stop].detach().to(
            dtype=geometry_dtype
        )
        if graph_positions.shape[0] <= dense_threshold:
            receiver, sender = _dense_edges(
                graph_positions,
                cutoff_squared=cutoff_squared,
                include_self=include_self,
                max_neighbors=max_neighbors,
                chunk_size=chunk_size,
            )
        else:
            receiver, sender = _cell_edges(
                graph_positions,
                cutoff=cutoff_value,
                cutoff_squared=cutoff_squared,
                include_self=include_self,
                max_neighbors=max_neighbors,
                dense_chunk_size=chunk_size,
                receiver_major=receiver_major,
            )
        receivers.append(receiver + start)
        senders.append(sender + start)
    if not receivers:
        empty = torch.empty(0, device=positions.device, dtype=torch.long)
        return empty, empty
    return torch.cat(receivers), torch.cat(senders)


def radius_graph(
    positions: torch.Tensor,
    *,
    cutoff: float,
    ptr: torch.Tensor | None = None,
    batch: torch.Tensor | None = None,
    max_neighbors: int | None = None,
    include_self: bool = False,
    dense_threshold: int = 256,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Build exact directed receiver/sender COO radius candidates."""

    receiver, sender = _radius_edges(
        positions,
        cutoff=cutoff,
        ptr=ptr,
        batch=batch,
        max_neighbors=max_neighbors,
        include_self=include_self,
        dense_threshold=dense_threshold,
        chunk_size=chunk_size,
        receiver_major=False,
    )
    return torch.stack((receiver, sender))


def _radius_graph_csr(
    positions: torch.Tensor,
    *,
    cutoff: float,
    ptr: torch.Tensor | None = None,
    batch: torch.Tensor | None = None,
    max_neighbors: int | None = None,
    include_self: bool = False,
    dense_threshold: int = 256,
    chunk_size: int = 1024,
    num_edge_relations: int = 0,
    prefer_int32: bool = True,
    _trusted_graph_counts: torch.Tensor | None = None,
) -> PackedNeighborGraph:
    """Build private receiver CSR directly from automatic radius discovery.

    Graph-major inputs leave every discovery backend receiver-major, so CSR
    offsets are formed without another receiver sort.  Interleaved membership
    remains supported through the validated COO packer because mapping grouped
    nodes back to their original node order necessarily destroys that order.
    """

    if (
        isinstance(num_edge_relations, bool)
        or not isinstance(num_edge_relations, int)
        or num_edge_relations not in {0, 1}
    ):
        raise ValueError("automatic radius CSR supports zero or one edge relation")
    receiver, sender = _radius_edges(
        positions,
        cutoff=cutoff,
        ptr=ptr,
        batch=batch,
        _trusted_graph_counts=_trusted_graph_counts,
        max_neighbors=max_neighbors,
        include_self=include_self,
        dense_threshold=dense_threshold,
        chunk_size=chunk_size,
        receiver_major=True,
    )
    relation = None
    if num_edge_relations == 1:
        relation = torch.zeros_like(receiver)
    is_grouped = _trusted_graph_counts is not None or batch is None or bool(
        batch.numel() < 2
        or (batch.to(dtype=torch.long)[1:] >= batch.to(dtype=torch.long)[:-1])
        .all()
        .item()
    )
    if not is_grouped:
        return build_receiver_csr(
            torch.stack((receiver, sender)),
            num_nodes=positions.shape[0],
            edge_relation_id=relation,
            prefer_int32=prefer_int32,
            build_ell=False,
        )
    return _build_receiver_csr_from_major(
        receiver,
        sender,
        num_nodes=positions.shape[0],
        edge_relation_id=relation,
        prefer_int32=prefer_int32,
    )


__all__ = ["radius_graph"]
