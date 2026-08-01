from __future__ import annotations

from itertools import product
from math import isfinite

import torch

_INTEGER_DTYPES = frozenset(
    {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
)
_CELL_OFFSETS = tuple(product((-1, 0, 1), repeat=3))
_INT64_SAFE = (1 << 62) - 1


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
    if max_neighbors is None or receiver.numel() == 0:
        return receiver, sender
    by_distance = torch.argsort(distance_squared, stable=True)
    receiver_distance = receiver[by_distance]
    by_receiver = torch.argsort(receiver_distance, stable=True)
    order = by_distance[by_receiver]
    receiver = receiver[order]
    sender = sender[order]
    counts = torch.bincount(receiver, minlength=nodes)
    starts = torch.repeat_interleave(
        counts.cumsum(0) - counts,
        counts,
        output_size=receiver.numel(),
    )
    rank = torch.arange(receiver.numel(), device=receiver.device) - starts
    keep = rank < max_neighbors
    return receiver[keep], sender[keep]


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
    return _limit_neighbors(
        receiver,
        sender,
        distance_squared,
        nodes=nodes,
        max_neighbors=max_neighbors,
    )


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
    """Build exact radius candidates without PyG.

    Small graphs use a chunked dense reference. Larger graphs use a cell list
    with exact distance filtering. Both paths return directed receiver/sender
    COO and never connect nodes from different graphs.
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
    graph_ptr = _normalize_ptr(positions, ptr=ptr, batch=batch)

    receivers: list[torch.Tensor] = []
    senders: list[torch.Tensor] = []
    cutoff_squared = cutoff_value * cutoff_value
    boundaries = [
        int(value) for value in graph_ptr.detach().to("cpu").tolist()
    ]
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        graph_positions = positions[start:stop]
        if graph_positions.shape[0] <= dense_threshold:
            receiver, sender = _dense_edges(
                graph_positions.detach(),
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
            )
        receivers.append(receiver + start)
        senders.append(sender + start)
    if not receivers:
        return torch.empty((2, 0), device=positions.device, dtype=torch.long)
    return torch.stack([torch.cat(receivers), torch.cat(senders)])


__all__ = ["radius_graph"]