from __future__ import annotations

from dataclasses import dataclass

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
_PACKED_INDEX_DTYPES = frozenset({torch.int32, torch.int64})


@dataclass(frozen=True)
class PackedNeighborGraph:
    """Receiver-major CSR with reversible edge-order metadata.

    ``edge_order[p]`` is the original COO column represented by packed edge
    ``p``. When reverse CSR is present, ``reverse_edge_order[r]`` is the
    forward-packed edge represented by reverse-packed edge ``r``.
    """

    num_nodes: int
    row_ptr: torch.Tensor
    sender: torch.Tensor
    edge_order: torch.Tensor
    reverse_row_ptr: torch.Tensor | None = None
    reverse_edge_order: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_nodes, bool)
            or not isinstance(self.num_nodes, int)
            or self.num_nodes < 0
        ):
            raise ValueError("num_nodes must be a nonnegative integer")
        tensors = (self.row_ptr, self.sender, self.edge_order)
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("packed neighbor fields must be tensors")
        if self.row_ptr.dtype not in _PACKED_INDEX_DTYPES:
            raise TypeError("packed neighbor indices must use int32 or int64")
        if any(value.dtype != self.row_ptr.dtype for value in tensors[1:]):
            raise TypeError("packed neighbor indices must share one dtype")
        if any(value.device != self.row_ptr.device for value in tensors[1:]):
            raise ValueError("packed neighbor indices must share one device")
        if self.row_ptr.shape != (self.num_nodes + 1,):
            raise ValueError("row_ptr must have shape (num_nodes + 1,)")
        if self.sender.ndim != 1:
            raise ValueError("sender must be one-dimensional")
        if self.edge_order.shape != self.sender.shape:
            raise ValueError("edge_order must match sender shape")
        if int(self.row_ptr[0].item()) != 0:
            raise ValueError("row_ptr must start at zero")
        if int(self.row_ptr[-1].item()) != self.num_edges:
            raise ValueError("row_ptr must end at num_edges")
        if bool((self.row_ptr[1:] < self.row_ptr[:-1]).any().item()):
            raise ValueError("row_ptr must be nondecreasing")
        if self.num_edges:
            sender = self.sender.to(dtype=torch.long)
            if bool((sender < 0).any().item()) or bool(
                (sender >= self.num_nodes).any().item()
            ):
                raise ValueError("sender values are out of range")
            expected_order = torch.arange(
                self.num_edges,
                device=self.device,
            )
            if not torch.equal(
                torch.sort(self.edge_order.to(dtype=torch.long)).values,
                expected_order,
            ):
                raise ValueError("edge_order must be a permutation")

        reverse_fields = (self.reverse_row_ptr, self.reverse_edge_order)
        if (reverse_fields[0] is None) != (reverse_fields[1] is None):
            raise ValueError("reverse CSR fields must be both present or both absent")
        if self.reverse_row_ptr is not None:
            if self.reverse_edge_order is None:
                raise RuntimeError("reverse CSR validation is incomplete")
            if self.reverse_row_ptr.dtype != self.index_dtype:
                raise TypeError("reverse CSR must share the packed index dtype")
            if self.reverse_edge_order.dtype != self.index_dtype:
                raise TypeError("reverse CSR must share the packed index dtype")
            if self.reverse_row_ptr.device != self.device:
                raise ValueError("reverse CSR must share the packed device")
            if self.reverse_edge_order.device != self.device:
                raise ValueError("reverse CSR must share the packed device")
            if self.reverse_row_ptr.shape != (self.num_nodes + 1,):
                raise ValueError(
                    "reverse_row_ptr must have shape (num_nodes + 1,)"
                )
            if self.reverse_edge_order.shape != self.sender.shape:
                raise ValueError("reverse_edge_order must match sender shape")
            if int(self.reverse_row_ptr[0].item()) != 0:
                raise ValueError("reverse_row_ptr must start at zero")
            if int(self.reverse_row_ptr[-1].item()) != self.num_edges:
                raise ValueError("reverse_row_ptr must end at num_edges")
            if bool(
                (
                    self.reverse_row_ptr[1:] < self.reverse_row_ptr[:-1]
                ).any().item()
            ):
                raise ValueError("reverse_row_ptr must be nondecreasing")
            if self.num_edges:
                expected_order = torch.arange(
                    self.num_edges,
                    device=self.device,
                )
                if not torch.equal(
                    torch.sort(
                        self.reverse_edge_order.to(dtype=torch.long)
                    ).values,
                    expected_order,
                ):
                    raise ValueError("reverse_edge_order must be a permutation")
                reverse_order = self.reverse_edge_order.to(dtype=torch.long)
                reverse_counts = (
                    self.reverse_row_ptr[1:] - self.reverse_row_ptr[:-1]
                )
                reverse_rows = torch.repeat_interleave(
                    torch.arange(
                        self.num_nodes,
                        dtype=torch.long,
                        device=self.device,
                    ),
                    reverse_counts.to(dtype=torch.long),
                    output_size=self.num_edges,
                )
                if not torch.equal(
                    self.sender.to(dtype=torch.long)[reverse_order],
                    reverse_rows,
                ):
                    raise ValueError(
                        "reverse CSR rows must match sender[reverse_edge_order]"
                    )

    @property
    def device(self) -> torch.device:
        return self.row_ptr.device

    @property
    def index_dtype(self) -> torch.dtype:
        return self.row_ptr.dtype

    @property
    def num_edges(self) -> int:
        return self.sender.numel()

    @property
    def has_reverse(self) -> bool:
        return self.reverse_row_ptr is not None

    def receiver_index(self) -> torch.Tensor:
        """Expand CSR rows into one receiver index per packed edge."""
        counts = self.row_ptr[1:] - self.row_ptr[:-1]
        receivers = torch.arange(
            self.num_nodes,
            dtype=self.index_dtype,
            device=self.device,
        )
        return torch.repeat_interleave(
            receivers,
            counts.to(dtype=torch.long),
            output_size=self.num_edges,
        )

    def packed_edge_index(self) -> torch.Tensor:
        """Return receiver-major COO in the packed index dtype."""
        return torch.stack([self.receiver_index(), self.sender])

    def original_edge_index(self) -> torch.Tensor:
        """Restore COO column order from which this graph was packed."""
        packed = self.packed_edge_index()
        restored = torch.empty_like(packed)
        return restored.index_copy(
            1,
            self.edge_order.to(dtype=torch.long),
            packed,
        )

    def reverse_edge_index(self) -> torch.Tensor:
        """Return sender-major reverse COO as ``sender <- receiver`` rows."""
        if self.reverse_row_ptr is None or self.reverse_edge_order is None:
            raise RuntimeError("reverse CSR was not built")
        reverse_order = self.reverse_edge_order.to(dtype=torch.long)
        return self.packed_edge_index().flip(0)[:, reverse_order]

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> PackedNeighborGraph:
        """Move packed metadata without changing its compact index dtype."""

        def move(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            return value.to(device=device, non_blocking=non_blocking)

        return PackedNeighborGraph(
            num_nodes=self.num_nodes,
            row_ptr=self.row_ptr.to(device=device, non_blocking=non_blocking),
            sender=self.sender.to(device=device, non_blocking=non_blocking),
            edge_order=self.edge_order.to(
                device=device,
                non_blocking=non_blocking,
            ),
            reverse_row_ptr=move(self.reverse_row_ptr),
            reverse_edge_order=move(self.reverse_edge_order),
        )


def pack_neighbor_graph(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    build_reverse: bool = False,
    prefer_int32: bool = True,
) -> PackedNeighborGraph:
    """Pack directed COO edges into stable receiver CSR.

    Edges retain their input order within each receiver row. Compact int32
    metadata is selected when every addressable node and edge offset fits;
    otherwise the representation falls back to int64.
    """
    _validate_pack_controls(
        num_nodes=num_nodes,
        build_reverse=build_reverse,
        prefer_int32=prefer_int32,
    )
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError("edge_index must be a tensor")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if edge_index.dtype not in _INTEGER_DTYPES:
        raise TypeError("edge_index must use an integer dtype")

    num_edges = edge_index.shape[1]
    receiver = edge_index[0].to(dtype=torch.long)
    original_sender = edge_index[1].to(dtype=torch.long)
    if num_edges:
        if bool((receiver < 0).any().item()) or bool(
            (original_sender < 0).any().item()
        ):
            raise ValueError("edge_index values must be nonnegative")
        if bool((receiver >= num_nodes).any().item()) or bool(
            (original_sender >= num_nodes).any().item()
        ):
            raise ValueError("edge_index values are out of range")

    index_dtype = _select_index_dtype(
        num_nodes=num_nodes,
        num_edges=num_edges,
        prefer_int32=prefer_int32,
    )
    edge_order_long = torch.argsort(receiver, stable=True)
    packed_receiver = receiver[edge_order_long]
    packed_sender_long = original_sender[edge_order_long]
    row_ptr = _row_ptr(
        packed_receiver,
        num_nodes=num_nodes,
        dtype=index_dtype,
    )

    reverse_row_ptr = None
    reverse_edge_order = None
    if build_reverse:
        reverse_edge_order_long = torch.argsort(packed_sender_long, stable=True)
        reverse_row_ptr = _row_ptr(
            packed_sender_long[reverse_edge_order_long],
            num_nodes=num_nodes,
            dtype=index_dtype,
        )
        reverse_edge_order = reverse_edge_order_long.to(dtype=index_dtype)

    return PackedNeighborGraph(
        num_nodes=num_nodes,
        row_ptr=row_ptr,
        sender=packed_sender_long.to(dtype=index_dtype),
        edge_order=edge_order_long.to(dtype=index_dtype),
        reverse_row_ptr=reverse_row_ptr,
        reverse_edge_order=reverse_edge_order,
    )


def _select_index_dtype(
    *,
    num_nodes: int,
    num_edges: int,
    prefer_int32: bool,
) -> torch.dtype:
    """Select compact storage without constructing an overflow-sized graph."""
    for name, value in (("num_nodes", num_nodes), ("num_edges", num_edges)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if not isinstance(prefer_int32, bool):
        raise TypeError("prefer_int32 must be boolean")
    int32_max = torch.iinfo(torch.int32).max
    largest_node = max(0, num_nodes - 1)
    if prefer_int32 and largest_node <= int32_max and num_edges <= int32_max:
        return torch.int32
    return torch.int64


def _row_ptr(
    sorted_row: torch.Tensor,
    *,
    num_nodes: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    counts = torch.bincount(sorted_row, minlength=num_nodes)
    zero = counts.new_zeros(1)
    return torch.cat([zero, counts.cumsum(dim=0)]).to(dtype=dtype)


def _validate_pack_controls(
    *,
    num_nodes: int,
    build_reverse: bool,
    prefer_int32: bool,
) -> None:
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int):
        raise TypeError("num_nodes must be an integer")
    if num_nodes < 0:
        raise ValueError("num_nodes must be nonnegative")
    if not isinstance(build_reverse, bool):
        raise TypeError("build_reverse must be boolean")
    if not isinstance(prefer_int32, bool):
        raise TypeError("prefer_int32 must be boolean")
