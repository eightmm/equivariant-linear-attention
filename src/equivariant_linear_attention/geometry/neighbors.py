from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, isfinite

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
    relation_id: torch.Tensor | None = None
    row_spans: tuple[tuple[int, int], ...] | None = None
    reverse_row_ptr: torch.Tensor | None = None
    reverse_edge_order: torch.Tensor | None = None
    degree: torch.Tensor | None = None
    degree_histogram: torch.Tensor | None = None
    degree_bucket: torch.Tensor | None = None
    max_degree: int | None = None
    degree_skew: float | None = None
    ell_sender: torch.Tensor | None = None
    ell_mask: torch.Tensor | None = None
    _validated: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )

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
        boundaries = tuple(
            int(value)
            for value in self.row_ptr.detach().to(device="cpu").tolist()
        )
        expected_row_spans = tuple(
            zip(boundaries[:-1], boundaries[1:])
        )
        if self.row_spans is None:
            object.__setattr__(self, "row_spans", expected_row_spans)
        elif self.row_spans != expected_row_spans:
            raise ValueError("row_spans must match receiver CSR offsets")
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
        if self.relation_id is not None:
            if not isinstance(self.relation_id, torch.Tensor):
                raise TypeError("relation_id must be a tensor")
            if self.relation_id.dtype != self.index_dtype:
                raise TypeError(
                    "relation_id must share the packed index dtype"
                )
            if self.relation_id.device != self.device:
                raise ValueError(
                    "relation_id must share the packed device"
                )
            if self.relation_id.shape != self.sender.shape:
                raise ValueError("relation_id must match sender shape")
            if self.relation_id.numel() and bool(
                (self.relation_id < 0).any().item()
            ):
                raise ValueError("relation_id must be nonnegative")

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
        metadata = (
            self.degree,
            self.degree_histogram,
            self.degree_bucket,
        )
        if self.degree is None:
            if any(value is not None for value in metadata[1:]):
                raise ValueError("degree metadata must be all present or all absent")
            if self.max_degree is not None or self.degree_skew is not None:
                raise ValueError(
                    "degree summary requires complete degree metadata"
                )
        else:
            if any(value is None for value in metadata[1:]):
                raise ValueError("degree metadata must be all present or all absent")
            if isinstance(self.max_degree, bool) or not isinstance(
                self.max_degree, int
            ):
                raise TypeError(
                    "max_degree must be an integer with degree metadata"
                )
            if self.max_degree < 0:
                raise ValueError("max_degree must be nonnegative")
            if (
                isinstance(self.degree_skew, bool)
                or not isinstance(self.degree_skew, (int, float))
                or not isfinite(float(self.degree_skew))
                or float(self.degree_skew) < 0.0
            ):
                raise ValueError(
                    "degree_skew must be finite and nonnegative"
                )
            if self.degree.dtype != self.index_dtype:
                raise TypeError("degree must share the packed index dtype")
            if self.degree.device != self.device:
                raise ValueError("degree must share the packed device")
            if self.degree.shape != (self.num_nodes,):
                raise ValueError("degree must have shape (num_nodes,)")
            expected_degree = self.row_ptr[1:] - self.row_ptr[:-1]
            if not torch.equal(self.degree, expected_degree):
                raise ValueError("degree must match receiver CSR rows")
            actual_max = (
                int(self.degree.max().item()) if self.num_nodes else 0
            )
            if self.max_degree != actual_max:
                raise ValueError("max_degree must match degree")
            if self.degree_histogram is None or self.degree_bucket is None:
                raise RuntimeError("degree metadata validation is incomplete")
            if self.degree_histogram.device != self.device:
                raise ValueError("degree_histogram must share the packed device")
            if self.degree_histogram.dtype != torch.int64:
                raise TypeError("degree_histogram must use int64")
            if self.degree_histogram.shape != (7,):
                raise ValueError("degree_histogram must have seven degree bins")
            if int(self.degree_histogram.sum().item()) != self.num_nodes:
                raise ValueError("degree_histogram must cover every node")
            if self.degree_bucket.device != self.device:
                raise ValueError("degree_bucket must share the packed device")
            if self.degree_bucket.dtype != torch.uint8:
                raise TypeError("degree_bucket must use uint8")
            if self.degree_bucket.shape != (self.num_nodes,):
                raise ValueError("degree_bucket must have shape (num_nodes,)")
            expected_bucket = _degree_bucket(self.degree)
            if not torch.equal(self.degree_bucket, expected_bucket):
                raise ValueError("degree_bucket must match degree boundaries")
            expected_histogram = torch.bincount(
                expected_bucket.to(dtype=torch.long),
                minlength=7,
            )
            if not torch.equal(
                self.degree_histogram,
                expected_histogram,
            ):
                raise ValueError(
                    "degree_histogram must match degree_bucket counts"
                )
            expected_skew = (
                float(actual_max)
                / (float(self.num_edges) / float(self.num_nodes))
                if self.num_nodes and self.num_edges
                else 0.0
            )
            if not isfinite(expected_skew) or not isclose(
                float(self.degree_skew),
                expected_skew,
                rel_tol=1e-12,
                abs_tol=0.0,
            ):
                raise ValueError("degree_skew must match receiver degrees")
        if (self.ell_sender is None) != (self.ell_mask is None):
            raise ValueError("ELL sender and mask must be both present or absent")
        if self.ell_sender is not None:
            if self.ell_mask is None or self.degree is None:
                raise RuntimeError("ELL validation is incomplete")
            if self.ell_sender.device != self.device:
                raise ValueError("ELL metadata must share the packed device")
            if self.ell_sender.dtype != self.index_dtype:
                raise TypeError("ELL sender must share the packed index dtype")
            if self.ell_mask.device != self.device:
                raise ValueError("ELL metadata must share the packed device")
            if self.ell_mask.dtype != torch.bool:
                raise TypeError("ELL mask must be boolean")
            if self.max_degree is None:
                raise RuntimeError("ELL max degree metadata is missing")
            expected_shape = (self.num_nodes, self.max_degree)
            if self.ell_sender.shape != expected_shape:
                raise ValueError("ELL sender shape must be (num_nodes, max_degree)")
            if self.ell_mask.shape != expected_shape:
                raise ValueError("ELL mask shape must be (num_nodes, max_degree)")
            if not torch.equal(
                self.ell_mask.sum(dim=1).to(dtype=self.index_dtype),
                self.degree,
            ):
                raise ValueError("ELL mask counts must match degree")
            if not torch.equal(self.ell_sender[self.ell_mask], self.sender):
                raise ValueError("ELL sender rows must match receiver CSR")
        object.__setattr__(self, "_validated", True)

    @classmethod
    def _from_trusted(
        cls,
        *,
        num_nodes: int,
        row_ptr: torch.Tensor,
        sender: torch.Tensor,
        edge_order: torch.Tensor,
        relation_id: torch.Tensor | None,
        row_spans: tuple[tuple[int, int], ...] | None,
        reverse_row_ptr: torch.Tensor | None,
        reverse_edge_order: torch.Tensor | None,
        degree: torch.Tensor | None,
        degree_histogram: torch.Tensor | None,
        degree_bucket: torch.Tensor | None,
        max_degree: int | None,
        degree_skew: float | None,
        ell_sender: torch.Tensor | None,
        ell_mask: torch.Tensor | None,
    ) -> PackedNeighborGraph:
        """Construct from already validated metadata without device sync."""
        value = object.__new__(cls)
        for name, field_value in (
            ("num_nodes", num_nodes),
            ("row_ptr", row_ptr),
            ("sender", sender),
            ("edge_order", edge_order),
            ("relation_id", relation_id),
            ("row_spans", row_spans),
            ("reverse_row_ptr", reverse_row_ptr),
            ("reverse_edge_order", reverse_edge_order),
            ("degree", degree),
            ("degree_histogram", degree_histogram),
            ("degree_bucket", degree_bucket),
            ("max_degree", max_degree),
            ("degree_skew", degree_skew),
            ("ell_sender", ell_sender),
            ("ell_mask", ell_mask),
            ("_validated", True),
        ):
            object.__setattr__(value, name, field_value)
        return value

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

    @property
    def validated(self) -> bool:
        return self._validated

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

    def original_relation_id(self) -> torch.Tensor:
        """Restore relation IDs to the original COO column order."""
        if self.relation_id is None:
            raise RuntimeError("packed graph has no relation metadata")
        restored = torch.empty_like(self.relation_id)
        return restored.index_copy(
            0,
            self.edge_order.to(dtype=torch.long),
            self.relation_id,
        )

    def reverse_relation_view(self) -> torch.Tensor:
        """View forward relation IDs in sender-major reduction order.

        This does not map IDs to their semantic reverse.  Reverse CSR is only
        another reduction ordering of the same directed edges.
        """
        if self.relation_id is None:
            raise RuntimeError("packed graph has no relation metadata")
        if self.reverse_edge_order is None:
            raise RuntimeError("reverse CSR was not built")
        return self.relation_id.index_select(
            0,
            self.reverse_edge_order.to(dtype=torch.long),
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

        target = torch.device(device)
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        if self.device == target:
            return self

        def move(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            return value.to(device=target, non_blocking=non_blocking)

        return PackedNeighborGraph._from_trusted(
            num_nodes=self.num_nodes,
            row_ptr=self.row_ptr.to(device=target, non_blocking=non_blocking),
            sender=self.sender.to(device=target, non_blocking=non_blocking),
            edge_order=self.edge_order.to(
                device=target,
                non_blocking=non_blocking,
            ),
            relation_id=move(self.relation_id),
            row_spans=self.row_spans,
            reverse_row_ptr=move(self.reverse_row_ptr),
            reverse_edge_order=move(self.reverse_edge_order),
            degree=move(self.degree),
            degree_histogram=move(self.degree_histogram),
            degree_bucket=move(self.degree_bucket),
            max_degree=self.max_degree,
            degree_skew=self.degree_skew,
            ell_sender=move(self.ell_sender),
            ell_mask=move(self.ell_mask),
        )

def build_receiver_csr(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    edge_relation_id: torch.Tensor | None = None,
    prefer_int32: bool = True,
    build_ell: bool = False,
    ell_max_degree: int = 64,
    ell_max_padding_ratio: float = 8.0,
    ell_max_elements: int = 16_777_216,
) -> PackedNeighborGraph:
    """Pack directed COO edges into stable receiver CSR only.

    Edges retain their input order within each receiver row. Compact int32
    metadata is selected when every addressable node and edge offset fits;
    otherwise the representation falls back to int64.
    """
    _validate_pack_controls(
        num_nodes=num_nodes,
        prefer_int32=prefer_int32,
        build_ell=build_ell,
        ell_max_degree=ell_max_degree,
        ell_max_padding_ratio=ell_max_padding_ratio,
        ell_max_elements=ell_max_elements,
    )
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError("edge_index must be a tensor")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if edge_index.dtype not in _INTEGER_DTYPES:
        raise TypeError("edge_index must use an integer dtype")

    num_edges = edge_index.shape[1]
    if edge_relation_id is not None:
        if not isinstance(edge_relation_id, torch.Tensor):
            raise TypeError("edge_relation_id must be a tensor")
        if edge_relation_id.dtype not in _INTEGER_DTYPES:
            raise TypeError("edge_relation_id must use an integer dtype")
        if edge_relation_id.device != edge_index.device:
            raise ValueError(
                "edge_relation_id and edge_index must share one device"
            )
        if edge_relation_id.shape != (num_edges,):
            raise ValueError(
                "edge_relation_id must have one relation per edge"
            )
        if edge_relation_id.numel() and bool(
            (edge_relation_id < 0).any().item()
        ):
            raise ValueError("edge_relation_id must be nonnegative")
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
    (
        degree,
        degree_histogram,
        degree_bucket,
        max_degree,
        degree_skew,
        ell_sender,
        ell_mask,
    ) = _receiver_execution_metadata(
        row_ptr,
        packed_receiver,
        packed_sender_long,
        num_nodes=num_nodes,
        index_dtype=index_dtype,
        build_ell=build_ell,
        ell_max_degree=ell_max_degree,
        ell_max_padding_ratio=ell_max_padding_ratio,
        ell_max_elements=ell_max_elements,
    )

    return PackedNeighborGraph(
        num_nodes=num_nodes,
        row_ptr=row_ptr,
        sender=packed_sender_long.to(dtype=index_dtype),
        edge_order=edge_order_long.to(dtype=index_dtype),
        relation_id=(
            None
            if edge_relation_id is None
            else edge_relation_id.index_select(
                0,
                edge_order_long,
            ).to(dtype=index_dtype)
        ),
        degree=degree,
        degree_histogram=degree_histogram,
        degree_bucket=degree_bucket,
        max_degree=max_degree,
        degree_skew=degree_skew,
        ell_sender=ell_sender,
        ell_mask=ell_mask,
    )


def _build_receiver_csr_from_major(
    receiver: torch.Tensor,
    sender: torch.Tensor,
    *,
    num_nodes: int,
    edge_relation_id: torch.Tensor | None = None,
    prefer_int32: bool = True,
) -> PackedNeighborGraph:
    """Build trusted CSR from an already receiver-major edge stream.

    Automatic radius discovery owns both tensors and guarantees their range,
    device, and stable receiver-major ordering.  Keeping this constructor
    private lets that path form CSR offsets directly instead of sending its
    output through :func:`build_receiver_csr` and sorting receivers again.
    """

    _validate_pack_controls(
        num_nodes=num_nodes,
        prefer_int32=prefer_int32,
        build_ell=False,
        ell_max_degree=64,
        ell_max_padding_ratio=8.0,
        ell_max_elements=16_777_216,
    )
    if not isinstance(receiver, torch.Tensor) or not isinstance(sender, torch.Tensor):
        raise TypeError("receiver and sender must be tensors")
    if receiver.ndim != 1 or sender.shape != receiver.shape:
        raise ValueError("receiver and sender must be equal-length vectors")
    if receiver.device != sender.device:
        raise ValueError("receiver and sender must share one device")
    if receiver.dtype not in _INTEGER_DTYPES or sender.dtype not in _INTEGER_DTYPES:
        raise TypeError("receiver and sender must use integer dtypes")
    if edge_relation_id is not None:
        if not isinstance(edge_relation_id, torch.Tensor):
            raise TypeError("edge_relation_id must be a tensor")
        if edge_relation_id.shape != receiver.shape:
            raise ValueError("edge_relation_id must have one relation per edge")
        if edge_relation_id.device != receiver.device:
            raise ValueError("edge_relation_id and edges must share one device")
        if edge_relation_id.dtype not in _INTEGER_DTYPES:
            raise TypeError("edge_relation_id must use an integer dtype")

    receiver_long = receiver.to(dtype=torch.long)
    sender_long = sender.to(dtype=torch.long)
    num_edges = receiver_long.numel()
    index_dtype = _select_index_dtype(
        num_nodes=num_nodes,
        num_edges=num_edges,
        prefer_int32=prefer_int32,
    )
    row_ptr = _row_ptr(
        receiver_long,
        num_nodes=num_nodes,
        dtype=index_dtype,
    )
    edge_order = torch.arange(
        num_edges,
        device=receiver.device,
        dtype=index_dtype,
    )
    return PackedNeighborGraph._from_trusted(
        num_nodes=num_nodes,
        row_ptr=row_ptr,
        sender=sender_long.to(dtype=index_dtype),
        edge_order=edge_order,
        relation_id=(
            None
            if edge_relation_id is None
            else edge_relation_id.to(dtype=index_dtype)
        ),
        # Legacy host row spans are not consumed by the tensor kernels. Keep
        # them lazy so GPU radius discovery does not synchronize row_ptr back
        # to the CPU merely to duplicate CSR offsets as Python tuples.
        row_spans=None,
        reverse_row_ptr=None,
        reverse_edge_order=None,
        # Degree summaries were only diagnostic inputs for retired backend
        # routing. Omitting them avoids a GPU ``max().item()`` synchronization
        # in the automatic-radius hot path.
        degree=None,
        degree_histogram=None,
        degree_bucket=None,
        max_degree=None,
        degree_skew=None,
        ell_sender=None,
        ell_mask=None,
    )


def build_reverse_csr(packed: PackedNeighborGraph) -> PackedNeighborGraph:
    """Attach sender-major reverse CSR to a validated receiver plan."""
    if not isinstance(packed, PackedNeighborGraph):
        raise TypeError("packed must be a PackedNeighborGraph")
    if packed.has_reverse:
        return packed
    reverse_edge_order_long = torch.argsort(
        packed.sender.to(dtype=torch.long),
        stable=True,
    )
    reverse_row_ptr = _row_ptr(
        packed.sender.to(dtype=torch.long)[reverse_edge_order_long],
        num_nodes=packed.num_nodes,
        dtype=packed.index_dtype,
    )
    return PackedNeighborGraph._from_trusted(
        num_nodes=packed.num_nodes,
        row_ptr=packed.row_ptr,
        sender=packed.sender,
        edge_order=packed.edge_order,
        relation_id=packed.relation_id,
        row_spans=packed.row_spans,
        reverse_row_ptr=reverse_row_ptr,
        reverse_edge_order=reverse_edge_order_long.to(dtype=packed.index_dtype),
        degree=packed.degree,
        degree_histogram=packed.degree_histogram,
        degree_bucket=packed.degree_bucket,
        max_degree=packed.max_degree,
        degree_skew=packed.degree_skew,
        ell_sender=packed.ell_sender,
        ell_mask=packed.ell_mask,
    )


def pack_neighbor_graph(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    edge_relation_id: torch.Tensor | None = None,
    build_reverse: bool = False,
    prefer_int32: bool = True,
    build_ell: bool = False,
    ell_max_degree: int = 64,
    ell_max_padding_ratio: float = 8.0,
    ell_max_elements: int = 16_777_216,
) -> PackedNeighborGraph:
    """Compatibility wrapper for receiver CSR plus optional reverse/ELL plans."""
    if not isinstance(build_reverse, bool):
        raise TypeError("build_reverse must be boolean")
    packed = build_receiver_csr(
        edge_index,
        num_nodes=num_nodes,
        edge_relation_id=edge_relation_id,
        prefer_int32=prefer_int32,
        build_ell=build_ell,
        ell_max_degree=ell_max_degree,
        ell_max_padding_ratio=ell_max_padding_ratio,
        ell_max_elements=ell_max_elements,
    )
    return build_reverse_csr(packed) if build_reverse else packed


def receiver_csr_reduce(
    packed: PackedNeighborGraph,
    value: torch.Tensor,
    *,
    reduce: str = "sum",
) -> torch.Tensor:
    """Reduce receiver-major edge values without expanding an ``E`` index.

    The compact CSR offsets remain int32 when addressable.  This is the
    receiver-owned primitive used by streamed/reference backends; unlike
    ``index_add`` it does not allocate or cast a receiver vector.
    """
    _validate_csr_reduce_inputs(packed, value, reduce=reduce)
    return torch.segment_reduce(
        value,
        reduce=reduce,
        offsets=packed.row_ptr,
    )


def sender_csr_reduce(
    packed: PackedNeighborGraph,
    value: torch.Tensor,
    *,
    reduce: str = "sum",
) -> torch.Tensor:
    """Deterministically reduce forward-packed edges by sender.

    A reverse CSR plan is mandatory.  Values are reordered by the stable
    forward-edge permutation and reduced by segment, avoiding atomic sender
    scatter and making the reduction order explicit.
    """
    _validate_csr_reduce_inputs(packed, value, reduce=reduce)
    if packed.reverse_row_ptr is None or packed.reverse_edge_order is None:
        raise ValueError("sender CSR reduction requires reverse metadata")
    return torch.segment_reduce(
        value.index_select(
            0,
            packed.reverse_edge_order.to(dtype=torch.long),
        ),
        reduce=reduce,
        offsets=packed.reverse_row_ptr,
    )


def _validate_csr_reduce_inputs(
    packed: PackedNeighborGraph,
    value: torch.Tensor,
    *,
    reduce: str,
) -> None:
    if not isinstance(packed, PackedNeighborGraph):
        raise TypeError("packed must be a PackedNeighborGraph")
    if not isinstance(value, torch.Tensor):
        raise TypeError("value must be a tensor")
    if value.ndim == 0 or value.shape[0] != packed.num_edges:
        raise ValueError("value must have one leading row per packed edge")
    if value.device != packed.device:
        raise ValueError("value and packed graph must use the same device")
    if reduce not in {"sum", "mean", "prod", "max", "min"}:
        raise ValueError("unsupported CSR reduction")


def _degree_bucket(degree: torch.Tensor) -> torch.Tensor:
    """Map receiver degree to stable ``0, 1-8, ..., 129+`` buckets."""
    boundaries = torch.tensor(
        [0, 8, 16, 32, 64, 128],
        dtype=torch.long,
        device=degree.device,
    )
    return torch.bucketize(
        degree.to(dtype=torch.long),
        boundaries,
        right=False,
    ).to(dtype=torch.uint8)


def _receiver_execution_metadata(
    row_ptr: torch.Tensor,
    packed_receiver: torch.Tensor,
    packed_sender_long: torch.Tensor,
    *,
    num_nodes: int,
    index_dtype: torch.dtype,
    build_ell: bool,
    ell_max_degree: int,
    ell_max_padding_ratio: float,
    ell_max_elements: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    float,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    degree = row_ptr[1:] - row_ptr[:-1]
    max_degree = int(degree.max().item()) if num_nodes else 0
    mean_degree = (
        float(packed_sender_long.numel()) / float(num_nodes)
        if num_nodes
        else 0.0
    )
    degree_skew = (
        float(max_degree) / mean_degree if mean_degree > 0.0 else 0.0
    )
    degree_bucket_long = _degree_bucket(degree).to(dtype=torch.long)
    degree_histogram = torch.bincount(
        degree_bucket_long,
        minlength=7,
    )
    degree_bucket = degree_bucket_long.to(dtype=torch.uint8)
    ell_sender = None
    ell_mask = None
    if build_ell:
        if max_degree > ell_max_degree:
            raise ValueError(
                "max receiver degree exceeds ell_max_degree for requested ELL"
            )
        padded_elements = num_nodes * max_degree
        padding_ratio = (
            float(padded_elements)
            / float(max(packed_sender_long.numel(), 1))
        )
        if padding_ratio > ell_max_padding_ratio:
            raise ValueError(
                "ELL padding ratio exceeds ell_max_padding_ratio"
            )
        if padded_elements > ell_max_elements:
            raise ValueError(
                "ELL allocation exceeds ell_max_elements"
            )
        ell_sender = torch.zeros(
            (num_nodes, max_degree),
            dtype=index_dtype,
            device=row_ptr.device,
        )
        ell_mask = torch.zeros(
            (num_nodes, max_degree),
            dtype=torch.bool,
            device=row_ptr.device,
        )
        if packed_sender_long.numel():
            position = torch.arange(
                packed_sender_long.numel(),
                device=row_ptr.device,
            ) - row_ptr.to(dtype=torch.long)[packed_receiver]
            ell_sender[packed_receiver, position] = packed_sender_long.to(
                dtype=index_dtype
            )
            ell_mask[packed_receiver, position] = True
    return (
        degree,
        degree_histogram,
        degree_bucket,
        max_degree,
        degree_skew,
        ell_sender,
        ell_mask,
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
    prefer_int32: bool,
    build_ell: bool,
    ell_max_degree: int,
    ell_max_padding_ratio: float,
    ell_max_elements: int,
) -> None:
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int):
        raise TypeError("num_nodes must be an integer")
    if num_nodes < 0:
        raise ValueError("num_nodes must be nonnegative")
    if not isinstance(prefer_int32, bool):
        raise TypeError("prefer_int32 must be boolean")
    if not isinstance(build_ell, bool):
        raise TypeError("build_ell must be boolean")
    if isinstance(ell_max_degree, bool) or not isinstance(ell_max_degree, int):
        raise TypeError("ell_max_degree must be an integer")
    if ell_max_degree < 0:
        raise ValueError("ell_max_degree must be nonnegative")
    if (
        isinstance(ell_max_padding_ratio, bool)
        or not isinstance(ell_max_padding_ratio, (int, float))
    ):
        raise TypeError("ell_max_padding_ratio must be a real number")
    if (
        not isfinite(float(ell_max_padding_ratio))
        or float(ell_max_padding_ratio) < 1.0
    ):
        raise ValueError(
            "ell_max_padding_ratio must be finite and at least one"
        )
    if isinstance(ell_max_elements, bool) or not isinstance(
        ell_max_elements, int
    ):
        raise TypeError("ell_max_elements must be an integer")
    if ell_max_elements < 0:
        raise ValueError("ell_max_elements must be nonnegative")
