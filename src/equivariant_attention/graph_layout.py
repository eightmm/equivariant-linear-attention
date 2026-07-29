from __future__ import annotations

from dataclasses import dataclass, field
import math

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
_FLOATING_DTYPES = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
)
_BACKENDS = frozenset({"auto", "feature_gemm", "outer_scatter"})
_STRUCTURES = frozenset(
    {"direct", "padded", "bucketed", "ragged", "extreme"}
)


def _move_tensor(
    value: torch.Tensor | None,
    device: torch.device,
    *,
    non_blocking: bool,
) -> torch.Tensor | None:
    if value is None:
        return None
    return value.to(device=device, non_blocking=non_blocking)


def _zero_padded_gather(
    value: torch.Tensor,
    node_index: torch.Tensor,
) -> torch.Tensor:
    zero = value.new_zeros((1, *value.shape[1:]))
    extended = torch.cat((value, zero), dim=0)
    return extended.index_select(0, node_index.reshape(-1).to(dtype=torch.long)).reshape(
        *node_index.shape,
        *value.shape[1:],
    )


@dataclass(frozen=True)
class _PackedGraphBucket:
    """One power-of-two graph-size bucket in grouped-node coordinates."""

    width: int
    graph_index: torch.Tensor
    node_index: torch.Tensor
    mask: torch.Tensor

    def gather(self, grouped_value: torch.Tensor) -> torch.Tensor:
        if grouped_value.ndim == 0:
            raise ValueError("grouped node values must have a leading node dimension")
        if grouped_value.device != self.node_index.device:
            raise ValueError("bucket and grouped node values must use the same device")
        return _zero_padded_gather(grouped_value, self.node_index)

    def to(
        self,
        device: torch.device,
        *,
        non_blocking: bool,
    ) -> _PackedGraphBucket:
        if self.node_index.device == device:
            return self
        return _PackedGraphBucket(
            width=self.width,
            graph_index=self.graph_index.to(
                device=device,
                non_blocking=non_blocking,
            ),
            node_index=self.node_index.to(
                device=device,
                non_blocking=non_blocking,
            ),
            mask=self.mask.to(device=device, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class PackedGraphLayout:
    """Immutable, prevalidated execution plan for graph-wise global reductions.

    The layout is deliberately not an ``nn.Module`` and must be supplied as
    per-batch runtime metadata. Consequently it cannot add buffers, parameters,
    or checkpoint keys to a model. Instances should be built with
    :func:`pack_graph_layout`; moving a built layout is a trusted tensor-only
    operation and does not repeat host-side validation.
    """

    batch: torch.Tensor
    graph_counts: torch.Tensor
    graph_ptr: torch.Tensor
    order: torch.Tensor | None
    inverse_order: torch.Tensor | None
    num_nodes: int
    num_graphs: int
    max_nodes: int
    structure: str
    graph_spans: tuple[tuple[int, int], ...]
    dense_index: torch.Tensor | None
    dense_mask: torch.Tensor | None
    buckets: tuple[_PackedGraphBucket, ...]
    packed_slots: int
    maximum_padding_ratio: float = 1.5
    maximum_bucket_padding_ratio: float = 2.0
    maximum_buckets: int = 8
    extreme_size_ratio: float = 128.0
    minimum_extreme_graphs: int = 8
    _validated: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate every public construction once.

        Execution plans are part of the numerical contract: accepting forged
        counts, permutations, or padding masks can silently mix graphs.  The
        public constructor therefore validates complete semantics.  Trusted
        device moves use :meth:`_from_trusted` and do not synchronize merely
        to repeat checks that already passed.
        """
        _validate_packed_graph_layout(self)
        object.__setattr__(self, "_validated", True)

    @classmethod
    def _from_trusted(
        cls,
        *,
        batch: torch.Tensor,
        graph_counts: torch.Tensor,
        graph_ptr: torch.Tensor,
        order: torch.Tensor | None,
        inverse_order: torch.Tensor | None,
        num_nodes: int,
        num_graphs: int,
        max_nodes: int,
        structure: str,
        graph_spans: tuple[tuple[int, int], ...],
        dense_index: torch.Tensor | None,
        dense_mask: torch.Tensor | None,
        buckets: tuple[_PackedGraphBucket, ...],
        packed_slots: int,
        maximum_padding_ratio: float,
        maximum_bucket_padding_ratio: float,
        maximum_buckets: int,
        extreme_size_ratio: float,
        minimum_extreme_graphs: int,
    ) -> PackedGraphLayout:
        value = object.__new__(cls)
        for name, field_value in (
            ("batch", batch),
            ("graph_counts", graph_counts),
            ("graph_ptr", graph_ptr),
            ("order", order),
            ("inverse_order", inverse_order),
            ("num_nodes", num_nodes),
            ("num_graphs", num_graphs),
            ("max_nodes", max_nodes),
            ("structure", structure),
            ("graph_spans", graph_spans),
            ("dense_index", dense_index),
            ("dense_mask", dense_mask),
            ("buckets", buckets),
            ("packed_slots", packed_slots),
            ("maximum_padding_ratio", maximum_padding_ratio),
            (
                "maximum_bucket_padding_ratio",
                maximum_bucket_padding_ratio,
            ),
            ("maximum_buckets", maximum_buckets),
            ("extreme_size_ratio", extreme_size_ratio),
            ("minimum_extreme_graphs", minimum_extreme_graphs),
            ("_validated", True),
        ):
            object.__setattr__(value, name, field_value)
        return value

    @property
    def device(self) -> torch.device:
        return self.batch.device

    @property
    def is_grouped(self) -> bool:
        return self.order is None

    @property
    def validated(self) -> bool:
        return self._validated

    def validate_batch(self, batch: torch.Tensor) -> None:
        """Require the exact membership tensor used to build this plan."""
        if batch is not self.batch:
            raise ValueError(
                "PackedGraphLayout requires the exact batch tensor used at packing"
            )

    def group_nodes(self, value: torch.Tensor) -> torch.Tensor:
        """Put node-major values in stable graph-major order."""
        self._validate_node_value(value)
        if self.order is None:
            return value
        return value.index_select(0, self.order)

    def ungroup_nodes(self, grouped_value: torch.Tensor) -> torch.Tensor:
        """Restore stable graph-major values to their original node order."""
        self._validate_node_value(grouped_value)
        if self.inverse_order is None:
            return grouped_value
        return grouped_value.index_select(0, self.inverse_order)

    def gather_dense(self, value: torch.Tensor) -> torch.Tensor:
        """Gather all graphs into the cached dense padded plan."""
        if self.dense_index is None or self.dense_mask is None:
            raise RuntimeError("this layout does not carry a dense padded plan")
        grouped = self.group_nodes(value)
        return _zero_padded_gather(grouped, self.dense_index)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> PackedGraphLayout:
        """Move cached tensors without rebuilding or revalidating the plan."""
        target = torch.device(device)
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        if target == self.device:
            return self
        return PackedGraphLayout._from_trusted(
            batch=self.batch.to(device=target, non_blocking=non_blocking),
            graph_counts=self.graph_counts.to(
                device=target,
                non_blocking=non_blocking,
            ),
            graph_ptr=self.graph_ptr.to(
                device=target,
                non_blocking=non_blocking,
            ),
            order=_move_tensor(self.order, target, non_blocking=non_blocking),
            inverse_order=_move_tensor(
                self.inverse_order,
                target,
                non_blocking=non_blocking,
            ),
            num_nodes=self.num_nodes,
            num_graphs=self.num_graphs,
            max_nodes=self.max_nodes,
            structure=self.structure,
            graph_spans=self.graph_spans,
            dense_index=_move_tensor(
                self.dense_index,
                target,
                non_blocking=non_blocking,
            ),
            dense_mask=_move_tensor(
                self.dense_mask,
                target,
                non_blocking=non_blocking,
            ),
            buckets=tuple(
                bucket.to(target, non_blocking=non_blocking)
                for bucket in self.buckets
            ),
            packed_slots=self.packed_slots,
            maximum_padding_ratio=self.maximum_padding_ratio,
            maximum_bucket_padding_ratio=(
                self.maximum_bucket_padding_ratio
            ),
            maximum_buckets=self.maximum_buckets,
            extreme_size_ratio=self.extreme_size_ratio,
            minimum_extreme_graphs=self.minimum_extreme_graphs,
        )

    def padded_widths(
        self,
        *,
        feature_width: int,
        augmented_value_width: int,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> tuple[int, int]:
        """Return exact zero-padding widths for the selected compute device."""
        _validate_positive_integer("feature_width", feature_width)
        _validate_positive_integer("augmented_value_width", augmented_value_width)
        if dtype not in _FLOATING_DTYPES:
            raise TypeError("dtype must be a supported floating point dtype")
        target = self.device if device is None else torch.device(device)
        multiple = 1
        if target.type == "cuda":
            if dtype in {torch.float16, torch.bfloat16}:
                multiple = 16
            elif dtype == torch.float32:
                multiple = 8
        return (
            _round_up(feature_width, multiple),
            _round_up(augmented_value_width, multiple),
        )

    def select_lane(
        self,
        *,
        backend: str,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
        num_heads: int,
        feature_width: int,
        value_width: int,
    ) -> str:
        """Select a deterministic reduction lane without inspecting tensor data.

        ``value_width`` excludes the appended denominator coordinate. ``auto``
        uses a conservative workspace model; its constants are dispatch
        heuristics rather than performance claims and should be calibrated by a
        registered device benchmark before promotion.
        """
        if not isinstance(backend, str) or backend not in _BACKENDS:
            choices = ", ".join(sorted(_BACKENDS))
            raise ValueError(f"backend must be one of: {choices}")
        if dtype not in _FLOATING_DTYPES:
            raise TypeError("dtype must be a supported floating point dtype")
        _validate_positive_integer("num_heads", num_heads)
        _validate_positive_integer("feature_width", feature_width)
        _validate_nonnegative_integer("value_width", value_width)
        target = self.device if device is None else torch.device(device)

        if backend == "outer_scatter":
            return "outer_scatter"
        if self.structure == "extreme":
            return "outer_scatter"
        feature_lane = {
            "direct": "direct",
            "padded": "padded_bmm",
            "bucketed": "bucket_bmm",
            "ragged": "ragged_gemm",
        }[self.structure]
        if backend == "feature_gemm":
            return feature_lane

        element_size = _element_size(dtype)
        augmented_value_width = value_width + 1
        outer_bytes = (
            self.num_nodes
            * num_heads
            * feature_width
            * augmented_value_width
            * element_size
        )
        summary_bytes = (
            self.num_graphs
            * num_heads
            * feature_width
            * augmented_value_width
            * element_size
        )
        packed_bytes = (
            self.packed_slots
            * num_heads
            * (2 * feature_width + augmented_value_width)
            * element_size
        )
        gemm_bytes = summary_bytes + packed_bytes
        minimum_workspace = _minimum_auto_workspace(target, dtype)
        if outer_bytes < minimum_workspace:
            return "outer_scatter"
        if 4 * gemm_bytes > 3 * outer_bytes:
            return "outer_scatter"
        return feature_lane

    def _validate_node_value(self, value: torch.Tensor) -> None:
        if value.ndim == 0 or value.shape[0] != self.num_nodes:
            raise ValueError(
                f"node values must have leading dimension {self.num_nodes}"
            )
        if value.device != self.device:
            raise ValueError("layout and node values must use the same device")


def _validate_packed_graph_layout(layout: PackedGraphLayout) -> None:
    _validate_pack_controls(
        assume_grouped=True,
        maximum_padding_ratio=layout.maximum_padding_ratio,
        maximum_bucket_padding_ratio=layout.maximum_bucket_padding_ratio,
        maximum_buckets=layout.maximum_buckets,
        extreme_size_ratio=layout.extreme_size_ratio,
        minimum_extreme_graphs=layout.minimum_extreme_graphs,
    )
    _validate_batch_tensor(layout.batch)
    _validate_positive_integer("num_nodes", layout.num_nodes)
    _validate_positive_integer("num_graphs", layout.num_graphs)
    _validate_positive_integer("max_nodes", layout.max_nodes)
    _validate_positive_integer("packed_slots", layout.packed_slots)
    if layout.num_nodes != layout.batch.numel():
        raise ValueError("num_nodes must match batch length")
    if layout.structure not in _STRUCTURES:
        choices = ", ".join(sorted(_STRUCTURES))
        raise ValueError(f"structure must be one of: {choices}")

    if not isinstance(layout.graph_counts, torch.Tensor):
        raise TypeError("graph_counts must be a tensor")
    if layout.graph_counts.dtype not in _INTEGER_DTYPES:
        raise TypeError("graph_counts must use an integer dtype")
    if layout.graph_counts.device != layout.device:
        raise ValueError("graph_counts must share the layout device")
    if layout.graph_counts.shape != (layout.num_graphs,):
        raise ValueError("graph_counts must have shape (num_graphs,)")
    counts = layout.graph_counts.to(dtype=torch.long)
    if bool((counts <= 0).any().item()):
        raise ValueError("graph_counts must be positive")
    if int(counts.sum().item()) != layout.num_nodes:
        raise ValueError("graph_counts must sum to num_nodes")
    if int(counts.max().item()) != layout.max_nodes:
        raise ValueError("max_nodes must match graph_counts")

    batch_long = layout.batch.to(dtype=torch.long)
    if bool((batch_long < 0).any().item()):
        raise ValueError("batch indices must be nonnegative")
    observed_counts = torch.bincount(batch_long, minlength=layout.num_graphs)
    if observed_counts.shape != counts.shape or not torch.equal(
        observed_counts,
        counts,
    ):
        raise ValueError(
            "batch membership must be contiguous and match graph_counts"
        )

    if not isinstance(layout.graph_ptr, torch.Tensor):
        raise TypeError("graph_ptr must be a tensor")
    if layout.graph_ptr.dtype not in {torch.int32, torch.int64}:
        raise TypeError("graph_ptr must use int32 or int64")
    if layout.graph_ptr.device != layout.device:
        raise ValueError("graph_ptr must share the layout device")
    if layout.graph_ptr.shape != (layout.num_graphs + 1,):
        raise ValueError("graph_ptr must have shape (num_graphs + 1,)")
    expected_ptr = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0))).to(
        dtype=layout.graph_ptr.dtype
    )
    if not torch.equal(layout.graph_ptr, expected_ptr):
        raise ValueError("graph_ptr must be the prefix sum of graph_counts")
    expected_spans: list[tuple[int, int]] = []
    span_start = 0
    for count in tuple(int(value) for value in counts.cpu().tolist()):
        expected_spans.append((span_start, count))
        span_start += count
    if layout.graph_spans != tuple(expected_spans):
        raise ValueError("graph_spans must match graph_counts")

    grouped_batch = torch.repeat_interleave(
        torch.arange(layout.num_graphs, device=layout.device),
        counts,
        output_size=layout.num_nodes,
    )
    if (layout.order is None) != (layout.inverse_order is None):
        raise ValueError("order and inverse_order must be both present or absent")
    if layout.order is None:
        if not torch.equal(batch_long, grouped_batch):
            raise ValueError(
                "a grouped layout requires graph-major batch membership"
            )
    else:
        if layout.inverse_order is None:
            raise RuntimeError("inverse_order validation is incomplete")
        for name, value in (
            ("order", layout.order),
            ("inverse_order", layout.inverse_order),
        ):
            if value.dtype not in _INTEGER_DTYPES:
                raise TypeError(f"{name} must use an integer dtype")
            if value.device != layout.device:
                raise ValueError(f"{name} must share the layout device")
            if value.shape != (layout.num_nodes,):
                raise ValueError(f"{name} must have shape (num_nodes,)")
        expected_order = torch.arange(layout.num_nodes, device=layout.device)
        order_long = layout.order.to(dtype=torch.long)
        inverse_long = layout.inverse_order.to(dtype=torch.long)
        if not torch.equal(torch.sort(order_long).values, expected_order):
            raise ValueError("order must be a permutation")
        if not torch.equal(
            inverse_long.index_select(0, order_long),
            expected_order,
        ):
            raise ValueError("inverse_order must invert order")
        if not torch.equal(batch_long.index_select(0, order_long), grouped_batch):
            raise ValueError("order must group nodes by graph")

    if (layout.dense_index is None) != (layout.dense_mask is None):
        raise ValueError("dense_index and dense_mask must be both present or absent")
    if layout.structure == "padded":
        if layout.dense_index is None or layout.dense_mask is None:
            raise ValueError("padded structure requires a dense plan")
        expected_dense_index, expected_dense_mask = _dense_indices(
            torch.arange(layout.num_graphs, device=layout.device),
            layout.max_nodes,
            layout.graph_ptr,
            counts,
            num_nodes=layout.num_nodes,
        )
        if layout.dense_index.device != layout.device:
            raise ValueError("dense_index must share the layout device")
        if layout.dense_index.dtype not in _INTEGER_DTYPES:
            raise TypeError("dense_index must use an integer dtype")
        if layout.dense_mask.device != layout.device:
            raise ValueError("dense_mask must share the layout device")
        if layout.dense_mask.dtype != torch.bool:
            raise TypeError("dense_mask must be boolean")
        if not torch.equal(
            layout.dense_index.to(dtype=torch.long),
            expected_dense_index,
        ) or not torch.equal(layout.dense_mask, expected_dense_mask):
            raise ValueError("dense plan must match graph counts and pointers")
    elif layout.dense_index is not None:
        raise ValueError("only padded structure may carry a dense plan")

    if not isinstance(layout.buckets, tuple) or not layout.buckets:
        raise ValueError("buckets must be a nonempty tuple")
    count_values = tuple(int(value) for value in counts.cpu().tolist())
    expected_specs = _bucket_specs(count_values)
    padded_slots = layout.num_graphs * layout.max_nodes
    expected_bucket_slots = sum(
        len(graph_indices) * width
        for width, graph_indices in expected_specs
    )
    extreme_skew = (
        layout.num_graphs >= layout.minimum_extreme_graphs
        and layout.max_nodes
        > layout.extreme_size_ratio * min(count_values)
    )
    if layout.num_graphs == 1:
        expected_structure = "direct"
    elif padded_slots <= layout.maximum_padding_ratio * layout.num_nodes:
        expected_structure = "padded"
    elif extreme_skew:
        expected_structure = "extreme"
    elif (
        len(expected_specs) <= layout.maximum_buckets
        and expected_bucket_slots
        <= layout.maximum_bucket_padding_ratio * layout.num_nodes
    ):
        expected_structure = "bucketed"
    else:
        expected_structure = "ragged"
    if layout.structure != expected_structure:
        raise ValueError(
            "structure must match the recorded graph-layout selection policy"
        )
    if len(layout.buckets) != len(expected_specs):
        raise ValueError("bucket count must match power-of-two graph groups")
    for bucket, (expected_width, expected_graphs) in zip(
        layout.buckets,
        expected_specs,
        strict=True,
    ):
        if not isinstance(bucket, _PackedGraphBucket):
            raise TypeError("buckets must contain packed graph buckets")
        expected_bucket = _build_bucket(
            expected_width,
            expected_graphs,
            layout.graph_ptr,
            counts,
            num_nodes=layout.num_nodes,
        )
        if bucket.width != expected_bucket.width:
            raise ValueError("bucket width must match graph counts")
        if any(
            value.device != layout.device
            for value in (bucket.graph_index, bucket.node_index, bucket.mask)
        ):
            raise ValueError("bucket tensors must share the layout device")
        if bucket.graph_index.dtype not in _INTEGER_DTYPES:
            raise TypeError("bucket graph_index must use an integer dtype")
        if bucket.node_index.dtype not in _INTEGER_DTYPES:
            raise TypeError("bucket node_index must use an integer dtype")
        if bucket.mask.dtype != torch.bool:
            raise TypeError("bucket mask must be boolean")
        if (
            not torch.equal(
                bucket.graph_index.to(dtype=torch.long),
                expected_bucket.graph_index,
            )
            or not torch.equal(
                bucket.node_index.to(dtype=torch.long),
                expected_bucket.node_index,
            )
            or not torch.equal(bucket.mask, expected_bucket.mask)
        ):
            raise ValueError("bucket plan must match graph counts and pointers")

    expected_slots = {
        "direct": layout.num_nodes,
        "padded": layout.num_graphs * layout.max_nodes,
        "bucketed": sum(bucket.node_index.numel() for bucket in layout.buckets),
        "ragged": layout.num_nodes,
        "extreme": sum(bucket.node_index.numel() for bucket in layout.buckets),
    }[layout.structure]
    if layout.packed_slots != expected_slots:
        raise ValueError("packed_slots must match the selected layout structure")


def pack_graph_layout(
    batch: torch.Tensor,
    *,
    graph_counts: torch.Tensor | None = None,
    assume_grouped: bool = False,
    maximum_padding_ratio: float = 1.5,
    maximum_bucket_padding_ratio: float = 2.0,
    maximum_buckets: int = 8,
    extreme_size_ratio: float = 128.0,
    minimum_extreme_graphs: int = 8,
) -> PackedGraphLayout:
    """Build graph grouping, padding, and size-bucket plans once per batch.

    ``assume_grouped=True`` is the collate fast path: supplied counts are
    trusted to describe an already graph-major membership tensor, avoiding the
    grouped scan and stable sort. Basic shape, positivity, and total-node
    validation still run once during packing.
    """
    _validate_pack_controls(
        assume_grouped=assume_grouped,
        maximum_padding_ratio=maximum_padding_ratio,
        maximum_bucket_padding_ratio=maximum_bucket_padding_ratio,
        maximum_buckets=maximum_buckets,
        extreme_size_ratio=extreme_size_ratio,
        minimum_extreme_graphs=minimum_extreme_graphs,
    )
    _validate_batch_tensor(batch)
    num_nodes = batch.numel()
    batch_long = batch.to(dtype=torch.long)

    if graph_counts is None:
        if bool((batch_long < 0).any().item()):
            raise ValueError("batch indices must be nonnegative")
        num_graphs = int(batch_long.max().item()) + 1
        resolved_counts = torch.bincount(batch_long, minlength=num_graphs)
        if bool((resolved_counts == 0).any().item()):
            raise ValueError("batch indices must be contiguous and start at zero")
    else:
        resolved_counts = _validated_graph_counts(
            graph_counts,
            num_nodes=num_nodes,
            device=batch.device,
        )
        num_graphs = resolved_counts.numel()
        if not assume_grouped:
            if bool((batch_long < 0).any().item()):
                raise ValueError("batch indices must be nonnegative")
            observed = torch.bincount(batch_long, minlength=num_graphs)
            if observed.shape != resolved_counts.shape or not torch.equal(
                observed,
                resolved_counts,
            ):
                raise ValueError("graph_counts must match batch membership")

    grouped = assume_grouped or bool(
        batch_long.numel() < 2
        or (batch_long[1:] >= batch_long[:-1]).all().item()
    )
    order = None
    inverse_order = None
    if not grouped:
        order = torch.argsort(batch_long, stable=True)
        inverse_order = torch.empty_like(order)
        inverse_order.scatter_(
            0,
            order,
            torch.arange(num_nodes, device=batch.device),
        )

    count_values = tuple(int(value) for value in resolved_counts.cpu().tolist())
    max_nodes = max(count_values)
    min_nodes = min(count_values)
    index_dtype = (
        torch.int32
        if num_nodes <= torch.iinfo(torch.int32).max
        else torch.int64
    )
    graph_ptr = torch.cat(
        (
            resolved_counts.new_zeros(1),
            resolved_counts.cumsum(dim=0),
        )
    ).to(dtype=index_dtype)

    bucket_specs = _bucket_specs(count_values)
    buckets = tuple(
        _build_bucket(
            width,
            graph_indices,
            graph_ptr,
            resolved_counts,
            num_nodes=num_nodes,
        )
        for width, graph_indices in bucket_specs
    )
    padded_slots = num_graphs * max_nodes
    bucket_slots = sum(bucket.node_index.numel() for bucket in buckets)
    extreme_skew = (
        num_graphs >= minimum_extreme_graphs
        and max_nodes > extreme_size_ratio * min_nodes
    )
    if num_graphs == 1:
        structure = "direct"
    elif padded_slots <= maximum_padding_ratio * num_nodes:
        structure = "padded"
    elif extreme_skew:
        structure = "extreme"
    elif (
        len(buckets) <= maximum_buckets
        and bucket_slots <= maximum_bucket_padding_ratio * num_nodes
    ):
        structure = "bucketed"
    else:
        structure = "ragged"
    if structure not in _STRUCTURES:
        raise RuntimeError("graph layout structure selection failed")

    dense_index = None
    dense_mask = None
    if structure == "padded":
        graph_index = torch.arange(num_graphs, device=batch.device)
        dense_index, dense_mask = _dense_indices(
            graph_index,
            max_nodes,
            graph_ptr,
            resolved_counts,
            num_nodes=num_nodes,
        )
    packed_slots = {
        "direct": num_nodes,
        "padded": padded_slots,
        "bucketed": bucket_slots,
        "ragged": num_nodes,
        "extreme": bucket_slots,
    }[structure]
    graph_spans: list[tuple[int, int]] = []
    span_start = 0
    for count in count_values:
        graph_spans.append((span_start, count))
        span_start += count
    return PackedGraphLayout(
        batch=batch,
        graph_counts=resolved_counts,
        graph_ptr=graph_ptr,
        order=order,
        inverse_order=inverse_order,
        num_nodes=num_nodes,
        num_graphs=num_graphs,
        max_nodes=max_nodes,
        structure=structure,
        graph_spans=tuple(graph_spans),
        dense_index=dense_index,
        dense_mask=dense_mask,
        buckets=buckets,
        packed_slots=packed_slots,
        maximum_padding_ratio=float(maximum_padding_ratio),
        maximum_bucket_padding_ratio=float(maximum_bucket_padding_ratio),
        maximum_buckets=maximum_buckets,
        extreme_size_ratio=float(extreme_size_ratio),
        minimum_extreme_graphs=minimum_extreme_graphs,
    )


def _validate_batch_tensor(batch: torch.Tensor) -> None:
    if not isinstance(batch, torch.Tensor):
        raise TypeError("batch must be a tensor")
    if batch.ndim != 1:
        raise ValueError("batch must be one-dimensional")
    if batch.numel() == 0:
        raise ValueError("batch must be nonempty")
    if batch.dtype not in _INTEGER_DTYPES:
        raise TypeError("batch must use an integer dtype")


def _validated_graph_counts(
    graph_counts: torch.Tensor,
    *,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(graph_counts, torch.Tensor):
        raise TypeError("graph_counts must be a tensor")
    if graph_counts.ndim != 1 or graph_counts.numel() == 0:
        raise ValueError("graph_counts must be a nonempty one-dimensional tensor")
    if graph_counts.dtype not in _INTEGER_DTYPES:
        raise TypeError("graph_counts must use an integer dtype")
    counts = graph_counts.to(device=device, dtype=torch.long)
    if bool((counts <= 0).any().item()):
        raise ValueError("graph_counts must be positive")
    if int(counts.sum().item()) != num_nodes:
        raise ValueError("graph_counts sum must equal the batch node count")
    return counts


def _bucket_specs(
    counts: tuple[int, ...],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    grouped: dict[int, list[int]] = {}
    for graph_index, count in enumerate(counts):
        width = 1 << (count - 1).bit_length()
        grouped.setdefault(width, []).append(graph_index)
    return tuple(
        (width, tuple(grouped[width]))
        for width in sorted(grouped)
    )


def _build_bucket(
    width: int,
    graph_indices: tuple[int, ...],
    graph_ptr: torch.Tensor,
    graph_counts: torch.Tensor,
    *,
    num_nodes: int,
) -> _PackedGraphBucket:
    graph_index = torch.tensor(
        graph_indices,
        dtype=torch.long,
        device=graph_counts.device,
    )
    node_index, mask = _dense_indices(
        graph_index,
        width,
        graph_ptr,
        graph_counts,
        num_nodes=num_nodes,
    )
    return _PackedGraphBucket(
        width=width,
        graph_index=graph_index,
        node_index=node_index,
        mask=mask,
    )


def _dense_indices(
    graph_index: torch.Tensor,
    width: int,
    graph_ptr: torch.Tensor,
    graph_counts: torch.Tensor,
    *,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = graph_ptr[:-1].to(dtype=torch.long).index_select(0, graph_index)
    counts = graph_counts.index_select(0, graph_index)
    position = torch.arange(width, device=graph_counts.device)
    mask = position.unsqueeze(0) < counts.unsqueeze(1)
    node_index = starts.unsqueeze(1) + position.unsqueeze(0)
    node_index = torch.where(mask, node_index, node_index.new_full((), num_nodes))
    return node_index, mask


def _validate_pack_controls(
    *,
    assume_grouped: bool,
    maximum_padding_ratio: float,
    maximum_bucket_padding_ratio: float,
    maximum_buckets: int,
    extreme_size_ratio: float,
    minimum_extreme_graphs: int,
) -> None:
    if not isinstance(assume_grouped, bool):
        raise TypeError("assume_grouped must be boolean")
    for name, value in (
        ("maximum_padding_ratio", maximum_padding_ratio),
        ("maximum_bucket_padding_ratio", maximum_bucket_padding_ratio),
        ("extreme_size_ratio", extreme_size_ratio),
    ):
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be finite and positive")
    _validate_positive_integer("maximum_buckets", maximum_buckets)
    _validate_positive_integer("minimum_extreme_graphs", minimum_extreme_graphs)


def _validate_positive_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_nonnegative_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _element_size(dtype: torch.dtype) -> int:
    return {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.float64: 8,
    }[dtype]


def _minimum_auto_workspace(device: torch.device, dtype: torch.dtype) -> int:
    if device.type == "cpu":
        return 128 * 1024
    if device.type == "cuda":
        if dtype in {torch.float16, torch.bfloat16}:
            return 256 * 1024
        if dtype == torch.float32:
            return 512 * 1024
        return 1024 * 1024
    return 1024 * 1024
