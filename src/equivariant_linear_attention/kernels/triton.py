from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
import os
from math import prod
from typing import Final

import torch

_BACKEND_ENV: Final = "ELA_KERNEL_BACKEND"
_ALLOWED_BACKENDS: Final = frozenset({"auto", "torch", "triton"})
_BACKEND_OVERRIDE: ContextVar[str | None] = ContextVar(
    "ela_kernel_backend_override",
    default=None,
)

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional runtime dependency
    triton = None
    tl = None


def _validate_backend_name(value: str, *, source: str) -> str:
    value = value.strip().lower()
    if value not in _ALLOWED_BACKENDS:
        raise ValueError(
            f"{source} must be one of {sorted(_ALLOWED_BACKENDS)}, got {value!r}"
        )
    return value


def backend_policy() -> str:
    """Return the task-local override or environment backend policy."""

    override = _BACKEND_OVERRIDE.get()
    if override is not None:
        return override
    return _validate_backend_name(
        os.environ.get(_BACKEND_ENV, "auto"),
        source=_BACKEND_ENV,
    )


@contextmanager
def kernel_backend(policy: str) -> Iterator[None]:
    """Temporarily select one backend without mutating process-global state."""

    if not isinstance(policy, str):
        raise TypeError("backend policy must be a string")
    resolved = _validate_backend_name(policy, source="backend policy")
    token = _BACKEND_OVERRIDE.set(resolved)
    try:
        yield
    finally:
        _BACKEND_OVERRIDE.reset(token)


def triton_available() -> bool:
    return triton is not None and tl is not None


def _basic_triton_support(
    value: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    payload_dtype: torch.dtype | None = None,
) -> bool:
    dtype = value.dtype if payload_dtype is None else payload_dtype
    return (
        triton_available()
        and value.device.type == "cuda"
        and row_ptr.device == value.device
        and dtype in {torch.float16, torch.bfloat16, torch.float32}
        and row_ptr.dtype in {torch.int32, torch.int64}
    )


def _can_use_triton(
    value: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    policy: str,
    payload_dtype: torch.dtype | None = None,
) -> bool:
    if policy == "torch":
        return False
    supported = _basic_triton_support(
        value,
        row_ptr,
        payload_dtype=payload_dtype,
    )
    if policy == "triton" and not supported:
        raise RuntimeError(
            "Triton CSR reduction was forced but the current device or dtype "
            "is unsupported"
        )
    # ``auto`` stays on the numerical reference until a complete-stack
    # hardware/runtime regime passes the documented promotion gate. Forced
    # Triton remains available for contract tests and explicit benchmarks.
    return supported and policy == "triton"


def active_backend(
    value: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    payload_dtype: torch.dtype | None = None,
) -> str:
    """Return the backend that would execute this CSR reduction."""

    policy = backend_policy()
    if policy == "torch":
        return "torch"
    if not _basic_triton_support(
        value,
        row_ptr,
        payload_dtype=payload_dtype,
    ):
        if policy == "triton":
            raise RuntimeError(
                "Triton CSR reduction was forced but the current device or "
                "dtype is unsupported"
            )
        return "torch"
    return (
        "triton"
        if _can_use_triton(
            value,
            row_ptr,
            policy=policy,
            payload_dtype=payload_dtype,
        )
        else "torch"
    )


if triton_available():

    @triton.jit
    def _csr_sum_kernel(
        value_ptr,
        row_ptr,
        output_ptr,
        num_features: tl.constexpr,
        block_edges: tl.constexpr,
        block_features: tl.constexpr,
    ):
        row = tl.program_id(0)
        feature_block = tl.program_id(1)
        feature = feature_block * block_features + tl.arange(0, block_features)
        feature_mask = feature < num_features
        start = tl.load(row_ptr + row).to(tl.int64)
        stop = tl.load(row_ptr + row + 1).to(tl.int64)
        accumulator = tl.zeros((block_features,), dtype=tl.float32)
        for base in tl.range(
            0,
            stop - start,
            block_edges,
            loop_unroll_factor=1,
        ):
            edge = start + base + tl.arange(0, block_edges)
            edge_mask = edge < stop
            pointer = value_ptr + edge[:, None] * num_features + feature[None, :]
            loaded = tl.load(
                pointer,
                mask=edge_mask[:, None] & feature_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.sum(loaded, axis=0)
        tl.store(
            output_ptr + row * num_features + feature,
            accumulator,
            mask=feature_mask,
        )

    @triton.jit
    def _csr_sum_pair_kernel(
        value0_ptr,
        value1_ptr,
        row_ptr,
        output0_ptr,
        output1_ptr,
        num_features0: tl.constexpr,
        num_features1: tl.constexpr,
        block_edges: tl.constexpr,
        block_features: tl.constexpr,
    ):
        row = tl.program_id(0)
        logical_block = tl.program_id(1)
        blocks0: tl.constexpr = tl.cdiv(num_features0, block_features)
        start = tl.load(row_ptr + row).to(tl.int64)
        stop = tl.load(row_ptr + row + 1).to(tl.int64)

        if logical_block < blocks0:
            feature = logical_block * block_features + tl.arange(0, block_features)
            feature_mask = feature < num_features0
            accumulator = tl.zeros((block_features,), dtype=tl.float32)
            for base in tl.range(
                0,
                stop - start,
                block_edges,
                loop_unroll_factor=1,
            ):
                edge = start + base + tl.arange(0, block_edges)
                edge_mask = edge < stop
                loaded = tl.load(
                    value0_ptr + edge[:, None] * num_features0 + feature[None, :],
                    mask=edge_mask[:, None] & feature_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                accumulator += tl.sum(loaded, axis=0)
            tl.store(
                output0_ptr + row * num_features0 + feature,
                accumulator,
                mask=feature_mask,
            )
        else:
            local_block = logical_block - blocks0
            feature = local_block * block_features + tl.arange(0, block_features)
            feature_mask = feature < num_features1
            accumulator = tl.zeros((block_features,), dtype=tl.float32)
            for base in tl.range(
                0,
                stop - start,
                block_edges,
                loop_unroll_factor=1,
            ):
                edge = start + base + tl.arange(0, block_edges)
                edge_mask = edge < stop
                loaded = tl.load(
                    value1_ptr + edge[:, None] * num_features1 + feature[None, :],
                    mask=edge_mask[:, None] & feature_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                accumulator += tl.sum(loaded, axis=0)
            tl.store(
                output1_ptr + row * num_features1 + feature,
                accumulator,
                mask=feature_mask,
            )

    @triton.jit
    def _csr_sum_triple_kernel(
        value0_ptr,
        value1_ptr,
        value2_ptr,
        row_ptr,
        output0_ptr,
        output1_ptr,
        output2_ptr,
        num_features0: tl.constexpr,
        num_features1: tl.constexpr,
        num_features2: tl.constexpr,
        block_edges: tl.constexpr,
        block_features: tl.constexpr,
    ):
        row = tl.program_id(0)
        logical_block = tl.program_id(1)
        blocks0: tl.constexpr = tl.cdiv(num_features0, block_features)
        blocks1: tl.constexpr = tl.cdiv(num_features1, block_features)
        start = tl.load(row_ptr + row).to(tl.int64)
        stop = tl.load(row_ptr + row + 1).to(tl.int64)

        if logical_block < blocks0:
            feature = logical_block * block_features + tl.arange(0, block_features)
            feature_mask = feature < num_features0
            accumulator = tl.zeros((block_features,), dtype=tl.float32)
            for base in tl.range(
                0,
                stop - start,
                block_edges,
                loop_unroll_factor=1,
            ):
                edge = start + base + tl.arange(0, block_edges)
                edge_mask = edge < stop
                loaded = tl.load(
                    value0_ptr + edge[:, None] * num_features0 + feature[None, :],
                    mask=edge_mask[:, None] & feature_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                accumulator += tl.sum(loaded, axis=0)
            tl.store(
                output0_ptr + row * num_features0 + feature,
                accumulator,
                mask=feature_mask,
            )
        elif logical_block < blocks0 + blocks1:
            local_block = logical_block - blocks0
            feature = local_block * block_features + tl.arange(0, block_features)
            feature_mask = feature < num_features1
            accumulator = tl.zeros((block_features,), dtype=tl.float32)
            for base in tl.range(
                0,
                stop - start,
                block_edges,
                loop_unroll_factor=1,
            ):
                edge = start + base + tl.arange(0, block_edges)
                edge_mask = edge < stop
                loaded = tl.load(
                    value1_ptr + edge[:, None] * num_features1 + feature[None, :],
                    mask=edge_mask[:, None] & feature_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                accumulator += tl.sum(loaded, axis=0)
            tl.store(
                output1_ptr + row * num_features1 + feature,
                accumulator,
                mask=feature_mask,
            )
        else:
            local_block = logical_block - blocks0 - blocks1
            feature = local_block * block_features + tl.arange(0, block_features)
            feature_mask = feature < num_features2
            accumulator = tl.zeros((block_features,), dtype=tl.float32)
            for base in tl.range(
                0,
                stop - start,
                block_edges,
                loop_unroll_factor=1,
            ):
                edge = start + base + tl.arange(0, block_edges)
                edge_mask = edge < stop
                loaded = tl.load(
                    value2_ptr + edge[:, None] * num_features2 + feature[None, :],
                    mask=edge_mask[:, None] & feature_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                accumulator += tl.sum(loaded, axis=0)
            tl.store(
                output2_ptr + row * num_features2 + feature,
                accumulator,
                mask=feature_mask,
            )

    @triton.jit
    def _weighted_gather_reduce_pair_kernel(  # pragma: no cover - Triton JIT
        weight_ptr,
        radial_gate_ptr,
        source0_ptr,
        source1_ptr,
        sender_ptr,
        row_ptr,
        output0_ptr,
        output1_ptr,
        rank: tl.constexpr,
        components0: tl.constexpr,
        components1: tl.constexpr,
        gate_lane0: tl.constexpr,
        gate_lane1: tl.constexpr,
        num_gates: tl.constexpr,
        block_edges: tl.constexpr,
        block_features: tl.constexpr,
    ):
        row = tl.program_id(0)
        logical_block = tl.program_id(1)
        width0: tl.constexpr = rank * components0
        width1: tl.constexpr = rank * components1
        blocks0: tl.constexpr = tl.cdiv(width0, block_features)
        start = tl.load(row_ptr + row).to(tl.int64)
        stop = tl.load(row_ptr + row + 1).to(tl.int64)

        if logical_block < blocks0:
            feature = logical_block * block_features + tl.arange(0, block_features)
            feature_mask = feature < width0
            rank_index = feature // components0
            accumulator = tl.zeros((block_features,), dtype=tl.float32)
            for base in tl.range(
                0,
                stop - start,
                block_edges,
                loop_unroll_factor=1,
            ):
                edge = start + base + tl.arange(0, block_edges)
                edge_mask = edge < stop
                sender = tl.load(sender_ptr + edge, mask=edge_mask, other=0).to(
                    tl.int64
                )
                coefficient = tl.load(
                    weight_ptr + edge[:, None] * rank + rank_index[None, :],
                    mask=edge_mask[:, None] & feature_mask[None, :],
                    other=0.0,
                ) * tl.load(
                    radial_gate_ptr
                    + (edge[:, None] * rank + rank_index[None, :]) * num_gates
                    + gate_lane0,
                    mask=edge_mask[:, None] & feature_mask[None, :],
                    other=0.0,
                )
                source = tl.load(
                    source0_ptr + sender[:, None] * width0 + feature[None, :],
                    mask=edge_mask[:, None] & feature_mask[None, :],
                    other=0.0,
                )
                accumulator += tl.sum(coefficient.to(tl.float32) * source, axis=0)
            tl.store(
                output0_ptr + row * width0 + feature,
                accumulator,
                mask=feature_mask,
            )
        else:
            local_block = logical_block - blocks0
            feature = local_block * block_features + tl.arange(0, block_features)
            feature_mask = feature < width1
            rank_index = feature // components1
            accumulator = tl.zeros((block_features,), dtype=tl.float32)
            for base in tl.range(
                0,
                stop - start,
                block_edges,
                loop_unroll_factor=1,
            ):
                edge = start + base + tl.arange(0, block_edges)
                edge_mask = edge < stop
                sender = tl.load(sender_ptr + edge, mask=edge_mask, other=0).to(
                    tl.int64
                )
                coefficient = tl.load(
                    weight_ptr + edge[:, None] * rank + rank_index[None, :],
                    mask=edge_mask[:, None] & feature_mask[None, :],
                    other=0.0,
                ) * tl.load(
                    radial_gate_ptr
                    + (edge[:, None] * rank + rank_index[None, :]) * num_gates
                    + gate_lane1,
                    mask=edge_mask[:, None] & feature_mask[None, :],
                    other=0.0,
                )
                source = tl.load(
                    source1_ptr + sender[:, None] * width1 + feature[None, :],
                    mask=edge_mask[:, None] & feature_mask[None, :],
                    other=0.0,
                )
                accumulator += tl.sum(coefficient.to(tl.float32) * source, axis=0)
            tl.store(
                output1_ptr + row * width1 + feature,
                accumulator,
                mask=feature_mask,
            )

    @triton.jit
    def _local_tensor_reduce_pair_kernel(  # pragma: no cover - Triton JIT
        weight_ptr,
        radial_gate_ptr,
        even_source_ptr,
        odd_source_ptr,
        direction_ptr,
        sender_ptr,
        row_ptr,
        even_output_ptr,
        odd_output_ptr,
        rank: tl.constexpr,
        even_gate_lane: tl.constexpr,
        odd_gate_lane: tl.constexpr,
        num_gates: tl.constexpr,
        block_edges: tl.constexpr,
        block_features: tl.constexpr,
    ):
        """Fused l=2 sender transport plus the edge ST carrier."""

        row = tl.program_id(0)
        logical_block = tl.program_id(1)
        components: tl.constexpr = 5
        width: tl.constexpr = rank * components
        blocks: tl.constexpr = tl.cdiv(width, block_features)
        feature = (logical_block % blocks) * block_features + tl.arange(
            0, block_features
        )
        feature_mask = feature < width
        rank_index = feature // components
        component = feature % components
        start = tl.load(row_ptr + row).to(tl.int64)
        stop = tl.load(row_ptr + row + 1).to(tl.int64)
        accumulator = tl.zeros((block_features,), dtype=tl.float32)

        for base in tl.range(
            0,
            stop - start,
            block_edges,
            loop_unroll_factor=1,
        ):
            edge = start + base + tl.arange(0, block_edges)
            edge_mask = edge < stop
            mask = edge_mask[:, None] & feature_mask[None, :]
            sender = tl.load(sender_ptr + edge, mask=edge_mask, other=0).to(tl.int64)
            gate_lane = even_gate_lane
            if logical_block >= blocks:
                gate_lane = odd_gate_lane
            coefficient = tl.load(
                weight_ptr + edge[:, None] * rank + rank_index[None, :],
                mask=mask,
                other=0.0,
            ) * tl.load(
                radial_gate_ptr
                + (edge[:, None] * rank + rank_index[None, :]) * num_gates
                + gate_lane,
                mask=mask,
                other=0.0,
            )

            if logical_block < blocks:
                source = tl.load(
                    even_source_ptr
                    + sender[:, None] * width
                    + feature[None, :],
                    mask=mask,
                    other=0.0,
                )
                x = tl.load(
                    direction_ptr + edge * 3,
                    mask=edge_mask,
                    other=0.0,
                )
                y = tl.load(
                    direction_ptr + edge * 3 + 1,
                    mask=edge_mask,
                    other=0.0,
                )
                z = tl.load(
                    direction_ptr + edge * 3 + 2,
                    mask=edge_mask,
                    other=0.0,
                )
                trace_third = (x * x + y * y + z * z) / 3.0
                direction_tensor = tl.where(
                    component[None, :] == 0,
                    x[:, None] * x[:, None] - trace_third[:, None],
                    tl.where(
                        component[None, :] == 1,
                        y[:, None] * y[:, None] - trace_third[:, None],
                        tl.where(
                            component[None, :] == 2,
                            x[:, None] * y[:, None],
                            tl.where(
                                component[None, :] == 3,
                                x[:, None] * z[:, None],
                                y[:, None] * z[:, None],
                            ),
                        ),
                    ),
                )
                payload = source + direction_tensor
            else:
                payload = tl.load(
                    odd_source_ptr
                    + sender[:, None] * width
                    + feature[None, :],
                    mask=mask,
                    other=0.0,
                )
            accumulator += tl.sum(
                coefficient.to(tl.float32) * payload,
                axis=0,
            )

        if logical_block < blocks:
            tl.store(
                even_output_ptr + row * width + feature,
                accumulator,
                mask=feature_mask,
            )
        else:
            tl.store(
                odd_output_ptr + row * width + feature,
                accumulator,
                mask=feature_mask,
            )

    @triton.jit
    def _direction_reduce_triple_kernel(  # pragma: no cover - Triton JIT
        weight_ptr,
        radial_gate_ptr,
        direction_gate_ptr,
        direction_ptr,
        sender_ptr,
        row_ptr,
        output0_ptr,
        output1_ptr,
        output2_ptr,
        rank: tl.constexpr,
        gate_lane0: tl.constexpr,
        gate_lane1: tl.constexpr,
        gate_lane2: tl.constexpr,
        num_gates: tl.constexpr,
        block_edges: tl.constexpr,
        block_features: tl.constexpr,
    ):
        """Fuse the three direction-gated l=1 CSR moments."""

        row = tl.program_id(0)
        logical_block = tl.program_id(1)
        components: tl.constexpr = 3
        width: tl.constexpr = rank * components
        blocks: tl.constexpr = tl.cdiv(width, block_features)
        moment = logical_block // blocks
        feature = (logical_block % blocks) * block_features + tl.arange(
            0, block_features
        )
        feature_mask = feature < width
        rank_index = feature // components
        component = feature % components
        start = tl.load(row_ptr + row).to(tl.int64)
        stop = tl.load(row_ptr + row + 1).to(tl.int64)
        accumulator = tl.zeros((block_features,), dtype=tl.float32)

        for base in tl.range(
            0,
            stop - start,
            block_edges,
            loop_unroll_factor=1,
        ):
            edge = start + base + tl.arange(0, block_edges)
            edge_mask = edge < stop
            mask = edge_mask[:, None] & feature_mask[None, :]
            sender = tl.load(sender_ptr + edge, mask=edge_mask, other=0).to(tl.int64)
            gate_lane = gate_lane0
            if moment == 1:
                gate_lane = gate_lane1
            elif moment == 2:
                gate_lane = gate_lane2
            coefficient = tl.load(
                weight_ptr + edge[:, None] * rank + rank_index[None, :],
                mask=mask,
                other=0.0,
            ) * tl.load(
                radial_gate_ptr
                + (edge[:, None] * rank + rank_index[None, :]) * num_gates
                + gate_lane,
                mask=mask,
                other=0.0,
            ) * tl.load(
                direction_gate_ptr
                + (sender[:, None] * 3 + moment) * rank
                + rank_index[None, :],
                mask=mask,
                other=0.0,
            )
            direction = tl.load(
                direction_ptr + edge[:, None] * components + component[None, :],
                mask=mask,
                other=0.0,
            )
            accumulator += tl.sum(
                coefficient.to(tl.float32) * direction,
                axis=0,
            )

        if moment == 0:
            tl.store(
                output0_ptr + row * width + feature,
                accumulator,
                mask=feature_mask,
            )
        elif moment == 1:
            tl.store(
                output1_ptr + row * width + feature,
                accumulator,
                mask=feature_mask,
            )
        else:
            tl.store(
                output2_ptr + row * width + feature,
                accumulator,
                mask=feature_mask,
            )

    @triton.jit
    def _receiver_gather_pair_kernel(  # pragma: no cover - Triton JIT
        grad0_ptr,
        grad1_ptr,
        receiver_ptr,
        output0_ptr,
        output1_ptr,
        num_edges: tl.constexpr,
        num_features0: tl.constexpr,
        num_features1: tl.constexpr,
        block_edges: tl.constexpr,
        block_features: tl.constexpr,
    ):
        edge = tl.program_id(0) * block_edges + tl.arange(0, block_edges)
        edge_mask = edge < num_edges
        receiver = tl.load(receiver_ptr + edge, mask=edge_mask, other=0).to(tl.int64)
        # Promoted like ``receiver`` above: the store offset below is
        # ``edge * num_features``, which wraps in int32 past 2^31 elements.
        edge = edge.to(tl.int64)
        logical_block = tl.program_id(1)
        blocks0: tl.constexpr = tl.cdiv(num_features0, block_features)
        if logical_block < blocks0:
            feature = logical_block * block_features + tl.arange(0, block_features)
            mask = edge_mask[:, None] & (feature[None, :] < num_features0)
            value = tl.load(
                grad0_ptr + receiver[:, None] * num_features0 + feature[None, :],
                mask=mask,
            )
            tl.store(
                output0_ptr + edge[:, None] * num_features0 + feature[None, :],
                value,
                mask=mask,
            )
        else:
            feature = (
                logical_block - blocks0
            ) * block_features + tl.arange(0, block_features)
            mask = edge_mask[:, None] & (feature[None, :] < num_features1)
            value = tl.load(
                grad1_ptr + receiver[:, None] * num_features1 + feature[None, :],
                mask=mask,
            )
            tl.store(
                output1_ptr + edge[:, None] * num_features1 + feature[None, :],
                value,
                mask=mask,
            )

    @triton.jit
    def _receiver_gather_triple_kernel(  # pragma: no cover - Triton JIT
        grad0_ptr,
        grad1_ptr,
        grad2_ptr,
        receiver_ptr,
        output0_ptr,
        output1_ptr,
        output2_ptr,
        num_edges: tl.constexpr,
        num_features0: tl.constexpr,
        num_features1: tl.constexpr,
        num_features2: tl.constexpr,
        block_edges: tl.constexpr,
        block_features: tl.constexpr,
    ):
        edge = tl.program_id(0) * block_edges + tl.arange(0, block_edges)
        edge_mask = edge < num_edges
        receiver = tl.load(receiver_ptr + edge, mask=edge_mask, other=0).to(tl.int64)
        # Promoted like ``receiver`` above: the store offset below is
        # ``edge * num_features``, which wraps in int32 past 2^31 elements.
        edge = edge.to(tl.int64)
        logical_block = tl.program_id(1)
        blocks0: tl.constexpr = tl.cdiv(num_features0, block_features)
        blocks1: tl.constexpr = tl.cdiv(num_features1, block_features)
        if logical_block < blocks0:
            feature = logical_block * block_features + tl.arange(0, block_features)
            mask = edge_mask[:, None] & (feature[None, :] < num_features0)
            value = tl.load(
                grad0_ptr + receiver[:, None] * num_features0 + feature[None, :],
                mask=mask,
            )
            tl.store(
                output0_ptr + edge[:, None] * num_features0 + feature[None, :],
                value,
                mask=mask,
            )
        elif logical_block < blocks0 + blocks1:
            feature = (
                logical_block - blocks0
            ) * block_features + tl.arange(0, block_features)
            mask = edge_mask[:, None] & (feature[None, :] < num_features1)
            value = tl.load(
                grad1_ptr + receiver[:, None] * num_features1 + feature[None, :],
                mask=mask,
            )
            tl.store(
                output1_ptr + edge[:, None] * num_features1 + feature[None, :],
                value,
                mask=mask,
            )
        else:
            feature = (
                logical_block - blocks0 - blocks1
            ) * block_features + tl.arange(0, block_features)
            mask = edge_mask[:, None] & (feature[None, :] < num_features2)
            value = tl.load(
                grad2_ptr + receiver[:, None] * num_features2 + feature[None, :],
                mask=mask,
            )
            tl.store(
                output2_ptr + edge[:, None] * num_features2 + feature[None, :],
                value,
                mask=mask,
            )

    @triton.jit
    def _csr_broadcast_kernel(  # pragma: no cover - Triton JIT
        grad_ptr,
        row_ptr,
        output_ptr,
        num_features: tl.constexpr,
        block_edges: tl.constexpr,
        block_features: tl.constexpr,
    ):
        row = tl.program_id(0)
        feature = tl.program_id(1) * block_features + tl.arange(0, block_features)
        feature_mask = feature < num_features
        value = tl.load(
            grad_ptr + row * num_features + feature,
            mask=feature_mask,
            other=0.0,
        )
        start = tl.load(row_ptr + row).to(tl.int64)
        stop = tl.load(row_ptr + row + 1).to(tl.int64)
        for base in tl.range(
            0,
            stop - start,
            block_edges,
            loop_unroll_factor=1,
        ):
            edge = start + base + tl.arange(0, block_edges)
            mask = (edge[:, None] < stop) & feature_mask[None, :]
            tl.store(
                output_ptr + edge[:, None] * num_features + feature[None, :],
                value[None, :],
                mask=mask,
            )


class _TorchCsrSum(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        value: torch.Tensor,
        row_ptr: torch.Tensor,
    ) -> torch.Tensor:
        reduce_value = (
            value.float() if value.dtype in {torch.float16, torch.bfloat16} else value
        )
        reduced = torch.segment_reduce(
            reduce_value,
            reduce="sum",
            offsets=row_ptr,
        )
        result = reduced.to(dtype=value.dtype)
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(value, reduced, row_ptr)
        ctx.input_size = value.shape[0]
        return result

    @staticmethod
    def backward(
        ctx: object,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        value, reduced, row_ptr = ctx.saved_tensors
        if torch.is_grad_enabled():
            counts = (row_ptr[1:] - row_ptr[:-1]).to(dtype=torch.long)
            grad_value = torch.repeat_interleave(
                grad_output,
                counts,
                dim=0,
                output_size=ctx.input_size,
            )
        else:
            # Match the pristine reference: ordinary first-order training uses
            # PyTorch's native fast backward. The differentiable gather above
            # is reserved for create_graph=True, where native segment-reduce
            # does not provide the required gradgrad contract.
            promoted = value.dtype in {torch.float16, torch.bfloat16}
            reduce_value = value.float() if promoted else value
            reduce_grad = grad_output.float() if promoted else grad_output
            grad_value = torch.ops.aten._segment_reduce_backward.default(
                reduce_grad,
                reduced,
                reduce_value,
                "sum",
                offsets=row_ptr,
            ).to(dtype=value.dtype)
        return grad_value, None


class _TritonCsrSum(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        value: torch.Tensor,
        row_ptr: torch.Tensor,
    ) -> torch.Tensor:
        if triton is None:
            raise RuntimeError("Triton is unavailable")
        value_contiguous = value.contiguous()
        rows = row_ptr.numel() - 1
        flattened = value_contiguous.reshape(value.shape[0], -1)
        features = flattened.shape[1]
        output = torch.empty(
            (rows, features),
            device=value.device,
            dtype=value.dtype,
        )
        if rows and features:
            block_features = min(32, triton.next_power_of_2(features))
            block_edges = 64
            grid = (rows, triton.cdiv(features, block_features))
            _csr_sum_kernel[grid](
                flattened,
                row_ptr,
                output,
                num_features=features,
                block_edges=block_edges,
                block_features=block_features,
                num_warps=4,
            )
        ctx.save_for_backward(row_ptr)
        ctx.input_size = value.shape[0]
        return output.reshape(rows, *value.shape[1:])

    @staticmethod
    def backward(
        ctx: object,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        (row_ptr,) = ctx.saved_tensors
        if torch.is_grad_enabled():
            counts = (row_ptr[1:] - row_ptr[:-1]).to(dtype=torch.long)
            grad_value = torch.repeat_interleave(
                grad_output,
                counts,
                dim=0,
                output_size=ctx.input_size,
            )
        else:
            if triton is None:  # pragma: no cover - guarded by forward
                raise RuntimeError("Triton is unavailable")
            contiguous = grad_output.contiguous().reshape(grad_output.shape[0], -1)
            features = contiguous.shape[1]
            grad_value = torch.empty(
                (ctx.input_size, features),
                device=grad_output.device,
                dtype=grad_output.dtype,
            )
            if ctx.input_size and features:
                block_features = min(32, triton.next_power_of_2(features))
                grid = (
                    row_ptr.numel() - 1,
                    triton.cdiv(features, block_features),
                )
                _csr_broadcast_kernel[grid](
                    contiguous,
                    row_ptr,
                    grad_value,
                    num_features=features,
                    block_edges=128,
                    block_features=block_features,
                    num_warps=1,
                )
            grad_value = grad_value.reshape(ctx.input_size, *grad_output.shape[1:])
        return grad_value, None


def _receiver_gradients(
    grad_outputs: Sequence[torch.Tensor | None],
    row_ptr: torch.Tensor,
    *,
    input_size: int,
    receiver: torch.Tensor | None,
) -> tuple[torch.Tensor | None, ...]:
    active = tuple(
        (index, grad_output)
        for index, grad_output in enumerate(grad_outputs)
        if grad_output is not None
    )
    if not active:
        return tuple(None for _ in grad_outputs)

    widths = tuple(grad_output.shape[1] for _, grad_output in active)
    packed = (
        active[0][1]
        if len(active) == 1
        else torch.cat(tuple(grad_output for _, grad_output in active), dim=-1)
    )
    if receiver is not None:
        gathered = packed.index_select(0, receiver)
    else:
        counts = (row_ptr[1:] - row_ptr[:-1]).to(dtype=torch.long)
        gathered = torch.repeat_interleave(
            packed,
            counts,
            dim=0,
            output_size=input_size,
        )
    parts = torch.split(gathered, widths, dim=-1)
    gradients: list[torch.Tensor | None] = [None] * len(grad_outputs)
    for (index, _), part in zip(active, parts, strict=True):
        gradients[index] = part
    return tuple(gradients)


def _triton_receiver_gradients(  # pragma: no cover - CUDA-only dispatch
    grad_outputs: Sequence[torch.Tensor],
    receiver: torch.Tensor,
    *,
    input_size: int,
) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    contiguous = tuple(grad_output.contiguous() for grad_output in grad_outputs)
    widths = tuple(grad_output.shape[1] for grad_output in contiguous)
    outputs = tuple(
        torch.empty(
            (input_size, width),
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        for grad_output, width in zip(contiguous, widths, strict=True)
    )
    if input_size == 0:
        return outputs
    block_edges = 128
    block_features = min(32, triton.next_power_of_2(max(widths)))
    grid = (
        triton.cdiv(input_size, block_edges),
        sum(triton.cdiv(width, block_features) for width in widths),
    )
    if len(contiguous) == 2:
        _receiver_gather_pair_kernel[grid](
            contiguous[0],
            contiguous[1],
            receiver,
            outputs[0],
            outputs[1],
            num_edges=input_size,
            num_features0=widths[0],
            num_features1=widths[1],
            block_edges=block_edges,
            block_features=block_features,
            num_warps=1,
        )
    elif len(contiguous) == 3:
        _receiver_gather_triple_kernel[grid](
            contiguous[0],
            contiguous[1],
            contiguous[2],
            receiver,
            outputs[0],
            outputs[1],
            outputs[2],
            num_edges=input_size,
            num_features0=widths[0],
            num_features1=widths[1],
            num_features2=widths[2],
            block_edges=block_edges,
            block_features=block_features,
            num_warps=1,
        )
    else:  # pragma: no cover - internal arity contract
        raise ValueError("Triton receiver gather supports pair or triple payloads")
    return outputs


class _TritonCsrSumPair(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        value0: torch.Tensor,
        value1: torch.Tensor,
        row_ptr: torch.Tensor,
        receiver: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if triton is None:
            raise RuntimeError("Triton is unavailable")
        ctx.set_materialize_grads(False)
        value0 = value0.contiguous()
        value1 = value1.contiguous()
        rows = row_ptr.numel() - 1
        features0 = value0.shape[1]
        features1 = value1.shape[1]
        output0 = torch.empty(
            (rows, features0),
            device=value0.device,
            dtype=value0.dtype,
        )
        output1 = torch.empty(
            (rows, features1),
            device=value1.device,
            dtype=value1.dtype,
        )
        if rows:
            block_features = 32
            grid = (
                rows,
                triton.cdiv(features0, block_features)
                + triton.cdiv(features1, block_features),
            )
            _csr_sum_pair_kernel[grid](
                value0,
                value1,
                row_ptr,
                output0,
                output1,
                num_features0=features0,
                num_features1=features1,
                block_edges=64,
                block_features=block_features,
                num_warps=4,
            )
        if receiver is None:
            ctx.save_for_backward(row_ptr)
        else:
            ctx.save_for_backward(row_ptr, receiver)
        ctx.has_receiver = receiver is not None
        ctx.input_size = value0.shape[0]
        return output0, output1

    @staticmethod
    def backward(
        ctx: object,
        grad_output0: torch.Tensor | None,
        grad_output1: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
    ]:
        row_ptr = ctx.saved_tensors[0]
        receiver = ctx.saved_tensors[1] if ctx.has_receiver else None
        if (
            not torch.is_grad_enabled()
            and receiver is not None
            and grad_output0 is not None
            and grad_output1 is not None
        ):
            gradients = _triton_receiver_gradients(
                (grad_output0, grad_output1),
                receiver,
                input_size=ctx.input_size,
            )
        else:
            gradients = _receiver_gradients(
                (grad_output0, grad_output1),
                row_ptr,
                input_size=ctx.input_size,
                receiver=receiver,
            )
        return gradients[0], gradients[1], None, None


class _TritonCsrSumTriple(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        value0: torch.Tensor,
        value1: torch.Tensor,
        value2: torch.Tensor,
        row_ptr: torch.Tensor,
        receiver: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if triton is None:
            raise RuntimeError("Triton is unavailable")
        ctx.set_materialize_grads(False)
        value0 = value0.contiguous()
        value1 = value1.contiguous()
        value2 = value2.contiguous()
        rows = row_ptr.numel() - 1
        features0 = value0.shape[1]
        features1 = value1.shape[1]
        features2 = value2.shape[1]
        output0 = torch.empty(
            (rows, features0),
            device=value0.device,
            dtype=value0.dtype,
        )
        output1 = torch.empty(
            (rows, features1),
            device=value1.device,
            dtype=value1.dtype,
        )
        output2 = torch.empty(
            (rows, features2),
            device=value2.device,
            dtype=value2.dtype,
        )
        if rows:
            block_features = 32
            grid = (
                rows,
                triton.cdiv(features0, block_features)
                + triton.cdiv(features1, block_features)
                + triton.cdiv(features2, block_features),
            )
            _csr_sum_triple_kernel[grid](
                value0,
                value1,
                value2,
                row_ptr,
                output0,
                output1,
                output2,
                num_features0=features0,
                num_features1=features1,
                num_features2=features2,
                block_edges=64,
                block_features=block_features,
                num_warps=4,
            )
        if receiver is None:
            ctx.save_for_backward(row_ptr)
        else:
            ctx.save_for_backward(row_ptr, receiver)
        ctx.has_receiver = receiver is not None
        ctx.input_size = value0.shape[0]
        return output0, output1, output2

    @staticmethod
    def backward(
        ctx: object,
        grad_output0: torch.Tensor | None,
        grad_output1: torch.Tensor | None,
        grad_output2: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
    ]:
        row_ptr = ctx.saved_tensors[0]
        receiver = ctx.saved_tensors[1] if ctx.has_receiver else None
        if (
            not torch.is_grad_enabled()
            and receiver is not None
            and grad_output0 is not None
            and grad_output1 is not None
            and grad_output2 is not None
        ):
            gradients = _triton_receiver_gradients(
                (grad_output0, grad_output1, grad_output2),
                receiver,
                input_size=ctx.input_size,
            )
        else:
            gradients = _receiver_gradients(
                (grad_output0, grad_output1, grad_output2),
                row_ptr,
                input_size=ctx.input_size,
                receiver=receiver,
            )
        return gradients[0], gradients[1], gradients[2], None, None


def _validated_row_ptr(
    value: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    trusted: bool,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("CSR values must be a tensor")
    if value.ndim == 0:
        raise ValueError("CSR values must have an edge dimension")
    if not isinstance(row_ptr, torch.Tensor):
        raise TypeError("row_ptr must be a tensor")
    if row_ptr.ndim != 1 or row_ptr.numel() == 0:
        raise ValueError("row_ptr must be a non-empty one-dimensional tensor")
    if row_ptr.dtype not in {torch.int32, torch.int64}:
        raise TypeError("row_ptr must use int32 or int64")
    if row_ptr.device != value.device:
        raise ValueError("row_ptr and CSR values must use the same device")
    contiguous = row_ptr.contiguous()
    if trusted:
        return contiguous
    if int(contiguous[0].item()) != 0:
        raise ValueError("row_ptr must start at zero")
    if int(contiguous[-1].item()) != value.shape[0]:
        raise ValueError("row_ptr must end at the CSR edge count")
    if bool((contiguous[1:] < contiguous[:-1]).any().item()):
        raise ValueError("row_ptr must be nondecreasing")
    return contiguous


def _csr_sum_resolved(
    value: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    policy: str,
) -> torch.Tensor:
    if policy == "torch":
        return _TorchCsrSum.apply(value, row_ptr)
    if _can_use_triton(value, row_ptr, policy=policy):
        return _TritonCsrSum.apply(value, row_ptr)
    return _TorchCsrSum.apply(value, row_ptr)


def csr_sum(
    value: torch.Tensor,
    row_ptr: torch.Tensor,
) -> torch.Tensor:
    """Safely validate and reduce arbitrary CSR rows."""

    contiguous = _validated_row_ptr(value, row_ptr, trusted=False)
    return _csr_sum_resolved(value, contiguous, policy=backend_policy())


def _trusted_csr_sum(
    value: torch.Tensor,
    row_ptr: torch.Tensor,
) -> torch.Tensor:
    """Reduce a CSR plan already validated by ``PackedNeighborGraph``."""

    contiguous = _validated_row_ptr(value, row_ptr, trusted=True)
    return _csr_sum_resolved(value, contiguous, policy=backend_policy())


def csr_sum_many(
    values: Sequence[torch.Tensor],
    row_ptr: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Safely reduce one lifetime-compatible payload group."""

    return _csr_sum_many(
        values,
        row_ptr,
        trusted=False,
        policy=backend_policy(),
        receiver=None,
    )


def _trusted_csr_sum_many(
    values: Sequence[torch.Tensor],
    row_ptr: torch.Tensor,
    *,
    policy: str,
    receiver: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Reduce payloads over a previously validated immutable CSR plan."""

    return _csr_sum_many(
        values,
        row_ptr,
        trusted=True,
        policy=policy,
        receiver=receiver,
    )


def _trusted_weighted_gather_reduce_pair(  # pragma: no cover - CUDA-only dispatch
    weight: torch.Tensor,
    radial_gate: torch.Tensor,
    source0: torch.Tensor,
    source1: torch.Tensor,
    sender: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    gate_lanes: tuple[int, int],
    policy: str,
    receiver: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse two sender gathers, invariant weights, and CSR sums.

    ``weight`` and every selected radial-gate lane are trusted ``0e`` scalars.
    ``rank`` is a multiplicity axis, so one coefficient is applied unchanged to
    every carrier component. ``sender``, weights, gates, and receiver-major CSR
    rows must share the immutable packed-edge order already validated by
    ``PackedNeighborGraph``. The operator is homogeneous (`N_source == N_rows`)
    During training the custom autograd wrapper saves node values and compact
    edge coefficients, not the expanded ``[E, R, C]`` message payloads.  Its
    backward is expressed with differentiable PyTorch gathers/scatters so
    first- and higher-order derivatives keep the reference semantics.
    """

    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if weight.ndim != 2 or radial_gate.ndim != 3:
        raise ValueError("weight and radial_gate must have [E,R] and [E,R,G] shapes")
    if radial_gate.shape[:2] != weight.shape:
        raise ValueError("weight and radial_gate edge/rank dimensions must match")
    if source0.ndim < 2 or source1.ndim < 2:
        raise ValueError("weighted sources must have [N,R,...] shapes")
    contiguous_row_ptr = _validated_row_ptr(weight, row_ptr, trusted=True)
    rows = contiguous_row_ptr.numel() - 1
    rank = weight.shape[1]
    if source0.shape[:2] != (rows, rank) or source1.shape[:2] != (rows, rank):
        raise ValueError("weighted sources must match the CSR node/rank dimensions")
    tensors = (radial_gate, source0, source1)
    if any(tensor.device != weight.device or tensor.dtype != weight.dtype for tensor in tensors):
        raise ValueError("weighted gather-reduce tensors must share device and dtype")
    if sender.ndim != 1 or sender.shape[0] != weight.shape[0]:
        raise ValueError("sender must match the CSR edge dimension")
    if sender.device != weight.device or sender.dtype not in {torch.int32, torch.int64}:
        raise ValueError("sender must use the CSR device and integer dtype")
    if receiver is not None:
        if receiver.ndim != 1 or receiver.shape[0] != weight.shape[0]:
            raise ValueError("receiver must match the CSR edge dimension")
        if receiver.device != weight.device or receiver.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise ValueError("receiver must use the CSR device and integer dtype")
        receiver = receiver.contiguous()
    if any(lane < 0 or lane >= radial_gate.shape[2] for lane in gate_lanes):
        raise ValueError("gate lanes must index the radial-gate dimension")
    if not _can_use_triton(weight, contiguous_row_ptr, policy=policy):
        raise RuntimeError("weighted gather-reduce requires the forced Triton backend")

    if torch.is_grad_enabled() and any(
        value.requires_grad for value in (weight, radial_gate, source0, source1)
    ):
        return _TritonWeightedGatherReducePair.apply(
            weight,
            radial_gate,
            source0,
            source1,
            sender.contiguous(),
            contiguous_row_ptr,
            receiver,
            gate_lanes[0],
            gate_lanes[1],
        )
    return _launch_weighted_gather_reduce_pair(
        weight,
        radial_gate,
        source0,
        source1,
        sender,
        contiguous_row_ptr,
        gate_lanes=gate_lanes,
    )


def _launch_weighted_gather_reduce_pair(  # pragma: no cover - CUDA-only dispatch
    weight: torch.Tensor,
    radial_gate: torch.Tensor,
    source0: torch.Tensor,
    source1: torch.Tensor,
    sender: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    gate_lanes: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the validated forward-only Triton primitive."""

    rows = row_ptr.numel() - 1
    rank = weight.shape[1]
    components0 = prod(source0.shape[2:]) if source0.ndim > 2 else 1
    components1 = prod(source1.shape[2:]) if source1.ndim > 2 else 1
    width0 = rank * components0
    width1 = rank * components1
    if not width0 or not width1:
        raise ValueError("weighted source feature dimensions must be nonzero")
    output0 = torch.empty(
        (rows, width0),
        device=weight.device,
        dtype=weight.dtype,
    )
    output1 = torch.empty(
        (rows, width1),
        device=weight.device,
        dtype=weight.dtype,
    )
    if rows:
        block_features = min(32, triton.next_power_of_2(max(width0, width1)))
        grid = (
            rows,
            triton.cdiv(width0, block_features)
            + triton.cdiv(width1, block_features),
        )
        _weighted_gather_reduce_pair_kernel[grid](
            weight.contiguous(),
            radial_gate.contiguous(),
            source0.contiguous(),
            source1.contiguous(),
            sender.contiguous(),
            row_ptr,
            output0,
            output1,
            rank=rank,
            components0=components0,
            components1=components1,
            gate_lane0=gate_lanes[0],
            gate_lane1=gate_lanes[1],
            num_gates=radial_gate.shape[2],
            block_edges=64,
            block_features=block_features,
            num_warps=4,
        )
    return (
        output0.reshape(rows, *source0.shape[1:]),
        output1.reshape(rows, *source1.shape[1:]),
    )


class _TritonWeightedGatherReducePair(torch.autograd.Function):
    """Memory-bounded Triton forward with an exact differentiable backward."""

    @staticmethod
    def forward(
        ctx: object,
        weight: torch.Tensor,
        radial_gate: torch.Tensor,
        source0: torch.Tensor,
        source1: torch.Tensor,
        sender: torch.Tensor,
        row_ptr: torch.Tensor,
        receiver: torch.Tensor | None,
        gate_lane0: int,
        gate_lane1: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)
        if receiver is None:
            ctx.save_for_backward(
                weight,
                radial_gate,
                source0,
                source1,
                sender,
                row_ptr,
            )
        else:
            ctx.save_for_backward(
                weight,
                radial_gate,
                source0,
                source1,
                sender,
                row_ptr,
                receiver,
            )
        ctx.has_receiver = receiver is not None
        ctx.gate_lanes = (gate_lane0, gate_lane1)
        return _launch_weighted_gather_reduce_pair(
            weight,
            radial_gate,
            source0,
            source1,
            sender,
            row_ptr,
            gate_lanes=(gate_lane0, gate_lane1),
        )

    @staticmethod
    def backward(
        ctx: object,
        grad_output0: torch.Tensor | None,
        grad_output1: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
    ]:
        weight, radial_gate, source0, source1, sender, row_ptr = (
            ctx.saved_tensors[:6]
        )
        if ctx.has_receiver:
            receiver = ctx.saved_tensors[6]
        else:
            counts = (row_ptr[1:] - row_ptr[:-1]).to(dtype=torch.long)
            receiver = torch.repeat_interleave(
                torch.arange(
                    counts.shape[0],
                    device=row_ptr.device,
                    dtype=torch.long,
                ),
                counts,
                output_size=weight.shape[0],
            )
        sender_index = sender.to(dtype=torch.long)
        receiver_index = receiver.to(dtype=torch.long)
        lane0, lane1 = ctx.gate_lanes
        gate0 = radial_gate[..., lane0]
        gate1 = radial_gate[..., lane1]
        coefficient0 = weight * gate0
        coefficient1 = weight * gate1

        source0_flat = source0.reshape(source0.shape[0], source0.shape[1], -1)
        source1_flat = source1.reshape(source1.shape[0], source1.shape[1], -1)
        edge_source0 = source0_flat.index_select(0, sender_index)
        edge_source1 = source1_flat.index_select(0, sender_index)

        if grad_output0 is None:
            edge_grad0 = torch.zeros_like(edge_source0)
        else:
            edge_grad0 = grad_output0.reshape(
                grad_output0.shape[0], grad_output0.shape[1], -1
            ).index_select(0, receiver_index)
        if grad_output1 is None:
            edge_grad1 = torch.zeros_like(edge_source1)
        else:
            edge_grad1 = grad_output1.reshape(
                grad_output1.shape[0], grad_output1.shape[1], -1
            ).index_select(0, receiver_index)

        grad_coefficient0 = (edge_grad0 * edge_source0).sum(dim=-1)
        grad_coefficient1 = (edge_grad1 * edge_source1).sum(dim=-1)
        grad_weight = grad_coefficient0 * gate0 + grad_coefficient1 * gate1

        lane_basis0 = torch.nn.functional.one_hot(
            torch.tensor(lane0, device=weight.device),
            num_classes=radial_gate.shape[-1],
        ).to(dtype=weight.dtype)
        lane_basis1 = torch.nn.functional.one_hot(
            torch.tensor(lane1, device=weight.device),
            num_classes=radial_gate.shape[-1],
        ).to(dtype=weight.dtype)
        grad_radial_gate = weight.unsqueeze(-1) * (
            grad_coefficient0.unsqueeze(-1) * lane_basis0
            + grad_coefficient1.unsqueeze(-1) * lane_basis1
        )

        grad_source0_flat = torch.zeros_like(source0_flat).index_add(
            0,
            sender_index,
            coefficient0.unsqueeze(-1) * edge_grad0,
        )
        grad_source1_flat = torch.zeros_like(source1_flat).index_add(
            0,
            sender_index,
            coefficient1.unsqueeze(-1) * edge_grad1,
        )
        return (
            grad_weight,
            grad_radial_gate,
            grad_source0_flat.reshape_as(source0),
            grad_source1_flat.reshape_as(source1),
            None,
            None,
            None,
            None,
            None,
        )


def _validate_local_fused_plan(
    weight: torch.Tensor,
    radial_gate: torch.Tensor,
    sender: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    gate_lanes: tuple[int, ...],
    policy: str,
    receiver: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int, int]:
    """Validate the shared compact inputs for private local fused kernels."""

    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if weight.ndim != 2 or radial_gate.ndim != 3:
        raise ValueError("weight and radial_gate must have [E,R] and [E,R,G] shapes")
    if radial_gate.shape[:2] != weight.shape:
        raise ValueError("weight and radial_gate edge/rank dimensions must match")
    if radial_gate.device != weight.device or radial_gate.dtype != weight.dtype:
        raise ValueError("fused local tensors must share device and dtype")
    if sender.ndim != 1 or sender.shape[0] != weight.shape[0]:
        raise ValueError("sender must match the CSR edge dimension")
    if sender.device != weight.device or sender.dtype not in {torch.int32, torch.int64}:
        raise ValueError("sender must use the CSR device and integer dtype")
    if receiver is not None:
        if receiver.ndim != 1 or receiver.shape[0] != weight.shape[0]:
            raise ValueError("receiver must match the CSR edge dimension")
        if receiver.device != weight.device or receiver.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise ValueError("receiver must use the CSR device and integer dtype")
        receiver = receiver.contiguous()
    if any(lane < 0 or lane >= radial_gate.shape[2] for lane in gate_lanes):
        raise ValueError("gate lanes must index the radial-gate dimension")
    contiguous_row_ptr = _validated_row_ptr(weight, row_ptr, trusted=True)
    if not _can_use_triton(weight, contiguous_row_ptr, policy=policy):
        raise RuntimeError("fused local reduction requires the forced Triton backend")
    return (
        contiguous_row_ptr,
        sender.contiguous(),
        receiver,
        contiguous_row_ptr.numel() - 1,
        weight.shape[1],
    )


def _direction_to_st(direction: torch.Tensor) -> torch.Tensor:
    """Reference ST map used only by the differentiable custom backward."""

    x, y, z = direction.unbind(dim=-1)
    trace_third = (x.square() + y.square() + z.square()) / 3.0
    return torch.stack(
        (
            x.square() - trace_third,
            y.square() - trace_third,
            x * y,
            x * z,
            y * z,
        ),
        dim=-1,
    )


def _trusted_local_tensor_reduce_pair(  # pragma: no cover - CUDA-only dispatch
    weight: torch.Tensor,
    radial_gate: torch.Tensor,
    even_source: torch.Tensor,
    odd_source: torch.Tensor,
    direction: torch.Tensor,
    sender: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    gate_lanes: tuple[int, int],
    policy: str,
    receiver: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse the two l=2 transports, including the edge-direction ST term."""

    row_ptr, sender, receiver, rows, rank = _validate_local_fused_plan(
        weight,
        radial_gate,
        sender,
        row_ptr,
        gate_lanes=gate_lanes,
        policy=policy,
        receiver=receiver,
    )
    if even_source.shape != (rows, rank, 5) or odd_source.shape != (rows, rank, 5):
        raise ValueError("tensor sources must have [N,R,5] shapes")
    if direction.shape != (weight.shape[0], 3):
        raise ValueError("direction must have [E,3] shape")
    if any(
        tensor.device != weight.device or tensor.dtype != weight.dtype
        for tensor in (even_source, odd_source, direction)
    ):
        raise ValueError("fused local tensors must share device and dtype")

    if torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (weight, radial_gate, even_source, odd_source, direction)
    ):
        return _TritonLocalTensorReducePair.apply(
            weight,
            radial_gate,
            even_source,
            odd_source,
            direction,
            sender,
            row_ptr,
            receiver,
            gate_lanes[0],
            gate_lanes[1],
        )
    return _launch_local_tensor_reduce_pair(
        weight,
        radial_gate,
        even_source,
        odd_source,
        direction,
        sender,
        row_ptr,
        gate_lanes=gate_lanes,
    )


def _launch_local_tensor_reduce_pair(  # pragma: no cover - CUDA-only dispatch
    weight: torch.Tensor,
    radial_gate: torch.Tensor,
    even_source: torch.Tensor,
    odd_source: torch.Tensor,
    direction: torch.Tensor,
    sender: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    gate_lanes: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the validated forward-only l=2 fused primitive."""

    if triton is None:
        raise RuntimeError("Triton is unavailable")
    rows = row_ptr.numel() - 1
    rank = weight.shape[1]
    width = rank * 5
    outputs = tuple(
        torch.empty((rows, width), device=weight.device, dtype=weight.dtype)
        for _ in range(2)
    )
    if rows:
        block_features = min(32, triton.next_power_of_2(width))
        blocks = triton.cdiv(width, block_features)
        _local_tensor_reduce_pair_kernel[(rows, 2 * blocks)](
            weight.contiguous(),
            radial_gate.contiguous(),
            even_source.contiguous(),
            odd_source.contiguous(),
            direction.contiguous(),
            sender,
            row_ptr,
            outputs[0],
            outputs[1],
            rank=rank,
            even_gate_lane=gate_lanes[0],
            odd_gate_lane=gate_lanes[1],
            num_gates=radial_gate.shape[2],
            block_edges=64,
            block_features=block_features,
            num_warps=4,
        )
    return outputs[0].reshape(rows, rank, 5), outputs[1].reshape(rows, rank, 5)


class _TritonLocalTensorReducePair(torch.autograd.Function):
    """Fused l=2 forward with exact, differentiable PyTorch recomputation."""

    @staticmethod
    def forward(
        ctx: object,
        weight: torch.Tensor,
        radial_gate: torch.Tensor,
        even_source: torch.Tensor,
        odd_source: torch.Tensor,
        direction: torch.Tensor,
        sender: torch.Tensor,
        row_ptr: torch.Tensor,
        receiver: torch.Tensor | None,
        even_gate_lane: int,
        odd_gate_lane: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)
        saved = (
            weight,
            radial_gate,
            even_source,
            odd_source,
            direction,
            sender,
            row_ptr,
        )
        ctx.save_for_backward(*(saved if receiver is None else (*saved, receiver)))
        ctx.has_receiver = receiver is not None
        ctx.gate_lanes = (even_gate_lane, odd_gate_lane)
        return _launch_local_tensor_reduce_pair(
            weight,
            radial_gate,
            even_source,
            odd_source,
            direction,
            sender,
            row_ptr,
            gate_lanes=(even_gate_lane, odd_gate_lane),
        )

    @staticmethod
    def backward(
        ctx: object,
        grad_even: torch.Tensor | None,
        grad_odd: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
    ]:
        weight, radial_gate, even_source, odd_source, direction, sender, row_ptr = (
            ctx.saved_tensors[:7]
        )
        if ctx.has_receiver:
            receiver = ctx.saved_tensors[7]
        else:
            counts = (row_ptr[1:] - row_ptr[:-1]).to(dtype=torch.long)
            receiver = torch.repeat_interleave(
                torch.arange(counts.shape[0], device=row_ptr.device),
                counts,
                output_size=weight.shape[0],
            )
        sender_index = sender.to(dtype=torch.long)
        receiver_index = receiver.to(dtype=torch.long)
        even_lane, odd_lane = ctx.gate_lanes
        even_gate = radial_gate[..., even_lane]
        odd_gate = radial_gate[..., odd_lane]
        even_coefficient = weight * even_gate
        odd_coefficient = weight * odd_gate

        edge_even = even_source.index_select(0, sender_index)
        edge_odd = odd_source.index_select(0, sender_index)
        direction_tensor = _direction_to_st(direction).unsqueeze(1)
        augmented_even = edge_even + direction_tensor
        edge_grad_even = (
            torch.zeros_like(augmented_even)
            if grad_even is None
            else grad_even.index_select(0, receiver_index)
        )
        edge_grad_odd = (
            torch.zeros_like(edge_odd)
            if grad_odd is None
            else grad_odd.index_select(0, receiver_index)
        )

        grad_even_coefficient = (edge_grad_even * augmented_even).sum(dim=-1)
        grad_odd_coefficient = (edge_grad_odd * edge_odd).sum(dim=-1)
        grad_weight = (
            grad_even_coefficient * even_gate + grad_odd_coefficient * odd_gate
        )
        even_basis = torch.nn.functional.one_hot(
            torch.tensor(even_lane, device=weight.device),
            num_classes=radial_gate.shape[-1],
        ).to(dtype=weight.dtype)
        odd_basis = torch.nn.functional.one_hot(
            torch.tensor(odd_lane, device=weight.device),
            num_classes=radial_gate.shape[-1],
        ).to(dtype=weight.dtype)
        grad_radial_gate = weight.unsqueeze(-1) * (
            grad_even_coefficient.unsqueeze(-1) * even_basis
            + grad_odd_coefficient.unsqueeze(-1) * odd_basis
        )

        grad_even_edge = even_coefficient.unsqueeze(-1) * edge_grad_even
        grad_odd_edge = odd_coefficient.unsqueeze(-1) * edge_grad_odd
        grad_even_source = torch.zeros_like(even_source).index_add(
            0, sender_index, grad_even_edge
        )
        grad_odd_source = torch.zeros_like(odd_source).index_add(
            0, sender_index, grad_odd_edge
        )

        st_gradient = grad_even_edge.sum(dim=1)
        g_xx, g_yy, g_xy, g_xz, g_yz = st_gradient.unbind(dim=-1)
        x, y, z = direction.unbind(dim=-1)
        grad_direction = torch.stack(
            (
                ((4.0 * g_xx - 2.0 * g_yy) / 3.0) * x + g_xy * y + g_xz * z,
                ((-2.0 * g_xx + 4.0 * g_yy) / 3.0) * y + g_xy * x + g_yz * z,
                (-2.0 * (g_xx + g_yy) / 3.0) * z + g_xz * x + g_yz * y,
            ),
            dim=-1,
        )
        return (
            grad_weight,
            grad_radial_gate,
            grad_even_source,
            grad_odd_source,
            grad_direction,
            None,
            None,
            None,
            None,
            None,
        )


def _trusted_direction_reduce_triple(  # pragma: no cover - CUDA-only dispatch
    weight: torch.Tensor,
    radial_gate: torch.Tensor,
    direction_gate: torch.Tensor,
    direction: torch.Tensor,
    sender: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    gate_lanes: tuple[int, int, int],
    policy: str,
    receiver: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse all three direction-gated l=1 receiver moments."""

    row_ptr, sender, receiver, rows, rank = _validate_local_fused_plan(
        weight,
        radial_gate,
        sender,
        row_ptr,
        gate_lanes=gate_lanes,
        policy=policy,
        receiver=receiver,
    )
    if direction_gate.shape != (rows, 3, rank):
        raise ValueError("direction_gate must have [N,3,R] shape")
    if direction.shape != (weight.shape[0], 3):
        raise ValueError("direction must have [E,3] shape")
    if any(
        tensor.device != weight.device or tensor.dtype != weight.dtype
        for tensor in (direction_gate, direction)
    ):
        raise ValueError("fused local tensors must share device and dtype")

    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (weight, radial_gate, direction_gate, direction)
    ):
        return _TritonDirectionReduceTriple.apply(
            weight,
            radial_gate,
            direction_gate,
            direction,
            sender,
            row_ptr,
            receiver,
            *gate_lanes,
        )
    return _launch_direction_reduce_triple(
        weight,
        radial_gate,
        direction_gate,
        direction,
        sender,
        row_ptr,
        gate_lanes=gate_lanes,
    )


def _launch_direction_reduce_triple(  # pragma: no cover - CUDA-only dispatch
    weight: torch.Tensor,
    radial_gate: torch.Tensor,
    direction_gate: torch.Tensor,
    direction: torch.Tensor,
    sender: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    gate_lanes: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the validated forward-only directional fused primitive."""

    if triton is None:
        raise RuntimeError("Triton is unavailable")
    rows = row_ptr.numel() - 1
    rank = weight.shape[1]
    width = rank * 3
    outputs = tuple(
        torch.empty((rows, width), device=weight.device, dtype=weight.dtype)
        for _ in range(3)
    )
    if rows:
        block_features = min(32, triton.next_power_of_2(width))
        blocks = triton.cdiv(width, block_features)
        _direction_reduce_triple_kernel[(rows, 3 * blocks)](
            weight.contiguous(),
            radial_gate.contiguous(),
            direction_gate.contiguous(),
            direction.contiguous(),
            sender,
            row_ptr,
            outputs[0],
            outputs[1],
            outputs[2],
            rank=rank,
            gate_lane0=gate_lanes[0],
            gate_lane1=gate_lanes[1],
            gate_lane2=gate_lanes[2],
            num_gates=radial_gate.shape[2],
            block_edges=64,
            block_features=block_features,
            num_warps=4,
        )
    shape = (rows, rank, 3)
    return tuple(output.reshape(shape) for output in outputs)  # type: ignore[return-value]


class _TritonDirectionReduceTriple(torch.autograd.Function):
    """Fused three-moment forward with differentiable exact recomputation."""

    @staticmethod
    def forward(
        ctx: object,
        weight: torch.Tensor,
        radial_gate: torch.Tensor,
        direction_gate: torch.Tensor,
        direction: torch.Tensor,
        sender: torch.Tensor,
        row_ptr: torch.Tensor,
        receiver: torch.Tensor | None,
        gate_lane0: int,
        gate_lane1: int,
        gate_lane2: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)
        saved = (weight, radial_gate, direction_gate, direction, sender, row_ptr)
        ctx.save_for_backward(*(saved if receiver is None else (*saved, receiver)))
        ctx.has_receiver = receiver is not None
        ctx.gate_lanes = (gate_lane0, gate_lane1, gate_lane2)
        return _launch_direction_reduce_triple(
            weight,
            radial_gate,
            direction_gate,
            direction,
            sender,
            row_ptr,
            gate_lanes=(gate_lane0, gate_lane1, gate_lane2),
        )

    @staticmethod
    def backward(
        ctx: object,
        grad_output0: torch.Tensor | None,
        grad_output1: torch.Tensor | None,
        grad_output2: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        weight, radial_gate, direction_gate, direction, sender, row_ptr = (
            ctx.saved_tensors[:6]
        )
        if ctx.has_receiver:
            receiver = ctx.saved_tensors[6]
        else:
            counts = (row_ptr[1:] - row_ptr[:-1]).to(dtype=torch.long)
            receiver = torch.repeat_interleave(
                torch.arange(counts.shape[0], device=row_ptr.device),
                counts,
                output_size=weight.shape[0],
            )
        sender_index = sender.to(dtype=torch.long)
        receiver_index = receiver.to(dtype=torch.long)
        edge_direction_gate = direction_gate.index_select(0, sender_index)
        edge_gradients = tuple(
            torch.zeros(
                (weight.shape[0], weight.shape[1], 3),
                device=weight.device,
                dtype=weight.dtype,
            )
            if grad_output is None
            else grad_output.index_select(0, receiver_index)
            for grad_output in (grad_output0, grad_output1, grad_output2)
        )
        direction_payload = direction.unsqueeze(1)

        grad_weight = torch.zeros_like(weight)
        grad_radial_gate = torch.zeros_like(radial_gate)
        edge_direction_gate_gradients: list[torch.Tensor] = []
        grad_direction = torch.zeros_like(direction)
        for moment, (lane, edge_gradient) in enumerate(
            zip(ctx.gate_lanes, edge_gradients, strict=True)
        ):
            radial = radial_gate[..., lane]
            node_gate = edge_direction_gate[:, moment]
            dot = (edge_gradient * direction_payload).sum(dim=-1)
            grad_weight = grad_weight + dot * radial * node_gate
            lane_basis = torch.nn.functional.one_hot(
                torch.tensor(lane, device=weight.device),
                num_classes=radial_gate.shape[-1],
            ).to(dtype=weight.dtype)
            grad_radial_gate = grad_radial_gate + (
                weight * dot * node_gate
            ).unsqueeze(-1) * lane_basis
            edge_direction_gate_gradients.append(weight * radial * dot)
            coefficient = weight * radial * node_gate
            grad_direction = grad_direction + (
                coefficient.unsqueeze(-1) * edge_gradient
            ).sum(dim=1)

        grad_direction_gate = torch.zeros_like(direction_gate).index_add(
            0,
            sender_index,
            torch.stack(edge_direction_gate_gradients, dim=1),
        )
        return (
            grad_weight,
            grad_radial_gate,
            grad_direction_gate,
            grad_direction,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _csr_sum_many(
    values: Sequence[torch.Tensor],
    row_ptr: torch.Tensor,
    *,
    trusted: bool,
    policy: str,
    receiver: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:

    if not values:
        return ()
    if values[0].ndim == 0:
        raise ValueError("values[0] has an incompatible edge dimension")
    edge_count = values[0].shape[0]
    device = values[0].device
    dtype = values[0].dtype
    shapes: list[tuple[int, ...]] = []
    widths: list[int] = []
    flattened: list[torch.Tensor] = []
    for index, value in enumerate(values):
        if value.ndim == 0 or value.shape[0] != edge_count:
            raise ValueError(f"values[{index}] has an incompatible edge dimension")
        if value.device != device or value.dtype != dtype:
            raise ValueError("all CSR payloads must share one device and dtype")
        shape = tuple(value.shape[1:])
        width = prod(shape) if shape else 1
        if width == 0:
            raise ValueError("CSR payload feature dimensions must be nonzero")
        shapes.append(shape)
        widths.append(width)
        flattened.append(value.reshape(edge_count, width))
    contiguous = _validated_row_ptr(values[0], row_ptr, trusted=trusted)
    if receiver is not None:
        if receiver.ndim != 1 or receiver.shape[0] != edge_count:
            raise ValueError("receiver must match the CSR edge dimension")
        if receiver.device != device or receiver.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise ValueError("receiver must use the CSR device and integer dtype")
        receiver = receiver.contiguous()
    rows = contiguous.numel() - 1
    use_triton = _can_use_triton(
        values[0],
        contiguous,
        policy=policy,
    )
    if use_triton and len(flattened) == 2:
        parts = _TritonCsrSumPair.apply(
            flattened[0],
            flattened[1],
            contiguous,
            receiver,
        )
    elif use_triton and len(flattened) == 3:
        parts = _TritonCsrSumTriple.apply(
            flattened[0],
            flattened[1],
            flattened[2],
            contiguous,
            receiver,
        )
    else:
        packed = torch.cat(flattened, dim=-1)
        reduced = _csr_sum_resolved(packed, contiguous, policy=policy)
        parts = torch.split(reduced, widths, dim=-1)
    return tuple(
        part.reshape(rows, *shape) for part, shape in zip(parts, shapes, strict=True)
    )


__all__ = [
    "active_backend",
    "backend_policy",
    "csr_sum",
    "csr_sum_many",
    "kernel_backend",
    "triton_available",
]
