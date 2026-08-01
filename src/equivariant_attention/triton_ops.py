from __future__ import annotations

import os
import weakref
from math import prod
from typing import Final, Sequence

import torch

_BACKEND_ENV: Final = "ELA_KERNEL_BACKEND"
_ALLOWED_BACKENDS: Final = frozenset({"auto", "torch", "triton"})
_MAX_DEGREE_ENV: Final = "ELA_TRITON_MAX_DEGREE"
_MIN_EDGES_ENV: Final = "ELA_TRITON_MIN_EDGES"
_MAX_DEGREE_CACHE: dict[
    int,
    tuple[weakref.ReferenceType[torch.Tensor], int, int],
] = {}
_BACKEND_INSTALLED = False

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional runtime dependency
    triton = None
    tl = None


def backend_policy() -> str:
    """Return the requested execution backend policy."""

    value = os.environ.get(_BACKEND_ENV, "auto").strip().lower()
    if value not in _ALLOWED_BACKENDS:
        raise ValueError(
            f"{_BACKEND_ENV} must be one of {sorted(_ALLOWED_BACKENDS)}, "
            f"got {value!r}"
        )
    return value


def triton_available() -> bool:
    return triton is not None and tl is not None


def _minimum_triton_edges() -> int:
    raw = os.environ.get(_MIN_EDGES_ENV, "256")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_MIN_EDGES_ENV} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{_MIN_EDGES_ENV} must be nonnegative")
    return value


def _max_supported_degree() -> int:
    raw = os.environ.get(_MAX_DEGREE_ENV, "2048")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_MAX_DEGREE_ENV} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{_MAX_DEGREE_ENV} must be positive")
    return value


def register_csr_metadata(row_ptr: torch.Tensor, max_degree: int) -> None:
    """Register trusted immutable CSR metadata without a device-to-host read."""

    if isinstance(max_degree, bool) or not isinstance(max_degree, int):
        raise TypeError("max_degree must be an integer")
    if max_degree < 0:
        raise ValueError("max_degree must be nonnegative")
    key = id(row_ptr)

    def remove(_: weakref.ReferenceType[torch.Tensor]) -> None:
        _MAX_DEGREE_CACHE.pop(key, None)

    if len(_MAX_DEGREE_CACHE) >= 1024:
        stale = [
            name
            for name, entry in _MAX_DEGREE_CACHE.items()
            if entry[0]() is None
        ]
        for name in stale:
            _MAX_DEGREE_CACHE.pop(name, None)
        if len(_MAX_DEGREE_CACHE) >= 1024:
            _MAX_DEGREE_CACHE.clear()
    _MAX_DEGREE_CACHE[key] = (
        weakref.ref(row_ptr, remove),
        row_ptr._version,
        max_degree,
    )


def _row_ptr_max_degree(row_ptr: torch.Tensor) -> int:
    entry = _MAX_DEGREE_CACHE.get(id(row_ptr))
    if (
        entry is not None
        and entry[0]() is row_ptr
        and entry[1] == row_ptr._version
    ):
        return entry[2]
    offsets = tuple(int(value) for value in row_ptr.detach().to("cpu").tolist())
    degree = (
        0
        if len(offsets) <= 1
        else max(
            stop - start
            for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
        )
    )
    register_csr_metadata(row_ptr, degree)
    return degree


def _compile_degree(max_degree: int) -> int:
    if max_degree <= 1:
        return 1
    if triton is None:
        return 1 << (max_degree - 1).bit_length()
    return triton.next_power_of_2(max_degree)


def _basic_triton_support(
    value: torch.Tensor,
    row_ptr: torch.Tensor,
) -> bool:
    return (
        triton_available()
        and value.device.type == "cuda"
        and row_ptr.device == value.device
        and value.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and row_ptr.dtype in {torch.int32, torch.int64}
    )


def _can_use_triton(
    value: torch.Tensor,
    row_ptr: torch.Tensor,
    max_degree: int,
) -> bool:
    policy = backend_policy()
    if policy == "torch":
        return False
    minimum_size = policy == "triton" or (
        value.shape[0] >= _minimum_triton_edges()
    )
    supported = (
        _basic_triton_support(value, row_ptr)
        and max_degree <= _max_supported_degree()
        and minimum_size
    )
    if policy == "triton" and not supported:
        raise RuntimeError(
            "Triton CSR reduction was forced but the current device, dtype, or "
            "maximum degree is unsupported"
        )
    return supported


def active_backend(
    value: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    max_degree: int | None = None,
) -> str:
    """Return the backend that would execute this CSR reduction."""

    policy = backend_policy()
    if policy == "torch":
        return "torch"
    if not _basic_triton_support(value, row_ptr):
        if policy == "triton":
            raise RuntimeError(
                "Triton CSR reduction was forced but the current device or "
                "dtype is unsupported"
            )
        return "torch"
    resolved_degree = (
        _row_ptr_max_degree(row_ptr)
        if max_degree is None
        else int(max_degree)
    )
    return (
        "triton"
        if _can_use_triton(value, row_ptr, resolved_degree)
        else "torch"
    )


if triton_available():

    @triton.jit
    def _csr_sum_kernel(
        value_ptr,
        row_ptr,
        output_ptr,
        num_features: tl.constexpr,
        compile_degree: tl.constexpr,
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
        for base in range(0, compile_degree, block_edges):
            edge = start + base + tl.arange(0, block_edges)
            edge_mask = edge < stop
            pointer = (
                value_ptr
                + edge[:, None] * num_features
                + feature[None, :]
            )
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


class _TorchCsrSum(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        value: torch.Tensor,
        row_ptr: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.segment_reduce(value, reduce="sum", offsets=row_ptr)
        ctx.save_for_backward(row_ptr)
        ctx.input_size = value.shape[0]
        return result

    @staticmethod
    def backward(
        ctx: object,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        (row_ptr,) = ctx.saved_tensors
        counts = (row_ptr[1:] - row_ptr[:-1]).to(dtype=torch.long)
        grad_value = torch.repeat_interleave(
            grad_output,
            counts,
            dim=0,
            output_size=ctx.input_size,
        )
        return grad_value, None


class _TritonCsrSum(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        value: torch.Tensor,
        row_ptr: torch.Tensor,
        max_degree: int,
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
            degree_bucket = _compile_degree(max_degree)
            block_features = min(32, triton.next_power_of_2(features))
            block_edges = min(64, degree_bucket)
            grid = (rows, triton.cdiv(features, block_features))
            _csr_sum_kernel[grid](
                flattened,
                row_ptr,
                output,
                num_features=features,
                compile_degree=degree_bucket,
                block_edges=max(1, block_edges),
                block_features=block_features,
                num_warps=4 if block_edges <= 128 else 8,
            )
        ctx.save_for_backward(row_ptr)
        ctx.input_size = value.shape[0]
        return output.reshape(rows, *value.shape[1:])

    @staticmethod
    def backward(
        ctx: object,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        (row_ptr,) = ctx.saved_tensors
        counts = (row_ptr[1:] - row_ptr[:-1]).to(dtype=torch.long)
        grad_value = torch.repeat_interleave(
            grad_output,
            counts,
            dim=0,
            output_size=ctx.input_size,
        )
        return grad_value, None, None


def csr_sum(
    value: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    max_degree: int | None = None,
) -> torch.Tensor:
    """CSR row sum with automatic Triton dispatch and a PyTorch fallback."""

    if value.ndim == 0:
        raise ValueError("CSR values must have an edge dimension")
    if row_ptr.ndim != 1 or row_ptr.numel() == 0:
        raise ValueError("row_ptr must be a non-empty one-dimensional tensor")
    if value.shape[0] == 0:
        return value.new_zeros((row_ptr.numel() - 1, *value.shape[1:]))
    resolved_degree = (
        _row_ptr_max_degree(row_ptr)
        if max_degree is None
        else int(max_degree)
    )
    if resolved_degree < 0:
        raise ValueError("max_degree must be nonnegative")
    if _can_use_triton(value, row_ptr, resolved_degree):
        return _TritonCsrSum.apply(value, row_ptr, resolved_degree)
    return _TorchCsrSum.apply(value, row_ptr)


def csr_sum_many(
    values: Sequence[torch.Tensor],
    row_ptr: torch.Tensor,
    *,
    max_degree: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Reduce one lifetime-compatible payload group in one backend launch."""

    if not values:
        return ()
    edge_count = values[0].shape[0]
    device = values[0].device
    dtype = values[0].dtype
    shapes: list[tuple[int, ...]] = []
    widths: list[int] = []
    flattened: list[torch.Tensor] = []
    for index, value in enumerate(values):
        if value.ndim == 0 or value.shape[0] != edge_count:
            raise ValueError(
                f"values[{index}] has an incompatible edge dimension"
            )
        if value.device != device or value.dtype != dtype:
            raise ValueError("all CSR payloads must share one device and dtype")
        shape = tuple(value.shape[1:])
        width = prod(shape) if shape else 1
        shapes.append(shape)
        widths.append(width)
        flattened.append(value.reshape(edge_count, width))
    packed = torch.cat(flattened, dim=-1)
    reduced = csr_sum(packed, row_ptr, max_degree=max_degree)
    parts = torch.split(reduced, widths, dim=-1)
    rows = row_ptr.numel() - 1
    return tuple(
        part.reshape(rows, *shape)
        for part, shape in zip(parts, shapes, strict=True)
    )


def install_triton_backend() -> None:
    """Install optimized primitives into the canonical numerical core once."""

    global _BACKEND_INSTALLED
    if _BACKEND_INSTALLED:
        return
    from . import canonical_se3, multipole_ops, parity_se3
    from .optimized_local import triton_local_message

    parity_se3._csr_sum = csr_sum
    canonical_se3._csr_sum = csr_sum
    if hasattr(multipole_ops, "_csr_sum"):
        multipole_ops._csr_sum = csr_sum

    block = canonical_se3._CanonicalMultipoleBlock
    if not hasattr(block, "_ela_torch_local_message"):
        block._ela_torch_local_message = block._local_message
    block._local_message = triton_local_message
    _BACKEND_INSTALLED = True


__all__ = [
    "active_backend",
    "backend_policy",
    "csr_sum",
    "csr_sum_many",
    "install_triton_backend",
    "register_csr_metadata",
    "triton_available",
]