from __future__ import annotations

import os
from functools import lru_cache
from typing import Final

import torch

_BACKEND_ENV: Final = "ELA_KERNEL_BACKEND"
_ALLOWED_BACKENDS: Final = frozenset({"auto", "torch", "triton"})
_MAX_DEGREE_ENV: Final = "ELA_TRITON_MAX_DEGREE"

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional runtime dependency
    triton = None
    tl = None


def _backend_policy() -> str:
    value = os.environ.get(_BACKEND_ENV, "auto").strip().lower()
    if value not in _ALLOWED_BACKENDS:
        raise ValueError(
            f"{_BACKEND_ENV} must be one of {sorted(_ALLOWED_BACKENDS)}, got {value!r}"
        )
    return value


def triton_available() -> bool:
    return triton is not None and tl is not None


def _max_supported_degree() -> int:
    raw = os.environ.get(_MAX_DEGREE_ENV, "2048")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_MAX_DEGREE_ENV} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{_MAX_DEGREE_ENV} must be positive")
    return value


@lru_cache(maxsize=1024)
def _cached_max_degree(
    device_type: str,
    device_index: int | None,
    data_ptr: int,
    numel: int,
    version: int,
    row_ptr_cpu: tuple[int, ...],
) -> int:
    del device_type, device_index, data_ptr, numel, version
    if len(row_ptr_cpu) <= 1:
        return 0
    return max(
        stop - start
        for start, stop in zip(row_ptr_cpu[:-1], row_ptr_cpu[1:], strict=True)
    )


def _row_ptr_max_degree(row_ptr: torch.Tensor) -> int:
    # Prepared graphs are immutable. The cache makes this a once-per-CSR
    # metadata transfer rather than a per-layer synchronization.
    cpu = tuple(int(value) for value in row_ptr.detach().to("cpu").tolist())
    return _cached_max_degree(
        row_ptr.device.type,
        row_ptr.device.index,
        row_ptr.data_ptr(),
        row_ptr.numel(),
        row_ptr._version,
        cpu,
    )


if triton_available():

    @triton.jit
    def _csr_sum_kernel(
        value_ptr,
        row_ptr,
        output_ptr,
        num_features: tl.constexpr,
        max_degree: tl.constexpr,
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
        for base in range(0, max_degree, block_edges):
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
            block_features = min(32, triton.next_power_of_2(features))
            block_edges = (
                64
                if max_degree >= 64
                else max(1, triton.next_power_of_2(max_degree))
            )
            grid = (rows, triton.cdiv(features, block_features))
            _csr_sum_kernel[grid](
                flattened,
                row_ptr,
                output,
                num_features=features,
                max_degree=max(1, max_degree),
                block_edges=block_edges,
                block_features=block_features,
                num_warps=4 if block_edges <= 128 else 8,
            )
        ctx.save_for_backward(row_ptr)
        ctx.input_shape = value.shape
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
            output_size=ctx.input_shape[0],
        )
        return grad_value, None, None


def _torch_csr_sum(value: torch.Tensor, row_ptr: torch.Tensor) -> torch.Tensor:
    return torch.segment_reduce(value, reduce="sum", offsets=row_ptr)


def _can_use_triton(
    value: torch.Tensor,
    row_ptr: torch.Tensor,
    max_degree: int,
) -> bool:
    policy = _backend_policy()
    if policy == "torch":
        return False
    supported = (
        triton_available()
        and value.device.type == "cuda"
        and row_ptr.device == value.device
        and value.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and row_ptr.dtype in {torch.int32, torch.int64}
        and max_degree <= _max_supported_degree()
    )
    if policy == "triton" and not supported:
        raise RuntimeError(
            "Triton CSR reduction was forced but the current device, dtype, or "
            "maximum degree is unsupported"
        )
    return supported


def csr_sum(
    value: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    max_degree: int | None = None,
) -> torch.Tensor:
    """CSR row sum with an automatic Triton backend and PyTorch fallback.

    Ordinary and higher-order autograd are preserved. Triton accumulates
    FP16/BF16 values in FP32 and stores the result in the input dtype.
    """

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
    return _torch_csr_sum(value, row_ptr)


def install_triton_backend() -> None:
    """Install the optimized primitive into the loaded numerical modules."""

    from . import canonical_se3, multipole_ops, parity_se3

    parity_se3._csr_sum = csr_sum
    canonical_se3._csr_sum = csr_sum
    if hasattr(multipole_ops, "_csr_sum"):
        multipole_ops._csr_sum = csr_sum


__all__ = ["csr_sum", "install_triton_backend", "triton_available"]
