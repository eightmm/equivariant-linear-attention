"""Execution backend policy and optional optimized kernels."""

from .triton import (
    active_backend,
    backend_policy,
    csr_sum,
    csr_sum_many,
    kernel_backend,
    triton_available,
)

__all__ = [
    "active_backend",
    "backend_policy",
    "csr_sum",
    "csr_sum_many",
    "kernel_backend",
    "triton_available",
]
