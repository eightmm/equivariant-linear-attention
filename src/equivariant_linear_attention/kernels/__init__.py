"""Execution backend policy and optional optimized kernels."""

from .triton import (
    active_backend,
    backend_policy,
    csr_sum,
    csr_sum_many,
    install_triton_backend,
    kernel_backend,
    triton_available,
    uninstall_triton_backend,
)

__all__ = [
    "active_backend",
    "backend_policy",
    "csr_sum",
    "csr_sum_many",
    "install_triton_backend",
    "kernel_backend",
    "triton_available",
    "uninstall_triton_backend",
]
