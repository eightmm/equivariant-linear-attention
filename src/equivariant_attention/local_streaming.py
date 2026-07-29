"""Pure-PyTorch references and dispatch metadata for sparse local transport.

The streamed references operate one receiver row and one bounded edge chunk at
a time.  They deliberately never construct a normalized ``[E, H]`` tensor or
an ``[E, H, ...]`` weighted-message tensor.  They are correctness and fallback
paths; a compiled backend may use the same semantics without inheriting the
Python row loop.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch


_BACKENDS = (
    "materialized",
    "segment_csr",
    "ell",
    "streamed_csr",
    "custom",
)
_OPERATIONS = ("positive", "softmax")
_LAYOUTS = ("none", "csr", "ell", "csr_or_ell")
_FLOAT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


@dataclass(frozen=True)
class LocalBackendCapability:
    """Static execution facts used by deterministic backend selection."""

    backend: str
    available: bool
    operations: tuple[str, ...]
    supports_gradgrad: bool
    layout: str
    supported_dtypes: tuple[torch.dtype, ...]
    supported_devices: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.backend not in _BACKENDS:
            raise ValueError("backend must name a registered local backend")
        if not isinstance(self.available, bool):
            raise TypeError("available must be boolean")
        if not isinstance(self.supports_gradgrad, bool):
            raise TypeError("supports_gradgrad must be boolean")
        if (
            not isinstance(self.operations, tuple)
            or not self.operations
            or len(set(self.operations)) != len(self.operations)
            or any(operation not in _OPERATIONS for operation in self.operations)
        ):
            raise ValueError("operations must be unique registered operations")
        if self.layout not in _LAYOUTS:
            raise ValueError("layout must be a registered sparse layout")
        if (
            not isinstance(self.supported_dtypes, tuple)
            or not self.supported_dtypes
            or len(set(self.supported_dtypes)) != len(self.supported_dtypes)
            or any(dtype not in _FLOAT_DTYPES for dtype in self.supported_dtypes)
        ):
            raise ValueError(
                "supported_dtypes must be unique floating torch dtypes",
            )
        if (
            not isinstance(self.supported_devices, tuple)
            or not self.supported_devices
            or len(set(self.supported_devices)) != len(self.supported_devices)
            or any(
                not isinstance(device, str) or not device
                for device in self.supported_devices
            )
        ):
            raise ValueError(
                "supported_devices must be unique nonempty device names",
            )


@dataclass(frozen=True)
class LocalBackendSelection:
    """Requested/effective backend metadata suitable for run provenance."""

    requested_backend: str
    effective_backend: str
    operation: str
    require_gradgrad: bool
    supports_gradgrad: bool
    fallback_reason: str | None

    @property
    def used_fallback(self) -> bool:
        return (
            self.requested_backend != "auto"
            and self.requested_backend != self.effective_backend
        )


def default_local_backend_capabilities() -> dict[str, LocalBackendCapability]:
    """Return a fresh capability registry for the built-in reference lanes."""

    all_dtypes = _FLOAT_DTYPES
    pytorch_devices = ("cpu", "cuda", "mps", "xpu")
    operations = _OPERATIONS
    return {
        "materialized": LocalBackendCapability(
            backend="materialized",
            available=True,
            operations=operations,
            supports_gradgrad=True,
            layout="none",
            supported_dtypes=all_dtypes,
            supported_devices=pytorch_devices,
        ),
        "segment_csr": LocalBackendCapability(
            backend="segment_csr",
            # No sparse-residual operator currently dispatches to
            # torch.segment_reduce.  Keep the name as a requested
            # compatibility value, but never claim it as an effective lane.
            available=False,
            operations=operations,
            supports_gradgrad=True,
            layout="csr",
            supported_dtypes=all_dtypes,
            supported_devices=pytorch_devices,
        ),
        "ell": LocalBackendCapability(
            backend="ell",
            available=True,
            operations=operations,
            supports_gradgrad=True,
            layout="ell",
            supported_dtypes=all_dtypes,
            supported_devices=pytorch_devices,
        ),
        "streamed_csr": LocalBackendCapability(
            backend="streamed_csr",
            available=True,
            operations=operations,
            supports_gradgrad=True,
            layout="csr",
            supported_dtypes=all_dtypes,
            supported_devices=pytorch_devices,
        ),
        "custom": LocalBackendCapability(
            backend="custom",
            available=False,
            operations=operations,
            supports_gradgrad=False,
            layout="csr_or_ell",
            supported_dtypes=(torch.bfloat16, torch.float32),
            supported_devices=("cuda",),
        ),
    }


def select_local_backend(
    requested_backend: str,
    *,
    operation: str,
    max_degree: int,
    has_csr: bool,
    has_ell: bool,
    require_gradgrad: bool = False,
    dtype: torch.dtype = torch.float32,
    device_type: str = "cpu",
    capabilities: Mapping[str, LocalBackendCapability] | None = None,
    small_degree_threshold: int = 8,
    streamed_degree_threshold: int = 32,
    allow_fallback: bool = True,
) -> LocalBackendSelection:
    """Select a local backend by a frozen, degree-aware priority.

    Small rows use the materialized PyTorch path to avoid launch overhead.
    Valid ELL layouts are preferred for bounded medium rows, receiver CSR is
    used for other medium rows, and large rows prefer a custom kernel before
    the streamed PyTorch reference.  Backends without a wired execution
    operator are unavailable even when the installed PyTorch exposes a
    similarly named primitive.  An explicit unavailable backend safely falls
    back unless ``allow_fallback=False``.
    """

    _validate_selector_controls(
        requested_backend=requested_backend,
        operation=operation,
        max_degree=max_degree,
        has_csr=has_csr,
        has_ell=has_ell,
        require_gradgrad=require_gradgrad,
        dtype=dtype,
        device_type=device_type,
        small_degree_threshold=small_degree_threshold,
        streamed_degree_threshold=streamed_degree_threshold,
        allow_fallback=allow_fallback,
    )
    registry = _capability_registry(capabilities)
    context = _BackendContext(
        operation=operation,
        require_gradgrad=require_gradgrad,
        dtype=dtype,
        device_type=device_type,
        has_csr=has_csr,
        has_ell=has_ell,
    )

    if requested_backend != "auto":
        requested_capability = registry[requested_backend]
        reason = _capability_rejection(requested_capability, context)
        if reason is None:
            return _selection(
                requested_backend=requested_backend,
                capability=requested_capability,
                operation=operation,
                require_gradgrad=require_gradgrad,
                fallback_reason=None,
            )
        if not allow_fallback:
            raise RuntimeError(
                f"{requested_backend} backend unavailable: {reason}",
            )
    else:
        reason = None

    candidates = (
        (
            "streamed_csr",
            "ell",
            "materialized",
            "custom",
            "segment_csr",
        )
        if requested_backend == "segment_csr"
        else _auto_candidates(
            max_degree=max_degree,
            has_ell=has_ell,
            small_degree_threshold=small_degree_threshold,
            streamed_degree_threshold=streamed_degree_threshold,
        )
    )
    for backend in candidates:
        capability = registry[backend]
        rejection = _capability_rejection(capability, context)
        if rejection is None:
            fallback_reason = None
            if requested_backend != "auto":
                fallback_reason = (
                    f"{requested_backend} backend {reason}; selected {backend}"
                )
            return _selection(
                requested_backend=requested_backend,
                capability=capability,
                operation=operation,
                require_gradgrad=require_gradgrad,
                fallback_reason=fallback_reason,
            )
    raise RuntimeError("no local backend satisfies the requested capabilities")


def streamed_positive_csr(
    score: torch.Tensor,
    value: torch.Tensor,
    cutoff: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    chunk_size: int = 64,
    accumulation_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Mass-damped positive transport over receiver-major CSR.

    The exact row equation is ``sum_e(score * cutoff * value) /
    (1 + sum_e(score * cutoff))``.  ``score`` has shape ``[E, H]`` and
    ``value`` has shape ``[E, H, ...]``.  Reductions use float32 for lower
    precision inputs and preserve float64.
    """

    accumulator_dtype, boundaries = _validate_csr_inputs(
        score,
        value,
        cutoff,
        row_ptr,
        positive_score=True,
        chunk_size=chunk_size,
        accumulation_dtype=accumulation_dtype,
    )
    rows = [
        _positive_row(
            score[start:stop],
            value[start:stop],
            cutoff[start:stop],
            mask=None,
            chunk_size=chunk_size,
            accumulation_dtype=accumulator_dtype,
        )
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    return _stack_rows(rows, value, boundaries, accumulator_dtype)


def streamed_softmax_csr(
    score: torch.Tensor,
    value: torch.Tensor,
    cutoff: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    chunk_size: int = 64,
    accumulation_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Two-pass stable receiver softmax over receiver-major CSR.

    The local-only weights are proportional to
    ``cutoff * exp(score)``.  Pass one streams each receiver maximum; pass two
    streams its exponential mass and weighted value.  A row with no positive
    cutoff returns zero.
    """

    accumulator_dtype, boundaries = _validate_csr_inputs(
        score,
        value,
        cutoff,
        row_ptr,
        positive_score=False,
        chunk_size=chunk_size,
        accumulation_dtype=accumulation_dtype,
    )
    rows = [
        _softmax_row(
            score[start:stop],
            value[start:stop],
            cutoff[start:stop],
            mask=None,
            chunk_size=chunk_size,
            accumulation_dtype=accumulator_dtype,
        )
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    return _stack_rows(rows, value, boundaries, accumulator_dtype)


def streamed_positive_ell(
    score: torch.Tensor,
    value: torch.Tensor,
    cutoff: torch.Tensor,
    mask: torch.Tensor,
    *,
    chunk_size: int = 64,
    accumulation_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Mass-damped positive transport over padded receiver-major ELL rows."""

    accumulator_dtype = _validate_ell_inputs(
        score,
        value,
        cutoff,
        mask,
        positive_score=True,
        chunk_size=chunk_size,
        accumulation_dtype=accumulation_dtype,
    )
    rows = [
        _positive_row(
            score[receiver],
            value[receiver],
            cutoff[receiver],
            mask=mask[receiver],
            chunk_size=chunk_size,
            accumulation_dtype=accumulator_dtype,
        )
        for receiver in range(score.shape[0])
    ]
    return _stack_ell_rows(rows, value, accumulator_dtype)


def streamed_softmax_ell(
    score: torch.Tensor,
    value: torch.Tensor,
    cutoff: torch.Tensor,
    mask: torch.Tensor,
    *,
    chunk_size: int = 64,
    accumulation_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Two-pass stable receiver softmax over padded ELL rows."""

    accumulator_dtype = _validate_ell_inputs(
        score,
        value,
        cutoff,
        mask,
        positive_score=False,
        chunk_size=chunk_size,
        accumulation_dtype=accumulation_dtype,
    )
    rows = [
        _softmax_row(
            score[receiver],
            value[receiver],
            cutoff[receiver],
            mask=mask[receiver],
            chunk_size=chunk_size,
            accumulation_dtype=accumulator_dtype,
        )
        for receiver in range(score.shape[0])
    ]
    return _stack_ell_rows(rows, value, accumulator_dtype)


@dataclass(frozen=True)
class _BackendContext:
    operation: str
    require_gradgrad: bool
    dtype: torch.dtype
    device_type: str
    has_csr: bool
    has_ell: bool


def _selection(
    *,
    requested_backend: str,
    capability: LocalBackendCapability,
    operation: str,
    require_gradgrad: bool,
    fallback_reason: str | None,
) -> LocalBackendSelection:
    return LocalBackendSelection(
        requested_backend=requested_backend,
        effective_backend=capability.backend,
        operation=operation,
        require_gradgrad=require_gradgrad,
        supports_gradgrad=capability.supports_gradgrad,
        fallback_reason=fallback_reason,
    )


def _auto_candidates(
    *,
    max_degree: int,
    has_ell: bool,
    small_degree_threshold: int,
    streamed_degree_threshold: int,
) -> tuple[str, ...]:
    if max_degree <= small_degree_threshold:
        return (
            "materialized",
            "ell",
            "segment_csr",
            "streamed_csr",
            "custom",
        )
    if max_degree < streamed_degree_threshold:
        if has_ell:
            return (
                "ell",
                "segment_csr",
                "streamed_csr",
                "materialized",
                "custom",
            )
        return (
            "segment_csr",
            "streamed_csr",
            "materialized",
            "custom",
            "ell",
        )
    return (
        "custom",
        "streamed_csr",
        "ell",
        "segment_csr",
        "materialized",
    )


def _capability_rejection(
    capability: LocalBackendCapability,
    context: _BackendContext,
) -> str | None:
    if not capability.available:
        return "is unavailable"
    if context.operation not in capability.operations:
        return f"does not support {context.operation}"
    if context.require_gradgrad and not capability.supports_gradgrad:
        return "does not support required double backward"
    if context.dtype not in capability.supported_dtypes:
        return f"does not support dtype {context.dtype}"
    if context.device_type not in capability.supported_devices:
        return f"does not support device {context.device_type}"
    if capability.layout == "csr" and not context.has_csr:
        return "requires receiver CSR"
    if capability.layout == "ell" and not context.has_ell:
        return "requires ELL"
    if (
        capability.layout == "csr_or_ell"
        and not context.has_csr
        and not context.has_ell
    ):
        return "requires receiver CSR or ELL"
    return None


def _capability_registry(
    overrides: Mapping[str, LocalBackendCapability] | None,
) -> dict[str, LocalBackendCapability]:
    registry = default_local_backend_capabilities()
    if overrides is None:
        return registry
    if not isinstance(overrides, Mapping):
        raise TypeError("capabilities must be a mapping")
    for backend, capability in overrides.items():
        if backend not in _BACKENDS:
            raise ValueError("capabilities contains an unknown backend")
        if not isinstance(capability, LocalBackendCapability):
            raise TypeError("capability values must be LocalBackendCapability")
        if capability.backend != backend:
            raise ValueError("capability key must match capability.backend")
        registry[backend] = capability
    return registry


def _validate_selector_controls(
    *,
    requested_backend: str,
    operation: str,
    max_degree: int,
    has_csr: bool,
    has_ell: bool,
    require_gradgrad: bool,
    dtype: torch.dtype,
    device_type: str,
    small_degree_threshold: int,
    streamed_degree_threshold: int,
    allow_fallback: bool,
) -> None:
    if not isinstance(requested_backend, str):
        raise TypeError("requested_backend must be a string")
    if requested_backend not in (*_BACKENDS, "auto"):
        raise ValueError("requested_backend must be auto or a registered backend")
    if not isinstance(operation, str):
        raise TypeError("operation must be a string")
    if operation not in _OPERATIONS:
        raise ValueError("operation must be positive or softmax")
    _nonnegative_integer(max_degree, "max_degree")
    _nonnegative_integer(small_degree_threshold, "small_degree_threshold")
    _nonnegative_integer(streamed_degree_threshold, "streamed_degree_threshold")
    if streamed_degree_threshold <= small_degree_threshold:
        raise ValueError(
            "streamed_degree_threshold must exceed small_degree_threshold",
        )
    for control, name in (
        (has_csr, "has_csr"),
        (has_ell, "has_ell"),
        (require_gradgrad, "require_gradgrad"),
        (allow_fallback, "allow_fallback"),
    ):
        if not isinstance(control, bool):
            raise TypeError(f"{name} must be boolean")
    if dtype not in _FLOAT_DTYPES:
        raise ValueError("dtype must be a supported floating torch dtype")
    if not isinstance(device_type, str) or not device_type:
        raise ValueError("device_type must be a nonempty string")


def _nonnegative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _validate_csr_inputs(
    score: torch.Tensor,
    value: torch.Tensor,
    cutoff: torch.Tensor,
    row_ptr: torch.Tensor,
    *,
    positive_score: bool,
    chunk_size: int,
    accumulation_dtype: torch.dtype | None,
) -> tuple[torch.dtype, list[int]]:
    accumulator_dtype = _validate_common_inputs(
        score,
        value,
        cutoff,
        score_dimensions=2,
        positive_score=positive_score,
        chunk_size=chunk_size,
        accumulation_dtype=accumulation_dtype,
    )
    if not isinstance(row_ptr, torch.Tensor):
        raise TypeError("row_ptr must be a tensor")
    if row_ptr.dtype not in (torch.int32, torch.int64):
        raise TypeError("row_ptr must use int32 or int64")
    if row_ptr.device != score.device:
        raise ValueError("row_ptr and inputs must use the same device")
    if row_ptr.ndim != 1 or row_ptr.numel() == 0:
        raise ValueError("row_ptr must be a nonempty one-dimensional tensor")
    boundaries = row_ptr.detach().cpu().tolist()
    if boundaries[0] != 0:
        raise ValueError("row_ptr must start at zero")
    if boundaries[-1] != score.shape[0]:
        raise ValueError("row_ptr must end at the edge count")
    if any(stop < start for start, stop in zip(boundaries[:-1], boundaries[1:])):
        raise ValueError("row_ptr must be nondecreasing")
    return accumulator_dtype, boundaries


def _validate_ell_inputs(
    score: torch.Tensor,
    value: torch.Tensor,
    cutoff: torch.Tensor,
    mask: torch.Tensor,
    *,
    positive_score: bool,
    chunk_size: int,
    accumulation_dtype: torch.dtype | None,
) -> torch.dtype:
    accumulator_dtype = _validate_common_inputs(
        score,
        value,
        cutoff,
        score_dimensions=3,
        positive_score=positive_score,
        chunk_size=chunk_size,
        accumulation_dtype=accumulation_dtype,
    )
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a tensor")
    if mask.dtype != torch.bool:
        raise TypeError("mask must use bool dtype")
    if mask.device != score.device:
        raise ValueError("mask and inputs must use the same device")
    if mask.shape != score.shape[:2]:
        raise ValueError("mask must match the ELL receiver and slot dimensions")
    return accumulator_dtype


def _validate_common_inputs(
    score: torch.Tensor,
    value: torch.Tensor,
    cutoff: torch.Tensor,
    *,
    score_dimensions: int,
    positive_score: bool,
    chunk_size: int,
    accumulation_dtype: torch.dtype | None,
) -> torch.dtype:
    tensors = (score, value, cutoff)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("score, value, and cutoff must be tensors")
    if score.ndim != score_dimensions:
        raise ValueError(f"score must have {score_dimensions} dimensions")
    if score.shape[-1] == 0:
        raise ValueError("score must contain at least one head")
    if value.ndim < score_dimensions:
        raise ValueError("value must include score leading dimensions")
    if value.shape[:score_dimensions] != score.shape:
        raise ValueError("value leading dimensions must match score")
    no_head_shape = score.shape[:-1]
    if cutoff.shape not in (no_head_shape, score.shape):
        raise ValueError("cutoff must omit or include the score head dimension")
    if any(tensor.device != score.device for tensor in tensors[1:]):
        raise ValueError("score, value, and cutoff must use the same device")
    if score.dtype not in _FLOAT_DTYPES:
        raise TypeError("score must use a floating dtype")
    if any(tensor.dtype != score.dtype for tensor in tensors[1:]):
        raise TypeError("score, value, and cutoff must share one dtype")
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise ValueError("score, value, and cutoff must be finite")
    if bool((cutoff < 0).any().item()):
        raise ValueError("cutoff must be nonnegative")
    if positive_score and bool((score < 0).any().item()):
        raise ValueError("score must be nonnegative for positive transport")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return _resolve_accumulation_dtype(
        score.dtype,
        accumulation_dtype=accumulation_dtype,
    )


def _resolve_accumulation_dtype(
    input_dtype: torch.dtype,
    *,
    accumulation_dtype: torch.dtype | None,
) -> torch.dtype:
    if accumulation_dtype is None:
        return torch.float64 if input_dtype == torch.float64 else torch.float32
    if accumulation_dtype not in (torch.float32, torch.float64):
        raise ValueError("accumulation_dtype must be float32 or float64")
    if input_dtype == torch.float64 and accumulation_dtype != torch.float64:
        raise ValueError("accumulation_dtype cannot downcast float64 inputs")
    return accumulation_dtype


def _positive_row(
    score: torch.Tensor,
    value: torch.Tensor,
    cutoff: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    chunk_size: int,
    accumulation_dtype: torch.dtype,
) -> torch.Tensor:
    mass = torch.zeros(
        score.shape[-1],
        dtype=accumulation_dtype,
        device=score.device,
    )
    numerator = torch.zeros(
        value.shape[1:],
        dtype=accumulation_dtype,
        device=value.device,
    )
    for start in range(0, score.shape[0], chunk_size):
        stop = min(start + chunk_size, score.shape[0])
        score_chunk = score[start:stop].to(dtype=accumulation_dtype)
        cutoff_chunk = _cutoff_chunk(
            cutoff[start:stop],
            score_chunk,
            mask=None if mask is None else mask[start:stop],
            accumulation_dtype=accumulation_dtype,
        )
        weight = score_chunk * cutoff_chunk
        mass = mass + weight.sum(dim=0)
        numerator = numerator + _weighted_value_sum(
            weight,
            value[start:stop].to(dtype=accumulation_dtype),
        )
    denominator = 1.0 + _broadcast_mass(mass, numerator)
    return numerator / denominator


def _softmax_row(
    score: torch.Tensor,
    value: torch.Tensor,
    cutoff: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    chunk_size: int,
    accumulation_dtype: torch.dtype,
) -> torch.Tensor:
    row_max = torch.full(
        (score.shape[-1],),
        -torch.inf,
        dtype=accumulation_dtype,
        device=score.device,
    )
    for start in range(0, score.shape[0], chunk_size):
        stop = min(start + chunk_size, score.shape[0])
        effective = _effective_logits(
            score[start:stop].to(dtype=accumulation_dtype),
            cutoff[start:stop],
            mask=None if mask is None else mask[start:stop],
            accumulation_dtype=accumulation_dtype,
        )
        row_max = torch.maximum(row_max, effective.max(dim=0).values)

    active_head = torch.isfinite(row_max)
    safe_max = torch.where(active_head, row_max, torch.zeros_like(row_max))
    mass = torch.zeros_like(row_max)
    numerator = torch.zeros(
        value.shape[1:],
        dtype=accumulation_dtype,
        device=value.device,
    )
    for start in range(0, score.shape[0], chunk_size):
        stop = min(start + chunk_size, score.shape[0])
        effective = _effective_logits(
            score[start:stop].to(dtype=accumulation_dtype),
            cutoff[start:stop],
            mask=None if mask is None else mask[start:stop],
            accumulation_dtype=accumulation_dtype,
        )
        weight = torch.exp(effective - safe_max)
        mass = mass + weight.sum(dim=0)
        numerator = numerator + _weighted_value_sum(
            weight,
            value[start:stop].to(dtype=accumulation_dtype),
        )
    safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
    return numerator / _broadcast_mass(safe_mass, numerator)


def _cutoff_chunk(
    cutoff: torch.Tensor,
    score: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    accumulation_dtype: torch.dtype,
) -> torch.Tensor:
    cutoff_chunk = cutoff.to(dtype=accumulation_dtype)
    if cutoff_chunk.ndim == score.ndim - 1:
        cutoff_chunk = cutoff_chunk.unsqueeze(-1)
    if mask is not None:
        cutoff_chunk = cutoff_chunk * mask.unsqueeze(-1).to(
            dtype=accumulation_dtype,
        )
    return cutoff_chunk


def _effective_logits(
    score: torch.Tensor,
    cutoff: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    accumulation_dtype: torch.dtype,
) -> torch.Tensor:
    cutoff_chunk = _cutoff_chunk(
        cutoff,
        score,
        mask=mask,
        accumulation_dtype=accumulation_dtype,
    )
    positive = cutoff_chunk > 0
    safe_cutoff = torch.where(
        positive,
        cutoff_chunk,
        torch.ones_like(cutoff_chunk),
    )
    effective = score + safe_cutoff.log()
    return torch.where(
        positive,
        effective,
        torch.full_like(score, -torch.inf),
    )


def _weighted_value_sum(
    weight: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    flat_value = value.reshape(value.shape[0], value.shape[1], -1)
    reduced = torch.einsum("eh,ehd->hd", weight, flat_value)
    return reduced.reshape(value.shape[1:])


def _broadcast_mass(
    mass: torch.Tensor,
    numerator: torch.Tensor,
) -> torch.Tensor:
    return mass.reshape((mass.shape[0],) + (1,) * (numerator.ndim - 1))


def _stack_rows(
    rows: list[torch.Tensor],
    value: torch.Tensor,
    boundaries: list[int],
    accumulation_dtype: torch.dtype,
) -> torch.Tensor:
    if rows:
        return torch.stack(rows)
    if boundaries != [0]:
        raise RuntimeError("empty CSR output received invalid boundaries")
    return torch.empty(
        (0, *value.shape[1:]),
        dtype=accumulation_dtype,
        device=value.device,
    )


def _stack_ell_rows(
    rows: list[torch.Tensor],
    value: torch.Tensor,
    accumulation_dtype: torch.dtype,
) -> torch.Tensor:
    if rows:
        return torch.stack(rows)
    return torch.empty(
        (0, *value.shape[2:]),
        dtype=accumulation_dtype,
        device=value.device,
    )


__all__ = [
    "LocalBackendCapability",
    "LocalBackendSelection",
    "default_local_backend_capabilities",
    "select_local_backend",
    "streamed_positive_csr",
    "streamed_positive_ell",
    "streamed_softmax_csr",
    "streamed_softmax_ell",
]
