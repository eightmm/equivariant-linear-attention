"""Immutable execution receipts and deterministic backend resolution.

This module does not execute model operators.  It converts requested policy
plus static batch/capability metadata into a machine-readable record of what a
run will execute and why any safe fallback was selected.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import json
from typing import Any, ClassVar

import torch

from .graph_layout import PackedGraphLayout
from .local_streaming import (
    default_local_backend_capabilities,
    select_local_backend,
)
from .neighbor_providers import NeighborCapabilities


_SCHEMA = "equivariant_attention.execution"
_SCHEMA_VERSION = 1
_GLOBAL_REQUESTS = frozenset({"outer_scatter", "feature_gemm", "auto"})
_GLOBAL_LANES = frozenset(
    {
        "outer_scatter",
        "direct",
        "padded_bmm",
        "bucket_bmm",
        "ragged_gemm",
    }
)
_LOCAL_REQUESTS = frozenset(
    {
        "materialized",
        "segment_csr",
        "streamed_csr",
        "ell",
        "custom",
        "auto",
    }
)
_LOCAL_EFFECTIVE = frozenset(
    {"materialized", "segment_csr", "streamed_csr", "ell", "custom"}
)
_LOCAL_OPERATIONS = frozenset({"positive", "softmax"})
_CACHE_MODES = frozenset({"full", "compact", "recompute", "auto"})
_EFFECTIVE_CACHE_MODES = frozenset({"full", "compact", "recompute"})
_SYMMETRY_GROUPS = frozenset({"O3", "SE3"})
_PROFILES = frozenset({"minimal", "standard", "chiral", "high_order", "expert"})
_GRAPH_STRUCTURES = frozenset({"direct", "padded", "bucketed", "ragged", "extreme"})
_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})


def _choice(name: str, value: object, choices: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value not in choices:
        options = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {options}")
    return value


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


@dataclass(frozen=True, slots=True)
class ProviderCapabilitySnapshot:
    """Serializable immutable copy of a neighbor provider's claimed scope."""

    provider: str
    complexity: str
    production_ready: bool
    deterministic_selection: bool
    supports_skin: bool = False
    supports_pbc: bool = False
    supports_cell_list: bool = False
    supports_hard_learned_topk: bool = False

    def __post_init__(self) -> None:
        _nonempty_string("provider", self.provider)
        _nonempty_string("complexity", self.complexity)
        for name in (
            "production_ready",
            "deterministic_selection",
            "supports_skin",
            "supports_pbc",
            "supports_cell_list",
            "supports_hard_learned_topk",
        ):
            _boolean(name, getattr(self, name))

    @classmethod
    def from_capabilities(
        cls,
        capabilities: NeighborCapabilities
        | ProviderCapabilitySnapshot
        | Mapping[str, object]
        | None,
    ) -> ProviderCapabilitySnapshot:
        if isinstance(capabilities, cls):
            return capabilities
        if capabilities is None:
            return cls(
                provider="unspecified",
                complexity="unknown",
                production_ready=False,
                deterministic_selection=False,
            )
        if isinstance(capabilities, NeighborCapabilities):
            return cls(**asdict(capabilities))
        if not isinstance(capabilities, Mapping):
            raise TypeError(
                "provider_capabilities must be NeighborCapabilities, a mapping, or None"
            )
        payload = _strict_mapping(
            dict(capabilities),
            expected={field.name for field in fields(cls)},
            required={
                "provider",
                "complexity",
                "production_ready",
                "deterministic_selection",
            },
            name="provider capabilities",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class FallbackDecision:
    """One requested backend that could not be used as requested."""

    subsystem: str
    requested: str
    effective: str
    reason: str

    def __post_init__(self) -> None:
        _choice(
            "subsystem",
            self.subsystem,
            frozenset({"global", "local", "geometry", "neighbor"}),
        )
        _nonempty_string("requested", self.requested)
        _nonempty_string("effective", self.effective)
        if self.requested == self.effective:
            raise ValueError("fallback requested and effective values must differ")
        _nonempty_string("reason", self.reason)


@dataclass(frozen=True, slots=True)
class ExecutionMetadata:
    """Frozen record of requested and effective runtime policy."""

    SCHEMA: ClassVar[str] = _SCHEMA
    SCHEMA_VERSION: ClassVar[int] = _SCHEMA_VERSION

    requested_global_lane: str
    effective_global_lane: str
    requested_local_backend: str
    effective_local_backend: str
    requested_cache_mode: str
    effective_cache_mode: str
    neighbor_policy: str
    provider: ProviderCapabilitySnapshot
    symmetry_group: str
    architecture_profile: str
    dtype: str
    device: str
    graph_structure: str | None
    node_count: int
    edge_count: int
    local_operation: str = "positive"
    fallbacks: tuple[FallbackDecision, ...] = ()

    def __post_init__(self) -> None:
        _choice(
            "requested_global_lane",
            self.requested_global_lane,
            _GLOBAL_REQUESTS,
        )
        _choice(
            "effective_global_lane",
            self.effective_global_lane,
            _GLOBAL_LANES,
        )
        _choice(
            "requested_local_backend",
            self.requested_local_backend,
            _LOCAL_REQUESTS,
        )
        _choice(
            "effective_local_backend",
            self.effective_local_backend,
            _LOCAL_EFFECTIVE,
        )
        _choice(
            "requested_cache_mode",
            self.requested_cache_mode,
            _CACHE_MODES,
        )
        _choice(
            "effective_cache_mode",
            self.effective_cache_mode,
            _EFFECTIVE_CACHE_MODES,
        )
        _nonempty_string("neighbor_policy", self.neighbor_policy)
        if not isinstance(self.provider, ProviderCapabilitySnapshot):
            raise TypeError("provider must be a ProviderCapabilitySnapshot")
        _choice("symmetry_group", self.symmetry_group, _SYMMETRY_GROUPS)
        _choice(
            "architecture_profile",
            self.architecture_profile,
            _PROFILES,
        )
        _choice("dtype", self.dtype, _DTYPES)
        _nonempty_string("device", self.device)
        if self.graph_structure is not None:
            _choice(
                "graph_structure",
                self.graph_structure,
                _GRAPH_STRUCTURES,
            )
        _integer("node_count", self.node_count)
        _integer("edge_count", self.edge_count)
        _choice("local_operation", self.local_operation, _LOCAL_OPERATIONS)
        if not isinstance(self.fallbacks, tuple):
            raise TypeError("fallbacks must be a tuple")
        if any(
            not isinstance(decision, FallbackDecision) for decision in self.fallbacks
        ):
            raise TypeError("fallbacks must contain FallbackDecision instances")
        subsystems = tuple(decision.subsystem for decision in self.fallbacks)
        if len(set(subsystems)) != len(subsystems):
            raise ValueError("fallbacks may contain at most one decision per subsystem")
        for decision in self.fallbacks:
            if decision.subsystem == "global":
                expected = (
                    self.requested_global_lane,
                    self.effective_global_lane,
                )
            elif decision.subsystem == "local":
                expected = (
                    self.requested_local_backend,
                    self.effective_local_backend,
                )
            elif decision.subsystem == "geometry":
                expected = (
                    self.requested_cache_mode,
                    self.effective_cache_mode,
                )
            else:
                continue
            if (decision.requested, decision.effective) != expected:
                raise ValueError(
                    f"{decision.subsystem} fallback does not match receipt fields"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "schema_version": self.SCHEMA_VERSION,
            "requested_global_lane": self.requested_global_lane,
            "effective_global_lane": self.effective_global_lane,
            "requested_local_backend": self.requested_local_backend,
            "effective_local_backend": self.effective_local_backend,
            "requested_cache_mode": self.requested_cache_mode,
            "effective_cache_mode": self.effective_cache_mode,
            "neighbor_policy": self.neighbor_policy,
            "provider": asdict(self.provider),
            "symmetry_group": self.symmetry_group,
            "architecture_profile": self.architecture_profile,
            "dtype": self.dtype,
            "device": self.device,
            "graph_structure": self.graph_structure,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "local_operation": self.local_operation,
            "fallbacks": [asdict(decision) for decision in self.fallbacks],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> ExecutionMetadata:
        return cls.from_dict(_strict_json_loads(payload))

    @classmethod
    def from_dict(cls, payload: object) -> ExecutionMetadata:
        expected = {
            "schema",
            "schema_version",
            *(field.name for field in fields(cls) if field.init),
        }
        values = _strict_mapping(
            payload,
            expected=expected,
            required=expected,
            name="execution metadata",
        )
        if values.pop("schema") != cls.SCHEMA:
            raise ValueError("unsupported execution schema")
        if values.pop("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        values["provider"] = ProviderCapabilitySnapshot.from_capabilities(
            values["provider"]
        )
        fallback_payload = values["fallbacks"]
        if not isinstance(fallback_payload, list):
            raise TypeError("fallbacks must be a JSON array")
        decisions: list[FallbackDecision] = []
        decision_fields = {field.name for field in fields(FallbackDecision)}
        for index, item in enumerate(fallback_payload):
            decision = _strict_mapping(
                item,
                expected=decision_fields,
                required=decision_fields,
                name=f"fallbacks[{index}]",
            )
            decisions.append(FallbackDecision(**decision))
        values["fallbacks"] = tuple(decisions)
        return cls(**values)


def resolve_execution_metadata(
    *,
    requested_global_lane: str,
    requested_local_backend: str,
    requested_cache_mode: str,
    graph_layout: PackedGraphLayout | None,
    node_count: int,
    edge_count: int,
    neighbor_policy: str,
    provider_capabilities: NeighborCapabilities
    | ProviderCapabilitySnapshot
    | Mapping[str, object]
    | None,
    symmetry_group: str,
    architecture_profile: str,
    dtype: torch.dtype | str,
    device: torch.device | str,
    num_heads: int = 4,
    feature_width: int = 40,
    value_width: int = 16,
    has_receiver_csr: bool = False,
    has_ell: bool = False,
    max_degree: int = 0,
    require_gradgrad: bool = False,
    custom_local_available: bool = False,
    custom_local_supports_gradgrad: bool = False,
    local_operation: str = "positive",
) -> ExecutionMetadata:
    """Resolve a deterministic receipt from static layout and capabilities."""
    requested_global = _choice(
        "requested_global_lane",
        requested_global_lane,
        _GLOBAL_REQUESTS,
    )
    requested_local = _choice(
        "requested_local_backend",
        requested_local_backend,
        _LOCAL_REQUESTS,
    )
    requested_cache = _choice(
        "requested_cache_mode",
        requested_cache_mode,
        _CACHE_MODES,
    )
    nodes = _integer("node_count", node_count)
    edges = _integer("edge_count", edge_count)
    _nonempty_string("neighbor_policy", neighbor_policy)
    _choice("symmetry_group", symmetry_group, _SYMMETRY_GROUPS)
    _choice("architecture_profile", architecture_profile, _PROFILES)
    heads = _integer("num_heads", num_heads, minimum=1)
    features = _integer("feature_width", feature_width, minimum=1)
    values = _integer("value_width", value_width)
    degree = _integer("max_degree", max_degree)
    operation = _choice(
        "local_operation",
        local_operation,
        _LOCAL_OPERATIONS,
    )
    for name, flag in (
        ("has_receiver_csr", has_receiver_csr),
        ("has_ell", has_ell),
        ("require_gradgrad", require_gradgrad),
        ("custom_local_available", custom_local_available),
        ("custom_local_supports_gradgrad", custom_local_supports_gradgrad),
    ):
        _boolean(name, flag)
    normalized_dtype, torch_dtype = _normalize_dtype(dtype)
    normalized_device = _normalize_device(device)
    if graph_layout is not None:
        if not isinstance(graph_layout, PackedGraphLayout):
            raise TypeError("graph_layout must be a PackedGraphLayout or None")
        if graph_layout.num_nodes != nodes:
            raise ValueError("graph_layout num_nodes must match node_count")

    fallbacks: list[FallbackDecision] = []
    effective_global, global_reason = _resolve_global_lane(
        requested_global,
        graph_layout=graph_layout,
        dtype=torch_dtype,
        device=normalized_device,
        num_heads=heads,
        feature_width=features,
        value_width=values,
    )
    if global_reason is not None:
        fallbacks.append(
            FallbackDecision(
                subsystem="global",
                requested=requested_global,
                effective=effective_global,
                reason=global_reason,
            )
        )

    effective_local, local_reason = _resolve_local_backend(
        requested_local,
        operation=operation,
        max_degree=degree,
        dtype=torch_dtype,
        device_type=torch.device(normalized_device).type,
        has_receiver_csr=has_receiver_csr,
        has_ell=has_ell,
        require_gradgrad=require_gradgrad,
        custom_local_available=custom_local_available,
        custom_local_supports_gradgrad=custom_local_supports_gradgrad,
    )
    if local_reason is not None:
        fallbacks.append(
            FallbackDecision(
                subsystem="local",
                requested=requested_local,
                effective=effective_local,
                reason=local_reason,
            )
        )

    effective_cache = _resolve_cache_mode(requested_cache, edges)
    return ExecutionMetadata(
        requested_global_lane=requested_global,
        effective_global_lane=effective_global,
        requested_local_backend=requested_local,
        effective_local_backend=effective_local,
        requested_cache_mode=requested_cache,
        effective_cache_mode=effective_cache,
        neighbor_policy=neighbor_policy,
        provider=ProviderCapabilitySnapshot.from_capabilities(provider_capabilities),
        symmetry_group=symmetry_group,
        architecture_profile=architecture_profile,
        dtype=normalized_dtype,
        device=normalized_device,
        graph_structure=(None if graph_layout is None else graph_layout.structure),
        node_count=nodes,
        edge_count=edges,
        local_operation=operation,
        fallbacks=tuple(fallbacks),
    )


def _resolve_global_lane(
    requested: str,
    *,
    graph_layout: PackedGraphLayout | None,
    dtype: torch.dtype,
    device: str,
    num_heads: int,
    feature_width: int,
    value_width: int,
) -> tuple[str, str | None]:
    if requested == "outer_scatter":
        return "outer_scatter", None
    if graph_layout is None:
        if requested == "auto":
            return "outer_scatter", None
        return (
            "outer_scatter",
            "feature_gemm requires a prepacked graph layout",
        )
    effective = graph_layout.select_lane(
        backend=requested,
        dtype=dtype,
        device=device,
        num_heads=num_heads,
        feature_width=feature_width,
        value_width=value_width,
    )
    if requested == "feature_gemm" and effective == "outer_scatter":
        return (
            effective,
            "graph layout is extreme-ragged and has no admitted GEMM lane",
        )
    return effective, None


def _resolve_local_backend(
    requested: str,
    *,
    operation: str,
    max_degree: int,
    dtype: torch.dtype,
    device_type: str,
    has_receiver_csr: bool,
    has_ell: bool,
    require_gradgrad: bool,
    custom_local_available: bool,
    custom_local_supports_gradgrad: bool,
) -> tuple[str, str | None]:
    capabilities = default_local_backend_capabilities()
    # Capability flags are retained in the public receipt builder for schema
    # compatibility, but this release has no callable custom sparse operator.
    # A claimed external capability must not become an effective backend until
    # the operator is explicitly registered and invoked by the model.
    del custom_local_available, custom_local_supports_gradgrad
    selection = select_local_backend(
        requested,
        operation=operation,
        max_degree=max_degree,
        has_csr=has_receiver_csr,
        has_ell=has_ell,
        require_gradgrad=require_gradgrad,
        dtype=dtype,
        device_type=device_type,
        capabilities=capabilities,
    )
    reason = selection.fallback_reason if selection.used_fallback else None
    return selection.effective_backend, reason


def _resolve_cache_mode(requested: str, edge_count: int) -> str:
    if requested != "auto":
        return requested
    if edge_count <= 4096:
        return "full"
    if edge_count <= 65_536:
        return "compact"
    return "recompute"


def _normalize_dtype(dtype: torch.dtype | str) -> tuple[str, torch.dtype]:
    if isinstance(dtype, torch.dtype):
        name = str(dtype).removeprefix("torch.")
    elif isinstance(dtype, str):
        name = dtype.removeprefix("torch.")
    else:
        raise TypeError("dtype must be a torch.dtype or string")
    _choice("dtype", name, _DTYPES)
    return name, getattr(torch, name)


def _normalize_device(device: torch.device | str) -> str:
    if not isinstance(device, (torch.device, str)):
        raise TypeError("device must be a torch.device or string")
    try:
        normalized = str(torch.device(device))
    except (RuntimeError, ValueError) as exc:
        raise ValueError("device must be a valid torch device") from exc
    return normalized


def _strict_mapping(
    payload: object,
    *,
    expected: set[str],
    required: set[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in payload):
        raise TypeError(f"{name} keys must be strings")
    unknown = set(payload) - expected
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {sorted(unknown)}")
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{name} is missing required fields: {sorted(missing)}")
    return dict(payload)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _strict_json_loads(payload: str | bytes | bytearray) -> object:
    if not isinstance(payload, (str, bytes, bytearray)):
        raise TypeError("JSON payload must be str, bytes, or bytearray")
    try:
        return json.loads(payload, object_pairs_hook=_reject_duplicate_pairs)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid execution JSON") from exc


__all__ = [
    "ExecutionMetadata",
    "FallbackDecision",
    "ProviderCapabilitySnapshot",
    "resolve_execution_metadata",
]
