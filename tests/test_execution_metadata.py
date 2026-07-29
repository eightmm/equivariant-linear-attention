from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest
import torch

from equivariant_attention.execution import (
    ExecutionMetadata,
    FallbackDecision,
    ProviderCapabilitySnapshot,
    resolve_execution_metadata,
)
from equivariant_attention.graph_layout import pack_graph_layout
from equivariant_attention.neighbor_providers import NeighborCapabilities


def _layout() -> object:
    return pack_graph_layout(
        torch.tensor([0] * 64 + [1] * 64, dtype=torch.int32),
        assume_grouped=True,
    )


def _provider() -> NeighborCapabilities:
    return NeighborCapabilities(
        provider="ReferenceRadiusNeighborProvider",
        complexity="quadratic_reference",
        production_ready=False,
        deterministic_selection=True,
        supports_skin=False,
        supports_pbc=False,
        supports_cell_list=False,
        supports_hard_learned_topk=False,
    )


def test_execution_receipt_is_immutable_and_sorted_json_round_trips() -> None:
    receipt = ExecutionMetadata(
        requested_global_lane="feature_gemm",
        effective_global_lane="direct",
        requested_local_backend="segment_csr",
        effective_local_backend="segment_csr",
        requested_cache_mode="auto",
        effective_cache_mode="compact",
        neighbor_policy="fixed",
        provider=ProviderCapabilitySnapshot.from_capabilities(_provider()),
        symmetry_group="O3",
        architecture_profile="standard",
        dtype="float32",
        device="cuda:0",
        graph_structure="direct",
        node_count=1024,
        edge_count=8192,
        fallbacks=(),
    )

    encoded = receipt.to_json()
    decoded = ExecutionMetadata.from_json(encoded)

    assert decoded == receipt
    assert encoded == decoded.to_json()
    assert encoded == json.dumps(
        json.loads(encoded),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    with pytest.raises(FrozenInstanceError):
        receipt.edge_count = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        receipt.provider.provider = "forged"  # type: ignore[misc]


def test_execution_receipt_strictly_rejects_unknown_and_tampered_json() -> None:
    valid = resolve_execution_metadata(
        requested_global_lane="outer_scatter",
        requested_local_backend="materialized",
        requested_cache_mode="full",
        graph_layout=None,
        node_count=4,
        edge_count=8,
        neighbor_policy="fixed",
        provider_capabilities=_provider(),
        symmetry_group="O3",
        architecture_profile="expert",
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    payload = json.loads(valid.to_json())

    payload["unknown"] = True
    with pytest.raises(ValueError, match="unknown"):
        ExecutionMetadata.from_dict(payload)

    payload.pop("unknown")
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        ExecutionMetadata.from_dict(payload)

    payload["schema_version"] = 1
    payload["provider"]["unknown"] = True
    with pytest.raises(ValueError, match="unknown"):
        ExecutionMetadata.from_dict(payload)

    encoded = valid.to_json().replace(
        '"schema_version":1',
        '"schema_version":1,"schema_version":1',
    )
    with pytest.raises(ValueError, match="duplicate"):
        ExecutionMetadata.from_json(encoded)


def test_resolver_records_lane_cache_provider_and_truthful_local_fallback() -> None:
    layout = _layout()

    receipt = resolve_execution_metadata(
        requested_global_lane="feature_gemm",
        requested_local_backend="segment_csr",
        requested_cache_mode="auto",
        graph_layout=layout,
        node_count=128,
        edge_count=20_000,
        neighbor_policy="fixed",
        provider_capabilities=_provider(),
        symmetry_group="O3",
        architecture_profile="standard",
        dtype=torch.float32,
        device="cpu",
        num_heads=4,
        feature_width=40,
        value_width=16,
        has_receiver_csr=True,
        max_degree=32,
    )

    assert receipt.effective_global_lane == "padded_bmm"
    assert receipt.effective_local_backend == "streamed_csr"
    assert receipt.effective_cache_mode == "compact"
    assert receipt.graph_structure == "padded"
    assert receipt.local_operation == "positive"
    assert receipt.provider.provider == "ReferenceRadiusNeighborProvider"
    assert receipt.provider.deterministic_selection
    assert not receipt.provider.production_ready
    assert len(receipt.fallbacks) == 1
    assert receipt.fallbacks[0].subsystem == "local"
    assert "unavailable" in receipt.fallbacks[0].reason


def test_gradgrad_and_missing_layout_fallbacks_are_explicit_and_serialized() -> None:
    receipt = resolve_execution_metadata(
        requested_global_lane="feature_gemm",
        requested_local_backend="custom",
        requested_cache_mode="auto",
        graph_layout=None,
        node_count=10,
        edge_count=70_000,
        neighbor_policy="rebuild",
        provider_capabilities={
            "provider": "external",
            "complexity": "external_unknown",
            "production_ready": False,
            "deterministic_selection": False,
            "supports_skin": False,
            "supports_pbc": False,
            "supports_cell_list": False,
            "supports_hard_learned_topk": False,
        },
        symmetry_group="SE3",
        architecture_profile="chiral",
        dtype="bfloat16",
        device="cuda:0",
        require_gradgrad=True,
        custom_local_available=True,
        custom_local_supports_gradgrad=False,
        has_receiver_csr=True,
        max_degree=96,
    )

    assert receipt.effective_global_lane == "outer_scatter"
    assert receipt.effective_local_backend == "streamed_csr"
    assert receipt.effective_cache_mode == "recompute"
    assert [decision.subsystem for decision in receipt.fallbacks] == [
        "global",
        "local",
    ]
    assert "layout" in receipt.fallbacks[0].reason
    assert "unavailable" in receipt.fallbacks[1].reason
    assert ExecutionMetadata.from_json(receipt.to_json()) == receipt


@pytest.mark.parametrize(
    ("requested", "has_csr", "has_ell", "expected"),
    [
        ("auto", True, True, "ell"),
        ("auto", True, False, "streamed_csr"),
        ("auto", False, False, "materialized"),
        ("ell", True, False, "streamed_csr"),
        ("segment_csr", False, False, "materialized"),
    ],
)
def test_local_resolution_is_deterministic_and_fallbacks_are_receipted(
    requested: str,
    has_csr: bool,
    has_ell: bool,
    expected: str,
) -> None:
    kwargs = dict(
        requested_global_lane="outer_scatter",
        requested_local_backend=requested,
        requested_cache_mode="full",
        graph_layout=None,
        node_count=12,
        edge_count=96,
        neighbor_policy="fixed",
        provider_capabilities=_provider(),
        symmetry_group="O3",
        architecture_profile="expert",
        dtype="float32",
        device="cpu",
        has_receiver_csr=has_csr,
        has_ell=has_ell,
        max_degree=16,
    )

    first = resolve_execution_metadata(**kwargs)
    second = resolve_execution_metadata(**kwargs)

    assert first == second
    assert first.effective_local_backend == expected
    assert bool(first.fallbacks) == (requested not in {"auto", expected})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"edge_count": -1}, "edge_count"),
        ({"node_count": True}, "node_count"),
        ({"requested_cache_mode": "unknown"}, "cache"),
        ({"requested_global_lane": "unknown"}, "global"),
        ({"requested_local_backend": "unknown"}, "local"),
        ({"symmetry_group": "E3"}, "symmetry"),
        ({"architecture_profile": "unknown"}, "profile"),
        ({"max_degree": -1}, "max_degree"),
        ({"local_operation": "global"}, "local_operation"),
    ],
)
def test_resolver_rejects_invalid_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "requested_global_lane": "outer_scatter",
        "requested_local_backend": "materialized",
        "requested_cache_mode": "full",
        "graph_layout": None,
        "node_count": 4,
        "edge_count": 8,
        "neighbor_policy": "fixed",
        "provider_capabilities": _provider(),
        "symmetry_group": "O3",
        "architecture_profile": "expert",
        "dtype": "float32",
        "device": "cpu",
        "max_degree": 2,
    }
    arguments.update(kwargs)

    with pytest.raises((TypeError, ValueError), match=message):
        resolve_execution_metadata(**arguments)


def test_fallback_decision_requires_a_real_change_and_reason() -> None:
    with pytest.raises(ValueError, match="differ"):
        FallbackDecision(
            subsystem="local",
            requested="materialized",
            effective="materialized",
            reason="not a fallback",
        )
    with pytest.raises(ValueError, match="reason"):
        FallbackDecision(
            subsystem="local",
            requested="custom",
            effective="streamed_csr",
            reason="",
        )
