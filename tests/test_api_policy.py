from __future__ import annotations

from equivariant_attention import ELA, ELAConfig, SparseGeometry
from equivariant_attention.experimental import (
    EquivariantAttentionResiduals,
    ImplicitGaussianSpatialKernel,
)
from equivariant_attention.legacy import (
    EquivariantAttention,
    UnifiedEquivariantAttention,
)


def test_canonical_api_is_available_from_package_root() -> None:
    config = ELAConfig(
        input_irreps="4x0e",
        geometry=SparseGeometry(cutoff=5.0),
    )
    model = ELA(config)
    assert model.attention_kind == "canonical_equivariant_linear_attention"


def test_noncanonical_mechanisms_have_explicit_namespaces() -> None:
    assert EquivariantAttentionResiduals.__module__.endswith(
        "attention_residuals"
    )
    assert ImplicitGaussianSpatialKernel.__module__.endswith("implicit_spatial")
    assert EquivariantAttention.__module__.endswith("moment")
    assert UnifiedEquivariantAttention.__module__.endswith("unified")
