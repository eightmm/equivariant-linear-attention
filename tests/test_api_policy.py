from __future__ import annotations

import inspect

import equivariant_attention as ela
from equivariant_attention import (
    ELA,
    ELAConfig,
    ELAContext,
    ELAFeatures,
    ELALayer,
    SparseGeometry,
)


def test_package_root_exposes_one_architecture_and_one_layer() -> None:
    exported = set(ela.__all__)
    assert {"ELA", "ELALayer"}.issubset(exported)

    forbidden = {
        "CanonicalEquivariantLinearAttention",
        "ConditionedELA",
        "ELACoordinateRefiner",
        "EquivariantAttention",
        "EquivariantAttentionResiduals",
        "EquivariantLinearAttention",
        "ImplicitGaussianSpatialKernel",
        "SpatialOperatorAblationModel",
        "UnifiedEquivariantAttention",
        "UnifiedEquivariantLayer",
    }
    assert exported.isdisjoint(forbidden)
    for name in forbidden:
        assert not hasattr(ela, name)

    public_module_classes = {
        name
        for name in exported
        if inspect.isclass(getattr(ela, name))
        and issubclass(getattr(ela, name), __import__("torch").nn.Module)
    }
    assert public_module_classes == {
        "CoordinateUpdateHead",
        "DirectVectorForceHead",
        "ELA",
        "ELALayer",
        "EquivariantVectorHead",
        "ScalarEnergyHead",
    }


def test_optional_capabilities_use_one_config_and_context() -> None:
    config = ELAConfig(
        input_irreps="4x0e",
        geometry=SparseGeometry(cutoff=5.0),
        features=ELAFeatures(
            condition_dim=8,
            order_dim=1,
            coordinate_refinement=True,
        ),
    )
    model = ELA(config)
    assert model.attention_kind == "equivariant_linear_attention"
    assert isinstance(model.layers[0], ELALayer)
    assert config.canonical_contract()["public_model"] == "ELA"
    assert config.canonical_contract()["public_layer"] == "ELALayer"
    assert ELAContext() == ELAContext(
        condition=None,
        order=None,
        refinement=None,
    )
