from __future__ import annotations

import inspect

import equivariant_attention as ela
from equivariant_attention import (
    ELA,
    ELABatch,
    ELAConfig,
    ELAFeatures,
    ELALayer,
    SparseGeometry,
)


def test_package_root_exposes_one_architecture_layer_and_graph_container() -> None:
    exported = set(ela.__all__)
    assert {"ELA", "ELALayer", "ELABatch"}.issubset(exported)

    forbidden = {
        "CanonicalEquivariantLinearAttention",
        "ConditionedELA",
        "ELAContext",
        "ELACoordinateRefiner",
        "EquivariantAttention",
        "EquivariantAttentionResiduals",
        "EquivariantLinearAttention",
        "ImplicitGaussianSpatialKernel",
        "PackedGraphLayout",
        "PackedNeighborGraph",
        "Prepared3DGraph",
        "SpatialOperatorAblationModel",
        "UnifiedEquivariantAttention",
        "UnifiedEquivariantLayer",
        "collate_graphs",
        "prepare_3d_graph",
        "radius_graph",
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


def test_representation_is_declared_only_with_irreps() -> None:
    parameters = inspect.signature(ELA.__init__).parameters
    assert "input_irreps" in parameters
    assert "output_irreps" in parameters
    assert "node_dim" not in parameters
    assert "output_dim" not in parameters
    assert not hasattr(ELA, "scalar")


def test_optional_capabilities_stay_in_one_model_and_batch() -> None:
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
    assert ELABatch.__name__ == "ELABatch"
