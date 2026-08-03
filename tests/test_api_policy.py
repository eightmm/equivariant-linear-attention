from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path

import equivariant_linear_attention as ela
from equivariant_linear_attention import (
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
        "_ELARuntime",
        "_BaseELALayer",
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


def test_every_shipped_core_module_imports_without_optional_dependencies() -> None:
    package_dir = Path(ela.__file__).resolve().parent
    modules: list[str] = []
    for path in package_dir.rglob("*.py"):
        relative = path.relative_to(package_dir).with_suffix("")
        parts = relative.parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            modules.append(".".join(parts))
    for module in modules:
        importlib.import_module(f"{ela.__name__}.{module}")


def test_retired_package_root_is_not_importable() -> None:
    assert importlib.util.find_spec("equivariant_attention") is None


def test_retired_architecture_modules_are_not_shipped() -> None:
    retired = {
        "_egnn_baseline",
        "_matched_vnext_arms",
        "benchmarking",
        "branch_fusion",
        "canonical",
        "canonical_se3",
        "config",
        "data",
        "diagnostics",
        "equivariant_linear_attention",
        "execution",
        "graph_layout",
        "heads",
        "high_order",
        "interface",
        "layered_se3",
        "local_streaming",
        "moment",
        "multiscale",
        "multipole_ops",
        "neighbor_providers",
        "neighbors",
        "optimized_local",
        "parity_se3",
        "pdbbind",
        "pooling",
        "radius",
        "reference_irreps",
        "reproducibility",
        "spherical",
        "stack",
        "tensor_product_executor",
        "training",
        "triton_ops",
        "unified",
        "runtime",
    }
    for module in retired:
        assert importlib.util.find_spec(f"{ela.__name__}.{module}") is None
