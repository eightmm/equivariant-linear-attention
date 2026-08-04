from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path

import torch

import equivariant_linear_attention as ela
from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.advanced import ELAConfig, ELAFeatures, SparseGeometry
from equivariant_linear_attention.model.ela import ELALayer


def test_package_root_is_one_model_and_one_graph() -> None:
    assert ela.__all__ == ["ELA", "ELAGraph"]
    assert ela.ELA is ELA
    assert ela.ELAGraph is ELAGraph

    forbidden = {
        "ELABatch",
        "ELAOutput",
        "ELALayer",
        "ELAConfig",
        "ELAFeatures",
        "SparseGeometry",
        "RefinementRequest",
        "ELARefiner",
        "conservative_forces",
        "pack_irreps",
        "split_irreps",
        "radius_graph",
        "prepare_3d_graph",
    }
    assert set(ela.__all__).isdisjoint(forbidden)
    for name in forbidden:
        assert not hasattr(ela, name)

    public_modules = {
        name
        for name in ela.__all__
        if inspect.isclass(getattr(ela, name))
        and issubclass(getattr(ela, name), torch.nn.Module)
    }
    assert public_modules == {"ELA"}


def test_model_has_one_declared_graph_call_contract() -> None:
    parameters = inspect.signature(ELA.__init__).parameters
    assert tuple(parameters) == (
        "self",
        "input_irreps",
        "output_irreps",
        "width",
        "depth",
        "cutoff",
        "max_neighbors",
        "edge_types",
        "condition_dim",
        "order_dim",
        "update_positions",
        "max_coordinate_step",
    )
    assert "node_dim" not in parameters
    assert "output_dim" not in parameters
    assert "coordinate_refinement" not in parameters
    assert "num_rbf" not in parameters
    assert "relation_cutoffs" not in parameters
    assert not hasattr(ELA, "batch")
    assert not hasattr(ELA, "padded")
    assert not hasattr(ELA, "as_batch")
    assert not hasattr(ELA, "refiner")
    assert not hasattr(ELA, "prepare")
    assert not hasattr(ELA, "forward_prepared")


def test_advanced_configuration_still_builds_the_same_model() -> None:
    config = ELAConfig(
        input_irreps="4x0e",
        output_irreps="1x0e",
        width=32,
        depth=2,
        geometry=SparseGeometry(cutoff=5.0, num_rbf=8),
        features=ELAFeatures(condition_dim=8, order_dim=1),
        coordinate_updates=2,
        max_coordinate_step=0.1,
    )
    model = ELA.from_config(config)
    assert model.attention_kind == "equivariant_linear_attention"
    assert model.updates_positions
    assert isinstance(model.layers[0], ELALayer)
    contract = config.canonical_contract()
    assert contract["public_model"] == "ELA"
    assert contract["public_graph"] == "ELAGraph"
    assert contract["message_fusion"] == "fixed_exact_global_plus_local_sum"


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


def test_retired_architecture_modules_do_not_reappear() -> None:
    retired = {
        "attention",
        "base",
        "conditioned",
        "equivariant_attention",
        "implicit_kernel",
        "linear_attention",
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
