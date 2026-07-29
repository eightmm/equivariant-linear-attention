from .benchmarking import (
    GraphBatch,
    GraphSample,
    SyntheticMoleculeDataset,
    collate_graphs,
    load_qm9_samples,
    split_dataset,
)
from .inference import autocast_dtype, prepare_for_inference
from .irreps import (
    CartesianIrreps,
    Irrep,
    IrrepBlock,
    IrrepLayout,
    TensorProductPath,
    TensorProductPlan,
)
from .moment import EquivariantAttention, EquivariantAttentionConfig
from .neighbors import PackedNeighborGraph, pack_neighbor_graph
from .training import TargetNormalizer, build_regression_model, evaluate_regression, fit_target_normalizer, train_regression_step

__all__ = [
    "CartesianIrreps",
    "EquivariantAttention",
    "EquivariantAttentionConfig",
    "GraphBatch",
    "GraphSample",
    "Irrep",
    "IrrepBlock",
    "IrrepLayout",
    "PackedNeighborGraph",
    "SyntheticMoleculeDataset",
    "TargetNormalizer",
    "TensorProductPath",
    "TensorProductPlan",
    "autocast_dtype",
    "build_regression_model",
    "collate_graphs",
    "evaluate_regression",
    "fit_target_normalizer",
    "load_qm9_samples",
    "pack_neighbor_graph",
    "prepare_for_inference",
    "split_dataset",
    "train_regression_step",
]


def main() -> None:
    print("equivariant-attention: import EquivariantAttention from Python")
