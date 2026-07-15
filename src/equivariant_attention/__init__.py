from .baselines import EGNNBaseline, EGNNBaselineConfig
from .benchmarking import (
    GraphBatch,
    GraphSample,
    SyntheticMoleculeDataset,
    collate_graphs,
    load_qm9_samples,
    split_dataset,
)
from .inference import autocast_dtype, prepare_for_inference
from .irreps import CartesianIrreps
from .model import EquivariantAttention, EquivariantAttentionConfig
from .moment import EquivariantMomentAttention, EquivariantMomentAttentionConfig
from .rich import RichEquivariantAttention, RichEquivariantAttentionConfig
from .training import TargetNormalizer, build_regression_model, evaluate_regression, fit_target_normalizer, train_regression_step

__all__ = [
    "CartesianIrreps",
    "EGNNBaseline",
    "EGNNBaselineConfig",
    "EquivariantAttention",
    "EquivariantAttentionConfig",
    "EquivariantMomentAttention",
    "EquivariantMomentAttentionConfig",
    "GraphBatch",
    "GraphSample",
    "RichEquivariantAttention",
    "RichEquivariantAttentionConfig",
    "SyntheticMoleculeDataset",
    "TargetNormalizer",
    "autocast_dtype",
    "build_regression_model",
    "collate_graphs",
    "evaluate_regression",
    "fit_target_normalizer",
    "load_qm9_samples",
    "prepare_for_inference",
    "split_dataset",
    "train_regression_step",
]


def main() -> None:
    print("equivariant-attention: import EquivariantAttention from Python")
