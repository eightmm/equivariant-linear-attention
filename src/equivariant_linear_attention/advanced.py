"""Advanced mathematical helpers for ELA."""

from .context import ELAFeatures, FourierOrderEncoder
from .inference import prepare_for_inference
from .irreps import (
    CartesianIrreps,
    Irrep,
    IrrepBlock,
    IrrepLayout,
    TensorProductPath,
    TensorProductPlan,
    matrix_to_st5,
    pack_irreps,
    project_symmetric_traceless,
    split_irreps,
    st5_inner,
    st5_mse,
    st5_norm,
    st5_to_matrix,
)
from .model import ELAConfig
from .physics import DirectVectorForceHead, ScalarEnergyHead, conservative_forces

__all__ = [
    "CartesianIrreps",
    "DirectVectorForceHead",
    "ELAConfig",
    "ELAFeatures",
    "FourierOrderEncoder",
    "Irrep",
    "IrrepBlock",
    "IrrepLayout",
    "ScalarEnergyHead",
    "TensorProductPath",
    "TensorProductPlan",
    "conservative_forces",
    "matrix_to_st5",
    "pack_irreps",
    "prepare_for_inference",
    "project_symmetric_traceless",
    "split_irreps",
    "st5_inner",
    "st5_mse",
    "st5_norm",
    "st5_to_matrix",
]
