"""Advanced configuration and mathematical helpers for ELA."""

from .context import ELAContext, ELAFeatures, FourierOrderEncoder, OrderContext
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
from .model.ela import ELAConfig, SparseGeometry
from .physics import DirectVectorForceHead, ScalarEnergyHead, conservative_forces

__all__ = [
    "CartesianIrreps",
    "DirectVectorForceHead",
    "ELAConfig",
    "ELAContext",
    "ELAFeatures",
    "FourierOrderEncoder",
    "Irrep",
    "IrrepBlock",
    "IrrepLayout",
    "OrderContext",
    "ScalarEnergyHead",
    "SparseGeometry",
    "TensorProductPath",
    "TensorProductPlan",
    "conservative_forces",
    "matrix_to_st5",
    "pack_irreps",
    "project_symmetric_traceless",
    "split_irreps",
    "st5_inner",
    "st5_mse",
    "st5_norm",
    "st5_to_matrix",
]
