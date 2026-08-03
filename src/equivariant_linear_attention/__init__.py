"""Public API for the single canonical equivariant linear-attention model."""

from .api import ELA
from .batch import ELABatch
from .context import ELAFeatures, GeometryRebuilder, OrderContext, RefinementRequest
from .irreps import (
    Irrep,
    IrrepBlock,
    IrrepLayout,
    matrix_to_st5,
    pack_irreps,
    split_irreps,
    st5_to_matrix,
)
from .model.ela import ELAConfig, ELALayer, SparseGeometry
from .nn.heads import CoordinateUpdateHead, EquivariantVectorHead
from .physics import DirectVectorForceHead, ScalarEnergyHead, conservative_forces

__all__ = [
    "CoordinateUpdateHead",
    "DirectVectorForceHead",
    "ELA",
    "ELABatch",
    "ELAConfig",
    "ELAFeatures",
    "ELALayer",
    "EquivariantVectorHead",
    "GeometryRebuilder",
    "Irrep",
    "IrrepBlock",
    "IrrepLayout",
    "OrderContext",
    "RefinementRequest",
    "ScalarEnergyHead",
    "SparseGeometry",
    "conservative_forces",
    "matrix_to_st5",
    "pack_irreps",
    "split_irreps",
    "st5_to_matrix",
]
