"""Public API for the single canonical equivariant linear-attention model."""

from .batch import ELABatch
from .canonical import ELAConfig, ELAFeatures, ELALayer, SparseGeometry
from .context import GeometryRebuilder, OrderContext, RefinementRequest
from .heads import CoordinateUpdateHead, EquivariantVectorHead
from .interface import ELA
from .irreps import (
    Irrep,
    IrrepBlock,
    IrrepLayout,
    matrix_to_st5,
    pack_irreps,
    split_irreps,
    st5_to_matrix,
)
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


def main() -> None:
    print("equivariant-attention: construct ELABatch and call ELA(batch)")
