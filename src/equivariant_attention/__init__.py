"""Public API for the single canonical equivariant linear-attention model."""

from .canonical import ELA, ELAConfig, ELALayer, SparseGeometry
from .context import (
    ELAContext,
    ELAFeatures,
    GeometryRebuilder,
    OrderContext,
    RefinementRequest,
)
from .heads import CoordinateUpdateHead, EquivariantVectorHead
from .irreps import (
    Irrep,
    IrrepBlock,
    IrrepLayout,
    matrix_to_st5,
    pack_irreps,
    split_irreps,
    st5_to_matrix,
)
from .neighbor_providers import (
    ExternalCallableNeighborProvider,
    NeighborCapabilities,
    NeighborProvider,
    PrecomputedNeighborProvider,
    ReferenceRadiusNeighborProvider,
    VerletRadiusNeighborProvider,
)
from .physics import (
    DirectVectorForceHead,
    ScalarEnergyHead,
    conservative_forces,
)
from .unified import (
    Prepared3DGraph,
    UnifiedSE3Context as ELAGeometryContext,
    UnifiedSE3State as ELAState,
    prepare_3d_graph,
)

__all__ = [
    "CoordinateUpdateHead",
    "DirectVectorForceHead",
    "ELA",
    "ELAConfig",
    "ELAContext",
    "ELAGeometryContext",
    "ELAFeatures",
    "ELALayer",
    "ELAState",
    "EquivariantVectorHead",
    "ExternalCallableNeighborProvider",
    "GeometryRebuilder",
    "Irrep",
    "IrrepBlock",
    "IrrepLayout",
    "NeighborCapabilities",
    "NeighborProvider",
    "OrderContext",
    "PrecomputedNeighborProvider",
    "Prepared3DGraph",
    "ReferenceRadiusNeighborProvider",
    "RefinementRequest",
    "ScalarEnergyHead",
    "SparseGeometry",
    "VerletRadiusNeighborProvider",
    "conservative_forces",
    "matrix_to_st5",
    "pack_irreps",
    "prepare_3d_graph",
    "split_irreps",
    "st5_to_matrix",
]


def main() -> None:
    print("equivariant-attention: import ELA, ELAConfig, and ELALayer")
