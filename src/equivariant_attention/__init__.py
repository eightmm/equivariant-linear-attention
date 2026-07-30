from .benchmarking import (
    GraphBatch,
    GraphSample,
    SyntheticMoleculeDataset,
    collate_graphs,
    load_qm9_samples,
    split_dataset,
)
from .annotations import DistanceBandSpec, RelationTable
from .config import (
    ArchitectureConfig,
    GlobalTransportConfig,
    LocalResidualConfig,
    NeighborConfig,
    RepresentationConfig,
)
from .execution import (
    ExecutionMetadata,
    FallbackDecision,
    ProviderCapabilitySnapshot,
    resolve_execution_metadata,
)
from .graph_layout import PackedGraphLayout, pack_graph_layout
from .heads import CoordinateUpdateHead, EquivariantVectorHead
from .high_order import TransientL3Workspace
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
from .multiscale import HierarchyAssignment
from .neighbors import (
    PackedNeighborGraph,
    build_receiver_csr,
    build_reverse_csr,
    pack_neighbor_graph,
    receiver_csr_reduce,
    sender_csr_reduce,
)
from .neighbor_providers import (
    ExternalCallableNeighborProvider,
    NeighborCapabilities,
    NeighborProvider,
    PrecomputedNeighborProvider,
    ReferenceRadiusNeighborProvider,
    VerletRadiusNeighborProvider,
    unsupported_neighbor_capability,
)
from .physics import (
    DirectVectorForceHead,
    ScalarEnergyHead,
    conservative_forces,
)
from .pooling import MaskedInvariantPooling
from .reference_irreps import (
    IrrepLinear,
    IrrepLinearPath,
    IrrepRMSNorm,
    ReferenceTensorProductPath,
    ScalarGatedIrreps,
    tensor_product_path,
    transform_irreps,
)
from .spherical import (
    cartesian_to_real_l1,
    matrix_to_real_l2,
    real_clebsch_gordan,
    real_l1_to_cartesian,
    real_l2_to_matrix,
    real_spherical_harmonics,
)
from .tensor_product_executor import (
    ExecutableTensorProductPlan,
    ReferenceTensorProduct,
    TensorProductInstruction,
    compile_executable_tensor_product,
)
from .training import (
    TargetNormalizer,
    build_regression_model,
    evaluate_regression,
    fit_target_normalizer,
    train_regression_step,
)
from .unified import (
    Prepared3DGraph,
    Unified3DConfig,
    UnifiedEquivariantAttention,
    prepare_3d_graph,
)

__all__ = [
    "ArchitectureConfig",
    "CartesianIrreps",
    "CoordinateUpdateHead",
    "DirectVectorForceHead",
    "DistanceBandSpec",
    "EquivariantAttention",
    "EquivariantAttentionConfig",
    "ExternalCallableNeighborProvider",
    "ExecutionMetadata",
    "ExecutableTensorProductPlan",
    "EquivariantVectorHead",
    "FallbackDecision",
    "GraphBatch",
    "GraphSample",
    "HierarchyAssignment",
    "Irrep",
    "IrrepBlock",
    "IrrepLayout",
    "IrrepLinear",
    "IrrepLinearPath",
    "IrrepRMSNorm",
    "GlobalTransportConfig",
    "LocalResidualConfig",
    "MaskedInvariantPooling",
    "NeighborCapabilities",
    "NeighborProvider",
    "NeighborConfig",
    "PackedGraphLayout",
    "PackedNeighborGraph",
    "Prepared3DGraph",
    "PrecomputedNeighborProvider",
    "ProviderCapabilitySnapshot",
    "ReferenceTensorProductPath",
    "ReferenceTensorProduct",
    "RelationTable",
    "ReferenceRadiusNeighborProvider",
    "RepresentationConfig",
    "ScalarEnergyHead",
    "ScalarGatedIrreps",
    "SyntheticMoleculeDataset",
    "TargetNormalizer",
    "TensorProductPath",
    "TensorProductPlan",
    "TensorProductInstruction",
    "TransientL3Workspace",
    "Unified3DConfig",
    "UnifiedEquivariantAttention",
    "VerletRadiusNeighborProvider",
    "autocast_dtype",
    "build_receiver_csr",
    "build_regression_model",
    "build_reverse_csr",
    "collate_graphs",
    "compile_executable_tensor_product",
    "conservative_forces",
    "evaluate_regression",
    "fit_target_normalizer",
    "load_qm9_samples",
    "pack_graph_layout",
    "pack_neighbor_graph",
    "prepare_3d_graph",
    "prepare_for_inference",
    "receiver_csr_reduce",
    "real_clebsch_gordan",
    "real_l1_to_cartesian",
    "real_l2_to_matrix",
    "real_spherical_harmonics",
    "resolve_execution_metadata",
    "sender_csr_reduce",
    "split_dataset",
    "tensor_product_path",
    "train_regression_step",
    "unsupported_neighbor_capability",
    "transform_irreps",
    "cartesian_to_real_l1",
    "matrix_to_real_l2",
]


def main() -> None:
    print("equivariant-attention: import EquivariantAttention from Python")
