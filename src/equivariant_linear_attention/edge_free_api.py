from __future__ import annotations

import torch

from .api import ELA as _SparseELA
from .batch import ELABatch
from .geometry.neighbors import build_receiver_csr
from .geometry.prepared import PreparationSpec, _prepare_trusted_3d_graph
from .model.edge_free import EdgeFreeELACore
from .model.ela import ELAConfig


class ELA(_SparseELA):
    """Public edge-free ELA with an optional explicit sparse-edge residual.

    Omitting ``edge_index`` no longer triggers radius-graph discovery. The model
    instead uses graphwise relative geometric moments and a shared implicit
    relation operator at Krylov orders one through three. Supplying explicit
    edges preserves the established typed sparse local residual.
    """

    def __init__(
        self,
        input_irreps: str,
        output_irreps: str = "1x0e",
        *,
        width: int = 128,
        depth: int = 8,
        cutoff: float = 5.0,
        max_neighbors: int | None = None,
        edge_types: int = 0,
        condition_dim: int = 0,
        order_dim: int = 0,
        update_positions: bool = False,
        max_coordinate_step: float = 0.25,
    ) -> None:
        super().__init__(
            input_irreps,
            output_irreps,
            width=width,
            depth=depth,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            edge_types=edge_types,
            condition_dim=condition_dim,
            order_dim=order_dim,
            update_positions=update_positions,
            max_coordinate_step=max_coordinate_step,
        )
        self._install_edge_free_core()

    @classmethod
    def from_config(cls, config: ELAConfig) -> ELA:
        model = super().from_config(config)
        model._install_edge_free_core()
        return model

    def _install_edge_free_core(self) -> None:
        self.core = EdgeFreeELACore(self.advanced_config)
        self._initialize_chiral_bridge()
        self.internal_irreps = self.core.internal_irreps
        self.output_irreps = self.core.output_irreps

    def _preparation_matches(self, batch: ELABatch) -> bool:
        if batch.edge_index is not None:
            return super()._preparation_matches(batch)

        graph = batch._prepared_graph
        if graph is None:
            return False
        if not graph.spec.matches(
            source="explicit",
            cutoff=None,
            max_neighbors=None,
            include_self=False,
            num_edge_relations=self.config.geometry.num_edge_relations,
            skin=0.0,
        ):
            return False
        if graph.neighbors.num_edges:
            return False
        if graph.neighbors.relation_id is not None:
            return False
        if batch._trusted_prepared:
            return True
        return torch.equal(graph.batch, batch.interaction_batch)

    def _prepare_packed(
        self,
        batch: ELABatch,
        *,
        prefer_int32: bool = True,
        force: bool = False,
    ) -> ELABatch:
        if batch.edge_index is not None:
            return super()._prepare_packed(
                batch,
                prefer_int32=prefer_int32,
                force=force,
            )
        if not isinstance(batch, ELABatch):
            raise TypeError("internal preparation expects ELABatch")
        if batch.edge_relation_id is not None:
            raise ValueError("edge_type cannot be used without edge_index")
        if batch.is_prepared and not force and self._preparation_matches(batch):
            return batch
        if batch.is_prepared:
            batch = batch.without_prepared_graph()

        graph_counts = None
        if batch.interaction_group is None:
            if batch.ptr is None:
                raise RuntimeError("packed graph pointer normalization failed")
            graph_counts = (batch.ptr[1:] - batch.ptr[:-1]).to(dtype=torch.long)

        empty_edges = torch.empty(
            (2, 0),
            device=batch.positions.device,
            dtype=torch.long,
        )
        neighbors = build_receiver_csr(
            empty_edges,
            num_nodes=batch.num_nodes,
            edge_relation_id=None,
            prefer_int32=prefer_int32,
            build_ell=False,
        )
        prepared = _prepare_trusted_3d_graph(
            batch.interaction_batch,
            neighbors,
            graph_counts=graph_counts,
            spec=PreparationSpec.explicit(
                num_edge_relations=self.config.geometry.num_edge_relations
            ),
        )
        return batch._with_prepared_graph_trusted(prepared)

    def describe(self) -> dict[str, object]:
        description = super().describe()
        description.update(
            {
                "default_geometry": "edge_free_relative_moments",
                "implicit_relation_orders": (1, 2, 3),
                "explicit_edge_residual": True,
                "automatic_radius_graph": False,
                "internal_graph_ir": (
                    "edge-free graph moments with optional explicit receiver-CSR"
                ),
            }
        )
        return description

    def canonical_contract(self) -> dict[str, object]:
        contract = self.config.canonical_contract()
        contract.update(
            {
                "spatial_policy": (
                    "edge_free_relative_moments_plus_implicit_krylov_relation"
                ),
                "message_fusion": (
                    "krylov_global_plus_optional_explicit_sparse_residual"
                ),
                "implicit_spatial": "receiver_centered_l0_l1_l2_graph_moments",
                "implicit_relation_orders": (1, 2, 3),
                "automatic_radius_graph": False,
            }
        )
        return contract


__all__ = ["ELA"]
