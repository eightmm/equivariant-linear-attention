from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .batch import ELABatch
from .canonical import ELA as _CanonicalELA
from .canonical import ELAConfig, ELAFeatures, ELALayer, SparseGeometry
from .data import radius_graph
from .triton_ops import install_triton_backend

# Install once at import. Unsupported devices and dtypes continue through the
# exact PyTorch reference without changing model configuration or checkpoints.
install_triton_backend()


class ELA(_CanonicalELA):
    """Single public ELA model consuming one canonical :class:`ELABatch`.

    Graph parsing, padding removal, target collation, and optional radius
    discovery happen when the batch is constructed or prepared. The numerical
    core always receives packed nodes and one receiver-major sparse graph.
    """

    def __init__(
        self,
        config: ELAConfig | None = None,
        *,
        input_irreps: str | None = None,
        output_irreps: str = "1x0e",
        width: int = 128,
        depth: int = 8,
        cutoff: float = 5.0,
        num_rbf: int = 16,
        relation_cutoffs: tuple[float, ...] = (),
        condition_dim: int = 0,
        order_dim: int = 0,
        coordinate_refinement: bool = False,
    ) -> None:
        if config is not None:
            if not isinstance(config, ELAConfig):
                raise TypeError("config must be an ELAConfig")
            if input_irreps is not None:
                raise ValueError(
                    "config and direct constructor fields are mutually exclusive"
                )
            resolved = config
        else:
            if input_irreps is None:
                raise ValueError("input_irreps is required when config is omitted")
            resolved = ELAConfig(
                input_irreps=input_irreps,
                output_irreps=output_irreps,
                width=width,
                depth=depth,
                geometry=SparseGeometry(
                    cutoff=cutoff,
                    num_rbf=num_rbf,
                    relation_cutoffs=relation_cutoffs,
                ),
                features=ELAFeatures(
                    condition_dim=condition_dim,
                    order_dim=order_dim,
                    coordinate_refinement=coordinate_refinement,
                ),
            )
        super().__init__(resolved)

    @classmethod
    def scalar(
        cls,
        node_dim: int,
        *,
        output_dim: int = 1,
        width: int = 128,
        depth: int = 8,
        cutoff: float = 5.0,
        num_rbf: int = 16,
        condition_dim: int = 0,
        order_dim: int = 0,
        coordinate_refinement: bool = False,
    ) -> ELA:
        if (
            isinstance(node_dim, bool)
            or not isinstance(node_dim, int)
            or node_dim <= 0
        ):
            raise ValueError("node_dim must be a positive integer")
        if (
            isinstance(output_dim, bool)
            or not isinstance(output_dim, int)
            or output_dim <= 0
        ):
            raise ValueError("output_dim must be a positive integer")
        return cls(
            input_irreps=f"{node_dim}x0e",
            output_irreps=f"{output_dim}x0e",
            width=width,
            depth=depth,
            cutoff=cutoff,
            num_rbf=num_rbf,
            condition_dim=condition_dim,
            order_dim=order_dim,
            coordinate_refinement=coordinate_refinement,
        )

    @staticmethod
    def collate(samples: Sequence[Mapping[str, Any]]) -> ELABatch:
        return ELABatch.collate(samples)

    def prepare(
        self,
        batch: ELABatch,
        *,
        max_neighbors: int | None = None,
        prefer_int32: bool = True,
        force: bool = False,
    ) -> ELABatch:
        """Return ``batch`` with a cached receiver-major execution graph."""

        if not isinstance(batch, ELABatch):
            raise TypeError("ELA.prepare expects an ELABatch")
        if batch.is_prepared and not force:
            return batch
        edge_index = batch.edge_index
        if edge_index is None:
            if batch.edge_relation_id is not None:
                raise ValueError("edge_relation_id cannot be used without edge_index")
            edge_index = radius_graph(
                batch.positions,
                cutoff=self.config.geometry.cutoff,
                batch=batch.batch,
                max_neighbors=max_neighbors,
                include_self=True,
            )
        graph = self.config.geometry.prepare(
            batch.batch,
            edge_index,
            edge_relation_id=batch.edge_relation_id,
            prefer_int32=prefer_int32,
        )
        return batch.with_prepared_graph(graph)

    def forward_prepared(self, batch: ELABatch) -> dict[str, torch.Tensor]:
        """Compile-friendly hot path for an already prepared immutable batch."""

        if not isinstance(batch, ELABatch):
            raise TypeError("ELA.forward_prepared expects an ELABatch")
        if batch._prepared_graph is None:
            raise ValueError("batch is not prepared; call model.prepare(batch) first")
        return dict(
            _CanonicalELA.forward(
                self,
                batch.node_irreps,
                batch.positions,
                batch._prepared_graph,
                context=batch.context,
            )
        )

    def forward(self, batch: ELABatch) -> dict[str, torch.Tensor]:
        if not isinstance(batch, ELABatch):
            raise TypeError(
                "ELA accepts one ELABatch. Use ELABatch(...), "
                "ELABatch.from_padded(...), or ELABatch.collate(...)."
            )
        prepared = batch if batch.is_prepared else self.prepare(batch)
        return self.forward_prepared(prepared)


__all__ = ["ELA", "ELAConfig", "ELAFeatures", "ELALayer", "SparseGeometry"]
