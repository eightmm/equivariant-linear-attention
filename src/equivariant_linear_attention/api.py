from __future__ import annotations

import torch

from .batch import ELABatch
from .context import ELAContext, ELAFeatures
from .geometry.prepared import PreparationSpec
from .geometry.radius import radius_graph
from .graph import ELAGraph
from .kernels import backend_policy, triton_available
from .model.ela import ELAConfig, SparseGeometry, _ELAEngine


def _nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


class ELA(_ELAEngine):
    """The one public equivariant linear-attention model.

    The complete public execution contract is::

        graph = ELAGraph(x, pos, edge_index=edges, batch=batch)
        graph = model(graph)

    The returned graph contains node predictions in ``x``, graph predictions in
    ``graph_x`` and ``graph_sum``, and final coordinates in ``pos``. Coordinate
    updates are a model property selected once with ``update_positions=True``.
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
        if not isinstance(input_irreps, str):
            raise TypeError("input_irreps must be a string")
        if not isinstance(output_irreps, str):
            raise TypeError("output_irreps must be a string")
        if not isinstance(update_positions, bool):
            raise TypeError("update_positions must be a bool")
        relation_count = _nonnegative_integer("edge_types", edge_types)
        relation_cutoffs = (float(cutoff),) * relation_count
        config = ELAConfig(
            input_irreps=input_irreps,
            output_irreps=output_irreps,
            width=width,
            depth=depth,
            geometry=SparseGeometry(
                cutoff=cutoff,
                num_rbf=16,
                max_neighbors=max_neighbors,
                skin=0.0,
                relation_cutoffs=relation_cutoffs,
            ),
            features=ELAFeatures(
                condition_dim=condition_dim,
                order_dim=order_dim,
            ),
            coordinate_updates=1 if update_positions else 0,
            max_coordinate_step=max_coordinate_step,
        )
        super().__init__(config)

    @classmethod
    def from_config(cls, config: ELAConfig) -> ELA:
        """Construct from the reproducibility-oriented advanced configuration."""

        if not isinstance(config, ELAConfig):
            raise TypeError("config must be an ELAConfig")
        model = cls.__new__(cls)
        _ELAEngine.__init__(model, config)
        return model

    @property
    def updates_positions(self) -> bool:
        return self.config.coordinate_updates > 0

    def extra_repr(self) -> str:
        return (
            f"input_irreps={self.config.input_irreps!r}, "
            f"output_irreps={self.config.output_irreps!r}, "
            f"width={self.config.width}, depth={self.config.depth}, "
            f"cutoff={self.config.geometry.cutoff}, "
            f"update_positions={self.updates_positions}"
        )

    def describe(self) -> dict[str, object]:
        return {
            "model": "ELA",
            "graph": "ELAGraph",
            "input_irreps": str(self.config.input_layout),
            "output_irreps": str(self.config.output_layout),
            "width": self.config.width,
            "depth": self.config.depth,
            "cutoff": self.config.geometry.cutoff,
            "max_neighbors": self.config.geometry.max_neighbors,
            "edge_types": self.config.geometry.num_edge_relations,
            "update_positions": self.updates_positions,
            "coordinate_updates": self.config.coordinate_updates,
            "max_coordinate_step": self.config.max_coordinate_step,
            "num_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "kernel_backend_policy": backend_policy(),
            "triton_available": triton_available(),
            "public_contract": "ELAGraph -> ELA -> ELAGraph",
            "internal_graph_ir": "packed receiver-major CSR",
        }

    def _preparation_matches(self, batch: ELABatch) -> bool:
        graph = batch._prepared_graph
        if graph is None:
            return False
        source = "explicit" if batch.edge_index is not None else "radius"
        if not torch.equal(graph.batch, batch.interaction_batch):
            return False
        if not graph.spec.matches(
            source=source,
            cutoff=None if source == "explicit" else self.config.geometry.cutoff,
            max_neighbors=self.config.geometry.max_neighbors,
            include_self=False,
            num_edge_relations=self.config.geometry.num_edge_relations,
            skin=0.0 if source == "explicit" else self.config.geometry.skin,
        ):
            return False
        if not graph.spec.can_reuse_positions(batch.positions):
            return False
        if source == "explicit":
            if batch.edge_index is None:
                return False
            if not torch.equal(
                graph.neighbors.original_edge_index().to(dtype=torch.long),
                batch.edge_index.to(dtype=torch.long),
            ):
                return False
            packed_relation = graph.neighbors.relation_id
            raw_relation = batch.edge_relation_id
            if packed_relation is None:
                return raw_relation is None
            original_relation = graph.neighbors.original_relation_id().to(
                dtype=torch.long
            )
            if raw_relation is None:
                return (
                    self.config.geometry.num_edge_relations == 1
                    and not bool(original_relation.any().item())
                )
            if not torch.equal(
                original_relation,
                raw_relation.to(dtype=torch.long),
            ):
                return False
        return True

    def _prepare_packed(
        self,
        batch: ELABatch,
        *,
        prefer_int32: bool = True,
        force: bool = False,
    ) -> ELABatch:
        if not isinstance(batch, ELABatch):
            raise TypeError("internal preparation expects ELABatch")
        if batch.is_prepared and not force and self._preparation_matches(batch):
            return batch
        if batch.is_prepared:
            batch = batch.without_prepared_graph()

        candidate_edges = batch.edge_index
        relation = batch.edge_relation_id
        relation_count = self.config.geometry.num_edge_relations
        if candidate_edges is None:
            if relation is not None:
                raise ValueError("edge_type cannot be used without edge_index")
            if relation_count > 1:
                raise ValueError(
                    "multiple edge types require explicit edges; "
                    "radius geometry cannot infer semantic relations"
                )
            spec = PreparationSpec.radius(
                batch.positions,
                cutoff=self.config.geometry.cutoff,
                max_neighbors=self.config.geometry.max_neighbors,
                include_self=False,
                num_edge_relations=relation_count,
                skin=self.config.geometry.skin,
            )
            candidate_edges = radius_graph(
                batch.positions,
                cutoff=float(spec.candidate_cutoff),
                batch=batch.interaction_batch,
                max_neighbors=self.config.geometry.max_neighbors,
                include_self=False,
            )
        else:
            spec = PreparationSpec.explicit(num_edge_relations=relation_count)

        if relation is None and relation_count == 1:
            relation = torch.zeros(
                candidate_edges.shape[1],
                device=candidate_edges.device,
                dtype=torch.long,
            )
        elif relation is None and relation_count > 1:
            raise ValueError("edge_type is required when edge_types > 1")
        if relation is not None:
            if relation_count == 0:
                raise ValueError(
                    "edge_type was supplied but edge_types=0 on the model"
                )
            if relation.numel() and (
                int(relation.min().item()) < 0
                or int(relation.max().item()) >= relation_count
            ):
                raise ValueError(f"edge_type values must be in [0, {relation_count})")

        prepared = self.config.geometry.prepare(
            batch.interaction_batch,
            candidate_edges,
            edge_relation_id=relation,
            prefer_int32=prefer_int32,
            spec=spec,
        )
        return batch.with_prepared_graph(prepared)

    def _prepare_graph(self, graph: ELAGraph) -> ELAGraph:
        """Prepare topology while preserving the one public graph type."""

        if not isinstance(graph, ELAGraph):
            raise TypeError("_prepare_graph expects an ELAGraph")
        packed = self._prepare_packed(graph._to_packed())
        if packed._prepared_graph is None:
            raise RuntimeError("graph preparation failed")
        return graph._with_prepared(packed._prepared_graph)

    @torch.compiler.disable
    def _pack_and_prepare(self, graph: ELAGraph) -> ELABatch:
        """Keep public validation and topology discovery outside compiled math."""

        packed = self._prepare_packed(graph._to_packed())
        if packed._prepared_graph is None:
            raise RuntimeError("graph preparation failed")
        graph._cache_prepared(packed._prepared_graph)
        return packed

    def _execute_packed(self, batch: ELABatch) -> dict[str, torch.Tensor]:
        """Numerical hot path for one already validated packed graph."""

        if batch._prepared_graph is None:
            raise RuntimeError("packed graph preparation invariant was violated")

        context = batch.context
        if (
            context is not None
            and context.condition is not None
            and batch.num_interactions != batch.num_graphs
            and context.condition.ndim == 2
            and context.condition.shape[0] == batch.num_graphs
        ):
            context = ELAContext(
                condition=context.condition[batch.batch],
                order=context.order,
            )
        raw = _ELAEngine.forward(
            self,
            batch.node_irreps,
            batch.positions,
            batch._prepared_graph,
            context=context,
            update_mask=batch.update_mask,
        )
        node = raw["node_irreps"]
        graph_sum = node.new_zeros((batch.num_graphs, node.shape[-1]))
        graph_sum.index_add_(0, batch.batch, node)
        counts = torch.bincount(
            batch.batch,
            minlength=batch.num_graphs,
        ).clamp_min(1)
        graph_mean = graph_sum / counts[:, None].to(dtype=node.dtype)
        final_positions = raw.get("positions", batch.positions)
        delta = raw.get("coordinate_delta", torch.zeros_like(batch.positions))
        return {
            "node_irreps": node,
            "graph_irreps": graph_mean,
            "graph_sum": graph_sum,
            "positions": final_positions,
            "coordinate_delta": delta,
        }

    def _forward_prepared(self, batch: ELABatch) -> dict[str, torch.Tensor]:
        """Checked private packed path used by kernel tests and benchmarks."""

        if not isinstance(batch, ELABatch):
            raise TypeError("_forward_prepared expects the private packed graph")
        if batch._prepared_graph is None:
            raise ValueError("packed graph is not prepared")
        if not self._preparation_matches(batch):
            raise ValueError("prepared topology is stale or incompatible")
        return self._execute_packed(batch)

    @torch.compiler.disable
    def _wrap_output(
        self,
        graph: ELAGraph,
        packed: ELABatch,
        output: dict[str, torch.Tensor],
    ) -> ELAGraph:
        # A fixed-position output can safely carry the prepared topology into a
        # subsequent model. Coordinate-updating execution may finish with a
        # rebuilt radius graph, so it deliberately drops the private cache.
        prepared = packed._prepared_graph if not self.updates_positions else None
        return graph._with_output(
            x=output["node_irreps"],
            graph_x=output["graph_irreps"],
            graph_sum=output["graph_sum"],
            pos=output["positions"],
            delta=output["coordinate_delta"],
            prepared=prepared,
        )

    def _forward_graph(self, graph: ELAGraph) -> ELAGraph:
        packed = self._pack_and_prepare(graph)
        output = self._execute_packed(packed)
        return self._wrap_output(graph, packed, output)

    def forward(self, graph: ELAGraph) -> ELAGraph:
        if not isinstance(graph, ELAGraph):
            raise TypeError(
                "ELA accepts exactly one public input type: "
                "ELAGraph(x, pos, edge_index=..., batch=...)"
            )
        return self._forward_graph(graph)


__all__ = ["ELA"]
