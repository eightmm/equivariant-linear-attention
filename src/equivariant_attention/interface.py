from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .batch import ELABatch
from .canonical import ELA as _CanonicalELA
from .canonical import ELAConfig, ELAFeatures, ELALayer, SparseGeometry
from .radius import radius_graph
from .triton_ops import (
    backend_policy,
    install_triton_backend,
    triton_available,
)

# Install once at import. Unsupported devices and dtypes continue through the
# exact PyTorch reference without changing model configuration or checkpoints.
install_triton_backend()


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class ELA(_CanonicalELA):
    """Single public ELA model consuming one canonical :class:`ELABatch`.

    ``ELA(node_dim=...)`` is the scalar-feature convenience constructor.
    Irrep-aware users may instead provide ``input_irreps`` and
    ``output_irreps``. The numerical core always receives packed nodes and one
    receiver-major sparse graph.
    """

    def __init__(
        self,
        config: ELAConfig | None = None,
        *,
        node_dim: int | None = None,
        output_dim: int | None = None,
        input_irreps: str | None = None,
        output_irreps: str | None = None,
        width: int | None = None,
        depth: int | None = None,
        cutoff: float | None = None,
        num_rbf: int | None = None,
        num_edge_types: int | None = None,
        relation_cutoffs: tuple[float, ...] | None = None,
        condition_dim: int | None = None,
        order_dim: int | None = None,
        coordinate_refinement: bool | None = None,
    ) -> None:
        direct_values = {
            "node_dim": node_dim,
            "output_dim": output_dim,
            "input_irreps": input_irreps,
            "output_irreps": output_irreps,
            "width": width,
            "depth": depth,
            "cutoff": cutoff,
            "num_rbf": num_rbf,
            "num_edge_types": num_edge_types,
            "relation_cutoffs": relation_cutoffs,
            "condition_dim": condition_dim,
            "order_dim": order_dim,
            "coordinate_refinement": coordinate_refinement,
        }
        if config is not None:
            if not isinstance(config, ELAConfig):
                raise TypeError("config must be an ELAConfig")
            supplied = [
                name for name, value in direct_values.items() if value is not None
            ]
            if supplied:
                raise ValueError(
                    "config and direct constructor fields are mutually exclusive; "
                    f"received {supplied}"
                )
            resolved = config
        else:
            resolved_width = 128 if width is None else width
            resolved_depth = 8 if depth is None else depth
            resolved_cutoff = 5.0 if cutoff is None else cutoff
            resolved_num_rbf = 16 if num_rbf is None else num_rbf
            resolved_edge_types = 0 if num_edge_types is None else num_edge_types
            resolved_condition_dim = 0 if condition_dim is None else condition_dim
            resolved_order_dim = 0 if order_dim is None else order_dim
            resolved_refinement = (
                False
                if coordinate_refinement is None
                else coordinate_refinement
            )
            if node_dim is not None and input_irreps is not None:
                raise ValueError("supply node_dim or input_irreps, not both")
            if output_dim is not None and output_irreps is not None:
                raise ValueError("supply output_dim or output_irreps, not both")
            if input_irreps is None:
                if node_dim is None:
                    raise ValueError(
                        "node_dim or input_irreps is required when config is omitted"
                    )
                input_irreps = f"{_positive_integer('node_dim', node_dim)}x0e"
            if output_irreps is None:
                output_irreps = (
                    "1x0e"
                    if output_dim is None
                    else f"{_positive_integer('output_dim', output_dim)}x0e"
                )
            if (
                isinstance(resolved_edge_types, bool)
                or not isinstance(resolved_edge_types, int)
                or resolved_edge_types < 0
            ):
                raise ValueError("num_edge_types must be a nonnegative integer")
            resolved_relation_cutoffs = (
                () if relation_cutoffs is None else relation_cutoffs
            )
            if (
                resolved_relation_cutoffs
                and resolved_edge_types
                and len(resolved_relation_cutoffs) != resolved_edge_types
            ):
                raise ValueError(
                    "num_edge_types must match relation_cutoffs when both are supplied"
                )
            if not resolved_relation_cutoffs and resolved_edge_types:
                resolved_relation_cutoffs = (
                    float(resolved_cutoff),
                ) * resolved_edge_types
            resolved = ELAConfig(
                input_irreps=input_irreps,
                output_irreps=output_irreps,
                width=resolved_width,
                depth=resolved_depth,
                geometry=SparseGeometry(
                    cutoff=resolved_cutoff,
                    num_rbf=resolved_num_rbf,
                    relation_cutoffs=resolved_relation_cutoffs,
                ),
                features=ELAFeatures(
                    condition_dim=resolved_condition_dim,
                    order_dim=resolved_order_dim,
                    coordinate_refinement=resolved_refinement,
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
        num_edge_types: int = 0,
        relation_cutoffs: tuple[float, ...] = (),
        condition_dim: int = 0,
        order_dim: int = 0,
        coordinate_refinement: bool = False,
    ) -> ELA:
        """Compatibility alias for ``ELA(node_dim=..., output_dim=...)``."""

        return cls(
            node_dim=node_dim,
            output_dim=output_dim,
            width=width,
            depth=depth,
            cutoff=cutoff,
            num_rbf=num_rbf,
            num_edge_types=num_edge_types,
            relation_cutoffs=relation_cutoffs,
            condition_dim=condition_dim,
            order_dim=order_dim,
            coordinate_refinement=coordinate_refinement,
        )

    @staticmethod
    def batch(
        x: torch.Tensor,
        pos: torch.Tensor,
        **kwargs: Any,
    ) -> ELABatch:
        """Build a packed :class:`ELABatch` from ordinary flat tensors."""

        return ELABatch.from_mapping({"x": x, "pos": pos, **kwargs})

    @staticmethod
    def padded(
        x: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ELABatch:
        """Pack a padded ``[B,M,...]`` input into one :class:`ELABatch`."""

        aliases = {"edge_type": "edge_relation_id", "y": "target"}
        normalized = dict(kwargs)
        for alias, canonical in aliases.items():
            if alias in normalized:
                if canonical in normalized:
                    raise ValueError(
                        f"{alias} and {canonical} are mutually exclusive"
                    )
                normalized[canonical] = normalized.pop(alias)
        return ELABatch.from_padded(x, pos, mask, **normalized)

    @staticmethod
    def collate(
        samples: Sequence[Mapping[str, Any] | ELABatch],
    ) -> ELABatch:
        return ELABatch.collate(samples)

    def extra_repr(self) -> str:
        return (
            f"input_irreps={self.config.input_irreps!r}, "
            f"output_irreps={self.config.output_irreps!r}, "
            f"width={self.config.width}, depth={self.config.depth}, "
            f"cutoff={self.config.geometry.cutoff}"
        )

    def describe(self) -> dict[str, object]:
        """Return a compact, serialization-friendly execution summary."""

        return {
            "model": "ELA",
            "layer": "ELALayer",
            "input_irreps": str(self.config.input_layout),
            "output_irreps": str(self.config.output_layout),
            "width": self.config.width,
            "depth": self.config.depth,
            "cutoff": self.config.geometry.cutoff,
            "num_rbf": self.config.geometry.num_rbf,
            "num_edge_types": self.config.geometry.num_edge_relations,
            "num_parameters": sum(
                parameter.numel() for parameter in self.parameters()
            ),
            "kernel_backend_policy": backend_policy(),
            "triton_available": triton_available(),
            "graph_input": "ELABatch",
        }

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
            batch.assert_prepared_fresh()
            return batch
        edge_index = batch.edge_index
        relation = batch.edge_relation_id
        relation_count = self.config.geometry.num_edge_relations
        from_radius = edge_index is None
        if edge_index is None:
            if relation is not None:
                raise ValueError("edge_relation_id cannot be used without edge_index")
            edge_index = radius_graph(
                batch.positions,
                cutoff=self.config.geometry.cutoff,
                ptr=batch.ptr,
                max_neighbors=max_neighbors,
                include_self=False,
            )
        if relation is None and relation_count == 1:
            relation = torch.zeros(
                edge_index.shape[1],
                device=edge_index.device,
                dtype=torch.long,
            )
        elif relation is None and relation_count > 1:
            raise ValueError(
                "multiple edge types are configured but no edge_type was supplied; "
                "automatic geometry cannot infer semantic relations"
            )
        if relation is not None:
            if relation_count == 0:
                raise ValueError(
                    "edge types were supplied but the model has no relation "
                    "capacity; construct ELA with num_edge_types or "
                    "relation_cutoffs"
                )
            if relation.numel() and (
                int(relation.min().item()) < 0
                or int(relation.max().item()) >= relation_count
            ):
                raise ValueError(
                    f"edge_relation_id values must be in [0, {relation_count})"
                )
        graph = self.config.geometry.prepare(
            batch.batch,
            edge_index,
            edge_relation_id=relation,
            prefer_int32=prefer_int32,
        )
        return batch.with_prepared_graph(
            graph,
            from_radius=from_radius,
            radius_signature=(
                (float(self.config.geometry.cutoff), max_neighbors)
                if from_radius
                else None
            ),
        )

    def forward_prepared(self, batch: ELABatch) -> dict[str, torch.Tensor]:
        """Compile-friendly hot path for an already prepared immutable batch."""

        if not isinstance(batch, ELABatch):
            raise TypeError("ELA.forward_prepared expects an ELABatch")
        batch.assert_prepared_fresh()
        if batch._prepared_graph is None:
            raise RuntimeError("prepared graph unexpectedly missing")
        output = dict(
            _CanonicalELA.forward(
                self,
                batch.node_irreps,
                batch.positions,
                batch._prepared_graph,
                context=batch.context,
            )
        )
        node = output["node_irreps"]
        graph_mean = output["graph_irreps"]
        graph_sum = output.get("graph_sum")
        if graph_sum is None:
            graph_sum = node.new_zeros((batch.num_graphs, node.shape[-1]))
            graph_sum.index_add_(0, batch.batch, node)
        positions = output.get("positions", batch.positions)
        delta = output.get("coordinate_delta", torch.zeros_like(batch.positions))

        # Friendly aliases preserve precise scientific names while making
        # ordinary downstream code concise.
        output.update(
            node=node,
            graph=graph_mean,
            graph_mean=graph_mean,
            graph_sum=graph_sum,
            pos=positions,
            delta=delta,
        )
        return output

    def forward(self, batch: ELABatch) -> dict[str, torch.Tensor]:
        if not isinstance(batch, ELABatch):
            raise TypeError(
                "ELA accepts one ELABatch. Use ELA.batch(...), ELA.padded(...), "
                "or ELA.collate(...)."
            )
        prepared = batch if batch.is_prepared else self.prepare(batch)
        return self.forward_prepared(prepared)


__all__ = ["ELA", "ELAConfig", "ELAFeatures", "ELALayer", "SparseGeometry"]