from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .canonical import (
    ELA as _CanonicalELA,
    ELAConfig,
    ELALayer,
    SparseGeometry,
)
from .context import (
    ELAContext,
    ELAFeatures,
    GeometryRebuilder,
    OrderContext,
    RefinementRequest,
)
from .data import (
    BatchLayout,
    collate_graphs,
    pack_edges,
    pack_node_input,
    radius_graph,
)
from .unified import Prepared3DGraph


class ELA(_CanonicalELA):
    """Ergonomic facade over the single canonical ELA architecture.

    The low-level ``ELA(ELAConfig(...))`` path remains available. Most users can
    construct the model directly and pass either packed graph tensors or padded
    ``[B,M,...]`` tensors with a boolean mask. PyG is not required.
    """

    def __init__(
        self,
        config: ELAConfig | None = None,
        *,
        input_irreps: str | None = None,
        output_irreps: str | None = None,
        width: int | None = None,
        depth: int | None = None,
        cutoff: float | None = None,
        num_rbf: int | None = None,
        relation_cutoffs: tuple[float, ...] | None = None,
        geometry: SparseGeometry | None = None,
        condition_dim: int | None = None,
        order_dim: int | None = None,
        coordinate_refinement: bool | None = None,
    ) -> None:
        direct_values = {
            "input_irreps": input_irreps,
            "output_irreps": output_irreps,
            "width": width,
            "depth": depth,
            "cutoff": cutoff,
            "num_rbf": num_rbf,
            "relation_cutoffs": relation_cutoffs,
            "geometry": geometry,
            "condition_dim": condition_dim,
            "order_dim": order_dim,
            "coordinate_refinement": coordinate_refinement,
        }
        if config is not None:
            if not isinstance(config, ELAConfig):
                raise TypeError("config must be an ELAConfig")
            supplied = [name for name, value in direct_values.items() if value is not None]
            if supplied:
                raise ValueError(
                    "ELAConfig and direct constructor options are mutually exclusive; "
                    f"received {supplied}"
                )
            resolved = config
        else:
            if input_irreps is None:
                raise ValueError("input_irreps is required when config is omitted")
            if geometry is not None and any(
                value is not None
                for value in (cutoff, num_rbf, relation_cutoffs)
            ):
                raise ValueError(
                    "geometry is mutually exclusive with cutoff, num_rbf, and "
                    "relation_cutoffs"
                )
            resolved_geometry = geometry or SparseGeometry(
                cutoff=5.0 if cutoff is None else cutoff,
                num_rbf=16 if num_rbf is None else num_rbf,
                relation_cutoffs=(
                    () if relation_cutoffs is None else relation_cutoffs
                ),
            )
            resolved = ELAConfig(
                input_irreps=input_irreps,
                output_irreps="1x0e" if output_irreps is None else output_irreps,
                width=128 if width is None else width,
                depth=8 if depth is None else depth,
                geometry=resolved_geometry,
                features=ELAFeatures(
                    condition_dim=0 if condition_dim is None else condition_dim,
                    order_dim=0 if order_dim is None else order_dim,
                    coordinate_refinement=(
                        False
                        if coordinate_refinement is None
                        else coordinate_refinement
                    ),
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
        """Construct the common scalar-input/scalar-output model."""

        if isinstance(node_dim, bool) or not isinstance(node_dim, int) or node_dim <= 0:
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
    def collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Use as ``DataLoader(..., collate_fn=ELA.collate)``."""

        return collate_graphs(samples)

    def prepare_graph(
        self,
        positions: torch.Tensor,
        *,
        batch: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_relation_id: torch.Tensor | None = None,
        max_neighbors: int | None = None,
        prefer_int32: bool = True,
    ) -> Prepared3DGraph:
        """Prepare and cache a flat graph without any PyG dependency."""

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("prepare_graph positions must have shape (N,3)")
        if batch is None:
            packed_batch = torch.zeros(
                positions.shape[0],
                device=positions.device,
                dtype=torch.long,
            )
        else:
            if batch.shape != (positions.shape[0],):
                raise ValueError("batch must have shape (N,)")
            packed_batch = batch.to(device=positions.device, dtype=torch.long)
        resolved_edges = edge_index
        if resolved_edges is None:
            if edge_relation_id is not None:
                raise ValueError(
                    "edge_relation_id cannot be inferred without edge_index"
                )
            resolved_edges = radius_graph(
                positions,
                cutoff=self.config.geometry.cutoff,
                batch=packed_batch,
                max_neighbors=max_neighbors,
                include_self=True,
            )
        return self.config.geometry.prepare(
            packed_batch,
            resolved_edges,
            edge_relation_id=edge_relation_id,
            prefer_int32=prefer_int32,
        )

    @staticmethod
    def _flatten_optional_node_tensor(
        value: torch.Tensor | None,
        layout: BatchLayout,
        *,
        name: str,
    ) -> torch.Tensor | None:
        if value is None or layout.kind == "flat":
            return value
        if layout.node_mask is None:
            raise RuntimeError("padded layout is incomplete")
        if value.shape[:2] == layout.node_mask.shape:
            return layout.flatten_node_tensor(value, name=name)
        return value

    def _normalize_order(
        self,
        order: OrderContext | torch.Tensor | None,
        *,
        layout: BatchLayout,
        order_group: torch.Tensor | None,
        order_periods: torch.Tensor | float | None,
        order_mask: torch.Tensor | None,
    ) -> OrderContext | None:
        if order is None:
            if any(
                value is not None
                for value in (order_group, order_periods, order_mask)
            ):
                raise ValueError(
                    "order_group, order_periods, and order_mask require order"
                )
            return None
        if isinstance(order, OrderContext):
            if any(
                value is not None
                for value in (order_group, order_periods, order_mask)
            ):
                raise ValueError(
                    "OrderContext is mutually exclusive with order shortcut fields"
                )
            coordinates = self._flatten_optional_node_tensor(
                order.coordinates,
                layout,
                name="order.coordinates",
            )
            group = self._flatten_optional_node_tensor(
                order.group_index,
                layout,
                name="order.group_index",
            )
            enabled = self._flatten_optional_node_tensor(
                order.enabled,
                layout,
                name="order.enabled",
            )
            if coordinates is None:
                raise RuntimeError("order coordinates unexpectedly missing")
            return OrderContext(
                coordinates=coordinates,
                group_index=group,
                periods=order.periods,
                enabled=enabled,
            )
        if not isinstance(order, torch.Tensor):
            raise TypeError("order must be a tensor or OrderContext")
        coordinates = self._flatten_optional_node_tensor(
            order,
            layout,
            name="order",
        )
        group = self._flatten_optional_node_tensor(
            order_group,
            layout,
            name="order_group",
        )
        enabled = self._flatten_optional_node_tensor(
            order_mask,
            layout,
            name="order_mask",
        )
        if coordinates is None:
            raise RuntimeError("order coordinates unexpectedly missing")
        if coordinates.ndim == 1:
            period = None
            if order_periods is not None:
                if isinstance(order_periods, torch.Tensor):
                    if order_periods.numel() != 1:
                        raise ValueError("sequence order_periods must be scalar")
                    period = float(order_periods.reshape(()).item())
                else:
                    period = float(order_periods)
            return OrderContext.sequence(
                coordinates,
                segment_id=group,
                period=period,
                enabled=enabled,
            )
        if coordinates.ndim != 2:
            raise ValueError("order must have shape (N,) or (N,K) after packing")
        periods = order_periods
        if periods is not None and not isinstance(periods, torch.Tensor):
            periods = coordinates.new_full(
                (coordinates.shape[1],),
                float(periods),
            )
        return OrderContext.grid(
            coordinates,
            segment_id=group,
            periods=periods,
            enabled=enabled,
        )

    def _normalize_context(
        self,
        context: ELAContext | None,
        *,
        layout: BatchLayout,
        condition: torch.Tensor | None,
        order: OrderContext | torch.Tensor | None,
        order_group: torch.Tensor | None,
        order_periods: torch.Tensor | float | None,
        order_mask: torch.Tensor | None,
        refine_steps: int | None,
        max_coordinate_step: float | None,
        refinement_centering: str | None,
        update_mask: torch.Tensor | None,
        graph_rebuilder: GeometryRebuilder | None,
    ) -> ELAContext | None:
        shortcuts_active = any(
            value is not None
            for value in (
                condition,
                order,
                order_group,
                order_periods,
                order_mask,
                refine_steps,
                max_coordinate_step,
                refinement_centering,
                update_mask,
                graph_rebuilder,
            )
        )
        if context is not None and shortcuts_active:
            raise ValueError(
                "ELAContext and convenience context keywords are mutually exclusive"
            )
        if context is not None:
            if not isinstance(context, ELAContext):
                raise TypeError("context must be an ELAContext")
            normalized_condition = self._flatten_optional_node_tensor(
                context.condition,
                layout,
                name="context.condition",
            )
            normalized_order = self._normalize_order(
                context.order,
                layout=layout,
                order_group=None,
                order_periods=None,
                order_mask=None,
            )
            refinement = context.refinement
            if refinement is not None and layout.kind == "padded":
                normalized_mask = self._flatten_optional_node_tensor(
                    refinement.update_mask,
                    layout,
                    name="refinement.update_mask",
                )
                refinement = RefinementRequest(
                    steps=refinement.steps,
                    max_step=refinement.max_step,
                    centering=refinement.centering,
                    update_mask=normalized_mask,
                    graph_rebuilder=refinement.graph_rebuilder,
                )
            if (
                normalized_condition is None
                and normalized_order is None
                and refinement is None
            ):
                return None
            return ELAContext(
                condition=normalized_condition,
                order=normalized_order,
                refinement=refinement,
            )

        normalized_condition = self._flatten_optional_node_tensor(
            condition,
            layout,
            name="condition",
        )
        normalized_order = self._normalize_order(
            order,
            layout=layout,
            order_group=order_group,
            order_periods=order_periods,
            order_mask=order_mask,
        )
        refinement_requested = any(
            value is not None
            for value in (
                refine_steps,
                max_coordinate_step,
                refinement_centering,
                update_mask,
                graph_rebuilder,
            )
        )
        refinement = None
        if refinement_requested:
            normalized_mask = self._flatten_optional_node_tensor(
                update_mask,
                layout,
                name="update_mask",
            )
            refinement = RefinementRequest(
                steps=1 if refine_steps is None else refine_steps,
                max_step=(
                    0.25
                    if max_coordinate_step is None
                    else max_coordinate_step
                ),
                centering=(
                    "selected"
                    if refinement_centering is None
                    else refinement_centering
                ),
                update_mask=normalized_mask,
                graph_rebuilder=graph_rebuilder,
            )
        if (
            normalized_condition is None
            and normalized_order is None
            and refinement is None
        ):
            return None
        return ELAContext(
            condition=normalized_condition,
            order=normalized_order,
            refinement=refinement,
        )

    def forward(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph | None = None,
        *,
        batch: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        edge_index: torch.Tensor | Sequence[torch.Tensor] | None = None,
        edge_mask: torch.Tensor | None = None,
        adjacency: torch.Tensor | None = None,
        edge_relation_id: torch.Tensor | Sequence[torch.Tensor] | None = None,
        max_neighbors: int | None = None,
        context: ELAContext | None = None,
        condition: torch.Tensor | None = None,
        order: OrderContext | torch.Tensor | None = None,
        order_group: torch.Tensor | None = None,
        order_periods: torch.Tensor | float | None = None,
        order_mask: torch.Tensor | None = None,
        refine_steps: int | None = None,
        max_coordinate_step: float | None = None,
        refinement_centering: str | None = None,
        update_mask: torch.Tensor | None = None,
        graph_rebuilder: GeometryRebuilder | None = None,
    ) -> dict[str, torch.Tensor]:
        packed = pack_node_input(
            node_irreps,
            pos,
            batch=batch,
            mask=mask,
        )
        if graph is not None:
            if any(
                value is not None
                for value in (
                    edge_index,
                    edge_mask,
                    adjacency,
                    edge_relation_id,
                    max_neighbors,
                )
            ):
                raise ValueError(
                    "prepared graph is mutually exclusive with edge inputs"
                )
            if not isinstance(graph, Prepared3DGraph):
                raise TypeError("graph must be a Prepared3DGraph")
            if graph.num_nodes != packed.layout.num_nodes:
                raise ValueError("prepared graph node count does not match input")
            if graph.device != packed.node_irreps.device:
                raise ValueError("prepared graph and inputs must share one device")
            if not torch.equal(graph.batch, packed.batch):
                raise ValueError("prepared graph membership does not match input")
            prepared = graph
        else:
            packed_edges, packed_relation = pack_edges(
                packed,
                edge_index=edge_index,
                edge_mask=edge_mask,
                adjacency=adjacency,
                edge_relation_id=edge_relation_id,
            )
            if packed_edges is None:
                packed_edges = radius_graph(
                    packed.positions,
                    cutoff=self.config.geometry.cutoff,
                    batch=packed.batch,
                    max_neighbors=max_neighbors,
                    include_self=True,
                )
            prepared = self.config.geometry.prepare(
                packed.batch,
                packed_edges,
                edge_relation_id=packed_relation,
            )

        normalized_context = self._normalize_context(
            context,
            layout=packed.layout,
            condition=condition,
            order=order,
            order_group=order_group,
            order_periods=order_periods,
            order_mask=order_mask,
            refine_steps=refine_steps,
            max_coordinate_step=max_coordinate_step,
            refinement_centering=refinement_centering,
            update_mask=update_mask,
            graph_rebuilder=graph_rebuilder,
        )
        output = dict(
            super().forward(
                packed.node_irreps,
                packed.positions,
                prepared,
                context=normalized_context,
            )
        )
        if packed.layout.kind == "padded":
            output["node_irreps"] = packed.layout.restore_node_tensor(
                output["node_irreps"]
            )
            output["positions"] = packed.layout.restore_node_tensor(
                output["positions"],
                template=pos,
            )
            output["coordinate_delta"] = packed.layout.restore_node_tensor(
                output["coordinate_delta"]
            )
            if packed.layout.node_mask is None:
                raise RuntimeError("padded node mask unexpectedly missing")
            output["node_mask"] = packed.layout.node_mask
        return output


__all__ = ["ELA", "ELAConfig", "ELALayer", "SparseGeometry"]
