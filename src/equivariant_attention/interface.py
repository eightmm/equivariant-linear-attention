from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .api import ELA as _TensorELA
from .context import ELAContext, GeometryRebuilder, OrderContext
from .unified import Prepared3DGraph


class ELA(_TensorELA):
    """User-facing ELA accepting tensors or a collated graph mapping.

    ``model(batch_dict)`` consumes the output of :func:`collate_graphs` directly;
    training labels and sample IDs remain in the mapping but are ignored by the
    model. Tensor callers retain the full typed keyword API of :class:`api.ELA`.
    """

    _MODEL_KEYS = frozenset(
        {
            "node_irreps",
            "x",
            "node_features",
            "pos",
            "positions",
            "graph",
            "batch",
            "mask",
            "node_mask",
            "edge_index",
            "edge_mask",
            "adjacency",
            "edge_relation_id",
            "max_neighbors",
            "context",
            "condition",
            "order",
            "order_group",
            "order_periods",
            "order_mask",
            "refine_steps",
            "max_coordinate_step",
            "refinement_centering",
            "update_mask",
            "graph_rebuilder",
            "target",
            "sample_ids",
        }
    )

    @staticmethod
    def _mapping_value(
        payload: Mapping[str, Any],
        name: str,
        aliases: tuple[str, ...] = (),
    ) -> Any:
        present = [key for key in (name, *aliases) if key in payload]
        if len(present) > 1:
            raise ValueError(f"batch mapping contains multiple aliases for {name}")
        return None if not present else payload[present[0]]

    def forward(
        self,
        node_irreps: torch.Tensor | Mapping[str, Any],
        pos: torch.Tensor | None = None,
        graph: Prepared3DGraph | None = None,
        *,
        batch: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        edge_index: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        edge_mask: torch.Tensor | None = None,
        adjacency: torch.Tensor | None = None,
        edge_relation_id: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
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
        if isinstance(node_irreps, Mapping):
            payload = node_irreps
            unknown = set(payload) - self._MODEL_KEYS
            if unknown:
                raise ValueError(f"unsupported batch mapping keys: {sorted(unknown)}")
            explicitly_supplied = {
                "pos": pos,
                "graph": graph,
                "batch": batch,
                "mask": mask,
                "edge_index": edge_index,
                "edge_mask": edge_mask,
                "adjacency": adjacency,
                "edge_relation_id": edge_relation_id,
                "max_neighbors": max_neighbors,
                "context": context,
                "condition": condition,
                "order": order,
                "order_group": order_group,
                "order_periods": order_periods,
                "order_mask": order_mask,
                "refine_steps": refine_steps,
                "max_coordinate_step": max_coordinate_step,
                "refinement_centering": refinement_centering,
                "update_mask": update_mask,
                "graph_rebuilder": graph_rebuilder,
            }
            conflicts = [
                key
                for key, value in explicitly_supplied.items()
                if value is not None and key in payload
            ]
            if conflicts:
                raise ValueError(
                    "batch mapping and explicit keywords both supplied: "
                    f"{conflicts}"
                )
            node_value = self._mapping_value(
                payload,
                "node_irreps",
                ("x", "node_features"),
            )
            pos_value = self._mapping_value(payload, "pos", ("positions",))
            if not isinstance(node_value, torch.Tensor):
                raise TypeError("batch mapping node_irreps must be a tensor")
            if not isinstance(pos_value, torch.Tensor):
                raise TypeError("batch mapping pos must be a tensor")
            node_irreps = node_value
            pos = pos_value
            graph = graph if graph is not None else payload.get("graph")
            batch = batch if batch is not None else payload.get("batch")
            mask = (
                mask
                if mask is not None
                else self._mapping_value(payload, "mask", ("node_mask",))
            )
            edge_index = (
                edge_index if edge_index is not None else payload.get("edge_index")
            )
            edge_mask = edge_mask if edge_mask is not None else payload.get("edge_mask")
            adjacency = adjacency if adjacency is not None else payload.get("adjacency")
            edge_relation_id = (
                edge_relation_id
                if edge_relation_id is not None
                else payload.get("edge_relation_id")
            )
            max_neighbors = (
                max_neighbors
                if max_neighbors is not None
                else payload.get("max_neighbors")
            )
            context = context if context is not None else payload.get("context")
            condition = condition if condition is not None else payload.get("condition")
            order = order if order is not None else payload.get("order")
            order_group = (
                order_group if order_group is not None else payload.get("order_group")
            )
            order_periods = (
                order_periods
                if order_periods is not None
                else payload.get("order_periods")
            )
            order_mask = (
                order_mask if order_mask is not None else payload.get("order_mask")
            )
            refine_steps = (
                refine_steps if refine_steps is not None else payload.get("refine_steps")
            )
            max_coordinate_step = (
                max_coordinate_step
                if max_coordinate_step is not None
                else payload.get("max_coordinate_step")
            )
            refinement_centering = (
                refinement_centering
                if refinement_centering is not None
                else payload.get("refinement_centering")
            )
            update_mask = (
                update_mask if update_mask is not None else payload.get("update_mask")
            )
            graph_rebuilder = (
                graph_rebuilder
                if graph_rebuilder is not None
                else payload.get("graph_rebuilder")
            )

        if not isinstance(node_irreps, torch.Tensor):
            raise TypeError("node_irreps must be a tensor or graph mapping")
        if pos is None or not isinstance(pos, torch.Tensor):
            raise TypeError("pos must be a tensor")
        # A padded scalar node condition is naturally written as [B,M]. Convert
        # it to the model's node-level [B,M,1] convention when unambiguous.
        if (
            condition is not None
            and isinstance(condition, torch.Tensor)
            and node_irreps.ndim == 3
            and condition.ndim == 2
            and condition.shape == node_irreps.shape[:2]
            and self.config.features.condition_dim == 1
        ):
            condition = condition.unsqueeze(-1)
        return super().forward(
            node_irreps,
            pos,
            graph,
            batch=batch,
            mask=mask,
            edge_index=edge_index,
            edge_mask=edge_mask,
            adjacency=adjacency,
            edge_relation_id=edge_relation_id,
            max_neighbors=max_neighbors,
            context=context,
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


__all__ = ["ELA"]
