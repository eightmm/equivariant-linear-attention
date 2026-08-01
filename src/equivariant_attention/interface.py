from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .api import ELA as _TensorELA
from .canonical import ELA as _CanonicalELA
from .context import ELAContext, GeometryRebuilder, OrderContext
from .data import BatchLayout
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
    _MAPPING_ALIASES = {
        "pos": ("pos", "positions"),
        "mask": ("mask", "node_mask"),
    }

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

    @classmethod
    def _mapping_contains_argument(
        cls,
        payload: Mapping[str, Any],
        name: str,
    ) -> bool:
        return any(key in payload for key in cls._MAPPING_ALIASES.get(name, (name,)))

    def _flatten_optional_node_tensor(
        self,
        value: torch.Tensor | None,
        layout: BatchLayout,
        *,
        name: str,
    ) -> torch.Tensor | None:
        # A graph condition is [B,C], whereas a padded node condition is
        # [B,M,C]. Scalar node conditions may use [B,M] only when condition_dim
        # is exactly one. This avoids mistaking graph [B,C] for node [B,M] when
        # C happens to equal M.
        if (
            value is not None
            and layout.kind == "padded"
            and name in {"condition", "context.condition"}
        ):
            if layout.node_mask is None:
                raise RuntimeError("padded layout is incomplete")
            if value.ndim >= 3 and value.shape[:2] == layout.node_mask.shape:
                return layout.flatten_node_tensor(value, name=name)
            if (
                value.ndim == 2
                and value.shape == layout.node_mask.shape
                and value.is_floating_point()
                and self.config.features.condition_dim == 1
            ):
                return layout.flatten_node_tensor(
                    value.unsqueeze(-1),
                    name=name,
                )
            return value
        return _TensorELA._flatten_optional_node_tensor(
            value,
            layout,
            name=name,
        )

    def forward_prepared(
        self,
        node_irreps: torch.Tensor,
        pos: torch.Tensor,
        graph: Prepared3DGraph,
        *,
        context: ELAContext | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the validated flat hot path without input packing or graph checks.

        Build ``graph`` once with :meth:`prepare_graph`. The caller must preserve
        the packed node order and graph membership. This method is intended for
        repeated training, inference, profiling, and ``torch.compile``.
        """

        if not isinstance(node_irreps, torch.Tensor):
            raise TypeError("node_irreps must be a tensor")
        if not isinstance(pos, torch.Tensor):
            raise TypeError("pos must be a tensor")
        if not isinstance(graph, Prepared3DGraph):
            raise TypeError("graph must be a Prepared3DGraph")
        return dict(
            _CanonicalELA.forward(
                self,
                node_irreps,
                pos,
                graph,
                context=context,
            )
        )

    def forward(
        self,
        node_irreps: torch.Tensor | Mapping[str, Any],
        pos: torch.Tensor | None = None,
        graph: Prepared3DGraph | None = None,
        *,
        batch: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        edge_index: (
            torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None
        ) = None,
        edge_mask: torch.Tensor | None = None,
        adjacency: torch.Tensor | None = None,
        edge_relation_id: (
            torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None
        ) = None,
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
                name
                for name, value in explicitly_supplied.items()
                if value is not None
                and self._mapping_contains_argument(payload, name)
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
            graph = payload.get("graph")
            batch = payload.get("batch")
            mask = self._mapping_value(payload, "mask", ("node_mask",))
            edge_index = payload.get("edge_index")
            edge_mask = payload.get("edge_mask")
            adjacency = payload.get("adjacency")
            edge_relation_id = payload.get("edge_relation_id")
            max_neighbors = payload.get("max_neighbors")
            context = payload.get("context")
            condition = payload.get("condition")
            order = payload.get("order")
            order_group = payload.get("order_group")
            order_periods = payload.get("order_periods")
            order_mask = payload.get("order_mask")
            refine_steps = payload.get("refine_steps")
            max_coordinate_step = payload.get("max_coordinate_step")
            refinement_centering = payload.get("refinement_centering")
            update_mask = payload.get("update_mask")
            graph_rebuilder = payload.get("graph_rebuilder")

        if not isinstance(node_irreps, torch.Tensor):
            raise TypeError("node_irreps must be a tensor or graph mapping")
        if pos is None or not isinstance(pos, torch.Tensor):
            raise TypeError("pos must be a tensor")
        # Padded scalar node conditions are naturally written as [B,M].
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
