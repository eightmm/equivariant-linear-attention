"""Canonical padded dense-pair layout and state containers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .ops import INTEGER_DTYPES


def _validate_layout_fields(
    *,
    node_mask: torch.Tensor,
    pair_mask: torch.Tensor,
    packed_batch: torch.Tensor,
    packed_slot: torch.Tensor,
    lengths: torch.Tensor,
) -> None:
    if node_mask.dtype != torch.bool or node_mask.ndim != 2:
        raise ValueError("node_mask must be boolean with shape (B,Nmax)")
    batch_size, max_nodes = node_mask.shape
    device = node_mask.device
    if pair_mask.dtype != torch.bool or pair_mask.shape != (
        batch_size,
        max_nodes,
        max_nodes,
    ):
        raise ValueError("pair_mask must be boolean with shape (B,Nmax,Nmax)")
    if pair_mask.device != device:
        raise ValueError("pair_mask and node_mask must share one device")
    for name, value in (
        ("packed_batch", packed_batch),
        ("packed_slot", packed_slot),
        ("lengths", lengths),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.dtype != torch.long:
            raise TypeError(f"{name} must use torch.long")
        if value.device != device:
            raise ValueError(f"{name} and node_mask must share one device")
    if packed_batch.ndim != 1 or packed_slot.shape != packed_batch.shape:
        raise ValueError("packed_batch and packed_slot must have shape (N,)")
    if lengths.shape != (batch_size,):
        raise ValueError("lengths must have shape (B,)")
    if bool((lengths < 0).any().item()) or bool((lengths > max_nodes).any().item()):
        raise ValueError("lengths must be between zero and Nmax")
    if int(lengths.sum().item()) != packed_batch.numel():
        raise ValueError("lengths must sum to the packed node count")

    expected_node_mask = torch.arange(max_nodes, device=device).unsqueeze(
        0
    ) < lengths.unsqueeze(1)
    if not torch.equal(node_mask, expected_node_mask):
        raise ValueError("node_mask must contain exactly the leading valid slots")
    expected_pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
    if not torch.equal(pair_mask, expected_pair_mask):
        raise ValueError("pair_mask must be the ordered outer mask of node_mask")

    if packed_batch.numel() == 0:
        return
    if batch_size == 0 or max_nodes == 0:
        raise ValueError("nonempty packed indices require a nonempty layout")
    if bool((packed_batch < 0).any().item()) or bool(
        (packed_batch >= batch_size).any().item()
    ):
        raise ValueError("packed_batch contains an out-of-range segment")
    if bool((packed_slot < 0).any().item()) or bool(
        (packed_slot >= max_nodes).any().item()
    ):
        raise ValueError("packed_slot contains an out-of-range slot")
    if not bool(node_mask[packed_batch, packed_slot].all().item()):
        raise ValueError("every packed node must address a valid dense slot")
    linear_slot = packed_batch * max_nodes + packed_slot
    if torch.unique(linear_slot).numel() != linear_slot.numel():
        raise ValueError("packed node assignments must be one-to-one")


def _validate_pair_tensor(
    z: torch.Tensor,
    *,
    pair_mask: torch.Tensor,
    device: torch.device,
) -> None:
    if not isinstance(z, torch.Tensor):
        raise TypeError("z must be a tensor")
    if not torch.is_floating_point(z):
        raise TypeError("z must use a floating-point dtype")
    if z.ndim != 4 or z.shape[:3] != pair_mask.shape:
        raise ValueError("z must have shape (B,Nmax,Nmax,Cz)")
    if z.device != device:
        raise ValueError("z and pair layout must share one device")


@dataclass(frozen=True, slots=True)
class DensePairLayout:
    """Mapping between packed nodes and padded ordered interaction pairs.

    The leading dense dimension indexes isolated interaction components.  With
    no ``interaction_group`` those components are the input graphs.  When a
    group is supplied, every distinct ``(graph, group)`` pair is packed into a
    separate component, so a dense contraction cannot leak across groups.
    """

    node_mask: torch.Tensor
    pair_mask: torch.Tensor
    packed_batch: torch.Tensor
    packed_slot: torch.Tensor
    lengths: torch.Tensor

    def __post_init__(self) -> None:
        _validate_layout_fields(
            node_mask=self.node_mask,
            pair_mask=self.pair_mask,
            packed_batch=self.packed_batch,
            packed_slot=self.packed_slot,
            lengths=self.lengths,
        )

    @classmethod
    def _from_validated(
        cls,
        *,
        node_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        packed_batch: torch.Tensor,
        packed_slot: torch.Tensor,
        lengths: torch.Tensor,
    ) -> DensePairLayout:
        instance = object.__new__(cls)
        object.__setattr__(instance, "node_mask", node_mask)
        object.__setattr__(instance, "pair_mask", pair_mask)
        object.__setattr__(instance, "packed_batch", packed_batch)
        object.__setattr__(instance, "packed_slot", packed_slot)
        object.__setattr__(instance, "lengths", lengths)
        return instance

    @property
    def num_nodes(self) -> int:
        return int(self.packed_batch.numel())

    @property
    def num_components(self) -> int:
        return int(self.node_mask.shape[0])

    @property
    def max_nodes(self) -> int:
        return int(self.node_mask.shape[1])

    @property
    def device(self) -> torch.device:
        return self.node_mask.device

    def gather_nodes(self, dense: torch.Tensor) -> torch.Tensor:
        """Gather ``[B,Nmax,...]`` values back into the packed node order."""

        if not isinstance(dense, torch.Tensor):
            raise TypeError("dense must be a tensor")
        if dense.ndim < 2 or dense.shape[:2] != self.node_mask.shape:
            raise ValueError("dense must begin with shape (B,Nmax)")
        if dense.device != self.device:
            raise ValueError("dense and layout must share one device")
        return dense[self.packed_batch, self.packed_slot]

    def unpack_node_tensor(self, packed: torch.Tensor) -> torch.Tensor:
        """Scatter ``[N,...]`` packed values into zero-padded dense slots."""

        if not isinstance(packed, torch.Tensor):
            raise TypeError("packed must be a tensor")
        if packed.ndim < 1 or packed.shape[0] != self.num_nodes:
            raise ValueError("packed must begin with the packed node count")
        if packed.device != self.device:
            raise ValueError("packed and layout must share one device")
        output = packed.new_zeros(
            (self.num_components, self.max_nodes, *packed.shape[1:])
        )
        if self.num_nodes == 0:
            return output
        return output.index_put(
            (self.packed_batch, self.packed_slot),
            packed,
        )

    def with_z(self, z: torch.Tensor) -> DensePairState:
        """Attach pair features without changing the structural layout."""

        _validate_pair_tensor(z, pair_mask=self.pair_mask, device=self.device)
        return DensePairState._from_validated(
            z=z,
            node_mask=self.node_mask,
            pair_mask=self.pair_mask,
            packed_batch=self.packed_batch,
            packed_slot=self.packed_slot,
            lengths=self.lengths,
        )


@dataclass(frozen=True, slots=True)
class DensePairState:
    """Directed pair features and their exact packed/padded correspondence."""

    z: torch.Tensor
    node_mask: torch.Tensor
    pair_mask: torch.Tensor
    packed_batch: torch.Tensor
    packed_slot: torch.Tensor
    lengths: torch.Tensor

    def __post_init__(self) -> None:
        _validate_layout_fields(
            node_mask=self.node_mask,
            pair_mask=self.pair_mask,
            packed_batch=self.packed_batch,
            packed_slot=self.packed_slot,
            lengths=self.lengths,
        )
        _validate_pair_tensor(
            self.z,
            pair_mask=self.pair_mask,
            device=self.node_mask.device,
        )

    @classmethod
    def _from_validated(
        cls,
        *,
        z: torch.Tensor,
        node_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        packed_batch: torch.Tensor,
        packed_slot: torch.Tensor,
        lengths: torch.Tensor,
    ) -> DensePairState:
        instance = object.__new__(cls)
        object.__setattr__(instance, "z", z)
        object.__setattr__(instance, "node_mask", node_mask)
        object.__setattr__(instance, "pair_mask", pair_mask)
        object.__setattr__(instance, "packed_batch", packed_batch)
        object.__setattr__(instance, "packed_slot", packed_slot)
        object.__setattr__(instance, "lengths", lengths)
        return instance

    @property
    def layout(self) -> DensePairLayout:
        return DensePairLayout._from_validated(
            node_mask=self.node_mask,
            pair_mask=self.pair_mask,
            packed_batch=self.packed_batch,
            packed_slot=self.packed_slot,
            lengths=self.lengths,
        )

    def masked_z(self) -> torch.Tensor:
        return self.z * self.pair_mask.unsqueeze(-1).to(dtype=self.z.dtype)

    def transpose(self) -> DensePairState:
        """Reverse every ordered pair while preserving packed node order."""

        return DensePairState._from_validated(
            z=self.z.transpose(1, 2),
            node_mask=self.node_mask,
            pair_mask=self.pair_mask.transpose(1, 2),
            packed_batch=self.packed_batch,
            packed_slot=self.packed_slot,
            lengths=self.lengths,
        )

    def gather_nodes(self, dense: torch.Tensor) -> torch.Tensor:
        if not isinstance(dense, torch.Tensor):
            raise TypeError("dense must be a tensor")
        if dense.ndim < 2 or dense.shape[:2] != self.node_mask.shape:
            raise ValueError("dense must begin with shape (B,Nmax)")
        if dense.device != self.z.device:
            raise ValueError("dense and pair state must share one device")
        return dense[self.packed_batch, self.packed_slot]

    def unpack_node_tensor(self, packed: torch.Tensor) -> torch.Tensor:
        if not isinstance(packed, torch.Tensor):
            raise TypeError("packed must be a tensor")
        if packed.ndim < 1 or packed.shape[0] != self.packed_batch.numel():
            raise ValueError("packed must begin with the packed node count")
        if packed.device != self.z.device:
            raise ValueError("packed and pair state must share one device")
        output = packed.new_zeros(
            (self.node_mask.shape[0], self.node_mask.shape[1], *packed.shape[1:])
        )
        if packed.shape[0] == 0:
            return output
        return output.index_put(
            (self.packed_batch, self.packed_slot),
            packed,
        )

    def with_z(self, z: torch.Tensor) -> DensePairState:
        _validate_pair_tensor(z, pair_mask=self.pair_mask, device=self.z.device)
        return DensePairState._from_validated(
            z=z,
            node_mask=self.node_mask,
            pair_mask=self.pair_mask,
            packed_batch=self.packed_batch,
            packed_slot=self.packed_slot,
            lengths=self.lengths,
        )


def build_dense_pair_layout(
    batch_index: torch.Tensor,
    interaction_group: torch.Tensor | None = None,
    max_pair_tokens: int = 512,
) -> DensePairLayout:
    """Build the one canonical padded layout for exact dense pair mixing."""

    if not isinstance(batch_index, torch.Tensor):
        raise TypeError("batch_index must be a tensor")
    if batch_index.ndim != 1:
        raise ValueError("batch_index must have shape (N,)")
    if batch_index.dtype not in INTEGER_DTYPES:
        raise TypeError("batch_index must use an integer dtype")
    if isinstance(max_pair_tokens, bool) or not isinstance(max_pair_tokens, int):
        raise TypeError("max_pair_tokens must be an integer")
    if max_pair_tokens <= 0:
        raise ValueError("max_pair_tokens must be positive")

    batch = batch_index.to(dtype=torch.long)
    if batch.numel() and bool((batch < 0).any().item()):
        raise ValueError("batch_index values must be nonnegative")
    if batch.numel():
        graph_count = int(batch.max().item()) + 1
        graph_lengths = torch.bincount(batch, minlength=graph_count)
        if bool((graph_lengths == 0).any().item()):
            raise ValueError("batch_index IDs must be contiguous from zero")

    if interaction_group is None:
        interaction = batch
    else:
        if not isinstance(interaction_group, torch.Tensor):
            raise TypeError("interaction_group must be a tensor")
        if interaction_group.shape != batch.shape:
            raise ValueError("interaction_group must have shape (N,)")
        if interaction_group.device != batch.device:
            raise ValueError("interaction_group and batch_index must share one device")
        if interaction_group.dtype not in INTEGER_DTYPES:
            raise TypeError("interaction_group must use an integer dtype")
        group = interaction_group.to(dtype=torch.long)
        if group.numel() and bool((group < 0).any().item()):
            raise ValueError("interaction_group values must be nonnegative")
        if batch.numel():
            _, interaction = torch.unique(
                torch.stack((batch, group), dim=-1),
                dim=0,
                sorted=True,
                return_inverse=True,
            )
            interaction = interaction.to(dtype=torch.long)
        else:
            interaction = batch

    component_count = (
        0 if interaction.numel() == 0 else int(interaction.max().item()) + 1
    )
    lengths = torch.zeros(
        component_count,
        dtype=torch.long,
        device=batch.device,
    )
    if interaction.numel():
        lengths.index_add_(0, interaction, torch.ones_like(interaction))
    max_nodes = 0 if component_count == 0 else int(lengths.max().item())
    if max_nodes > max_pair_tokens:
        raise ValueError(
            "dense pair component length "
            f"{max_nodes} exceeds max_pair_tokens={max_pair_tokens}"
        )

    packed_slot = torch.empty_like(interaction)
    for component in range(component_count):
        selected = interaction == component
        packed_slot[selected] = torch.arange(
            int(lengths[component].item()),
            device=batch.device,
            dtype=torch.long,
        )
    node_mask = torch.arange(max_nodes, device=batch.device).unsqueeze(
        0
    ) < lengths.unsqueeze(1)
    pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
    return DensePairLayout._from_validated(
        node_mask=node_mask,
        pair_mask=pair_mask,
        packed_batch=interaction,
        packed_slot=packed_slot,
        lengths=lengths,
    )


@dataclass(frozen=True, slots=True)
class BiomolecularPairContext:
    """Optional ordered-pair metadata aligned to the packed node order."""

    token_index: torch.Tensor | None = None
    chain_id: torch.Tensor | None = None
    entity_id: torch.Tensor | None = None
    molecule_type: torch.Tensor | None = None
    residue_type: torch.Tensor | None = None
    bond_index: torch.Tensor | None = None
    bond_type: torch.Tensor | None = None
    pair_features: torch.Tensor | None = None

    def __post_init__(self) -> None:
        node_count: int | None = None
        device: torch.device | None = None
        for name in (
            "token_index",
            "chain_id",
            "entity_id",
            "molecule_type",
            "residue_type",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.ndim != 1:
                raise ValueError(f"{name} must have shape (N,)")
            if value.dtype not in INTEGER_DTYPES:
                raise TypeError(f"{name} must use an integer dtype")
            node_count = value.shape[0] if node_count is None else node_count
            if value.shape[0] != node_count:
                raise ValueError("node-level metadata must share one node count")
            device = value.device if device is None else device
            if value.device != device:
                raise ValueError("all pair context tensors must share one device")

        if (self.bond_index is None) != (self.bond_type is None):
            raise ValueError("bond_index and bond_type must be supplied together")
        if self.bond_index is not None and self.bond_type is not None:
            if not isinstance(self.bond_index, torch.Tensor):
                raise TypeError("bond_index must be a tensor")
            if not isinstance(self.bond_type, torch.Tensor):
                raise TypeError("bond_type must be a tensor")
            if self.bond_index.dtype not in INTEGER_DTYPES:
                raise TypeError("bond_index must use an integer dtype")
            if self.bond_index.ndim != 2 or self.bond_index.shape[0] != 2:
                raise ValueError("bond_index must have shape (2,E)")
            if self.bond_type.dtype not in INTEGER_DTYPES:
                raise TypeError("bond_type must use an integer dtype")
            if self.bond_type.shape != (self.bond_index.shape[1],):
                raise ValueError("bond_type must have shape (E,)")
            device = self.bond_index.device if device is None else device
            if self.bond_index.device != device or self.bond_type.device != device:
                raise ValueError("all pair context tensors must share one device")

        if self.pair_features is not None:
            if not isinstance(self.pair_features, torch.Tensor):
                raise TypeError("pair_features must be a tensor")
            if not torch.is_floating_point(self.pair_features):
                raise TypeError("pair_features must use a floating-point dtype")
            if not bool(torch.isfinite(self.pair_features).all().item()):
                raise ValueError("pair_features must contain only finite values")
            if (
                self.pair_features.ndim != 3
                or self.pair_features.shape[0] != self.pair_features.shape[1]
            ):
                raise ValueError("pair_features must have shape (N,N,D)")
            node_count = (
                self.pair_features.shape[0] if node_count is None else node_count
            )
            if self.pair_features.shape[:2] != (node_count, node_count):
                raise ValueError("pair_features must match the packed node count")
            device = self.pair_features.device if device is None else device
            if self.pair_features.device != device:
                raise ValueError("all pair context tensors must share one device")

    def validate(
        self,
        *,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> BiomolecularPairContext:
        """Validate metadata against the graph that will consume it."""

        if num_nodes < 0:
            raise ValueError("num_nodes must be nonnegative")
        for name in (
            "token_index",
            "chain_id",
            "entity_id",
            "molecule_type",
            "residue_type",
        ):
            value = getattr(self, name)
            if value is not None and value.shape != (num_nodes,):
                raise ValueError(f"{name} must have shape ({num_nodes},)")
            if value is not None and value.device != device:
                raise ValueError(f"{name} and graph must share one device")
        if self.bond_index is not None:
            if self.bond_index.device != device or self.bond_type is None:
                raise ValueError("bond metadata and graph must share one device")
            if self.bond_index.numel() and (
                bool((self.bond_index < 0).any().item())
                or bool((self.bond_index >= num_nodes).any().item())
            ):
                raise ValueError("bond_index contains an out-of-range node")
        if self.pair_features is not None:
            if self.pair_features.shape[:2] != (num_nodes, num_nodes):
                raise ValueError(
                    "pair_features must match the graph's packed node count"
                )
            if self.pair_features.device != device:
                raise ValueError("pair_features and graph must share one device")
            if dtype is not None and self.pair_features.dtype != dtype:
                raise ValueError("pair_features and graph must share one dtype")
        return self

    @property
    def num_nodes(self) -> int | None:
        """Return the represented node count when node data determines it."""

        for name in (
            "token_index",
            "chain_id",
            "entity_id",
            "molecule_type",
            "residue_type",
        ):
            value = getattr(self, name)
            if value is not None:
                return int(value.shape[0])
        if self.pair_features is not None:
            return int(self.pair_features.shape[0])
        return None

    def to(self, *args: Any, **kwargs: Any) -> BiomolecularPairContext:
        """Move metadata with tensor-like semantics while preserving indices."""

        tensors = tuple(
            value
            for value in (
                self.token_index,
                self.chain_id,
                self.entity_id,
                self.molecule_type,
                self.residue_type,
                self.bond_index,
                self.bond_type,
                self.pair_features,
            )
            if value is not None
        )
        if not tensors:
            return self
        probe = torch.empty(0, device=tensors[0].device).to(*args, **kwargs)
        device = probe.device

        def indexed(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.to(device=device)

        pair_features = (
            None
            if self.pair_features is None
            else self.pair_features.to(*args, **kwargs)
        )
        return BiomolecularPairContext(
            token_index=indexed(self.token_index),
            chain_id=indexed(self.chain_id),
            entity_id=indexed(self.entity_id),
            molecule_type=indexed(self.molecule_type),
            residue_type=indexed(self.residue_type),
            bond_index=indexed(self.bond_index),
            bond_type=indexed(self.bond_type),
            pair_features=pair_features,
        )

    @classmethod
    def collate(
        cls,
        contexts: Iterable[BiomolecularPairContext],
        *,
        node_counts: Sequence[int] | None = None,
    ) -> BiomolecularPairContext:
        """Collate sample contexts and offset their packed bond indices."""

        values = tuple(contexts)
        if not values:
            raise ValueError("cannot collate an empty context collection")
        if any(not isinstance(value, cls) for value in values):
            raise TypeError("BiomolecularPairContext.collate accepts only contexts")
        if node_counts is None:
            inferred = tuple(value.num_nodes for value in values)
            if any(count is None for count in inferred):
                raise ValueError("node_counts are required for metadata-free contexts")
            counts = tuple(int(count) for count in inferred if count is not None)
        else:
            counts = tuple(node_counts)
            if len(counts) != len(values):
                raise ValueError("node_counts length must match the context count")
            if any(
                isinstance(count, bool) or not isinstance(count, int)
                for count in counts
            ):
                raise TypeError("node_counts must contain integers")
            if any(count < 0 for count in counts):
                raise ValueError("node_counts must be non-negative")
        for value, count in zip(values, counts, strict=True):
            if value.num_nodes is not None and value.num_nodes != count:
                raise ValueError("node_counts must match every context")

        device = next(
            (
                tensor.device
                for value in values
                for tensor in (
                    value.token_index,
                    value.chain_id,
                    value.entity_id,
                    value.molecule_type,
                    value.residue_type,
                    value.bond_index,
                    value.bond_type,
                    value.pair_features,
                )
                if tensor is not None
            ),
            torch.device("cpu"),
        )
        if any(
            tensor.device != device
            for value in values
            for tensor in (
                value.token_index,
                value.chain_id,
                value.entity_id,
                value.molecule_type,
                value.residue_type,
                value.bond_index,
                value.bond_type,
                value.pair_features,
            )
            if tensor is not None
        ):
            raise ValueError("all collated contexts must share one device")

        node_fields: dict[str, torch.Tensor | None] = {}
        for name in (
            "token_index",
            "chain_id",
            "entity_id",
            "molecule_type",
            "residue_type",
        ):
            field_values = tuple(getattr(value, name) for value in values)
            if all(item is None for item in field_values):
                node_fields[name] = None
            elif any(item is None for item in field_values):
                raise ValueError(f"{name} must be present on every collated context")
            else:
                node_fields[name] = torch.cat(
                    tuple(item for item in field_values if item is not None)
                )

        pair_values = tuple(value.pair_features for value in values)
        pair_features: torch.Tensor | None = None
        if not all(item is None for item in pair_values):
            if any(item is None for item in pair_values):
                raise ValueError(
                    "pair_features must be present on every collated context"
                )
            present = tuple(item for item in pair_values if item is not None)
            feature_widths = {item.shape[-1] for item in present}
            dtypes = {item.dtype for item in present}
            if len(feature_widths) != 1 or len(dtypes) != 1:
                raise ValueError("collated pair_features must share width and dtype")
            total = sum(counts)
            width = present[0].shape[-1]
            pair_features = present[0].new_zeros((total, total, width))
            offset = 0
            for item, count in zip(present, counts, strict=True):
                pair_features[offset : offset + count, offset : offset + count] = item
                offset += count

        bond_indices: list[torch.Tensor] = []
        bond_types: list[torch.Tensor] = []
        offset = 0
        for value, count in zip(values, counts, strict=True):
            if value.bond_index is not None:
                assert value.bond_type is not None
                bond_indices.append(value.bond_index + offset)
                bond_types.append(value.bond_type)
            offset += count
        return cls(
            **node_fields,
            bond_index=(torch.cat(bond_indices, dim=1) if bond_indices else None),
            bond_type=(torch.cat(bond_types) if bond_types else None),
            pair_features=pair_features,
        )

    def dense_pair_features(
        self,
        layout: DensePairLayout,
    ) -> torch.Tensor | None:
        """Pack global ordered features and zero every padded pair."""

        if self.pair_features is None:
            return None
        self.validate(
            num_nodes=layout.num_nodes,
            device=layout.device,
            dtype=self.pair_features.dtype,
        )
        packed_index = torch.arange(
            layout.num_nodes,
            device=layout.device,
            dtype=torch.long,
        )
        dense_index = layout.unpack_node_tensor(packed_index)
        dense = self.pair_features[
            dense_index[:, :, None],
            dense_index[:, None, :],
        ]
        return dense * layout.pair_mask.unsqueeze(-1).to(dtype=dense.dtype)


__all__ = [
    "BiomolecularPairContext",
    "DensePairLayout",
    "DensePairState",
    "build_dense_pair_layout",
]
