"""Invariant ordered-pair features for initialization and stage refresh."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .ops import st_inner
from .pair_state import BiomolecularPairContext, DensePairLayout, DensePairState
from .state import ParityState, state_invariants


def _zero_pair_state(
    layout: DensePairLayout,
    *,
    pair_width: int,
    reference: torch.Tensor,
) -> DensePairState:
    shape = (*layout.pair_mask.shape, pair_width)
    return DensePairState(
        z=reference.new_zeros(shape),
        node_mask=layout.node_mask,
        pair_mask=layout.pair_mask,
        packed_batch=layout.packed_batch,
        packed_slot=layout.packed_slot,
        lengths=layout.lengths,
    )


@dataclass(frozen=True)
class PairFeatureOutput:
    features: torch.Tensor
    pair: DensePairState


class InvariantPairFeatures(nn.Module):
    """Build directional pair features without exposing Cartesian components."""

    def __init__(
        self,
        *,
        scalar_width: int,
        num_heads: int,
        pair_width: int,
        rbf_bins: int,
        max_distance: float,
        pair_feature_dim: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.pair_width = int(pair_width)
        self.pair_feature_dim = int(pair_feature_dim)
        self.max_distance = float(max_distance)
        self.eps = float(eps)
        invariant_width = scalar_width + 5 * num_heads
        self.left = nn.Linear(invariant_width, pair_width)
        self.right = nn.Linear(invariant_width, pair_width)
        self.contraction = nn.Linear(5 * num_heads, pair_width, bias=False)
        self.rbf = nn.Linear(rbf_bins, pair_width, bias=False)
        self.relative_token = nn.Embedding(65, pair_width)
        self.molecule_left = nn.Embedding(64, pair_width)
        self.molecule_right = nn.Embedding(64, pair_width)
        self.residue_left = nn.Embedding(64, pair_width)
        self.residue_right = nn.Embedding(64, pair_width)
        self.bond = nn.Embedding(64, pair_width)
        self.relation_flags = nn.Linear(5, pair_width, bias=False)
        self.external: nn.Linear | None = None
        if pair_feature_dim:
            self.external = nn.Linear(pair_feature_dim, pair_width, bias=False)
        self.register_buffer(
            "rbf_centers",
            torch.linspace(0.0, float(max_distance), rbf_bins),
            persistent=False,
        )
        spacing = float(max_distance) / max(1, rbf_bins - 1)
        self.rbf_inverse_width = 1.0 / max(spacing * spacing, eps)
        self.norm = nn.LayerNorm(pair_width)

    @staticmethod
    def _dense_node(
        pair: DensePairState,
        value: torch.Tensor | None,
        *,
        name: str,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        if value.ndim != 1 or value.shape[0] != pair.packed_batch.shape[0]:
            raise ValueError(f"{name} must have shape (N,)")
        if value.device != pair.z.device:
            raise ValueError(f"{name} and pair state must share one device")
        return pair.unpack_node_tensor(value.to(dtype=torch.long))

    def _metadata(
        self,
        pair: DensePairState,
        context: BiomolecularPairContext | None,
    ) -> torch.Tensor:
        output = pair.z.new_zeros((*pair.pair_mask.shape, self.pair_width))
        if context is None:
            return output
        token = self._dense_node(pair, context.token_index, name="token_index")
        chain = self._dense_node(pair, context.chain_id, name="chain_id")
        entity = self._dense_node(pair, context.entity_id, name="entity_id")
        molecule = self._dense_node(
            pair,
            context.molecule_type,
            name="molecule_type",
        )
        residue = self._dense_node(pair, context.residue_type, name="residue_type")

        flags = pair.z.new_zeros((*pair.pair_mask.shape, 5))
        if token is not None:
            relative = (token[:, None, :] - token[:, :, None]).clamp(-32, 32) + 32
            output = output + self.relative_token(relative)
            flags[..., 0] = ((relative - 32).abs() == 1).to(dtype=flags.dtype)
        if chain is not None:
            flags[..., 1] = (chain[:, :, None] == chain[:, None, :]).to(
                dtype=flags.dtype
            )
            flags[..., 3] = torch.sign(
                (chain[:, None, :] - chain[:, :, None]).to(dtype=flags.dtype)
            )
        if entity is not None:
            flags[..., 2] = (entity[:, :, None] == entity[:, None, :]).to(
                dtype=flags.dtype
            )
        if molecule is not None:
            output = (
                output
                + self.molecule_left(molecule.remainder(64))[:, :, None, :]
                + self.molecule_right(molecule.remainder(64))[:, None, :, :]
            )
            flags[..., 4] = (molecule[:, :, None] == molecule[:, None, :]).to(
                dtype=flags.dtype
            )
        if residue is not None:
            output = (
                output
                + self.residue_left(residue.remainder(64))[:, :, None, :]
                + self.residue_right(residue.remainder(64))[:, None, :, :]
            )
        output = output + self.relation_flags(flags)

        if context.bond_index is not None or context.bond_type is not None:
            if context.bond_index is None or context.bond_type is None:
                raise ValueError("bond_index and bond_type must be provided together")
            edge = context.bond_index
            kind = context.bond_type
            if edge.ndim != 2 or edge.shape[0] != 2:
                raise ValueError("bond_index must have shape (2,E)")
            if kind.shape != (edge.shape[1],):
                raise ValueError("bond_type must have shape (E,)")
            source, target = edge.to(dtype=torch.long)
            if source.numel() and (
                bool((source < 0).any().item())
                or bool((target < 0).any().item())
                or bool((source >= pair.packed_batch.numel()).any().item())
                or bool((target >= pair.packed_batch.numel()).any().item())
            ):
                raise ValueError("bond_index contains an invalid packed node index")
            batch = pair.packed_batch[source]
            if source.numel() and not bool(
                (batch == pair.packed_batch[target]).all().item()
            ):
                raise ValueError("bonds must not cross batch samples")
            output = output.index_put(
                (
                    batch,
                    pair.packed_slot[source],
                    pair.packed_slot[target],
                ),
                self.bond(
                    kind.to(device=output.device, dtype=torch.long).remainder(64)
                ),
                accumulate=True,
            )

        if context.pair_features is not None:
            if self.external is None:
                raise ValueError("pair_features were supplied but pair_feature_dim=0")
            dense_pair_features = context.dense_pair_features(pair.layout)
            assert dense_pair_features is not None
            if dense_pair_features.shape[-1] != self.pair_feature_dim:
                raise ValueError(
                    f"pair_features final dimension must be {self.pair_feature_dim}"
                )
            output = output + self.external(
                dense_pair_features.to(device=output.device, dtype=output.dtype)
            )
        elif self.external is not None:
            output = output + 0.0 * self.external.weight.sum()
        return output

    def forward(
        self,
        state: ParityState,
        positions: torch.Tensor,
        layout: DensePairLayout,
        context: BiomolecularPairContext | None,
    ) -> PairFeatureOutput:
        pair = _zero_pair_state(
            layout,
            pair_width=self.pair_width,
            reference=state.even_scalar,
        )
        node_invariant = state_invariants(state, self.eps)
        dense_invariant = pair.unpack_node_tensor(node_invariant)
        features = (
            self.left(dense_invariant)[:, :, None, :]
            + self.right(dense_invariant)[:, None, :, :]
        )

        odd = pair.unpack_node_tensor(state.odd_scalar)
        polar = pair.unpack_node_tensor(state.polar_vector)
        axial = pair.unpack_node_tensor(state.axial_vector)
        even_tensor = pair.unpack_node_tensor(state.even_tensor)
        odd_tensor = pair.unpack_node_tensor(state.odd_tensor)
        contractions = torch.cat(
            (
                odd[:, :, None] * odd[:, None, :],
                (polar[:, :, None] * polar[:, None, :]).sum(dim=-1),
                (axial[:, :, None] * axial[:, None, :]).sum(dim=-1),
                st_inner(even_tensor[:, :, None], even_tensor[:, None, :]) / 5.0,
                st_inner(odd_tensor[:, :, None], odd_tensor[:, None, :]) / 5.0,
            ),
            dim=-1,
        )
        features = features + self.contraction(contractions)

        dense_position = pair.unpack_node_tensor(positions)
        work_dtype = (
            torch.float32
            if dense_position.dtype in (torch.float16, torch.bfloat16)
            else dense_position.dtype
        )
        work_position = dense_position.to(dtype=work_dtype)
        displacement = work_position[:, :, None, :] - work_position[:, None, :, :]
        # The positive epsilon keeps coincident/self-pair distances smooth, so
        # the pair embedding supports finite second derivatives.  Unlike
        # torch.cdist this path also preserves genuine float64 execution.
        distance = torch.sqrt(
            displacement.square().sum(dim=-1) + self.eps * self.eps
        ).to(dtype=features.dtype)
        center = self.rbf_centers.to(device=distance.device, dtype=distance.dtype)
        rbf = torch.exp(
            -(distance[..., None] - center).square() * self.rbf_inverse_width
        )
        features = features + self.rbf(rbf) + self._metadata(pair, context)
        features = self.norm(features)
        features = features * pair.pair_mask[..., None].to(dtype=features.dtype)
        return PairFeatureOutput(features=features, pair=pair)


class PairEmbedding(nn.Module):
    def __init__(self, **feature_kwargs: object) -> None:
        super().__init__()
        self.features = InvariantPairFeatures(**feature_kwargs)
        pair_width = int(feature_kwargs["pair_width"])
        self.projection = nn.Sequential(
            nn.Linear(pair_width, 2 * pair_width),
            nn.SiLU(),
            nn.Linear(2 * pair_width, pair_width),
        )

    def forward(
        self,
        state: ParityState,
        positions: torch.Tensor,
        layout: DensePairLayout,
        context: BiomolecularPairContext | None,
    ) -> DensePairState:
        output = self.features(state, positions, layout, context)
        z = self.projection(output.features)
        z = z * output.pair.pair_mask[..., None].to(dtype=z.dtype)
        return output.pair.with_z(z)


class NodeGeometryToPair(nn.Module):
    """Zero-initialized stage-boundary refresh of persistent pair memory."""

    def __init__(self, **feature_kwargs: object) -> None:
        super().__init__()
        self.features = InvariantPairFeatures(**feature_kwargs)
        pair_width = int(feature_kwargs["pair_width"])
        self.projection = nn.Linear(pair_width, pair_width)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        state: ParityState,
        positions: torch.Tensor,
        pair: DensePairState,
        context: BiomolecularPairContext | None,
    ) -> torch.Tensor:
        output = self.features(state, positions, pair.layout, context)
        return self.projection(output.features) * pair.pair_mask[..., None].to(
            dtype=pair.z.dtype
        )


__all__ = [
    "InvariantPairFeatures",
    "NodeGeometryToPair",
    "PairEmbedding",
    "PairFeatureOutput",
]
