from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention.neighbor_providers import (
    ExternalCallableNeighborProvider,
    NeighborCapabilities,
    NeighborProvider,
    PrecomputedNeighborProvider,
    ReferenceRadiusNeighborProvider,
    VerletRadiusNeighborProvider,
    neighbor_provider_capabilities,
    unsupported_neighbor_capability,
)


def _geometry() -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    return pos, batch


def test_reference_radius_uses_strict_squared_admission_and_row_major_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pos, batch = _geometry()
    provider = ReferenceRadiusNeighborProvider()

    def forbidden(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("reference radius admission must not take a square root")

    monkeypatch.setattr(torch, "sqrt", forbidden)
    first = provider(pos, batch, cutoff=1.0)
    second = provider(pos, batch, cutoff=1.0)

    expected = torch.tensor(
        [
            [0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
            [0, 1, 0, 1, 2, 1, 2, 3, 4, 3, 4],
        ],
        dtype=torch.long,
    )
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)
    assert torch.equal(first[:, first[0] == first[1]], torch.arange(5).repeat(2, 1))
    assert not bool(((first[0] < 3) & (first[1] >= 3)).any())
    assert not bool(((first[0] >= 3) & (first[1] < 3)).any())
    assert isinstance(provider, NeighborProvider)


def test_precomputed_provider_is_fixed_and_owns_immutable_storage() -> None:
    pos, batch = _geometry()
    source = ReferenceRadiusNeighborProvider()(pos, batch, cutoff=1.0)
    expected = source.clone()
    provider = PrecomputedNeighborProvider(source, num_nodes=pos.shape[0])
    source.zero_()

    first = provider(pos, batch, cutoff=0.1)
    first.zero_()
    second = provider(pos, batch, cutoff=100.0)

    assert torch.equal(second, expected)
    with pytest.raises(FrozenInstanceError):
        provider.num_nodes = 2  # type: ignore[misc]


def test_reference_precomputed_external_and_zero_skin_verlet_are_equivalent() -> None:
    pos, batch = _geometry()
    reference_provider = ReferenceRadiusNeighborProvider()
    reference = reference_provider(pos, batch, cutoff=0.8)
    precomputed = PrecomputedNeighborProvider(reference, num_nodes=pos.shape[0])
    external = ExternalCallableNeighborProvider(
        lambda value, assignment, *, cutoff: reference_provider(
            value,
            assignment,
            cutoff=cutoff,
        )
    )
    verlet = VerletRadiusNeighborProvider(skin=0.0)

    assert torch.equal(precomputed(pos, batch, cutoff=0.8), reference)
    assert torch.equal(external(pos, batch, cutoff=0.8), reference)
    assert torch.equal(verlet(pos, batch, cutoff=0.8), reference)


def test_external_adapter_canonicalizes_edge_order() -> None:
    pos, batch = _geometry()
    expected = ReferenceRadiusNeighborProvider()(pos, batch, cutoff=0.8)
    provider = ExternalCallableNeighborProvider(
        lambda _pos, _batch, *, cutoff: expected.flip(1)
    )

    assert torch.equal(provider(pos, batch, cutoff=0.8), expected)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: ExternalCallableNeighborProvider(
                lambda _pos, _batch, *, cutoff: torch.tensor([0, 1])
            ),
            "shape",
        ),
        (
            lambda: ExternalCallableNeighborProvider(
                lambda _pos, _batch, *, cutoff: torch.eye(2)
            ),
            "integer",
        ),
        (
            lambda: ExternalCallableNeighborProvider(
                lambda _pos, _batch, *, cutoff: torch.tensor(
                    [[0, 1, 2, 3, 4], [0, 1, 2, 3, 5]]
                )
            ),
            "range",
        ),
        (
            lambda: ExternalCallableNeighborProvider(
                lambda _pos, _batch, *, cutoff: torch.tensor(
                    [[0, 1, 2, 3, 4, 0], [0, 1, 2, 3, 4, 3]]
                )
            ),
            "graph",
        ),
        (
            lambda: ExternalCallableNeighborProvider(
                lambda _pos, _batch, *, cutoff: torch.tensor(
                    [[0, 1, 2, 3], [0, 1, 2, 3]]
                )
            ),
            "self edge",
        ),
    ],
)
def test_external_adapter_validates_edge_contract(
    factory: object,
    message: str,
) -> None:
    pos, batch = _geometry()
    provider = factory()  # type: ignore[operator]

    with pytest.raises((TypeError, ValueError), match=message):
        provider(pos, batch, cutoff=1.0)


def test_external_adapter_rejects_wrong_device() -> None:
    pos, batch = _geometry()
    provider = ExternalCallableNeighborProvider(
        lambda _pos, _batch, *, cutoff: torch.empty(
            (2, 0),
            dtype=torch.long,
            device="meta",
        )
    )

    with pytest.raises(ValueError, match="device"):
        provider(pos, batch, cutoff=1.0)


@pytest.mark.parametrize(
    ("pos", "batch", "exception", "message"),
    [
        (
            torch.zeros(3, 2),
            torch.zeros(3, dtype=torch.long),
            ValueError,
            "shape",
        ),
        (
            torch.zeros(3, 3, dtype=torch.long),
            torch.zeros(3, dtype=torch.long),
            TypeError,
            "floating",
        ),
        (
            torch.tensor(
                [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]
            ),
            torch.zeros(2, dtype=torch.long),
            ValueError,
            "finite",
        ),
        (
            torch.zeros(3, 3),
            torch.zeros(2, dtype=torch.long),
            ValueError,
            "batch",
        ),
        (
            torch.zeros(3, 3),
            torch.zeros(3),
            TypeError,
            "integer",
        ),
        (
            torch.zeros(3, 3),
            torch.tensor([0, 2, 2]),
            ValueError,
            "contiguous",
        ),
    ],
)
def test_reference_radius_validates_position_and_batch(
    pos: torch.Tensor,
    batch: torch.Tensor,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        ReferenceRadiusNeighborProvider()(pos, batch, cutoff=1.0)


@pytest.mark.parametrize("cutoff", [0.0, -1.0, float("inf"), float("nan"), True])
def test_provider_rejects_invalid_cutoff(cutoff: object) -> None:
    pos, batch = _geometry()

    with pytest.raises((TypeError, ValueError), match="cutoff"):
        ReferenceRadiusNeighborProvider()(
            pos,
            batch,
            cutoff=cutoff,  # type: ignore[arg-type]
        )


def test_verlet_keeps_candidates_fixed_across_a_subthreshold_cutoff_crossing() -> (
    None
):
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
        dtype=torch.float64,
    )
    batch = torch.zeros(2, dtype=torch.long)
    provider = VerletRadiusNeighborProvider(skin=0.4)

    initial_candidates = provider(pos, batch, cutoff=1.0)
    initial_active = ReferenceRadiusNeighborProvider()(pos, batch, cutoff=1.0)
    moved = pos.clone()
    moved[1, 0] = 0.95
    moved_candidates = provider(moved, batch, cutoff=1.0)
    moved_active = ReferenceRadiusNeighborProvider()(moved, batch, cutoff=1.0)

    assert torch.equal(moved_candidates, initial_candidates)
    assert provider.rebuild_count == 1
    assert initial_active.shape[1] == 2
    assert moved_active.shape[1] == 4
    assert moved_candidates.shape[1] == 4


def test_verlet_rebuilds_after_half_skin_motion_and_can_be_invalidated() -> None:
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]],
        dtype=torch.float64,
    )
    batch = torch.zeros(2, dtype=torch.long)
    provider = VerletRadiusNeighborProvider(skin=0.4)

    initial = provider(pos, batch, cutoff=1.0)
    moved = pos.clone()
    moved[1, 0] = 0.9
    rebuilt = provider(moved, batch, cutoff=1.0)

    assert initial.shape[1] == 2
    assert rebuilt.shape[1] == 4
    assert provider.rebuild_count == 2
    assert not provider.needs_rebuild(moved, batch, cutoff=1.0)

    provider.invalidate()
    assert provider.needs_rebuild(moved, batch, cutoff=1.0)
    assert torch.equal(provider(moved, batch, cutoff=1.0), rebuilt)
    assert provider.rebuild_count == 3


def test_verlet_invalidates_cache_when_cutoff_or_batch_changes() -> None:
    pos, batch = _geometry()
    provider = VerletRadiusNeighborProvider(skin=0.25)

    provider(pos, batch, cutoff=0.8)
    assert provider.rebuild_count == 1
    provider(pos, batch, cutoff=0.9)
    assert provider.rebuild_count == 2

    changed_batch = torch.zeros_like(batch)
    provider(pos, changed_batch, cutoff=0.9)
    assert provider.rebuild_count == 3


@pytest.mark.parametrize(
    "capability",
    ["hard_learned_topk", "pbc", "cell_list"],
)
def test_unimplemented_neighbor_capabilities_are_rejected(capability: str) -> None:
    with pytest.raises(NotImplementedError, match="not implemented.*not production"):
        unsupported_neighbor_capability(capability)


def test_neighbor_capability_receipts_do_not_overclaim_reference_builders() -> None:
    pos, batch = _geometry()
    edge_index = ReferenceRadiusNeighborProvider()(pos, batch, cutoff=1.0)
    providers = (
        PrecomputedNeighborProvider(edge_index, num_nodes=pos.shape[0]),
        ReferenceRadiusNeighborProvider(),
        ExternalCallableNeighborProvider(
            lambda *_args, **_kwargs: edge_index,
        ),
        VerletRadiusNeighborProvider(skin=0.5),
    )

    receipts = tuple(neighbor_provider_capabilities(provider) for provider in providers)

    assert all(isinstance(receipt, NeighborCapabilities) for receipt in receipts)
    assert receipts[0].complexity == "precomputed"
    assert receipts[1].complexity == "quadratic_reference"
    assert receipts[2].complexity == "external_unknown"
    assert receipts[3].supports_skin
    assert not receipts[0].supports_skin
    assert all(not receipt.production_ready for receipt in receipts[1:])
    assert all(not receipt.supports_pbc for receipt in receipts)
    assert all(not receipt.supports_cell_list for receipt in receipts)
    assert receipts[1].deterministic_selection
    assert not receipts[2].deterministic_selection


def test_unknown_protocol_provider_gets_conservative_capabilities() -> None:
    class CustomProvider:
        def __call__(
            self,
            pos: torch.Tensor,
            batch: torch.Tensor,
            *,
            cutoff: float,
        ) -> torch.Tensor:
            del cutoff
            nodes = torch.arange(pos.shape[0], device=pos.device)
            return torch.stack([nodes, nodes])

    receipt = neighbor_provider_capabilities(CustomProvider())

    assert receipt.provider == "CustomProvider"
    assert receipt.complexity == "unknown"
    assert not receipt.production_ready
    assert not receipt.deterministic_selection


def _provider_model(
    *,
    coordinate_updates: bool = False,
    coordinate_neighbor_policy: str = "error",
) -> EquivariantAttention:
    return EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            hidden_irreps="8x0e + 2x1o",
            output_irreps="1x0e + 1x1o",
            num_layers=2,
            num_heads=2,
            local_head_counts=(2, 2),
            local_cutoff=2.0,
            use_gated_local_transport=True,
            coordinate_updates=coordinate_updates,
            coordinate_neighbor_policy=coordinate_neighbor_policy,
        )
    ).double()


def test_model_neighbor_provider_matches_explicit_edges_and_gradients() -> None:
    pos, batch = _geometry()
    feats = torch.randn(
        pos.shape[0],
        4,
        generator=torch.Generator().manual_seed(812),
        dtype=torch.float64,
    )
    provider = ReferenceRadiusNeighborProvider()
    edge_index = provider(pos, batch, cutoff=2.0)
    torch.manual_seed(813)
    reference = _provider_model()
    torch.manual_seed(813)
    candidate = _provider_model()
    reference_feats = feats.clone().requires_grad_(True)
    candidate_feats = feats.clone().requires_grad_(True)
    reference_pos = pos.clone().requires_grad_(True)
    candidate_pos = pos.clone().requires_grad_(True)

    expected = reference(
        reference_feats,
        reference_pos,
        batch=batch,
        edge_index=edge_index,
    )
    actual = candidate(
        candidate_feats,
        candidate_pos,
        batch=batch,
        neighbor_provider=provider,
    )

    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    expected_loss = sum(value.square().sum() for value in expected.values())
    actual_loss = sum(value.square().sum() for value in actual.values())
    expected_gradients = torch.autograd.grad(
        expected_loss,
        (reference_feats, reference_pos),
    )
    actual_gradients = torch.autograd.grad(
        actual_loss,
        (candidate_feats, candidate_pos),
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=0,
            atol=0,
        )


def test_model_rebuild_policy_queries_provider_at_each_coordinate_refresh() -> None:
    class CountingProvider:
        def __init__(self) -> None:
            self.calls = 0
            self.reference = ReferenceRadiusNeighborProvider()

        def __call__(
            self,
            pos: torch.Tensor,
            batch: torch.Tensor,
            *,
            cutoff: float,
        ) -> torch.Tensor:
            self.calls += 1
            return self.reference(pos, batch, cutoff=cutoff)

    pos, batch = _geometry()
    feats = torch.randn(pos.shape[0], 4, dtype=torch.float64)
    provider = CountingProvider()
    model = _provider_model(
        coordinate_updates=True,
        coordinate_neighbor_policy="rebuild",
    )

    output = model(
        feats,
        pos,
        batch=batch,
        neighbor_provider=provider,
    )

    assert provider.calls == model.config.num_layers
    assert torch.isfinite(output["node_positions"]).all()


def test_model_neighbor_provider_conflicts_are_explicit() -> None:
    pos, batch = _geometry()
    feats = torch.randn(pos.shape[0], 4, dtype=torch.float64)
    provider = ReferenceRadiusNeighborProvider()
    edge_index = provider(pos, batch, cutoff=2.0)
    model = _provider_model()

    with pytest.raises(ValueError, match="mutually exclusive"):
        model(
            feats,
            pos,
            batch=batch,
            edge_index=edge_index,
            neighbor_provider=provider,
        )
