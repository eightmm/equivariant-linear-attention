from dataclasses import FrozenInstanceError

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention.annotations import DistanceBandSpec


def test_distance_band_gates_are_overlapping_additive_not_a_partition() -> None:
    spec = DistanceBandSpec(cutoffs=(1.0, 2.0, 4.0))
    physical_squared_distance = torch.tensor(
        [0.0, 0.25, 1.0, 2.25, 16.0],
        dtype=torch.float64,
    )

    gates = spec.gates(physical_squared_distance)

    assert gates.shape == (5, 3)
    assert torch.equal(gates[0], torch.ones(3, dtype=torch.float64))
    assert torch.equal(gates[-1], torch.zeros(3, dtype=torch.float64))
    assert (gates >= 0).all() and (gates <= 1).all()
    assert not torch.allclose(
        gates.sum(dim=-1),
        torch.ones(5, dtype=torch.float64),
    )
    assert spec.additive_not_partition
    with pytest.raises(FrozenInstanceError):
        spec.cutoffs = (2.0,)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("cutoffs", "message"),
    [
        ((), "at least one"),
        ((0.0,), "positive"),
        ((2.0, 1.0), "strictly increasing"),
        ((1.0, float("inf")), "finite"),
    ],
)
def test_distance_band_spec_rejects_ambiguous_cutoffs(
    cutoffs: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        DistanceBandSpec(cutoffs=cutoffs)


def _model(
    *,
    distance_band_cutoffs: tuple[float, ...] = (),
) -> EquivariantAttention:
    return EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            hidden_irreps="8x0e + 2x1o",
            output_irreps="1x0e + 1x1o",
            num_layers=2,
            num_heads=2,
            local_head_counts=(0, 0),
            local_cutoff=4.0,
            use_sparse_low_rank_local_residual=True,
            local_residual_rank=2,
            distance_band_cutoffs=distance_band_cutoffs,
        )
    ).double()


def test_enabled_distance_bands_are_zero_init_additive_and_receive_gradient() -> None:
    torch.manual_seed(20260806)
    reference = _model()
    reference_rng = torch.random.get_rng_state()
    torch.manual_seed(20260806)
    candidate = _model(distance_band_cutoffs=(1.0, 2.0))
    assert torch.equal(torch.random.get_rng_state(), reference_rng)
    candidate.load_state_dict(
        {
            name: value
            for name, value in reference.state_dict().items()
            if name in candidate.state_dict()
        },
        strict=False,
    )
    generator = torch.Generator().manual_seed(20260807)
    features = torch.randn(4, 4, generator=generator, dtype=torch.float64)
    positions = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    batch = torch.zeros(4, dtype=torch.long)
    receiver = torch.arange(4).repeat_interleave(4)
    sender = torch.arange(4).repeat(4)
    edge_index = torch.stack([receiver, sender])

    expected = reference(
        features,
        positions,
        batch=batch,
        edge_index=edge_index,
    )
    actual = candidate(
        features,
        positions,
        batch=batch,
        edge_index=edge_index,
    )

    assert (
        candidate.state_dict().keys()
        != reference.state_dict().keys()
    )
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    for name, parameter in candidate.named_parameters():
        if (
            "sparse_low_rank_local_residual" in name
            and name.endswith("_out.weight")
        ):
            torch.nn.init.constant_(parameter, 0.1)
    active = candidate(
        features,
        positions,
        batch=batch,
        edge_index=edge_index,
    )
    loss = sum(value.square().sum() for value in active.values())
    loss.backward()
    gradients = [
        parameter.grad
        for name, parameter in candidate.named_parameters()
        if name.endswith("distance_band_score_bias")
    ]
    assert len(gradients) == candidate.config.num_layers
    assert all(gradient is not None for gradient in gradients)
    assert any(bool((gradient != 0).any()) for gradient in gradients)


def test_distance_band_bias_changes_sparse_residual_without_new_edge_lists() -> None:
    model = _model(distance_band_cutoffs=(1.0, 2.0))
    for name, parameter in model.named_parameters():
        if (
            "sparse_low_rank_local_residual" in name
            and name.endswith("_out.weight")
        ):
            torch.nn.init.constant_(parameter, 0.1)
        if name.endswith("distance_band_score_bias"):
            torch.nn.init.constant_(parameter, 0.5)
    features = torch.randn(
        3,
        4,
        generator=torch.Generator().manual_seed(20260808),
        dtype=torch.float64,
    )
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [2.5, 0.0, 0.0]],
        dtype=torch.float64,
    )
    batch = torch.zeros(3, dtype=torch.long)
    receiver = torch.arange(3).repeat_interleave(3)
    sender = torch.arange(3).repeat(3)
    edge_index = torch.stack([receiver, sender])

    enabled = model(
        features,
        positions,
        batch=batch,
        edge_index=edge_index,
    )
    for name, parameter in model.named_parameters():
        if name.endswith("distance_band_score_bias"):
            torch.nn.init.zeros_(parameter)
    disabled = model(
        features,
        positions,
        batch=batch,
        edge_index=edge_index,
    )

    assert not torch.allclose(
        enabled["node_scalars"],
        disabled["node_scalars"],
    )
