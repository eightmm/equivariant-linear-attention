from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention import ELA
from equivariant_linear_attention.model.stack import (
    EquivariantLinearAttention,
    EquivariantLinearAttentionConfig,
)
from equivariant_linear_attention.migration import (
    canonical_config_from_advanced,
    load_advanced_ela_state,
)


def _advanced(**overrides: object) -> EquivariantLinearAttentionConfig:
    values: dict[str, object] = {
        "input_irreps": "4x0e",
        "output_irreps": "1x0e",
        "hidden_dim": 32,
        "num_layers": 2,
        "num_heads": 2,
        "local_rank": 2,
        "local_cutoff": 6.0,
        "num_rbf": 8,
    }
    values.update(overrides)
    return EquivariantLinearAttentionConfig(**values)


def test_compatible_advanced_config_converts_to_single_ela_config() -> None:
    candidate = canonical_config_from_advanced(_advanced())
    assert candidate.width == 32
    assert candidate.depth == 2
    assert candidate.num_heads == 2
    assert candidate.local_rank == 2
    assert candidate.geometry.cutoff == 6.0
    assert candidate.features.condition_dim == 0


def test_conditioned_advanced_config_maps_to_ela_features() -> None:
    candidate = canonical_config_from_advanced(_advanced(condition_dim=8))
    assert candidate.features.condition_dim == 8
    assert candidate.to_advanced_config().condition_dim == 8


def test_state_migration_allows_only_new_branch_router_keys() -> None:
    advanced = _advanced()
    control = EquivariantLinearAttention(advanced)
    model = ELA(canonical_config_from_advanced(advanced))
    receipt = load_advanced_ela_state(model, control.state_dict())

    assert receipt.router_initialized is True
    assert receipt.missing_keys
    assert not receipt.unexpected_keys
    assert all(".branch_fusion." in key for key in receipt.missing_keys)


def test_conditioned_state_migration_preserves_conditioner_schema() -> None:
    advanced = _advanced(condition_dim=8)
    control = EquivariantLinearAttention(advanced)
    model = ELA(canonical_config_from_advanced(advanced))
    receipt = load_advanced_ela_state(model, control.state_dict())

    assert receipt.router_initialized is True
    assert all(".branch_fusion." in key for key in receipt.missing_keys)
    assert any(layer.conditioner is not None for layer in model.layers)


def test_migration_rejects_historical_per_layer_coordinate_update() -> None:
    advanced = _advanced(coordinate_updates=True)
    with pytest.raises(ValueError, match="ELAContext.refinement"):
        canonical_config_from_advanced(advanced)


def test_migration_rejects_silent_unexpected_checkpoint_keys() -> None:
    advanced = _advanced()
    model = ELA(canonical_config_from_advanced(advanced))
    state = dict(EquivariantLinearAttention(advanced).state_dict())
    changed_key = next(iter(state))
    before = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    state[changed_key] = torch.full_like(state[changed_key], 7)
    state["unexpected.weight"] = state[next(iter(state))].clone()
    with pytest.raises(RuntimeError, match="schema-compatible"):
        load_advanced_ela_state(model, state)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("norm_eps", 0.125),
        ("eps", 0.25),
        ("residual_scale_init", 0.7),
        ("max_coordinate_step", 0.75),
    ],
)
def test_migration_rejects_noncanonical_runtime_options(
    field: str,
    value: float,
) -> None:
    advanced = _advanced(**{field: value})
    with pytest.raises(ValueError, match=field):
        canonical_config_from_advanced(advanced)


def test_migration_rejects_partial_router_state_without_mutation() -> None:
    advanced = _advanced()
    model = ELA(canonical_config_from_advanced(advanced))
    state = dict(EquivariantLinearAttention(advanced).state_dict())
    router_key = next(
        key for key in model.state_dict() if ".branch_fusion." in key
    )
    state[router_key] = torch.ones_like(model.state_dict()[router_key])
    before = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }

    with pytest.raises(RuntimeError, match="partial branch_fusion"):
        load_advanced_ela_state(model, state)

    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0.0, rtol=0.0)


def test_migration_stages_tensor_copies_before_mutating_model() -> None:
    advanced = _advanced()
    model = ELA(canonical_config_from_advanced(advanced))
    state = dict(model.state_dict())
    keys = [key for key, value in state.items() if value.ndim == 2]
    state[keys[0]] = torch.full_like(state[keys[0]], 7)
    state[keys[-1]] = state[keys[-1]].to_sparse()
    before = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }

    with pytest.raises(RuntimeError, match="could not be staged"):
        load_advanced_ela_state(model, state)

    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0.0, rtol=0.0)


def test_advanced_migration_restores_identity_on_reused_target() -> None:
    advanced = _advanced()
    model = ELA(canonical_config_from_advanced(advanced))
    fusion = model.core.blocks[0].branch_fusion
    with torch.no_grad():
        fusion.router[-1].bias.fill_(1)
        fusion.balance_strength.fill_(0.5)

    receipt = load_advanced_ela_state(
        model,
        EquivariantLinearAttention(advanced).state_dict(),
    )

    assert receipt.router_initialized is True
    torch.testing.assert_close(
        fusion.router[-1].weight,
        torch.zeros_like(fusion.router[-1].weight),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        fusion.router[-1].bias,
        torch.zeros_like(fusion.router[-1].bias),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        fusion.balance_strength,
        torch.zeros_like(fusion.balance_strength),
        atol=0.0,
        rtol=0.0,
    )


def test_migration_accepts_complete_canonical_state() -> None:
    advanced = _advanced()
    source = ELA(canonical_config_from_advanced(advanced))
    target = ELA(canonical_config_from_advanced(advanced))

    receipt = load_advanced_ela_state(target, source.state_dict())

    assert receipt.router_initialized is False
    assert not receipt.missing_keys
    assert not receipt.unexpected_keys
