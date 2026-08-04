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


def test_complete_advanced_state_loads_without_canonical_initialization() -> None:
    advanced = _advanced()
    control = EquivariantLinearAttention(advanced)
    model = ELA.from_config(canonical_config_from_advanced(advanced))
    receipt = load_advanced_ela_state(model, control.state_dict())

    assert receipt.canonical_initialized is False
    assert not receipt.missing_keys
    assert not receipt.unexpected_keys


def test_conditioned_state_migration_preserves_conditioner_schema() -> None:
    advanced = _advanced(condition_dim=8)
    control = EquivariantLinearAttention(advanced)
    model = ELA.from_config(canonical_config_from_advanced(advanced))
    receipt = load_advanced_ela_state(model, control.state_dict())

    assert receipt.canonical_initialized is False
    assert not receipt.missing_keys
    assert any(layer.conditioner is not None for layer in model.layers)


def test_legacy_branch_router_requires_explicit_lossy_opt_in() -> None:
    advanced = _advanced()
    source = dict(EquivariantLinearAttention(advanced).state_dict())
    source["core.blocks.0.branch_fusion.router.2.weight"] = torch.randn(12, 12)
    source["core.blocks.0.branch_fusion.balance_strength"] = torch.randn(6)
    model = ELA.from_config(canonical_config_from_advanced(advanced))

    with pytest.raises(RuntimeError, match="allow_drop_learned_fusion=True"):
        load_advanced_ela_state(model, source)

    receipt = load_advanced_ela_state(
        model,
        source,
        allow_drop_learned_fusion=True,
    )

    assert receipt.canonical_initialized is True
    assert not receipt.missing_keys
    assert not receipt.unexpected_keys
    assert receipt.dropped_keys == (
        "core.blocks.0.branch_fusion.balance_strength",
        "core.blocks.0.branch_fusion.router.2.weight",
    )
    assert not any("branch_fusion" in key for key in model.state_dict())


def test_missing_new_canonical_field_keeps_target_initialization() -> None:
    advanced = _advanced()
    model = ELA.from_config(canonical_config_from_advanced(advanced))
    source = dict(EquivariantLinearAttention(advanced).state_dict())
    missing_key = next(key for key in source if key.endswith(".raw_odd_alignment"))
    del source[missing_key]
    before = model.state_dict()[missing_key].detach().clone()

    receipt = load_advanced_ela_state(model, source)

    assert receipt.canonical_initialized is True
    assert receipt.missing_keys == (missing_key,)
    torch.testing.assert_close(
        model.state_dict()[missing_key],
        before,
        atol=0.0,
        rtol=0.0,
    )


def test_missing_cg12_projection_fields_keep_zero_initialization() -> None:
    advanced = _advanced()
    source_model = ELA.from_config(canonical_config_from_advanced(advanced))
    target = ELA.from_config(canonical_config_from_advanced(advanced))
    source = dict(source_model.state_dict())
    suffixes = (
        ".l1_l2_polar_out.weight",
        ".l1_l2_axial_out.weight",
        ".l1_l2_even_tensor_out.weight",
        ".l1_l2_odd_tensor_out.weight",
    )
    removed = tuple(sorted(key for key in source if key.endswith(suffixes)))
    for key in removed:
        del source[key]

    receipt = load_advanced_ela_state(target, source)

    assert receipt.missing_keys == removed
    assert receipt.canonical_initialized is True
    for key in removed:
        torch.testing.assert_close(
            target.state_dict()[key],
            torch.zeros_like(target.state_dict()[key]),
        )


def test_migration_rejects_historical_per_layer_coordinate_update() -> None:
    advanced = _advanced(coordinate_updates=True)
    with pytest.raises(ValueError, match="update_positions=True"):
        canonical_config_from_advanced(advanced)


def test_migration_rejects_silent_unexpected_checkpoint_keys() -> None:
    advanced = _advanced()
    model = ELA.from_config(canonical_config_from_advanced(advanced))
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


def test_migration_stages_tensor_copies_before_mutating_model() -> None:
    advanced = _advanced()
    model = ELA.from_config(canonical_config_from_advanced(advanced))
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


def test_migration_accepts_complete_canonical_state() -> None:
    advanced = _advanced()
    source = ELA.from_config(canonical_config_from_advanced(advanced))
    target = ELA.from_config(canonical_config_from_advanced(advanced))

    receipt = load_advanced_ela_state(target, source.state_dict())

    assert receipt.canonical_initialized is False
    assert not receipt.missing_keys
    assert not receipt.unexpected_keys
