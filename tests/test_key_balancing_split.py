from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention import moment


def _config(**overrides: object) -> EquivariantAttentionConfig:
    base = EquivariantAttentionConfig(
        node_dim=4,
        hidden_irreps="16x0e + 2x1o",
        num_layers=2,
        num_heads=2,
        local_head_counts=(1, 0),
    )
    return replace(base, **overrides)


@pytest.mark.parametrize("legacy", [False, True])
def test_split_balancing_defaults_inherit_legacy_control_bitwise(legacy: bool) -> None:
    torch.manual_seed(33)
    inherited = EquivariantAttention(_config(use_key_balancing=legacy))
    state_after_inherited = torch.random.get_rng_state()
    torch.manual_seed(33)
    explicit = EquivariantAttention(
        _config(
            use_key_balancing=not legacy,
            use_local_key_balancing=legacy,
            use_global_key_balancing=legacy,
        )
    )
    state_after_explicit = torch.random.get_rng_state()

    assert torch.equal(state_after_inherited, state_after_explicit)
    assert inherited.state_dict().keys() == explicit.state_dict().keys()
    for first, second in zip(
        inherited.state_dict().values(),
        explicit.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    for layer in inherited.layers:
        assert layer.use_local_key_balancing is legacy
        assert layer.use_global_key_balancing is legacy
    for layer in explicit.layers:
        assert layer.use_local_key_balancing is legacy
        assert layer.use_global_key_balancing is legacy


def test_split_balancing_routes_independent_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, list[bool]] = {"local": [], "global": []}
    original_local = moment._local_attention_weights
    original_global = moment._global_moment_messages

    def wrapped_local(*args: object, **kwargs: object) -> object:
        observed["local"].append(bool(kwargs["balanced"]))
        return original_local(*args, **kwargs)

    def wrapped_global(*args: object, **kwargs: object) -> object:
        observed["global"].append(bool(kwargs["balanced"]))
        return original_global(*args, **kwargs)

    monkeypatch.setattr(moment, "_local_attention_weights", wrapped_local)
    monkeypatch.setattr(moment, "_global_moment_messages", wrapped_global)
    model = EquivariantAttention(
        _config(
            use_local_key_balancing=False,
            use_global_key_balancing=True,
        )
    )
    node_feats = torch.randn(6, 4)
    pos = torch.randn(6, 3)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    model(node_feats, pos, batch=batch)

    assert observed["local"] == [False]
    assert observed["global"] == [True, True]


def test_inverse_floor_checks_only_resolved_global_balancing() -> None:
    EquivariantAttention(
        _config(
            kernel_floor_mode="inverse_graph_size",
            use_key_balancing=True,
            use_local_key_balancing=True,
            use_global_key_balancing=False,
        )
    )
    with pytest.raises(ValueError, match="global key balancing"):
        EquivariantAttention(
            _config(
                kernel_floor_mode="inverse_graph_size",
                use_key_balancing=False,
                use_local_key_balancing=False,
                use_global_key_balancing=True,
            )
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("use_local_key_balancing", 1),
        ("use_global_key_balancing", "yes"),
    ],
)
def test_split_balancing_rejects_non_boolean_overrides(
    name: str,
    value: object,
) -> None:
    with pytest.raises(TypeError, match=name):
        EquivariantAttention(_config(**{name: value}))
