from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

import torch

from .context import ELAFeatures
from .model.ela import ELAConfig, SparseGeometry, _ELAEngine
from .model.stack import EquivariantLinearAttentionConfig


@dataclass(frozen=True, slots=True)
class ELAMigrationReceipt:
    """Auditable result of loading an advanced ELA state into canonical ELA."""

    loaded_keys: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    dropped_keys: tuple[str, ...]
    canonical_initialized: bool

    @property
    def router_initialized(self) -> bool:
        """Compatibility alias for receipts created before fixed fusion."""

        return self.canonical_initialized


def canonical_config_from_advanced(
    config: EquivariantLinearAttentionConfig,
) -> ELAConfig:
    """Convert an advanced config into the one public ELA configuration.

    Invariant conditioning maps to ``ELAFeatures.condition_dim``. Historical
    per-layer coordinate mutation cannot be converted automatically because the
    canonical model owns bounded model-declared coordinate updates with a
    different head schema. Derived canonical heads and local rank must match
    exactly.
    """

    if not isinstance(config, EquivariantLinearAttentionConfig):
        raise TypeError("config must be an EquivariantLinearAttentionConfig")
    if config.num_node_roles:
        raise ValueError(
            "num_node_roles belongs in an input adapter and cannot be migrated"
        )
    if config.coordinate_updates:
        raise ValueError(
            "historical per-layer coordinate updates cannot be migrated "
            "automatically; construct ELA(update_positions=True) or set "
            "ELAConfig.coordinate_updates and retrain the coordinate head"
        )
    if config.residual_dropout or config.drop_path_rate:
        raise ValueError(
            "dropout and DropPath are training policy, not ELAConfig fields"
        )

    candidate = ELAConfig(
        input_irreps=config.input_irreps,
        output_irreps=config.output_irreps,
        width=config.hidden_dim,
        depth=config.num_layers,
        geometry=SparseGeometry(
            cutoff=config.local_cutoff,
            num_rbf=config.num_rbf,
            relation_cutoffs=config.relation_cutoffs,
        ),
        features=ELAFeatures(condition_dim=config.condition_dim),
    )
    canonical_advanced = candidate.to_advanced_config()
    incompatible = tuple(
        field.name
        for field in fields(EquivariantLinearAttentionConfig)
        if getattr(canonical_advanced, field.name) != getattr(config, field.name)
    )
    if incompatible:
        raise ValueError(
            "advanced config does not match fixed canonical values for "
            + ", ".join(incompatible)
        )
    return candidate


def load_advanced_ela_state(
    model: _ELAEngine,
    state_dict: Mapping[str, torch.Tensor],
    *,
    allow_drop_learned_fusion: bool = False,
) -> ELAMigrationReceipt:
    """Load schema-compatible historical weights without partial mutation.

    The retired learned branch-router changed the represented function, so its
    state is discarded only after explicit opt-in. New canonical
    parity/radial/relation parameters may be absent and retain their
    deterministic initialization; every other missing or unexpected key fails
    closed.
    """

    if not isinstance(model, _ELAEngine):
        raise TypeError("model must be an ELA")
    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping")
    if not isinstance(allow_drop_learned_fusion, bool):
        raise TypeError("allow_drop_learned_fusion must be a bool")
    provided = dict(state_dict)
    if any(not isinstance(key, str) for key in provided):
        raise TypeError("state_dict keys must be strings")
    if any(not isinstance(value, torch.Tensor) for value in provided.values()):
        raise TypeError("state_dict values must be tensors")

    legacy_router = {
        key for key in provided if ".branch_fusion." in key
    }
    if legacy_router and not allow_drop_learned_fusion:
        raise RuntimeError(
            "checkpoint contains learned branch-fusion parameters; dropping "
            "them changes the function, so pass allow_drop_learned_fusion=True "
            "to acknowledge the conversion"
        )
    provided = {
        key: value for key, value in provided.items() if key not in legacy_router
    }
    target = model.state_dict()
    target_keys = set(target)
    provided_keys = set(provided)
    allowed_missing_suffixes = (
        ".query_odd_scalar.weight",
        ".key_odd_scalar.weight",
        ".raw_odd_alignment",
        ".global_radial_centers",
        ".raw_global_radial_alignment",
        ".relation_radial_scale",
        ".relation_value_gate",
        ".local_scale_score_mix",
        ".local_scale_value_mix",
        ".l1_l2_polar_out.weight",
        ".l1_l2_axial_out.weight",
        ".l1_l2_even_tensor_out.weight",
        ".l1_l2_odd_tensor_out.weight",
        ".second_moment_chiral_mix",
    )
    unexpected = tuple(sorted(provided_keys - target_keys))
    missing = tuple(sorted(target_keys - provided_keys))
    invalid_missing = tuple(
        key
        for key in missing
        if not key.endswith(allowed_missing_suffixes)
    )
    shape_mismatches = tuple(
        key
        for key in sorted(provided_keys & target_keys)
        if provided[key].shape != target[key].shape
    )
    if unexpected or invalid_missing or shape_mismatches:
        raise RuntimeError(
            "advanced ELA state is not schema-compatible with canonical ELA; "
            f"missing={invalid_missing}, unexpected={unexpected}, "
            f"shape_mismatches={shape_mismatches}"
        )

    staged = {key: value.detach().clone() for key, value in target.items()}
    try:
        with torch.no_grad():
            for key, value in provided.items():
                staged[key].copy_(value)
    except (RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(
            "advanced ELA state could not be staged without mutation"
        ) from error

    result = model.load_state_dict(staged, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("state schema changed during validated migration")
    return ELAMigrationReceipt(
        loaded_keys=len(provided),
        missing_keys=missing,
        unexpected_keys=unexpected,
        dropped_keys=tuple(sorted(legacy_router)),
        canonical_initialized=bool(missing or legacy_router),
    )


__all__ = [
    "ELAMigrationReceipt",
    "canonical_config_from_advanced",
    "load_advanced_ela_state",
]
