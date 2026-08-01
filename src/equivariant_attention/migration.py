from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

import torch

from .branch_fusion import RMSAwareBranchFusion
from .canonical import ELA, ELAConfig, SparseGeometry
from .context import ELAFeatures
from .equivariant_linear_attention import EquivariantLinearAttentionConfig


@dataclass(frozen=True, slots=True)
class ELAMigrationReceipt:
    """Auditable result of loading an advanced ELA state into canonical ELA."""

    loaded_keys: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    router_initialized: bool


def canonical_config_from_advanced(
    config: EquivariantLinearAttentionConfig,
) -> ELAConfig:
    """Convert an advanced config into the one public ELA configuration.

    Invariant conditioning maps to ``ELAFeatures.condition_dim``. Historical
    per-layer coordinate mutation cannot be converted automatically because the
    canonical model uses an explicit outer refinement loop and a different head
    schema. Derived canonical heads and local rank must match exactly.
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
            "automatically to ELAContext.refinement"
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
    model: ELA,
    state_dict: Mapping[str, torch.Tensor],
) -> ELAMigrationReceipt:
    """Load shared advanced-ELA weights and initialize only the new router.

    Unexpected keys and missing non-router keys are rejected. This prevents
    partial loads from silently creating a different function.
    """

    if not isinstance(model, ELA):
        raise TypeError("model must be an ELA")
    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping")
    provided = dict(state_dict)
    if any(not isinstance(key, str) for key in provided):
        raise TypeError("state_dict keys must be strings")
    if any(not isinstance(value, torch.Tensor) for value in provided.values()):
        raise TypeError("state_dict values must be tensors")

    target = model.state_dict()
    target_keys = set(target)
    provided_keys = set(provided)
    branch_keys = {key for key in target_keys if ".branch_fusion." in key}
    provided_branch_keys = provided_keys & branch_keys
    unexpected = tuple(sorted(provided_keys - target_keys))
    missing = tuple(sorted(target_keys - provided_keys))
    invalid_missing = tuple(key for key in missing if key not in branch_keys)
    partial_branch = bool(provided_branch_keys) and (
        provided_branch_keys != branch_keys
    )
    shape_mismatches = tuple(
        key
        for key in sorted(provided_keys & target_keys)
        if provided[key].shape != target[key].shape
    )
    if partial_branch:
        raise RuntimeError(
            "advanced ELA state contains a partial branch_fusion state; "
            "provide either no router keys or the complete router state"
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
    if not provided_branch_keys:
        for module in model.modules():
            if isinstance(module, RMSAwareBranchFusion):
                module.reset_identity()
    return ELAMigrationReceipt(
        loaded_keys=len(provided),
        missing_keys=missing,
        unexpected_keys=unexpected,
        router_initialized=not provided_branch_keys,
    )


__all__ = [
    "ELAMigrationReceipt",
    "canonical_config_from_advanced",
    "load_advanced_ela_state",
]
