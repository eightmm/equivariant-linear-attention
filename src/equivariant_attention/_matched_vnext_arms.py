"""Parameter-matched generic 3D arms used by QM9/LBA experiment runners."""

from __future__ import annotations

from dataclasses import replace

from .config import ArchitectureConfig


MATCHED_VNEXT_ARMS = (
    "lgl",
    "lgl_2e",
    "lgl_2e_l3",
    "global_only",
    "global_local",
    "high_order_sparse",
)


def build_matched_vnext_config(
    arm: str,
    *,
    node_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    local_cutoff: float,
    global_backend: str = "outer_scatter",
    geometry_cache_mode: str = "full",
    checkpoint_gated_local_mlp: bool = False,
) -> ArchitectureConfig:
    """Build one frozen, same-feature architecture-comparison arm.

    ``global_only`` and ``global_local`` are the primary homogeneous
    attribution pair: every block retains every exact global head and the
    latter adds a rank-4 sparse local residual at the first and last blocks.
    ``lgl_2e`` is the persistent-tensor control for ``lgl_2e_l3``.  The
    high-order sparse arm is a bounded capability diagnostic rather than a
    clean attribution control.
    """

    if arm not in MATCHED_VNEXT_ARMS:
        choices = ", ".join(MATCHED_VNEXT_ARMS)
        raise ValueError(f"unknown matched vNext arm {arm!r}; expected: {choices}")
    profile = {
        "lgl": "minimal",
        "lgl_2e": "standard",
        "lgl_2e_l3": "high_order",
        "global_only": "minimal",
        "global_local": "minimal",
        "high_order_sparse": "high_order",
    }[arm]
    config = ArchitectureConfig.for_profile(
        profile,
        node_dim=node_dim,
        width=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    config = replace(
        config,
        global_transport=replace(
            config.global_transport,
            use_key_balancing=False,
            use_global_key_balancing=False,
            reduction_backend=global_backend,
        ),
        neighbor=replace(
            config.neighbor,
            provider_kind="precomputed",
            geometry_cache_mode=geometry_cache_mode,
        ),
    )

    if arm in {"global_only", "global_local"}:
        use_local_residual = arm == "global_local"
        return replace(
            config,
            representation=replace(
                config.representation,
                hidden_irreps=f"{hidden_dim}x0e + {num_heads}x1o",
            ),
            local=replace(
                config.local,
                local_head_counts=(0,) * num_layers,
                local_cutoff=local_cutoff,
                use_sparse_low_rank_local_residual=use_local_residual,
                local_residual_rank=4,
                residual_layers=(
                    _first_and_last_layers(num_layers)
                    if use_local_residual
                    else None
                ),
                sparse_residual_normalization="positive",
                requested_backend="materialized",
            ),
        )

    if arm == "high_order_sparse":
        return replace(
            config,
            representation=replace(
                config.representation,
                hidden_irreps=(
                    f"{hidden_dim}x0e + {num_heads + 1}x1o + 1x2e"
                ),
                use_irrep_rms_normalization=True,
                angular_bandwidth=2,
                use_tensor_product_kernel=True,
                transient_workspace_channels=1,
                transient_workspace_layers=(0,),
            ),
            local=replace(
                config.local,
                local_head_counts=(0,) * num_layers,
                local_cutoff=local_cutoff,
                use_sparse_low_rank_local_residual=True,
                local_residual_rank=2,
                residual_layers=(0,),
                sparse_residual_normalization="positive",
                requested_backend="auto",
                distance_bands=(0.5 * local_cutoff, local_cutoff),
            ),
        )

    persistent_tensor_channels = 0 if arm == "lgl" else 1
    transient_layers = (
        _first_and_last_layers(num_layers)
        if arm == "lgl_2e_l3"
        else None
    )
    tensor_suffix = (
        "" if persistent_tensor_channels == 0 else " + 1x2e"
    )
    return replace(
        config,
        representation=replace(
            config.representation,
            hidden_irreps=(
                f"{hidden_dim}x0e + {num_heads}x1o{tensor_suffix}"
            ),
            use_irrep_rms_normalization=arm != "lgl",
            transient_workspace_channels=(
                2 if arm == "lgl_2e_l3" else 1
            ),
            transient_workspace_layers=transient_layers,
        ),
        local=replace(
            config.local,
            local_head_counts=_lgl_schedule(num_layers, num_heads),
            local_cutoff=local_cutoff,
            use_gated_local_transport=True,
            use_grouped_invariant_normalization=True,
            checkpoint_gated_local_mlp=checkpoint_gated_local_mlp,
            requested_backend="materialized",
        ),
    )


def _lgl_schedule(num_layers: int, num_heads: int) -> tuple[int, ...]:
    if num_layers == 1:
        return (num_heads,)
    return (num_heads, *((0,) * (num_layers - 2)), num_heads)


def _first_and_last_layers(num_layers: int) -> tuple[int, ...]:
    if num_layers == 1:
        return (0,)
    return (0, num_layers - 1)
