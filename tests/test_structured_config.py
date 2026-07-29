from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import json

import pytest

from equivariant_attention.config import (
    ArchitectureConfig,
    GlobalTransportConfig,
    LocalResidualConfig,
    NeighborConfig,
    RepresentationConfig,
)
from equivariant_attention.irreps import CartesianIrreps
from equivariant_attention.moment import EquivariantAttentionConfig


def test_structured_default_round_trips_every_legacy_field_without_reordering() -> None:
    legacy = EquivariantAttentionConfig(node_dim=11)

    structured = ArchitectureConfig.from_legacy(legacy)
    restored = structured.to_legacy()

    assert restored == legacy
    assert [field.name for field in fields(restored)] == [
        field.name for field in fields(legacy)
    ]
    assert structured.profile == "expert"
    assert structured.deferred_features == ()


def test_structured_nondefault_round_trip_preserves_tuples_and_new_relation_fields() -> (
    None
):
    legacy = EquivariantAttentionConfig(
        node_dim=9,
        hidden_irreps="48x0e + 3x1o + 3x2e",
        output_irreps="2x0e + 1x1o",
        num_layers=4,
        num_heads=3,
        local_cutoff=4.0,
        local_head_counts=(0, 0, 0, 0),
        use_sparse_low_rank_local_residual=True,
        local_residual_layers=(0, 2),
        use_static_tensor_carrier=True,
        symmetry_group="SE3",
        global_reduction_backend="auto",
        geometry_cache_mode="compact",
        num_edge_relations=3,
        relation_cutoffs=(2.0, 3.0, 4.0),
    )

    restored = ArchitectureConfig.from_legacy(legacy).to_legacy()

    assert restored == legacy


def test_sorted_versioned_json_is_byte_stable_and_round_trips() -> None:
    config = ArchitectureConfig.for_profile(
        "standard",
        node_dim=17,
        width=96,
        num_heads=6,
        num_layers=5,
    )

    encoded = config.to_json()
    decoded = ArchitectureConfig.from_json(encoded)

    assert encoded == decoded.to_json()
    assert decoded == config
    assert encoded == json.dumps(
        json.loads(encoded),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert json.loads(encoded)["schema_version"] == 1
    assert json.loads(encoded)["schema"] == "equivariant_attention.architecture"
    assert decoded.representation.input_irreps == "17x0e"


def test_json_preserves_cartesian_irreps_object_representation() -> None:
    legacy = EquivariantAttentionConfig(
        node_dim=8,
        hidden_irreps=CartesianIrreps(scalars=16, vectors=4),
        output_irreps=CartesianIrreps(scalars=2),
    )
    config = ArchitectureConfig.from_legacy(legacy)

    restored = ArchitectureConfig.from_json(config.to_json())

    assert restored == config
    assert restored.to_legacy() == legacy


def test_explicit_input_irreps_must_match_the_cartesian_executor_contract() -> None:
    representation = RepresentationConfig(
        input_irreps="7x0e + 2x1o + 1x2e",
        hidden_irreps="64x0e + 4x1o + 2x2e",
        input_vector_dim=2,
        input_tensor_dim=1,
    )
    config = ArchitectureConfig(
        node_dim=7,
        representation=representation,
    )

    assert ArchitectureConfig.from_json(config.to_json()) == config
    assert config.to_legacy().input_vector_dim == 2
    assert config.to_legacy().input_tensor_dim == 1

    with pytest.raises(ValueError, match="exactly match"):
        ArchitectureConfig(
            node_dim=7,
            representation=replace(
                representation,
                input_irreps="7x0e + 3x1o + 1x2e",
            ),
        )

    with pytest.raises(ValueError, match="Cartesian"):
        RepresentationConfig(input_irreps="7x0e + 1x3o")


def test_persistent_irrep_multiplicity_is_decoupled_from_attention_heads() -> None:
    from equivariant_attention import EquivariantAttention

    structured = ArchitectureConfig(
        node_dim=7,
        num_layers=1,
        num_heads=4,
        representation=RepresentationConfig(
            hidden_irreps="16x0e + 7x1o + 2x2e",
            output_irreps="1x0e + 3x1o + 1x2e",
        ),
        local=LocalResidualConfig(
            use_sparse_low_rank_local_residual=True,
            residual_layers=(0,),
        ),
    )

    model = EquivariantAttention(structured.to_legacy())

    assert model.hidden_irreps.vectors == 7
    assert model.hidden_irreps.tensors == 2
    assert model.layers[0].query_vector.out_channels == 4
    assert model.layers[0].vector_update.out_channels == 7


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema":"wrong","schema_version":1,"config":{}}',
        (
            '{"schema":"equivariant_attention.architecture",'
            '"schema_version":99,"config":{}}'
        ),
        (
            '{"schema":"equivariant_attention.architecture",'
            '"schema_version":1,"config":{},"unknown":1}'
        ),
        (
            '{"schema":"equivariant_attention.architecture",'
            '"schema_version":1,"config":{"node_dim":4,"unknown":1}}'
        ),
        (
            '{"schema":"equivariant_attention.architecture",'
            '"schema_version":1,"config":{"node_dim":4,'
            '"representation":{"unknown":1}}}'
        ),
        (
            '{"schema":"equivariant_attention.architecture",'
            '"schema_version":1,"schema_version":1,"config":{}}'
        ),
    ],
)
def test_json_rejects_tampered_versions_unknown_fields_and_duplicate_keys(
    payload: str,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ArchitectureConfig.from_json(payload)


def test_profiles_have_frozen_explicit_capability_snapshots() -> None:
    minimal = ArchitectureConfig.for_profile("minimal", node_dim=8)
    standard = ArchitectureConfig.for_profile("standard", node_dim=8)
    chiral = ArchitectureConfig.for_profile("chiral", node_dim=8)
    high_order = ArchitectureConfig.for_profile("high_order", node_dim=8)
    expert = ArchitectureConfig.for_profile("expert", node_dim=8)

    assert minimal.representation.hidden_irreps == "64x0e + 4x1o"
    assert minimal.representation.transient_max_degree == 2
    assert standard.representation.hidden_irreps.endswith("4x2e")
    assert standard.symmetry_group == "O3"
    assert chiral.symmetry_group == "SE3"
    assert high_order.representation.transient_max_degree == 3
    assert high_order.representation.persistent_max_degree == 2
    assert high_order.deferred_features == ()
    assert expert.deferred_features == ()
    high_order_legacy = high_order.to_legacy()
    assert high_order_legacy.use_transient_l3_workspace
    assert high_order_legacy.transient_l3_channels == 1


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            lambda: ArchitectureConfig(
                node_dim=8,
                num_layers=0,
            ),
            "num_layers",
        ),
        (
            lambda: ArchitectureConfig(
                node_dim=8,
                profile="minimal",
                representation=RepresentationConfig(
                    hidden_irreps="64x0e + 4x1o + 4x2e",
                ),
            ),
            "minimal",
        ),
        (
            lambda: ArchitectureConfig(
                node_dim=8,
                profile="chiral",
                symmetry_group="O3",
                representation=RepresentationConfig(
                    hidden_irreps="64x0e + 4x1o + 4x2e",
                ),
            ),
            "chiral",
        ),
        (
            lambda: ArchitectureConfig(
                node_dim=8,
                profile="standard",
            ),
            "standard",
        ),
        (
            lambda: ArchitectureConfig(
                node_dim=8,
                local=LocalResidualConfig(
                    use_sparse_low_rank_local_residual=True,
                    local_residual_rank=0,
                ),
            ),
            "local_residual_rank",
        ),
        (
            lambda: ArchitectureConfig(
                node_dim=8,
                num_layers=3,
                local=LocalResidualConfig(
                    use_sparse_low_rank_local_residual=True,
                    residual_stride=4,
                ),
            ),
            "residual_stride",
        ),
        (
            lambda: ArchitectureConfig(
                node_dim=8,
                local=LocalResidualConfig(
                    requested_backend="unknown",
                ),
            ),
            "requested_backend",
        ),
        (
            lambda: ArchitectureConfig(
                node_dim=8,
                neighbor=NeighborConfig(geometry_cache_mode="unknown"),
            ),
            "geometry_cache_mode",
        ),
        (
            lambda: ArchitectureConfig(
                node_dim=8,
                global_transport=GlobalTransportConfig(
                    reduction_backend="unknown",
                ),
            ),
            "reduction_backend",
        ),
        (
            lambda: ArchitectureConfig(
                node_dim=8,
                profile="minimal",
                representation=RepresentationConfig(
                    tensor_product_instructions=("0e,1o->1o",),
                ),
            ),
            "expert",
        ),
    ],
)
def test_invalid_profile_depth_rank_stride_backend_matrix_is_rejected_early(
    config,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        config()


def test_expert_only_and_runtime_only_options_are_never_silently_downconverted() -> (
    None
):
    expert = ArchitectureConfig(
        node_dim=8,
        profile="expert",
        representation=RepresentationConfig(
            tensor_product_instructions=("0e,1o->1o",),
        ),
    )
    automatic_local = ArchitectureConfig(
        node_dim=8,
        local=LocalResidualConfig(
            use_sparse_low_rank_local_residual=True,
            requested_backend="auto",
        ),
    )
    bands = ArchitectureConfig(
        node_dim=8,
        local=LocalResidualConfig(
            local_cutoff=6.0,
            use_sparse_low_rank_local_residual=True,
            distance_bands=(2.5, 6.0),
        ),
    )

    assert expert.deferred_features == ("expert_tensor_product_instructions",)
    assert automatic_local.deferred_features == ()
    assert bands.deferred_features == ()
    with pytest.raises(NotImplementedError):
        expert.to_legacy()
    assert automatic_local.to_legacy().sparse_residual_backend == "auto"
    assert bands.to_legacy().distance_band_cutoffs == (2.5, 6.0)


def test_structured_configs_are_deeply_immutable() -> None:
    config = ArchitectureConfig.for_profile("standard", node_dim=8)

    with pytest.raises(FrozenInstanceError):
        config.num_layers = 9  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        config.representation.angular_bandwidth = 1  # type: ignore[misc]


def test_explicit_stride_compiles_to_a_deterministic_legacy_layer_schedule() -> None:
    config = ArchitectureConfig(
        node_dim=8,
        num_layers=5,
        local=LocalResidualConfig(
            use_sparse_low_rank_local_residual=True,
            residual_stride=2,
        ),
    )

    legacy = config.to_legacy()

    assert legacy.local_residual_layers == (0, 2, 4)


def test_from_legacy_rejects_the_wrong_type() -> None:
    with pytest.raises(TypeError, match="EquivariantAttentionConfig"):
        ArchitectureConfig.from_legacy(object())  # type: ignore[arg-type]


def test_replace_still_runs_cross_group_validation() -> None:
    config = ArchitectureConfig.for_profile("standard", node_dim=8)

    with pytest.raises(ValueError, match="standard"):
        replace(
            config,
            representation=replace(
                config.representation,
                hidden_irreps="64x0e + 4x1o",
            ),
        )


def test_regression_builder_accepts_structured_configuration() -> None:
    from equivariant_attention.training import build_regression_model

    config = ArchitectureConfig(
        node_dim=7,
        local=LocalResidualConfig(
            local_cutoff=4.0,
            use_sparse_low_rank_local_residual=True,
            requested_backend="streamed_csr",
            residual_layers=(0,),
            distance_bands=(2.0, 4.0),
        ),
        neighbor=NeighborConfig(
            num_edge_relations=2,
            relation_cutoffs=(2.0, 4.0),
        ),
    )

    model = build_regression_model(
        node_dim=7,
        architecture_config=config,
    )

    assert model.config == config.to_legacy()
    assert model.config.sparse_residual_backend == "streamed_csr"
    assert model.config.num_edge_relations == 2


def test_regression_builder_rejects_structured_node_dimension_mismatch() -> None:
    from equivariant_attention.training import build_regression_model

    config = ArchitectureConfig(node_dim=8)

    with pytest.raises(ValueError, match="node_dim"):
        build_regression_model(node_dim=7, architecture_config=config)
