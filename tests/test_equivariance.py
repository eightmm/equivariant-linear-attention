from __future__ import annotations

import pytest
import torch
from conftest import orthogonal, transform_irreps

from equivariant_linear_attention import (
    BiomolecularPairContext,
    ELAGraph,
    TriELA,
)
from equivariant_linear_attention.nn.pair_state import DensePairState


def _tiny_model(
    input_irreps: str,
    output_irreps: str,
    *,
    update_positions: bool = False,
    pair_feature_dim: int = 0,
) -> TriELA:
    return (
        TriELA(
            input_irreps,
            output_irreps,
            width=16,
            pair_width=8,
            triangle_hidden=8,
            num_stages=1,
            pair_blocks_per_stage=1,
            local_blocks_per_stage=1,
            pair_transition_factor=2,
            pair_dropout=0.0,
            local_points=3,
            max_pair_tokens=8,
            pair_feature_dim=pair_feature_dim,
            distance_rbf_bins=4,
            distogram_bins=6,
            update_positions=update_positions,
            max_coordinate_step=0.15,
        )
        .double()
        .eval()
    )


def _global_pair_tensor(pair: DensePairState) -> torch.Tensor:
    nodes = pair.packed_batch.numel()
    packed_index = torch.arange(nodes, device=pair.z.device)
    dense_index = pair.unpack_node_tensor(packed_index)
    left = dense_index[:, :, None].expand_as(pair.pair_mask)
    right = dense_index[:, None, :].expand_as(pair.pair_mask)
    output = pair.z.new_zeros((nodes, nodes, pair.z.shape[-1]))
    return output.index_put(
        (left[pair.pair_mask], right[pair.pair_mask]),
        pair.z[pair.pair_mask],
    )


@pytest.mark.parametrize("reflection", [False, True])
def test_full_model_pair_axes_and_coordinate_update_obey_o3_and_translation(
    reflection: bool,
) -> None:
    generator = torch.Generator().manual_seed(101)
    input_irreps = "2x0e + 1x0o + 1x1o + 1x1e + 1x2e + 1x2o"
    output_irreps = "1x0e + 1x0o + 1x1o + 1x1e + 1x2e + 1x2o"
    model = _tiny_model(
        input_irreps,
        output_irreps,
        update_positions=True,
    )
    with torch.no_grad():
        for stage in model.stages:
            assert stage.coordinate_updates is not None
            for updater in stage.coordinate_updates:
                updater.vector.base_weight.fill_(0.2)

    x = torch.randn(
        5,
        model.input_irreps.dim,
        generator=generator,
        dtype=torch.float64,
    )
    pos = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    transform = orthogonal(reflection=reflection, seed=103)
    translation = torch.tensor([1.2, -0.7, 0.4], dtype=torch.float64)
    reference = model.forward_with_aux(ELAGraph(x, pos))
    moved = model.forward_with_aux(
        ELAGraph(
            transform_irreps(x, model.input_irreps, transform),
            pos @ transform.T + translation,
        )
    )

    torch.testing.assert_close(
        moved.graph.x,
        transform_irreps(reference.graph.x, model.output_irreps, transform),
        atol=3e-7,
        rtol=3e-7,
    )
    torch.testing.assert_close(
        moved.graph.pos,
        reference.graph.pos @ transform.T + translation,
        atol=3e-7,
        rtol=3e-7,
    )
    assert reference.graph.delta is not None and moved.graph.delta is not None
    torch.testing.assert_close(
        moved.graph.delta,
        reference.graph.delta @ transform.T,
        atol=3e-7,
        rtol=3e-7,
    )
    assert float(reference.graph.delta.detach().abs().max()) > 0.0
    torch.testing.assert_close(
        moved.pair_state.z,
        reference.pair_state.z,
        atol=3e-7,
        rtol=3e-7,
    )
    torch.testing.assert_close(
        moved.distogram_logits,
        reference.distogram_logits,
        atol=3e-7,
        rtol=3e-7,
    )


def test_node_permutation_moves_both_dense_pair_axes_and_metadata() -> None:
    generator = torch.Generator().manual_seed(107)
    model = _tiny_model("3x0e", "2x0e", pair_feature_dim=2)
    nodes = 6
    x = torch.randn(nodes, 3, generator=generator, dtype=torch.float64)
    pos = torch.randn(nodes, 3, generator=generator, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    group = torch.tensor([0, 1, 0, 0, 0, 1])
    pair_feature = torch.randn(
        nodes,
        nodes,
        2,
        generator=generator,
        dtype=torch.float64,
    )
    context = BiomolecularPairContext(
        token_index=torch.tensor([0, 1, 2, 0, 1, 2]),
        chain_id=torch.tensor([0, 1, 0, 0, 0, 1]),
        molecule_type=torch.tensor([0, 1, 0, 2, 2, 1]),
        pair_features=pair_feature,
    )
    reference = model.forward_with_aux(
        ELAGraph(x, pos, batch=batch, group=group),
        context,
    )
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    moved_context = BiomolecularPairContext(
        token_index=context.token_index[permutation],
        chain_id=context.chain_id[permutation],
        molecule_type=context.molecule_type[permutation],
        pair_features=pair_feature[permutation][:, permutation],
    )
    moved = model.forward_with_aux(
        ELAGraph(
            x[permutation],
            pos[permutation],
            batch=batch[permutation],
            group=group[permutation],
        ),
        moved_context,
    )
    torch.testing.assert_close(moved.graph.x, reference.graph.x[permutation])
    torch.testing.assert_close(moved.graph.pos, reference.graph.pos[permutation])
    reference_pair = _global_pair_tensor(reference.pair_state)
    moved_pair = _global_pair_tensor(moved.pair_state)
    torch.testing.assert_close(
        moved_pair,
        reference_pair[permutation][:, permutation],
        atol=3e-10,
        rtol=3e-10,
    )


def test_coincident_cutoff_ties_do_not_depend_on_packed_node_order() -> None:
    generator = torch.Generator().manual_seed(108)
    model = _tiny_model("3x0e", "2x0e", pair_feature_dim=2)
    nodes = 6
    x = torch.randn(nodes, 3, generator=generator, dtype=torch.float64)
    positions = torch.zeros(nodes, 3, dtype=torch.float64)
    pair_features = torch.randn(
        nodes,
        nodes,
        2,
        generator=generator,
        dtype=torch.float64,
    )
    reference = model.forward_with_aux(
        ELAGraph(x, positions),
        BiomolecularPairContext(pair_features=pair_features),
    )

    permutation = torch.tensor([4, 0, 5, 2, 1, 3])
    moved = model.forward_with_aux(
        ELAGraph(x[permutation], positions[permutation]),
        BiomolecularPairContext(
            pair_features=pair_features[permutation][:, permutation]
        ),
    )
    torch.testing.assert_close(
        moved.graph.x,
        reference.graph.x[permutation],
        atol=3e-10,
        rtol=3e-10,
    )
    torch.testing.assert_close(
        _global_pair_tensor(moved.pair_state),
        _global_pair_tensor(reference.pair_state)[permutation][:, permutation],
        atol=3e-10,
        rtol=3e-10,
    )


def test_interaction_groups_have_no_node_or_pair_leakage() -> None:
    generator = torch.Generator().manual_seed(109)
    model = _tiny_model("3x0e", "1x0e")
    x = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    pos = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    group = torch.tensor([0, 0, 0, 1, 1, 1])
    reference = model.forward_with_aux(ELAGraph(x, pos, group=group))
    changed_x = x.clone()
    changed_pos = pos.clone()
    changed_x[3:] = 100.0 * torch.randn(
        3,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    changed_pos[3:] = 100.0 * torch.randn(
        3,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    changed = model.forward_with_aux(ELAGraph(changed_x, changed_pos, group=group))
    torch.testing.assert_close(
        changed.graph.x[:3],
        reference.graph.x[:3],
        atol=2e-10,
        rtol=2e-10,
    )
    reference_pair = _global_pair_tensor(reference.pair_state)
    changed_pair = _global_pair_tensor(changed.pair_state)
    torch.testing.assert_close(
        changed_pair[:3, :3],
        reference_pair[:3, :3],
        atol=2e-10,
        rtol=2e-10,
    )
    assert torch.count_nonzero(reference_pair[:3, 3:]) == 0
    assert torch.count_nonzero(reference_pair[3:, :3]) == 0
