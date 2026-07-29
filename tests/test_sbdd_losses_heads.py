from __future__ import annotations

import pytest
import torch

from equivariant_attention.sbdd import (
    AffinityHead,
    LabelDirection,
    LabelKind,
    LabelQualifier,
    PoseAuxiliaryPredictions,
    PoseAuxiliaryTargets,
    PoseRefinementHead,
    PoseRefinementRequest,
    RefinementOutputKind,
    ScientificLabel,
    censored_affinity_loss,
    grouped_pose_ranking_loss,
)


@pytest.mark.parametrize("method", ["listwise", "rmsd_aware"])
def test_grouped_pose_ranking_prefers_near_native_within_each_pair(
    method: str,
) -> None:
    rmsd = torch.tensor([0.8, 2.5, 6.0, 1.2, 4.0], dtype=torch.float64)
    good = torch.tensor([3.0, 1.0, -1.0, 2.0, 0.0], dtype=torch.float64)
    bad = -good

    good_loss = grouped_pose_ranking_loss(
        good,
        rmsd,
        group_ids=("a", "a", "a", "b", "b"),
        method=method,
    )
    bad_loss = grouped_pose_ranking_loss(
        bad,
        rmsd,
        group_ids=("a", "a", "a", "b", "b"),
        method=method,
    )

    assert good_loss.ranking < bad_loss.ranking
    assert good_loss.total == good_loss.ranking


def test_grouped_pose_loss_reports_clash_strain_and_contact_auxiliaries() -> None:
    scores = torch.tensor([2.0, 0.0, 1.0, -1.0], requires_grad=True)
    predictions = PoseAuxiliaryPredictions(
        clash=torch.tensor([0.1, 0.9, 0.2, 0.8], requires_grad=True),
        strain=torch.tensor([0.2, 1.0, 0.3, 0.7], requires_grad=True),
        contact_logits=torch.tensor([2.0, -2.0, 1.0, -1.0], requires_grad=True),
    )
    targets = PoseAuxiliaryTargets(
        clash=torch.tensor([0.0, 1.0, 0.0, 1.0]),
        strain=torch.tensor([0.0, 1.0, 0.0, 1.0]),
        contact=torch.tensor([1.0, 0.0, 1.0, 0.0]),
    )
    result = grouped_pose_ranking_loss(
        scores,
        torch.tensor([1.0, 5.0, 1.5, 6.0]),
        group_ids=("a", "a", "b", "b"),
        method="listwise",
        auxiliary_predictions=predictions,
        auxiliary_targets=targets,
        auxiliary_weights=(0.2, 0.3, 0.4),
    )
    result.total.backward()

    assert result.clash.item() > 0
    assert result.strain.item() > 0
    assert result.contact.item() > 0
    assert scores.grad is not None
    assert predictions.contact_logits is not None
    assert predictions.contact_logits.grad is not None


def test_pose_ranking_rejects_cross_pair_singletons() -> None:
    with pytest.raises(ValueError, match="at least two poses"):
        grouped_pose_ranking_loss(
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.0, 2.0]),
            group_ids=("a", "b"),
        )


def test_affinity_head_exposes_separate_base_interaction_and_strain_components() -> (
    None
):
    torch.manual_seed(7)
    head = AffinityHead(
        interface_dim=3,
        global_dim=2,
        interaction_dim=4,
        include_strain=True,
    ).double()
    interface = torch.randn(5, 3, dtype=torch.float64)
    global_repr = torch.randn(5, 2, dtype=torch.float64)
    interaction = torch.randn(5, 4, dtype=torch.float64)
    strain = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)

    output = head(
        interface,
        global_repr,
        same_bound_geometry_interaction=interaction,
        strain=strain,
    )

    assert torch.allclose(
        output.affinity,
        output.base + output.interaction_residual + output.strain_contribution,
    )
    assert output.affinity.shape == (5,)
    assert head.output_direction is LabelDirection.HIGHER_IS_STRONGER
    with pytest.raises(ValueError, match="same-bound"):
        head(interface, global_repr, strain=strain)


def test_affinity_loss_requires_explicit_matching_prediction_direction() -> None:
    lower_is_stronger = ScientificLabel(
        kind=LabelKind.AFFINITY,
        qualifier=LabelQualifier.EXACT,
        direction=LabelDirection.LOWER_IS_STRONGER,
        unit="nM",
        raw_value="5.0",
        value=5.0,
    )

    with pytest.raises(ValueError, match="prediction direction"):
        censored_affinity_loss(torch.tensor([5.0]), (lower_is_stronger,))

    loss = censored_affinity_loss(
        torch.tensor([5.0]),
        (lower_is_stronger,),
        prediction_direction=LabelDirection.LOWER_IS_STRONGER,
    )
    assert loss.item() == pytest.approx(0.0)


def test_pose_refinement_head_is_rotation_equivariant_and_respects_masks_and_time() -> (
    None
):
    torch.manual_seed(8)
    head = PoseRefinementHead(
        scalar_dim=3,
        vector_channels=2,
        time_conditioned=True,
    ).double()
    scalars = torch.randn(5, 3, dtype=torch.float64)
    vectors = torch.randn(5, 2, 3, dtype=torch.float64)
    ligand = torch.tensor([False, False, True, True, True])
    protein = ~ligand
    flexible = torch.tensor([False, True, False, False, False])
    request = PoseRefinementRequest(
        ligand_mask=ligand,
        protein_mask=protein,
        flexible_protein_mask=flexible,
        time=torch.tensor(0.3, dtype=torch.float64),
    )
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(rotation) < 0:
        rotation[:, 0] = -rotation[:, 0]

    first = head(scalars, vectors, request)
    rotated = head(scalars, torch.einsum("ab,ncb->nca", rotation, vectors), request)

    assert torch.allclose(
        rotated.displacement,
        torch.einsum("ab,nb->na", rotation, first.displacement),
        atol=1e-10,
        rtol=1e-10,
    )
    assert torch.count_nonzero(first.displacement[0]) == 0
    assert torch.count_nonzero(first.displacement[1]) > 0
    assert torch.count_nonzero(first.displacement[ligand]) > 0
    assert torch.equal(first.update_mask, ligand | flexible)


def test_pose_refinement_requires_explicit_time_when_conditioned() -> None:
    head = PoseRefinementHead(2, 1, time_conditioned=True)
    request = PoseRefinementRequest(
        ligand_mask=torch.tensor([False, True]),
        protein_mask=torch.tensor([True, False]),
    )
    with pytest.raises(ValueError, match="time"):
        head(torch.randn(2, 2), torch.randn(2, 1, 3), request)


def test_pose_refinement_rejects_string_output_kind() -> None:
    with pytest.raises(TypeError, match="RefinementOutputKind"):
        PoseRefinementHead(
            2,
            1,
            output_kind="displacement",  # type: ignore[arg-type]
        )
    head = PoseRefinementHead(
        2,
        1,
        output_kind=RefinementOutputKind.DISPLACEMENT,
    )
    assert head.output_kind is RefinementOutputKind.DISPLACEMENT
