from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention.nn.heads import (
    CoordinateUpdateHead,
    EquivariantVectorHead,
)
from equivariant_linear_attention.physics import (
    DirectVectorForceHead,
    ScalarEnergyHead,
    conservative_forces,
)
from equivariant_linear_attention.nn.pooling import MaskedInvariantPooling


def _orthogonal(*, reflection: bool = False) -> torch.Tensor:
    torch.manual_seed(2901 + int(reflection))
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if bool((torch.linalg.det(matrix) < 0).item()) != reflection:
        matrix[:, 0].neg_()
    return matrix


def test_masked_invariant_pooling_keeps_global_and_interface_scopes_separate() -> (
    None
):
    value = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [20.0, 30.0], [40.0, 50.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    interface = torch.tensor([True, False, False, True])
    pooling = MaskedInvariantPooling(reduction="mean", empty_policy="error")

    global_value = pooling.global_pool(value, batch)
    interface_value = pooling.interface_pool(value, batch, interface)

    assert torch.equal(
        global_value,
        torch.tensor([[2.0, 3.0], [30.0, 40.0]], dtype=torch.float64),
    )
    assert torch.equal(
        interface_value,
        torch.tensor([[1.0, 2.0], [40.0, 50.0]], dtype=torch.float64),
    )
    (global_value.sum() + interface_value.sum()).backward()
    assert value.grad is not None
    assert torch.equal(
        value.grad,
        torch.tensor(
            [[1.5, 1.5], [0.5, 0.5], [0.5, 0.5], [1.5, 1.5]],
            dtype=torch.float64,
        ),
    )


def test_masked_invariant_pooling_is_permutation_safe_and_graph_isolated() -> None:
    torch.manual_seed(2903)
    value = torch.randn(7, 4, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    mask = torch.tensor([True, False, True, True, False, True, False])
    permutation = torch.tensor([2, 0, 1, 6, 4, 3, 5])
    pooling = MaskedInvariantPooling(reduction="sum", empty_policy="error")

    reference = pooling.interface_pool(value, batch, mask)
    permuted = pooling.interface_pool(
        value[permutation],
        batch[permutation],
        mask[permutation],
    )
    perturbed = value.clone()
    perturbed[batch == 1] += 1000.0
    isolated = pooling.interface_pool(perturbed, batch, mask)

    assert torch.equal(permuted, reference)
    assert torch.equal(isolated[0], reference[0])
    assert not torch.equal(isolated[1], reference[1])


def test_masked_invariant_pooling_empty_policy_is_explicit() -> None:
    value = torch.tensor([[1.0], [2.0]])
    batch = torch.tensor([0, 1])
    mask = torch.tensor([True, False])

    with pytest.raises(ValueError, match="empty"):
        MaskedInvariantPooling(empty_policy="error").interface_pool(
            value,
            batch,
            mask,
        )
    output = MaskedInvariantPooling(
        reduction="mean",
        empty_policy="zero",
    ).interface_pool(value, batch, mask)
    assert torch.equal(output, torch.tensor([[1.0], [0.0]]))

    with pytest.raises(TypeError, match="boolean"):
        MaskedInvariantPooling().interface_pool(value, batch, mask.long())
    with pytest.raises(ValueError, match="same length"):
        MaskedInvariantPooling().global_pool(value[:1], batch)


@pytest.mark.parametrize("reflection", [False, True])
def test_equivariant_vector_head_obeys_o3_and_node_permutations(
    reflection: bool,
) -> None:
    torch.manual_seed(2907)
    head = EquivariantVectorHead(
        scalar_channels=4,
        vector_channels=3,
        output_channels=2,
        hidden_channels=7,
    ).double()
    scalars = torch.randn(6, 4, dtype=torch.float64)
    vectors = torch.randn(6, 3, 3, dtype=torch.float64)
    transform = _orthogonal(reflection=reflection)
    permutation = torch.tensor([3, 1, 5, 0, 2, 4])

    reference = head(scalars, vectors)
    moved = head(scalars, vectors @ transform.T)
    permuted = head(scalars[permutation], vectors[permutation])

    assert torch.allclose(moved, reference @ transform.T, atol=1e-12)
    torch.testing.assert_close(
        permuted,
        reference[permutation],
        rtol=0.0,
        atol=1e-15,
    )


def _set_identity_coordinate_direction(head: CoordinateUpdateHead) -> None:
    with torch.no_grad():
        head.vector_head.base_weight.fill_(1.0)
        for parameter in head.vector_head.scalar_mixer.parameters():
            parameter.zero_()


def test_coordinate_update_head_has_exact_mask_and_explicit_centering() -> None:
    scalars = torch.zeros(5, 1, dtype=torch.float64)
    vectors = torch.tensor(
        [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0],
         [2.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        dtype=torch.float64,
    ).unsqueeze(1)
    positions = torch.randn(5, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1])
    selected = torch.tensor([True, False, True, True, False])

    outputs: dict[str, torch.Tensor] = {}
    for centering in ("none", "graph", "selected"):
        head = CoordinateUpdateHead(
            scalar_channels=1,
            vector_channels=1,
            centering=centering,
        ).double()
        _set_identity_coordinate_direction(head)
        outputs[centering] = head(
            scalars,
            vectors,
            positions,
            batch,
            update_mask=selected,
        )
        assert torch.equal(
            outputs[centering][~selected],
            positions[~selected],
        )

    assert torch.equal(
        outputs["none"][selected] - positions[selected],
        vectors[selected, 0],
    )
    expected_graph = vectors[:, 0].clone()
    expected_graph[:3] -= vectors[:3, 0].mean(dim=0)
    expected_graph[3:] -= vectors[3:, 0].mean(dim=0)
    assert torch.equal(
        outputs["graph"][selected] - positions[selected],
        expected_graph[selected],
    )
    selected_step = outputs["selected"] - positions
    zero = torch.zeros(3, dtype=selected_step.dtype)
    assert torch.allclose(selected_step[batch == 0].sum(dim=0), zero)
    assert torch.allclose(selected_step[batch == 1].sum(dim=0), zero)


@pytest.mark.parametrize("reflection", [False, True])
def test_coordinate_update_head_is_o3_translation_and_permutation_equivariant(
    reflection: bool,
) -> None:
    torch.manual_seed(2911)
    head = CoordinateUpdateHead(
        scalar_channels=3,
        vector_channels=2,
        centering="selected",
        hidden_channels=5,
    ).double()
    scalars = torch.randn(7, 3, dtype=torch.float64)
    vectors = torch.randn(7, 2, 3, dtype=torch.float64)
    positions = torch.randn(7, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    selected = torch.tensor([True, False, True, True, False, True, True])
    transform = _orthogonal(reflection=reflection)
    translation = torch.randn(1, 3, dtype=torch.float64)
    permutation = torch.tensor([2, 0, 1, 6, 4, 3, 5])

    reference = head(
        scalars,
        vectors,
        positions,
        batch,
        update_mask=selected,
    )
    moved = head(
        scalars,
        vectors @ transform.T,
        positions @ transform.T + translation,
        batch,
        update_mask=selected,
    )
    permuted = head(
        scalars[permutation],
        vectors[permutation],
        positions[permutation],
        batch[permutation],
        update_mask=selected[permutation],
    )

    assert torch.allclose(
        moved,
        reference @ transform.T + translation,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        permuted,
        reference[permutation],
        atol=1e-12,
        rtol=1e-12,
    )


def test_scalar_energy_head_is_additive_masked_and_graph_isolated() -> None:
    torch.manual_seed(2913)
    head = ScalarEnergyHead(scalar_channels=4, hidden_channels=6).double()
    scalars = torch.randn(7, 4, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    mask = torch.tensor([True, False, True, True, True, False, False])
    permutation = torch.tensor([2, 0, 1, 6, 4, 3, 5])

    node_energy = head.node_energies(scalars)
    energy = head(scalars, batch, mask=mask)
    permuted = head(
        scalars[permutation],
        batch[permutation],
        mask=mask[permutation],
    )
    perturbed = scalars.clone()
    perturbed[batch == 1] += 100.0
    isolated = head(perturbed, batch, mask=mask)

    assert torch.equal(
        energy,
        torch.stack(
            [
                node_energy[(batch == graph) & mask].sum()
                for graph in range(2)
            ]
        ),
    )
    assert torch.equal(permuted, energy)
    assert torch.equal(isolated[0], energy[0])
    assert not torch.equal(isolated[1], energy[1])


def _pair_energy(positions: torch.Tensor) -> torch.Tensor:
    displacement = positions[1] - positions[0]
    return 0.5 * displacement.square().sum()


def test_conservative_forces_are_negative_gradient_and_match_finite_difference() -> (
    None
):
    positions = torch.tensor(
        [[0.2, -0.4, 0.7], [1.1, 0.3, -0.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    energy = _pair_energy(positions)
    forces = conservative_forces(
        energy,
        positions,
        create_graph=True,
    )
    analytic = torch.stack(
        [positions[1] - positions[0], positions[0] - positions[1]]
    )

    assert torch.allclose(forces, analytic, atol=1e-12, rtol=1e-12)
    assert torch.allclose(forces.sum(dim=0), torch.zeros(3, dtype=torch.float64))
    step = 1e-6
    plus = positions.detach().clone()
    minus = positions.detach().clone()
    plus[0, 1] += step
    minus[0, 1] -= step
    finite_difference = (
        _pair_energy(plus) - _pair_energy(minus)
    ) / (2.0 * step)
    assert torch.allclose(
        forces[0, 1],
        -finite_difference,
        atol=1e-9,
        rtol=1e-9,
    )

    force_penalty = forces.square().sum()
    second_order = torch.autograd.grad(force_penalty, positions)[0]
    assert torch.isfinite(second_order).all()
    assert torch.count_nonzero(second_order)


@pytest.mark.parametrize("reflection", [False, True])
def test_conservative_and_direct_forces_transform_as_vectors(
    reflection: bool,
) -> None:
    torch.manual_seed(2917)
    transform = _orthogonal(reflection=reflection)
    translation = torch.randn(1, 3, dtype=torch.float64)
    positions = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    force = conservative_forces(_pair_energy(positions), positions)
    moved_positions = (
        positions.detach() @ transform.T + translation
    ).requires_grad_()
    moved_force = conservative_forces(
        _pair_energy(moved_positions),
        moved_positions,
    )
    assert torch.allclose(moved_force, force @ transform.T, atol=1e-12)

    head = DirectVectorForceHead(
        scalar_channels=3,
        vector_channels=2,
        hidden_channels=5,
    ).double()
    scalars = torch.randn(4, 3, dtype=torch.float64)
    vectors = torch.randn(4, 2, 3, dtype=torch.float64)
    direct = head(scalars, vectors)
    moved_direct = head(scalars, vectors @ transform.T)

    assert head.metadata["force_semantics"] == "non_conservative_auxiliary"
    assert head.metadata["conservative"] is False
    assert torch.allclose(moved_direct, direct @ transform.T, atol=1e-12)


def test_generic_primitives_validate_shapes_and_centering() -> None:
    with pytest.raises(ValueError, match="centering"):
        CoordinateUpdateHead(1, 1, centering="unknown")
    with pytest.raises(ValueError, match="final dimension"):
        EquivariantVectorHead(1, 1)(
            torch.randn(2, 1),
            torch.randn(2, 1, 2),
        )


def test_generic_primitives_support_second_order_autograd() -> None:
    torch.manual_seed(2919)
    head = CoordinateUpdateHead(
        scalar_channels=2,
        vector_channels=2,
        centering="selected",
        hidden_channels=4,
    ).double()
    scalars = torch.randn(
        4,
        2,
        dtype=torch.float64,
        requires_grad=True,
    )
    vectors = torch.randn(
        4,
        2,
        3,
        dtype=torch.float64,
        requires_grad=True,
    )
    positions = torch.randn(
        4,
        3,
        dtype=torch.float64,
        requires_grad=True,
    )
    batch = torch.tensor([0, 0, 1, 1])
    selected = torch.tensor([True, True, True, False])

    updated = head(
        scalars,
        vectors,
        positions,
        batch,
        update_mask=selected,
    )
    first = torch.autograd.grad(
        updated.square().sum(),
        (scalars, vectors, positions),
        create_graph=True,
    )
    second = torch.autograd.grad(
        sum(gradient.square().sum() for gradient in first),
        (scalars, vectors, positions),
    )

    assert all(torch.isfinite(gradient).all() for gradient in first)
    assert all(torch.isfinite(gradient).all() for gradient in second)


def test_explicit_empty_graph_pooling_is_well_defined() -> None:
    pooling = MaskedInvariantPooling(empty_policy="zero")
    pooled = pooling.global_pool(
        torch.empty(0, 2),
        torch.empty(0, dtype=torch.long),
        num_graphs=2,
    )
    assert torch.equal(pooled, torch.zeros(2, 2))

    with pytest.raises(ValueError, match="num_graphs is required"):
        pooling.global_pool(
            torch.empty(0, 2),
            torch.empty(0, dtype=torch.long),
        )


def test_generic_primitive_validation_rejects_ambiguous_metadata() -> None:
    with pytest.raises(ValueError, match="reduction"):
        MaskedInvariantPooling(reduction="median")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty_policy"):
        MaskedInvariantPooling(empty_policy="nan")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="floating"):
        MaskedInvariantPooling().global_pool(
            torch.ones(2, 1, dtype=torch.long),
            torch.tensor([0, 0]),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        MaskedInvariantPooling().global_pool(
            torch.ones(2, 1),
            torch.tensor([0, -1]),
        )
    with pytest.raises(ValueError, match="outside"):
        MaskedInvariantPooling().global_pool(
            torch.ones(2, 1),
            torch.tensor([0, 1]),
            num_graphs=1,
        )

    with pytest.raises(ValueError, match="positive"):
        EquivariantVectorHead(1, 0)
    with pytest.raises(TypeError, match="same dtype"):
        EquivariantVectorHead(1, 1).double()(
            torch.ones(2, 1, dtype=torch.float64),
            torch.ones(2, 1, 3, dtype=torch.float32),
        )
    coordinate_head = CoordinateUpdateHead(1, 1)
    with pytest.raises(TypeError, match="boolean"):
        coordinate_head(
            torch.ones(2, 1),
            torch.ones(2, 1, 3),
            torch.ones(2, 3),
            torch.tensor([0, 0]),
            update_mask=torch.ones(2, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="contiguous"):
        coordinate_head(
            torch.ones(2, 1),
            torch.ones(2, 1, 3),
            torch.ones(2, 3),
            torch.tensor([0, 2]),
        )

    with pytest.raises(ValueError, match="positive"):
        ScalarEnergyHead(0)
    with pytest.raises(ValueError, match="require gradients"):
        conservative_forces(
            torch.tensor(1.0),
            torch.ones(2, 3),
        )
