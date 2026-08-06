from __future__ import annotations

import pytest
import torch

from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.edge_free_api import _quotient_rigid_shape_step
from equivariant_linear_attention.model.edge_free import (
    _EdgeFreeRelativeMomentBank,
    _LatentAtlasOperator,
    _TransientL3Closure,
    _bounded_stf3,
    _message_node_inner,
    _orthogonalize_message,
    _stf3,
)
from equivariant_linear_attention.nn.multipoles import (
    _matrix_to_st,
    _st_to_matrix,
)
from equivariant_linear_attention.nn.parity import _ParityState


def _orthogonal(*, reflection: bool) -> torch.Tensor:
    generator = torch.Generator().manual_seed(8101 + int(reflection))
    matrix, _ = torch.linalg.qr(
        torch.randn(3, 3, generator=generator, dtype=torch.float64)
    )
    if bool((torch.linalg.det(matrix) < 0).item()) != reflection:
        matrix[:, 0].neg_()
    return matrix


def _normalize_positions(
    positions: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(positions)
    for graph in batch.unique(sorted=True):
        selected = batch == graph
        centered = positions[selected] - positions[selected].mean(dim=0)
        radius = centered.square().sum(dim=-1).mean().sqrt().clamp_min(1e-12)
        output[selected] = centered / radius
    return output


def _random_message(
    nodes: int,
    heads: int,
    head_dim: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(nodes, heads, head_dim, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, 3, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, 3, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, 5, generator=generator, dtype=torch.float64),
        torch.randn(nodes, heads, 5, generator=generator, dtype=torch.float64),
    )


def _transform_st(value: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    matrix = _st_to_matrix(value)
    moved = torch.einsum(
        "ia,...ab,jb->...ij",
        transform,
        matrix,
        transform,
    )
    return _matrix_to_st(moved)


def test_edge_free_third_moment_matches_explicit_tuple_oracle() -> None:
    generator = torch.Generator().manual_seed(8001)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    positions = _normalize_positions(
        torch.randn(7, 3, generator=generator, dtype=torch.float64),
        batch,
    )
    bank = _EdgeFreeRelativeMomentBank(rank=4, eps=1e-12).double()
    moments = bank(positions, batch, num_graphs=2)

    radius_squared = positions.square().sum(dim=-1)
    radial_coordinate = radius_squared / (1.0 + radius_squared)
    radial = torch.exp(
        (radial_coordinate.unsqueeze(-1) - 0.5)
        * bank._radial_scales.to(dtype=positions.dtype)
    )
    expected = torch.zeros_like(moments.third_tensor)
    mass = torch.zeros_like(moments.mass)
    for receiver in range(positions.shape[0]):
        for sender in range(positions.shape[0]):
            if receiver == sender or batch[receiver] != batch[sender]:
                continue
            displacement = positions[sender] - positions[receiver]
            cube = torch.einsum(
                "a,b,c->abc",
                displacement,
                displacement,
                displacement,
            )
            expected[receiver] += (
                radial[sender, :, None, None, None] * _stf3(cube)
            )
            mass[receiver] += radial[sender]
    expected = expected / (1.0 + mass)[..., None, None, None]
    expected = _bounded_stf3(expected, 1e-12)

    torch.testing.assert_close(
        moments.third_tensor,
        expected,
        atol=2e-12,
        rtol=2e-12,
    )
    trace = torch.einsum("...aac->...c", moments.third_tensor)
    torch.testing.assert_close(trace, torch.zeros_like(trace), atol=2e-12, rtol=0.0)


def test_graphwise_krylov_basis_is_invariantly_orthogonal() -> None:
    generator = torch.Generator().manual_seed(8003)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    counts = torch.tensor([3, 4], dtype=torch.long)
    first = _random_message(7, 3, 4, generator=generator)
    candidate = _random_message(7, 3, 4, generator=generator)
    orthogonal = _orthogonalize_message(
        candidate,
        (first,),
        batch=batch,
        num_graphs=2,
        graph_counts=counts,
        eps=1e-12,
    )

    inner = _message_node_inner(first, orthogonal)
    graph_inner = inner.new_zeros((2, inner.shape[1])).index_add(0, batch, inner)
    torch.testing.assert_close(
        graph_inner,
        torch.zeros_like(graph_inner),
        atol=2e-10,
        rtol=0.0,
    )
    assert all(torch.isfinite(sector).all() for sector in orthogonal)


@pytest.mark.parametrize("reflection", [False, True])
def test_latent_atlas_is_psd_and_o3_equivariant(reflection: bool) -> None:
    generator = torch.Generator().manual_seed(8005)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
    positions = _normalize_positions(
        torch.randn(7, 3, generator=generator, dtype=torch.float64),
        batch,
    )
    scalar = torch.randn(7, 6, generator=generator, dtype=torch.float64)
    value = torch.randn(7, 2, 3, generator=generator, dtype=torch.float64)
    operator = _LatentAtlasOperator(
        scalar_width=6,
        num_charts=3,
        eps=1e-12,
    ).double()
    transform = _orthogonal(reflection=reflection)

    factors = operator.factor(scalar, positions, batch, num_graphs=2)
    output = operator.apply(factors, value, batch, num_graphs=2)
    moved_factors = operator.factor(
        scalar,
        positions @ transform.T,
        batch,
        num_graphs=2,
    )
    moved_output = operator.apply(
        moved_factors,
        value @ transform.T,
        batch,
        num_graphs=2,
    )

    torch.testing.assert_close(
        moved_factors.assignment,
        factors.assignment,
        atol=2e-10,
        rtol=2e-10,
    )
    torch.testing.assert_close(
        moved_output,
        output @ transform.T,
        atol=2e-10,
        rtol=2e-10,
    )

    for graph in range(2):
        selected = batch == graph
        assignment = factors.assignment[selected]
        mass = factors.mass[graph]
        relation = assignment @ torch.diag_embed(mass.reciprocal()) @ assignment.T
        torch.testing.assert_close(
            relation,
            relation.T,
            atol=1e-12,
            rtol=0.0,
        )
        eigenvalues = torch.linalg.eigvalsh(relation)
        assert float(eigenvalues.min()) >= -1e-10


@pytest.mark.parametrize("reflection", [False, True])
def test_transient_l3_closure_obeys_o3_and_parity(reflection: bool) -> None:
    generator = torch.Generator().manual_seed(8007)
    nodes = 5
    heads = 3
    multipole_rank = 4
    closure = _TransientL3Closure(
        multipole_rank=multipole_rank,
        num_heads=heads,
        rank=2,
        eps=1e-12,
    ).double()
    with torch.no_grad():
        closure.polar_out.weight.normal_(generator=generator)
        closure.axial_out.weight.normal_(generator=generator)
        closure.even_tensor_out.weight.normal_(generator=generator)
        closure.odd_tensor_out.weight.normal_(generator=generator)

    third_seed = torch.randn(
        nodes,
        multipole_rank,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    third = _stf3(
        torch.einsum(
            "nra,nrb,nrc->nrabc",
            third_seed,
            third_seed,
            third_seed,
        )
    )
    state = _ParityState(
        even_scalar=torch.randn(nodes, 8, generator=generator, dtype=torch.float64),
        odd_scalar=torch.randn(nodes, heads, generator=generator, dtype=torch.float64),
        polar_vector=torch.randn(
            nodes, heads, 3, generator=generator, dtype=torch.float64
        ),
        axial_vector=torch.randn(
            nodes, heads, 3, generator=generator, dtype=torch.float64
        ),
        even_tensor=torch.randn(
            nodes, heads, 5, generator=generator, dtype=torch.float64
        ),
        odd_tensor=torch.randn(
            nodes, heads, 5, generator=generator, dtype=torch.float64
        ),
    )
    transform = _orthogonal(reflection=reflection)
    determinant = torch.linalg.det(transform)
    moved_third = torch.einsum(
        "ia,jb,kc,nrabc->nrijk",
        transform,
        transform,
        transform,
        third,
    )
    moved_state = _ParityState(
        even_scalar=state.even_scalar,
        odd_scalar=determinant * state.odd_scalar,
        polar_vector=state.polar_vector @ transform.T,
        axial_vector=determinant * (state.axial_vector @ transform.T),
        even_tensor=_transform_st(state.even_tensor, transform),
        odd_tensor=determinant * _transform_st(state.odd_tensor, transform),
    )

    reference = closure(state, third)
    moved = closure(moved_state, moved_third)
    torch.testing.assert_close(
        moved.polar_vector,
        reference.polar_vector @ transform.T,
        atol=3e-10,
        rtol=3e-10,
    )
    torch.testing.assert_close(
        moved.axial_vector,
        determinant * (reference.axial_vector @ transform.T),
        atol=3e-10,
        rtol=3e-10,
    )
    torch.testing.assert_close(
        moved.even_tensor,
        _transform_st(reference.even_tensor, transform),
        atol=3e-10,
        rtol=3e-10,
    )
    torch.testing.assert_close(
        moved.odd_tensor,
        determinant * _transform_st(reference.odd_tensor, transform),
        atol=3e-10,
        rtol=3e-10,
    )


@pytest.mark.parametrize("reflection", [False, True])
def test_quotient_step_removes_gauge_but_keeps_partial_rigid_pose(
    reflection: bool,
) -> None:
    generator = torch.Generator().manual_seed(8009)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
    positions = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    centers = torch.stack(
        [positions[batch == graph].mean(dim=0) for graph in range(2)]
    )
    translation = torch.tensor(
        [[0.4, -0.2, 0.1], [-0.3, 0.5, 0.2]],
        dtype=torch.float64,
    )
    angular = torch.tensor(
        [[0.2, -0.1, 0.3], [-0.4, 0.2, 0.1]],
        dtype=torch.float64,
    )
    relative = positions - centers[batch]
    rigid = translation[batch] + torch.cross(
        angular[batch],
        relative,
        dim=-1,
    )
    gates = torch.ones(2, 3, dtype=torch.float64)

    full = torch.ones(7, dtype=torch.bool)
    gauge_removed = _quotient_rigid_shape_step(
        rigid,
        positions,
        batch,
        full,
        gates,
        max_step=100.0,
        eps=1e-12,
    )
    torch.testing.assert_close(
        gauge_removed,
        torch.zeros_like(gauge_removed),
        atol=2e-10,
        rtol=0.0,
    )

    partial = torch.tensor([True, True, False, False, True, False, False])
    preserved = _quotient_rigid_shape_step(
        rigid,
        positions,
        batch,
        partial,
        gates,
        max_step=100.0,
        eps=1e-12,
    )
    torch.testing.assert_close(
        preserved[partial],
        rigid[partial],
        atol=2e-10,
        rtol=2e-10,
    )
    assert torch.count_nonzero(preserved[~partial]) == 0

    transform = _orthogonal(reflection=reflection)
    moved = _quotient_rigid_shape_step(
        rigid @ transform.T,
        positions @ transform.T,
        batch,
        partial,
        gates,
        max_step=100.0,
        eps=1e-12,
    )
    torch.testing.assert_close(
        moved,
        preserved @ transform.T,
        atol=3e-10,
        rtol=3e-10,
    )


def test_vnext_model_exposes_latent_edges_without_pair_state() -> None:
    generator = torch.Generator().manual_seed(8011)
    model = ELA(
        "4x0e",
        "1x0e",
        width=32,
        depth=2,
        update_positions=True,
    ).double()
    graph = ELAGraph(
        x=torch.randn(9, 4, generator=generator, dtype=torch.float64),
        pos=torch.randn(9, 3, generator=generator, dtype=torch.float64),
        update_mask=torch.tensor(
            [True, True, True, False, False, True, True, False, False]
        ),
    )
    description = model.describe()
    assert description["relative_moment_order"] == 3
    assert description["krylov_basis"] == "graphwise_irrep_orthogonal"
    assert description["latent_edge_relation"] == "soft_atlas_incidence"
    assert description["coordinate_manifold"] == "SE3_quotient_auto"

    output = model(graph)
    assert output.x.shape == (9, 1)
    assert torch.isfinite(output.x).all()
    assert torch.isfinite(output.pos).all()
    assert all(not hasattr(layer, "edge_state") for layer in model.layers)
