import pytest
import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
import equivariant_attention.moment as moment


def _kernel_inputs(seed: int = 307) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    q0 = (
        moment._normalize_positive_features(
            torch.rand(6, 2, 3, dtype=torch.float64),
            eps=1e-12,
        )
        .detach()
        .requires_grad_()
    )
    k0 = (
        moment._normalize_positive_features(
            torch.rand(6, 2, 3, dtype=torch.float64),
            eps=1e-12,
        )
        .detach()
        .requires_grad_()
    )
    q1 = (
        moment._unit_ball(
            torch.randn(6, 2, 3, dtype=torch.float64),
            eps=1e-12,
        )
        .detach()
        .requires_grad_()
    )
    k1 = (
        moment._unit_ball(
            torch.randn(6, 2, 3, dtype=torch.float64),
            eps=1e-12,
        )
        .detach()
        .requires_grad_()
    )
    gamma = torch.tensor([0.2, 0.5], dtype=torch.float64, requires_grad=True)
    beta = torch.tensor([0.1, 0.3], dtype=torch.float64, requires_grad=True)
    value = torch.randn(6, 2, 4, dtype=torch.float64, requires_grad=True)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.1, 0.0],
            [2.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.4, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    return q0, k0, q1, k1, gamma, beta, value, pos, batch


def _dense_local_weights(
    q0: torch.Tensor,
    k0: torch.Tensor,
    q1: torch.Tensor,
    k1: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    *,
    cutoff: float,
    balanced: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    content = torch.einsum("ihd,jhd->hij", q0, k0)
    angular = torch.einsum("iha,jha->hij", q1, k1)
    kernel = (
        0.5
        + content
        + beta[:, None, None] * (1.0 + angular)
        + gamma[:, None, None] * angular.square()
    )
    displacement = (pos[None, :, :] - pos[:, None, :]) / cutoff
    u = displacement.square().sum(dim=-1)
    same_graph = batch[:, None] == batch[None, :]
    mask = same_graph & (u < 1.0)
    cutoff_weight = torch.where(
        mask,
        0.5 * (torch.cos(torch.pi * u) + 1.0),
        torch.zeros_like(u),
    )
    radial_floor = 1e-3
    radial_gate = cutoff_weight * (radial_floor + (1.0 - radial_floor) * 0.5)
    weighted = kernel * radial_gate.unsqueeze(0)
    if balanced:
        weighted = weighted / weighted.sum(dim=1, keepdim=True)
    weights = weighted / weighted.sum(dim=2, keepdim=True)
    return weights, displacement


@pytest.mark.parametrize("balanced", [False, True])
def test_sparse_local_weights_match_dense_mask_and_gradients(balanced: bool) -> None:
    q0, k0, q1, k1, gamma, beta, value, pos, batch = _kernel_inputs()
    receiver, sender, weights, _, _ = moment._local_attention_weights(
        q0,
        k0,
        q1,
        k1,
        gamma,
        pos,
        batch,
        num_graphs=2,
        balanced=balanced,
        alignment_scale=beta,
        alignment_dot_scale=beta,
        kernel_floor=0.5,
        cutoff=2.5,
        num_rbf=16,
        radial_weight=torch.zeros(2, 16, dtype=torch.float64),
        radial_bias=torch.zeros(2, dtype=torch.float64),
    )
    actual = value.new_zeros(value.shape)
    actual.index_add_(0, receiver, weights.unsqueeze(-1) * value[sender])
    dense_weights, _ = _dense_local_weights(
        q0,
        k0,
        q1,
        k1,
        gamma,
        beta,
        pos,
        batch,
        cutoff=2.5,
        balanced=balanced,
    )
    expected = torch.einsum("hij,jhf->ihf", dense_weights, value)
    probe = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(
        (actual * probe).sum(),
        (q0, k0, q1, k1, gamma, beta, value, pos),
        retain_graph=True,
    )
    expected_grad = torch.autograd.grad(
        (expected * probe).sum(),
        (q0, k0, q1, k1, gamma, beta, value, pos),
    )

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-9)
    for left, right in zip(actual_grad, expected_grad, strict=True):
        assert torch.allclose(left, right, atol=1e-10, rtol=1e-9)


def test_cosine_of_squared_distance_cutoff_has_zero_value_and_first_derivative_at_boundary() -> (
    None
):
    u = torch.tensor(
        [0.0, 1.0 - 1e-8, 1.0, 1.0 + 1e-8, float("inf")],
        dtype=torch.float64,
        requires_grad=True,
    )
    value = moment._cosine_of_squared_distance_cutoff(u)
    gradient = torch.autograd.grad(value.sum(), u)[0]

    assert value[0] == 1.0
    assert value[2] == 0.0 and value[3] == 0.0 and value[4] == 0.0
    assert abs(float(gradient[1])) < 1e-7
    assert gradient[2] == 0.0 and gradient[3] == 0.0 and gradient[4] == 0.0


def test_tiny_positive_local_cutoff_keeps_exact_self_edges() -> None:
    q0 = torch.ones(2, 1, 1, dtype=torch.float32)
    q1 = torch.zeros(2, 1, 3, dtype=torch.float32)
    receiver, sender, weights, displacement, squared_distance = (
        moment._local_attention_weights(
            q0,
            q0,
            q1,
            q1,
            torch.zeros(1),
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.zeros(2, dtype=torch.long),
            num_graphs=1,
            balanced=False,
            alignment_scale=torch.zeros(1),
            alignment_dot_scale=torch.zeros(1),
            kernel_floor=1.0,
            cutoff=1e-40,
            num_rbf=2,
            radial_weight=torch.zeros(1, 2),
            radial_bias=torch.zeros(1),
        )
    )

    assert torch.equal(receiver, torch.tensor([0, 1]))
    assert torch.equal(sender, receiver)
    assert torch.equal(weights, torch.ones_like(weights))
    assert torch.equal(displacement, torch.zeros_like(displacement))
    assert torch.equal(squared_distance, torch.zeros_like(squared_distance))


def test_repeated_local_stages_reuse_one_geometry_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _hybrid_model((2, 2, 2))
    original = moment._local_geometry
    calls = 0

    def counted(*args: object, **kwargs: object) -> tuple[torch.Tensor, ...]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(moment, "_local_geometry", counted)
    model(
        torch.randn(6, 4, dtype=torch.float64),
        torch.randn(6, 3, dtype=torch.float64),
    )

    assert calls == 1


def _hybrid_model(
    local_head_counts: tuple[int, ...], *, memory_count: int = 1
) -> EquivariantAttention:
    torch.manual_seed(311)
    return (
        EquivariantAttention(
            EquivariantAttentionConfig(
                node_dim=4,
                hidden_irreps="8x0e + 2x1o",
                output_irreps="2x0e + 1x1o + 1x2e",
                num_layers=3,
                num_heads=2,
                local_head_counts=local_head_counts,
                global_memory_count=memory_count,
                use_memory_interaction=memory_count > 1,
                use_key_balancing=False,
            )
        )
        .double()
        .eval()
    )


def _orthogonal(reflection: bool) -> torch.Tensor:
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if (torch.linalg.det(rotation) < 0) != reflection:
        rotation[:, 0] *= -1
    return rotation


@pytest.mark.parametrize("reflection", [False, True])
@pytest.mark.parametrize(
    "route, memory_count",
    [
        ((2, 2, 2), 1),
        ((2, 0, 2), 1),
        ((2, 0, 2), 4),
    ],
)
def test_local_and_lgl_routes_preserve_o3_translation_and_permutation(
    reflection: bool,
    route: tuple[int, ...],
    memory_count: int,
) -> None:
    model = _hybrid_model(route, memory_count=memory_count)
    node_feats = torch.randn(7, 4, dtype=torch.float64)
    pos = torch.randn(7, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    reference = model(node_feats, pos, batch=batch)
    rotation = _orthogonal(reflection)
    translation = torch.randn(1, 3, dtype=torch.float64)
    moved = model(node_feats, pos @ rotation.T + translation, batch=batch)
    permutation = torch.tensor([2, 0, 1, 6, 4, 3, 5])
    inverse = torch.argsort(permutation)
    permuted = model(
        node_feats[permutation], pos[permutation], batch=batch[permutation]
    )

    assert torch.allclose(
        moved["node_scalars"], reference["node_scalars"], atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        moved["node_vectors"],
        torch.einsum("nca,ba->ncb", reference["node_vectors"], rotation),
        atol=1e-6,
        rtol=1e-6,
    )
    expected_tensor = torch.einsum(
        "ab,nkbc,dc->nkad", rotation, reference["node_tensors"], rotation
    )
    assert torch.allclose(moved["node_tensors"], expected_tensor, atol=1e-6, rtol=1e-6)
    for key in ("node_scalars", "node_vectors", "node_tensors"):
        assert torch.allclose(
            permuted[key][inverse], reference[key], atol=1e-6, rtol=1e-6
        )


def test_local_only_nodes_do_not_leak_global_fragment_separation() -> None:
    model = _hybrid_model((2, 2, 2))
    node_feats = torch.randn(6, 4, dtype=torch.float64)
    fragment = torch.tensor(
        [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 0.7, 0.0]],
        dtype=torch.float64,
    )
    pos = torch.cat([fragment, fragment + torch.tensor([10.0, 0.0, 0.0])])
    moved_pos = pos.clone()
    moved_pos[3:] += torch.tensor([100.0, -20.0, 30.0])

    reference = model(node_feats, pos)
    moved = model(node_feats, moved_pos)

    for key in ("node_scalars", "node_vectors", "node_tensors"):
        assert torch.allclose(moved[key], reference[key], atol=1e-8, rtol=1e-8)


def test_lgl_global_stage_communicates_between_distant_fragments() -> None:
    model = _hybrid_model((2, 0, 2))
    node_feats = torch.randn(6, 4, dtype=torch.float64)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.7, 0.0, 0.0],
            [0.0, 0.7, 0.0],
            [10.0, 0.0, 0.0],
            [10.7, 0.0, 0.0],
            [10.0, 0.7, 0.0],
        ],
        dtype=torch.float64,
    )
    changed = node_feats.clone()
    changed[3:] += 3.0

    reference = model(node_feats, pos)["node_scalars"][:3]
    influenced = model(changed, pos)["node_scalars"][:3]

    assert not torch.allclose(influenced, reference, atol=1e-10, rtol=0.0)


def test_exact_radial_trace_matches_dense_identity_and_gradients() -> None:
    torch.manual_seed(313)
    pos = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    weights = torch.rand(2, 5, 5, dtype=torch.float64)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    gate = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    mass = torch.einsum("hij,hj->ih", weights, gate)
    first = torch.einsum("hij,hj,ja->iha", weights, gate, pos)
    second = torch.einsum("hij,hj,j->ih", weights, gate, pos.square().sum(dim=-1))
    actual = moment._relative_radial_trace(second, first, mass, pos[:, None, :])
    relative_square = (pos[None, :, :] - pos[:, None, :]).square().sum(dim=-1)
    expected = torch.einsum("hij,hj,ij->ih", weights, gate, relative_square)
    actual_grad = torch.autograd.grad(
        actual.square().sum(), (pos, gate), retain_graph=True
    )
    expected_grad = torch.autograd.grad(expected.square().sum(), (pos, gate))

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-9)
    for left, right in zip(actual_grad, expected_grad, strict=True):
        assert torch.allclose(left, right, atol=1e-10, rtol=1e-9)


def test_radial_trace_flag_keeps_identical_state_schema() -> None:
    torch.manual_seed(317)
    off = EquivariantAttention(
        EquivariantAttentionConfig(node_dim=4, use_radial_trace=False)
    )
    torch.manual_seed(317)
    on = EquivariantAttention(
        EquivariantAttentionConfig(node_dim=4, use_radial_trace=True)
    )

    assert list(off.state_dict()) == list(on.state_dict())
    for name in off.state_dict():
        assert off.state_dict()[name].shape == on.state_dict()[name].shape


def _memory_dense(
    q0: torch.Tensor,
    k0: torch.Tensor,
    q1: torch.Tensor,
    k1: torch.Tensor,
    gamma: torch.Tensor,
    value: torch.Tensor,
    assignment: torch.Tensor,
    coupling: torch.Tensor,
    batch: torch.Tensor,
    beta: torch.Tensor,
    *,
    balanced: bool,
    kernel_floor_mode: str = "fixed",
) -> torch.Tensor:
    output = torch.empty_like(value)
    for graph in range(2):
        index = batch == graph
        content = torch.einsum("ihd,jhd->hij", q0[index], k0[index])
        angular = torch.einsum("iha,jha->hij", q1[index], k1[index])
        graph_scale = 1.0 if kernel_floor_mode == "fixed" else 1.0 / int(index.sum())
        kernel = (
            0.5 * graph_scale
            + content
            + graph_scale * beta[:, None, None] * (1.0 + angular)
            + gamma[:, None, None] * angular.square()
        )
        gate = torch.einsum(
            "ihm,hmn,jhn->hij",
            assignment[index],
            coupling[graph],
            assignment[index],
        )
        weighted = kernel * gate
        if balanced:
            weighted = weighted / weighted.sum(dim=1, keepdim=True)
        weights = weighted / weighted.sum(dim=2, keepdim=True)
        output[index] = torch.einsum("hij,jhf->ihf", weights, value[index])
    return output


def test_multi_memory_inverse_graph_baseline_matches_dense_forward_and_gradient() -> (
    None
):
    q0, k0, q1, k1, gamma, beta, value, _, _ = _kernel_inputs(seed=329)
    batch = torch.tensor([0, 0, 1, 1, 1, 1])
    assignment = (
        torch.softmax(torch.randn(6, 2, 3, dtype=torch.float64), dim=-1)
        .detach()
        .requires_grad_()
    )
    raw_coupling = torch.rand(2, 2, 3, 3, dtype=torch.float64)
    coupling = (
        (0.5 * (raw_coupling + raw_coupling.transpose(-1, -2)) + 0.5)
        .detach()
        .requires_grad_()
    )
    actual = moment._memory_factorized_attention(
        q0,
        k0,
        q1,
        k1,
        gamma,
        value,
        assignment,
        coupling,
        batch,
        num_graphs=2,
        balanced=False,
        alignment_scale=beta,
        alignment_dot_scale=beta,
        kernel_floor=0.5,
        kernel_floor_mode="inverse_graph_size",
        graph_counts=torch.tensor([2, 4]),
    )
    expected = _memory_dense(
        q0,
        k0,
        q1,
        k1,
        gamma,
        value,
        assignment,
        coupling,
        batch,
        beta,
        balanced=False,
        kernel_floor_mode="inverse_graph_size",
    )
    probe = torch.randn_like(actual)
    targets = (q0, k0, q1, k1, gamma, beta, value, assignment, coupling)
    actual_grad = torch.autograd.grad(
        (actual * probe).sum(), targets, retain_graph=True
    )
    expected_grad = torch.autograd.grad((expected * probe).sum(), targets)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-9)
    for left, right in zip(actual_grad, expected_grad, strict=True):
        assert torch.allclose(left, right, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("balanced", [False, True])
def test_single_memory_is_exact_incumbent_forward_and_gradient(balanced: bool) -> None:
    q0, k0, q1, k1, gamma, beta, value, _, batch = _kernel_inputs(seed=331)
    assignment = torch.ones(6, 2, 1, dtype=torch.float64)
    coupling = torch.ones(2, 2, 1, 1, dtype=torch.float64)
    baseline = moment._factorized_moment_attention(
        q0,
        k0,
        q1,
        k1,
        gamma,
        value,
        batch,
        num_graphs=2,
        balanced=balanced,
        alignment_scale=beta,
        alignment_dot_scale=beta,
        kernel_floor=0.5,
    )
    actual = moment._memory_factorized_attention(
        q0,
        k0,
        q1,
        k1,
        gamma,
        value,
        assignment,
        coupling,
        batch,
        num_graphs=2,
        balanced=balanced,
        alignment_scale=beta,
        alignment_dot_scale=beta,
        kernel_floor=0.5,
    )
    probe = torch.randn_like(actual)
    targets = (q0, k0, q1, k1, gamma, beta, value)
    baseline_grad = torch.autograd.grad(
        (baseline * probe).sum(), targets, retain_graph=True
    )
    actual_grad = torch.autograd.grad((actual * probe).sum(), targets)

    assert torch.equal(actual, baseline)
    for left, right in zip(actual_grad, baseline_grad, strict=True):
        assert torch.equal(left, right)


@pytest.mark.parametrize("balanced", [False, True])
def test_multi_memory_structured_attention_matches_dense_forward_and_gradient(
    balanced: bool,
) -> None:
    q0, k0, q1, k1, gamma, beta, value, _, batch = _kernel_inputs(seed=337)
    logits = torch.randn(6, 2, 3, dtype=torch.float64)
    assignment = torch.softmax(logits, dim=-1).detach().requires_grad_()
    raw_coupling = torch.rand(2, 2, 3, 3, dtype=torch.float64)
    coupling = (
        (0.5 * (raw_coupling + raw_coupling.transpose(-1, -2)) + 0.5)
        .detach()
        .requires_grad_()
    )
    actual = moment._memory_factorized_attention(
        q0,
        k0,
        q1,
        k1,
        gamma,
        value,
        assignment,
        coupling,
        batch,
        num_graphs=2,
        balanced=balanced,
        alignment_scale=beta,
        alignment_dot_scale=beta,
        kernel_floor=0.5,
    )
    expected = _memory_dense(
        q0,
        k0,
        q1,
        k1,
        gamma,
        value,
        assignment,
        coupling,
        batch,
        beta,
        balanced=balanced,
    )
    probe = torch.randn_like(actual)
    targets = (q0, k0, q1, k1, gamma, beta, value, assignment, coupling)
    actual_grad = torch.autograd.grad(
        (actual * probe).sum(), targets, retain_graph=True
    )
    expected_grad = torch.autograd.grad((expected * probe).sum(), targets)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-9)
    for left, right in zip(actual_grad, expected_grad, strict=True):
        assert torch.allclose(left, right, atol=1e-10, rtol=1e-9)


def test_memory_permutation_leaves_output_unchanged() -> None:
    q0, k0, q1, k1, gamma, beta, value, _, batch = _kernel_inputs(seed=347)
    assignment = torch.softmax(torch.randn(6, 2, 3, dtype=torch.float64), dim=-1)
    coupling = torch.rand(2, 2, 3, 3, dtype=torch.float64)
    coupling = 0.5 * (coupling + coupling.transpose(-1, -2)) + 0.5
    permutation = torch.tensor([2, 0, 1])
    reference = moment._memory_factorized_attention(
        q0,
        k0,
        q1,
        k1,
        gamma,
        value,
        assignment,
        coupling,
        batch,
        num_graphs=2,
        balanced=False,
        alignment_scale=beta,
        alignment_dot_scale=beta,
        kernel_floor=0.5,
    )
    permuted = moment._memory_factorized_attention(
        q0,
        k0,
        q1,
        k1,
        gamma,
        value,
        assignment[..., permutation],
        coupling[..., permutation, :][..., :, permutation],
        batch,
        num_graphs=2,
        balanced=False,
        alignment_scale=beta,
        alignment_dot_scale=beta,
        kernel_floor=0.5,
    )

    assert torch.allclose(permuted, reference, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("degeneration", ["ones_coupling", "uniform_identity"])
def test_multi_memory_safe_degenerations_reduce_to_incumbent(
    balanced: bool,
    degeneration: str,
) -> None:
    q0, k0, q1, k1, gamma, beta, value, _, batch = _kernel_inputs(seed=348)
    memory_count = 3
    if degeneration == "ones_coupling":
        assignment = torch.softmax(
            torch.randn(6, 2, memory_count, dtype=torch.float64), dim=-1
        )
        coupling = torch.ones(2, 2, memory_count, memory_count, dtype=torch.float64)
    else:
        assignment = torch.full(
            (6, 2, memory_count), 1.0 / memory_count, dtype=torch.float64
        )
        coupling = torch.eye(memory_count, dtype=torch.float64).expand(2, 2, -1, -1)
    baseline = moment._factorized_moment_attention(
        q0,
        k0,
        q1,
        k1,
        gamma,
        value,
        batch,
        num_graphs=2,
        balanced=balanced,
        alignment_scale=beta,
        alignment_dot_scale=beta,
        kernel_floor=0.5,
    )
    actual = moment._memory_factorized_attention(
        q0,
        k0,
        q1,
        k1,
        gamma,
        value,
        assignment,
        coupling,
        batch,
        num_graphs=2,
        balanced=balanced,
        alignment_scale=beta,
        alignment_dot_scale=beta,
        kernel_floor=0.5,
    )

    assert torch.allclose(actual, baseline, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("reflection", [False, True])
def test_memory_assignment_and_coupling_are_geometric_invariants(
    reflection: bool,
) -> None:
    q0, _, _, _, _, _, _, pos, batch = _kernel_inputs(seed=349)
    rotation = _orthogonal(reflection)
    translation = torch.randn(1, 3, dtype=torch.float64)
    assignment, coupling, centers = moment._memory_assignments_and_coupling(
        q0,
        pos,
        batch,
        num_graphs=2,
        memory_count=3,
        temperature=1.0,
        assignment_scale=2.5,
        interaction_cutoff=2.5,
        interact=True,
    )
    moved_assignment, moved_coupling, moved_centers = (
        moment._memory_assignments_and_coupling(
            q0,
            pos @ rotation.T + translation,
            batch,
            num_graphs=2,
            memory_count=3,
            temperature=1.0,
            assignment_scale=2.5,
            interaction_cutoff=2.5,
            interact=True,
        )
    )

    assert torch.allclose(moved_assignment, assignment, atol=1e-10, rtol=1e-9)
    assert torch.allclose(moved_coupling, coupling, atol=1e-10, rtol=1e-9)
    expected_centers = torch.einsum("ghma,ba->ghmb", centers, rotation) + translation
    assert torch.allclose(moved_centers, expected_centers, atol=1e-10, rtol=1e-9)
    assert torch.all((coupling >= 0.0) & (coupling <= 1.0))
    assert torch.allclose(
        coupling.diagonal(dim1=-2, dim2=-1),
        torch.ones_like(coupling.diagonal(dim1=-2, dim2=-1)),
        atol=1e-12,
        rtol=0.0,
    )


def test_memory_controls_stay_finite_at_small_positive_float32_scales() -> None:
    q0, _, _, _, _, _, _, pos, batch = _kernel_inputs(seed=351)
    q0 = q0.float().detach().requires_grad_()
    pos = pos.float().detach().requires_grad_()
    assignment, coupling, centers = moment._memory_assignments_and_coupling(
        q0,
        pos,
        batch,
        num_graphs=2,
        memory_count=4,
        temperature=1e-40,
        assignment_scale=1e-40,
        interaction_cutoff=1e-40,
        interact=True,
    )
    gradients = torch.autograd.grad(
        assignment.square().sum() + coupling.square().sum() + centers.square().sum(),
        (q0, pos),
    )

    assert torch.isfinite(assignment).all()
    assert torch.isfinite(coupling).all()
    assert torch.isfinite(centers).all()
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_full_lgl_memory_stays_finite_at_small_positive_float32_scales() -> None:
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            hidden_irreps="8x0e + 2x1o",
            num_heads=2,
            local_head_counts=(2, 0, 2),
            global_memory_count=4,
            use_memory_interaction=True,
            memory_assignment_temperature=1e-40,
            memory_assignment_scale=1e-40,
            memory_interaction_cutoff=1e-40,
        )
    )
    node_feats = torch.randn(6, 4, requires_grad=True)
    pos = torch.randn(6, 3, requires_grad=True)
    outputs = model(node_feats, pos)
    gradients = torch.autograd.grad(
        sum(value.square().sum() for value in outputs.values()),
        (node_feats, pos),
    )

    assert all(torch.isfinite(value).all() for value in outputs.values())
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("memory_count", [4, 8])
def test_deterministic_memory_codes_do_not_alias_for_one_dimensional_heads(
    memory_count: int,
) -> None:
    key_scalar = torch.linspace(0.1, 0.9, 7, dtype=torch.float64).reshape(7, 1, 1)
    pos = torch.zeros(7, 3, dtype=torch.float64)
    assignment, _, _ = moment._memory_assignments_and_coupling(
        key_scalar,
        pos,
        torch.zeros(7, dtype=torch.long),
        num_graphs=1,
        memory_count=memory_count,
        temperature=1.0,
        assignment_scale=2.5,
        interaction_cutoff=2.5,
        interact=False,
    )

    flattened_slots = assignment[:, 0].T
    for left in range(memory_count):
        for right in range(left + 1, memory_count):
            assert not torch.equal(flattened_slots[left], flattened_slots[right])


def test_single_memory_interaction_skips_unused_assignment_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=4,
            hidden_irreps="8x0e + 2x1o",
            num_heads=2,
            local_head_counts=(2, 0, 2),
            global_memory_count=1,
            use_memory_interaction=True,
        )
    ).double()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("M=1 must dispatch before memory assignment")

    monkeypatch.setattr(moment, "_memory_assignments_and_coupling", forbidden)
    outputs = model(
        torch.randn(5, 4, dtype=torch.float64),
        torch.randn(5, 3, dtype=torch.float64),
    )

    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_memory_count_does_not_change_parameter_schema_or_initialization() -> None:
    torch.manual_seed(353)
    one = _hybrid_model((2, 0, 2), memory_count=1)
    torch.manual_seed(353)
    four = _hybrid_model((2, 0, 2), memory_count=4)

    assert list(one.state_dict()) == list(four.state_dict())
    for name in one.state_dict():
        assert torch.equal(one.state_dict()[name], four.state_dict()[name])


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"local_head_counts": [0, 0, 0]}, TypeError, "local_head_counts"),
        ({"local_head_counts": (0, 0)}, ValueError, "num_layers"),
        ({"local_head_counts": (0, 5, 0)}, ValueError, "num_heads"),
        ({"num_rbf": True}, TypeError, "num_rbf"),
        ({"global_memory_count": 0}, ValueError, "global_memory_count"),
        ({"local_cutoff": float("nan")}, ValueError, "local_cutoff"),
        (
            {"memory_assignment_temperature": 0.0},
            ValueError,
            "memory_assignment_temperature",
        ),
        ({"memory_assignment_scale": 1e-100}, ValueError, "memory_assignment_scale"),
        ({"use_radial_trace": 1}, TypeError, "use_radial_trace"),
        (
            {"global_memory_count": 4, "use_memory_interaction": True},
            ValueError,
            "middle global stage",
        ),
    ],
)
def test_local_memory_config_rejects_ambiguous_or_unregistered_controls(
    kwargs: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        EquivariantAttention(EquivariantAttentionConfig(node_dim=4, **kwargs))
