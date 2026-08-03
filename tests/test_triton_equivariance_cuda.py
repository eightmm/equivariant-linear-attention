from __future__ import annotations

import pytest
import torch

from equivariant_attention import (
    ELA,
    ELABatch,
    IrrepLayout,
    RefinementRequest,
    conservative_forces,
    matrix_to_st5,
    pack_irreps,
    split_irreps,
    st5_to_matrix,
)
from equivariant_attention.triton_ops import kernel_backend, triton_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_available(),
    reason="CUDA Triton runtime is unavailable",
)

FULL_IRREPS = "1x0e + 1x0o + 1x1o + 1x1e + 1x2e + 1x2o"


def _ragged_graph(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = (5, 8, 4)
    ptr = torch.tensor((0, 5, 13, 17), device=device, dtype=torch.int64)
    edge_parts: list[torch.Tensor] = []
    offset = 0
    for graph_index, count in enumerate(counts):
        receiver_parts: list[torch.Tensor] = []
        sender_parts: list[torch.Tensor] = []
        for local_receiver in range(count):
            if graph_index == 2 and local_receiver == count - 1:
                degree = 0
            elif graph_index == 1:
                degree = 1 + (3 * local_receiver) % count
            else:
                degree = count
            if degree == 0:
                continue
            receiver_parts.append(
                torch.full(
                    (degree,),
                    offset + local_receiver,
                    device=device,
                    dtype=torch.long,
                )
            )
            sender_parts.append(
                offset
                + (
                    torch.arange(degree, device=device, dtype=torch.long)
                    + 2 * local_receiver
                )
                % count
            )
        edge_parts.append(
            torch.stack(
                [torch.cat(receiver_parts), torch.cat(sender_parts)],
            )
        )
        offset += count
    edge_index = torch.cat(edge_parts, dim=1)
    relation = (edge_index[0] + 2 * edge_index[1]).remainder(2)
    return ptr, edge_index, relation


def _transform_irreps(
    layout: str | IrrepLayout,
    value: torch.Tensor,
    orthogonal: torch.Tensor,
) -> torch.Tensor:
    parsed = IrrepLayout.parse(layout)
    determinant = torch.linalg.det(orthogonal)
    source = split_irreps(parsed, value)
    transformed: dict[str, torch.Tensor] = {}
    for block in parsed.blocks:
        irrep = block.irrep
        block_value = source[str(irrep)]
        parity_factor = determinant ** (
            irrep.degree + (1 if irrep.parity == "o" else 0)
        )
        if irrep.degree == 0:
            result = block_value
        elif irrep.degree == 1:
            result = torch.einsum(
                "...c,dc->...d",
                block_value,
                orthogonal,
            )
        else:
            matrix = st5_to_matrix(block_value)
            result = matrix_to_st5(
                torch.einsum(
                    "ab,...bc,dc->...ad",
                    orthogonal,
                    matrix,
                    orthogonal,
                )
            )
        transformed[str(irrep)] = parity_factor * result
    return pack_irreps(parsed, transformed)


def _activate_every_local_sector(model: ELA) -> None:
    with torch.no_grad():
        for layer in model.layers:
            for name in (
                "local_scalar_out",
                "local_odd_out",
                "local_polar_out",
                "local_axial_out",
                "local_even_tensor_out",
                "local_odd_tensor_out",
                "local_chiral_scalar_out",
                "local_chiral_axial_out",
                "local_chiral_tensor_out",
                "local_mass_out",
            ):
                module = getattr(layer, name)
                module.weight.normal_(mean=0.0, std=0.05)
            layer.local_radial_value.weight.normal_(mean=0.0, std=0.03)
            layer.branch_fusion.router[-1].weight.normal_(mean=0.0, std=0.04)
            layer.branch_fusion.router[-1].bias.normal_(mean=0.0, std=0.04)
            layer.branch_fusion.balance_strength.fill_(0.1)
        if model.coordinate_head is not None:
            model.coordinate_head.base_weight.fill_(0.15)


def _model(*, coordinate_refinement: bool = False) -> ELA:
    model = ELA(
        input_irreps=FULL_IRREPS,
        output_irreps=FULL_IRREPS,
        width=16,
        depth=1,
        cutoff=10.0,
        num_rbf=8,
        num_edge_types=2,
        coordinate_refinement=coordinate_refinement,
    ).cuda()
    _activate_every_local_sector(model)
    return model


def _local_parameters(model: ELA) -> tuple[torch.Tensor, ...]:
    return tuple(
        parameter
        for name, parameter in model.layers[0].named_parameters()
        if name.startswith("local_") or name == "relation_score_bias"
    )


def test_forced_triton_full_irreps_matches_torch_and_all_local_vjps() -> None:
    torch.manual_seed(401)
    device = torch.device("cuda")
    reference_model = _model()
    candidate_model = _model()
    candidate_model.load_state_dict(reference_model.state_dict(), strict=True)
    ptr, edge_index, relation = _ragged_graph(device)
    features = torch.randn(
        int(ptr[-1].item()),
        reference_model.config.input_layout.dim,
        device=device,
    )
    positions = torch.randn(features.shape[0], 3, device=device)
    cotangent = torch.randn(
        features.shape[0],
        reference_model.config.output_layout.dim,
        device=device,
    )

    reference_features = features.clone().requires_grad_(True)
    reference_positions = positions.clone().requires_grad_(True)
    reference_inputs = (
        reference_features,
        reference_positions,
        *_local_parameters(reference_model),
    )
    with kernel_backend("torch"):
        reference = reference_model(
            ELABatch(
                reference_features,
                reference_positions,
                ptr=ptr,
                edge_index=edge_index,
                edge_relation_id=relation,
            )
        )["node_irreps"]
        reference_gradients = torch.autograd.grad(
            (reference * cotangent).sum(),
            reference_inputs,
        )

    candidate_features = features.clone().requires_grad_(True)
    candidate_positions = positions.clone().requires_grad_(True)
    candidate_inputs = (
        candidate_features,
        candidate_positions,
        *_local_parameters(candidate_model),
    )
    with kernel_backend("triton"):
        candidate = candidate_model(
            ELABatch(
                candidate_features,
                candidate_positions,
                ptr=ptr,
                edge_index=edge_index,
                edge_relation_id=relation,
            )
        )["node_irreps"]
        candidate_gradients = torch.autograd.grad(
            (candidate * cotangent).sum(),
            candidate_inputs,
        )

    torch.testing.assert_close(candidate, reference, atol=8e-4, rtol=8e-4)
    for actual, expected in zip(
        candidate_gradients,
        reference_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        assert torch.isfinite(actual).all()

    chiral_names = {
        "local_chiral_scalar_out.weight",
        "local_chiral_axial_out.weight",
        "local_chiral_tensor_out.weight",
    }
    gradient_by_name = dict(
        zip(
            (
                name
                for name, _ in candidate_model.layers[0].named_parameters()
                if name.startswith("local_") or name == "relation_score_bias"
            ),
            candidate_gradients[2:],
            strict=True,
        )
    )
    for name in chiral_names:
        assert torch.count_nonzero(gradient_by_name[name]) > 0


def test_forced_triton_obeys_o3_translation_permutation_and_refinement() -> None:
    torch.manual_seed(409)
    device = torch.device("cuda")
    model = _model(coordinate_refinement=True).eval()
    ptr, edge_index, relation = _ragged_graph(device)
    nodes = int(ptr[-1].item())
    features = torch.randn(nodes, model.config.input_layout.dim, device=device)
    positions = torch.randn(nodes, 3, device=device)
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
    if torch.linalg.det(orthogonal) > 0:
        orthogonal[:, 0].neg_()
    translation = torch.tensor([0.7, -1.1, 0.3], device=device)
    refinement = RefinementRequest(steps=1, max_step=0.1, centering="graph")

    with torch.inference_mode(), kernel_backend("triton"):
        reference = model(
            ELABatch(
                features,
                positions,
                ptr=ptr,
                edge_index=edge_index,
                edge_relation_id=relation,
                refinement=refinement,
            )
        )
        transformed = model(
            ELABatch(
                _transform_irreps(
                    model.config.input_layout,
                    features,
                    orthogonal,
                ),
                positions @ orthogonal.T + translation,
                ptr=ptr,
                edge_index=edge_index,
                edge_relation_id=relation,
                refinement=refinement,
            )
        )

    assert torch.count_nonzero(reference["coordinate_delta"]) > 0
    expected_output = _transform_irreps(
        model.config.output_layout,
        reference["node_irreps"],
        orthogonal,
    )
    torch.testing.assert_close(
        transformed["node_irreps"],
        expected_output,
        atol=3e-3,
        rtol=3e-3,
    )
    torch.testing.assert_close(
        transformed["positions"],
        reference["positions"] @ orthogonal.T + translation,
        atol=3e-4,
        rtol=3e-4,
    )
    torch.testing.assert_close(
        transformed["coordinate_delta"],
        reference["coordinate_delta"] @ orthogonal.T,
        atol=3e-4,
        rtol=3e-4,
    )

    permutation = torch.cat(
        [
            torch.arange(start, stop, device=device)[
                torch.randperm(stop - start, device=device)
            ]
            for start, stop in zip(ptr[:-1].tolist(), ptr[1:].tolist(), strict=True)
        ]
    )
    old_to_new = torch.empty_like(permutation)
    old_to_new[permutation] = torch.arange(nodes, device=device)
    edge_shuffle = torch.randperm(edge_index.shape[1], device=device)
    with torch.inference_mode(), kernel_backend("triton"):
        permuted = model(
            ELABatch(
                features[permutation],
                positions[permutation],
                ptr=ptr,
                edge_index=old_to_new[edge_index[:, edge_shuffle]],
                edge_relation_id=relation[edge_shuffle],
                refinement=refinement,
            )
        )
    torch.testing.assert_close(
        permuted["node_irreps"],
        reference["node_irreps"][permutation],
        atol=2e-3,
        rtol=2e-3,
    )
    torch.testing.assert_close(
        permuted["coordinate_delta"],
        reference["coordinate_delta"][permutation],
        atol=5e-4,
        rtol=5e-4,
    )


def test_forced_triton_conservative_force_double_backward_matches_torch() -> None:
    torch.manual_seed(419)
    device = torch.device("cuda")
    reference_model = ELA(
        input_irreps="4x0e",
        output_irreps="1x0e",
        width=16,
        depth=1,
        cutoff=10.0,
        num_rbf=8,
    ).cuda()
    _activate_every_local_sector(reference_model)
    candidate_model = ELA(
        input_irreps="4x0e",
        output_irreps="1x0e",
        width=16,
        depth=1,
        cutoff=10.0,
        num_rbf=8,
    ).cuda()
    candidate_model.load_state_dict(reference_model.state_dict(), strict=True)
    nodes = 7
    receiver = torch.arange(nodes, device=device).repeat_interleave(nodes)
    sender = torch.arange(nodes, device=device).repeat(nodes)
    edge_index = torch.stack([receiver, sender])
    features = torch.randn(nodes, 4, device=device)
    positions = torch.randn(nodes, 3, device=device)

    def evaluate(
        model: ELA,
        position: torch.Tensor,
        backend: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_parameter = model.layers[0].local_radial_score.weight
        with kernel_backend(backend):
            energy = model(ELABatch(features, position, edge_index=edge_index))[
                "node_irreps"
            ]
            forces = conservative_forces(energy, position, create_graph=True)
            position_hvp, parameter_hvp = torch.autograd.grad(
                forces.square().sum(),
                (position, local_parameter),
            )
        return forces, position_hvp, parameter_hvp

    reference_positions = positions.clone().requires_grad_(True)
    reference = evaluate(reference_model, reference_positions, "torch")
    candidate_positions = positions.clone().requires_grad_(True)
    candidate = evaluate(candidate_model, candidate_positions, "triton")

    for actual, expected in zip(candidate, reference, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
        assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        candidate[0].sum(dim=0),
        torch.zeros(3, device=device),
        atol=2e-4,
        rtol=0.0,
    )
