from __future__ import annotations

import torch

from equivariant_linear_attention.nn.local_geometry import PointwiseLocalFeatures
from equivariant_linear_attention.nn.local_jet_types import LocalFeatureJet
from equivariant_linear_attention.nn.local_moments import LocalCumulants
from equivariant_linear_attention.nn.local_projection import LocalFeatureProjection


def test_local_projection_supports_unequal_moment_and_body_ranks() -> None:
    generator = torch.Generator().manual_seed(601)
    nodes = 5
    moment_rank = 8
    body_rank = 6
    scales = 3
    probes = 4
    heads = 8

    def sample(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    moments = LocalCumulants(
        mass=sample(nodes, moment_rank),
        polar=sample(nodes, moment_rank, 3),
        second_scalar=sample(nodes, moment_rank),
        even_tensor=sample(nodes, moment_rank, 5),
        axial=sample(nodes, moment_rank, 3),
        odd_scalar=sample(nodes, moment_rank),
        odd_tensor=sample(nodes, moment_rank, 5),
        third_tensor=sample(nodes, moment_rank, 7),
        fourth_scalar=sample(nodes, moment_rank),
        fourth_tensor=sample(nodes, moment_rank, 5),
        fourth_rank4=sample(nodes, moment_rank, 9),
        scale=torch.rand(nodes, moment_rank) + 0.5,
    )
    jet = LocalFeatureJet(
        value=sample(nodes, scales, probes),
        gradient=sample(nodes, scales, probes, 3),
        laplacian=sample(nodes, scales, probes),
        hessian=sample(nodes, scales, probes, 5),
        wavelet_value=sample(nodes, scales - 1, probes),
        wavelet_gradient=sample(nodes, scales - 1, probes, 3),
        wavelet_laplacian=sample(nodes, scales - 1, probes),
        wavelet_hessian=sample(nodes, scales - 1, probes, 5),
        confidence=torch.rand(nodes, scales),
        scale=torch.rand(nodes, scales) + 0.5,
    )
    local = PointwiseLocalFeatures(
        moments=moments,
        jet=jet,
        support_scale=torch.rand(nodes) + 0.5,
    )
    projection = LocalFeatureProjection(
        scalar_width=128,
        num_heads=heads,
        moment_rank=moment_rank,
        num_scales=scales,
        probe_rank=probes,
        body_rank=body_rank,
        eps=1e-8,
    )
    output = projection(local)
    assert output.even_scalar.shape == (nodes, 128)
    assert output.odd_scalar.shape == (nodes, heads)
    assert output.polar_vector.shape == (nodes, heads, 3)
    assert output.axial_vector.shape == (nodes, heads, 3)
    assert output.even_tensor.shape == (nodes, heads, 5)
    assert output.odd_tensor.shape == (nodes, heads, 5)
