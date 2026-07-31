from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import torch

from .irreps import IrrepLayout, pack_irreps


SpatialSyntheticTask = Literal[
    "local_directional",
    "smooth_gaussian",
    "mixed",
]


@dataclass(frozen=True)
class SyntheticSpatialBatch:
    """One deterministic full-batch synthetic spatial-regression fixture."""

    node_irreps: torch.Tensor
    positions: torch.Tensor
    batch: torch.Tensor
    edge_index: torch.Tensor
    targets: torch.Tensor
    input_irreps: str
    task: SpatialSyntheticTask
    cutoff: float
    gaussian_scale: float

    @property
    def num_graphs(self) -> int:
        return int(self.targets.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> SyntheticSpatialBatch:
        target = torch.device(device)
        floating_dtype = self.node_irreps.dtype if dtype is None else dtype
        return SyntheticSpatialBatch(
            node_irreps=self.node_irreps.to(
                device=target,
                dtype=floating_dtype,
            ),
            positions=self.positions.to(
                device=target,
                dtype=floating_dtype,
            ),
            batch=self.batch.to(device=target),
            edge_index=self.edge_index.to(device=target),
            targets=self.targets.to(device=target, dtype=floating_dtype),
            input_irreps=self.input_irreps,
            task=self.task,
            cutoff=self.cutoff,
            gaussian_scale=self.gaussian_scale,
        )


def _c2_cutoff(distance: torch.Tensor, cutoff: float) -> torch.Tensor:
    ratio = (distance / cutoff).square()
    inside = ratio < 1.0
    u = ratio.clamp(min=0.0, max=1.0)
    value = 1.0 - 10.0 * u.pow(3) + 15.0 * u.pow(4) - 6.0 * u.pow(5)
    return torch.where(inside, value, torch.zeros_like(value))


def _pair_target(
    scalar: torch.Tensor,
    polar: torch.Tensor,
    positions: torch.Tensor,
    *,
    task: SpatialSyntheticTask,
    cutoff: float,
    gaussian_scale: float,
) -> torch.Tensor:
    nodes = positions.shape[0]
    receiver, sender = torch.triu_indices(nodes, nodes, offset=1)
    displacement = positions[sender] - positions[receiver]
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    direction = displacement / distance.clamp_min(1e-8).unsqueeze(-1)

    scalar_content = scalar[receiver, 0] * scalar[sender, 0]
    scalar_content = scalar_content + 0.25 * (
        scalar[receiver, 1] * scalar[sender, 1]
    )
    axis_content = (
        (polar[receiver] * direction).sum(dim=-1)
        * (polar[sender] * direction).sum(dim=-1)
    )
    vector_content = (polar[receiver] * polar[sender]).sum(dim=-1)

    local_weight = _c2_cutoff(distance, cutoff)
    gaussian_weight = torch.exp(
        -distance.square() / (2.0 * gaussian_scale**2)
    )
    local_target = (
        local_weight * (scalar_content + 0.5 * axis_content)
    ).sum() / nodes
    smooth_target = (
        gaussian_weight * (scalar_content + 0.2 * vector_content)
    ).sum() / nodes

    if task == "local_directional":
        return local_target
    if task == "smooth_gaussian":
        return smooth_target
    if task == "mixed":
        return local_target + 0.5 * smooth_target
    raise ValueError(f"unsupported task: {task}")


def _directed_radius_edges(
    positions: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    displacement = positions[:, None, :] - positions[None, :, :]
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    mask = (distance < cutoff) & ~torch.eye(
        positions.shape[0],
        dtype=torch.bool,
        device=positions.device,
    )
    receiver, sender = mask.nonzero(as_tuple=True)
    return torch.stack([receiver, sender])


def make_synthetic_spatial_batch(
    *,
    task: SpatialSyntheticTask,
    num_graphs: int,
    nodes_per_graph: int,
    seed: int,
    scalar_dim: int = 4,
    cutoff: float = 1.75,
    candidate_skin: float = 0.25,
    gaussian_scale: float = 2.5,
    coordinate_scale: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> SyntheticSpatialBatch:
    """Create matched local, smooth, or mixed synthetic regression data.

    The exact sparse candidate graph contains every pair inside
    ``cutoff + candidate_skin``. The smooth target is graph-global and is not
    truncated by that graph. All target construction happens before training and
    is intentionally allowed to use dense pair computations.
    """

    if task not in {"local_directional", "smooth_gaussian", "mixed"}:
        raise ValueError("unsupported synthetic task")
    for name, value in (
        ("num_graphs", num_graphs),
        ("nodes_per_graph", nodes_per_graph),
        ("scalar_dim", scalar_dim),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if scalar_dim < 2:
        raise ValueError("scalar_dim must be at least two")
    if cutoff <= 0.0 or candidate_skin < 0.0:
        raise ValueError("cutoff must be positive and skin nonnegative")
    if gaussian_scale <= 0.0 or coordinate_scale <= 0.0:
        raise ValueError("scales must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    input_irreps = f"{scalar_dim}x0e + 1x1o"
    layout = IrrepLayout.parse(input_irreps)

    feature_blocks: list[torch.Tensor] = []
    position_blocks: list[torch.Tensor] = []
    batch_blocks: list[torch.Tensor] = []
    edge_blocks: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    offset = 0

    for graph_index in range(num_graphs):
        # Anisotropic cloud with random graph-level scale and center. Targets use
        # only pair differences, so center is a nuisance variable.
        graph_scale = coordinate_scale * (
            0.75
            + 0.5
            * torch.rand((), generator=generator, dtype=torch.float64)
        )
        axes = torch.tensor([1.0, 0.8, 1.2], dtype=torch.float64)
        positions = graph_scale * axes * torch.randn(
            nodes_per_graph,
            3,
            generator=generator,
            dtype=torch.float64,
        )
        center = 3.0 * torch.randn(
            1,
            3,
            generator=generator,
            dtype=torch.float64,
        )
        positions = positions + center
        scalar = torch.randn(
            nodes_per_graph,
            scalar_dim,
            generator=generator,
            dtype=torch.float64,
        )
        polar = torch.randn(
            nodes_per_graph,
            3,
            generator=generator,
            dtype=torch.float64,
        )

        feature_blocks.append(
            pack_irreps(
                layout,
                {
                    "0e": scalar.unsqueeze(-1),
                    "1o": polar.unsqueeze(1),
                },
            )
        )
        position_blocks.append(positions)
        batch_blocks.append(
            torch.full(
                (nodes_per_graph,),
                graph_index,
                dtype=torch.long,
            )
        )
        local_edges = _directed_radius_edges(
            positions,
            cutoff + candidate_skin,
        )
        edge_blocks.append(local_edges + offset)
        offset += nodes_per_graph
        targets.append(
            _pair_target(
                scalar,
                polar,
                positions,
                task=task,
                cutoff=cutoff,
                gaussian_scale=gaussian_scale,
            )
        )

    edge_index = (
        torch.cat(edge_blocks, dim=1)
        if edge_blocks
        else torch.empty((2, 0), dtype=torch.long)
    )
    return SyntheticSpatialBatch(
        node_irreps=torch.cat(feature_blocks, dim=0).to(dtype=dtype),
        positions=torch.cat(position_blocks, dim=0).to(dtype=dtype),
        batch=torch.cat(batch_blocks, dim=0),
        edge_index=edge_index,
        targets=torch.stack(targets).reshape(num_graphs, 1).to(dtype=dtype),
        input_irreps=input_irreps,
        task=task,
        cutoff=float(cutoff),
        gaussian_scale=float(gaussian_scale),
    )


def synthetic_batch_sha256(batch: SyntheticSpatialBatch) -> str:
    digest = sha256()
    for name in (
        "node_irreps",
        "positions",
        "batch",
        "edge_index",
        "targets",
    ):
        tensor = getattr(batch, name).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    digest.update(batch.task.encode("ascii"))
    digest.update(str(batch.cutoff).encode("ascii"))
    digest.update(str(batch.gaussian_scale).encode("ascii"))
    return digest.hexdigest()


__all__ = [
    "SpatialSyntheticTask",
    "SyntheticSpatialBatch",
    "make_synthetic_spatial_batch",
    "synthetic_batch_sha256",
]
