from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal

import torch
from torch import nn

from .layered_se3 import UnifiedSE3State
from .multipole_ops import _st_orthonormal
from .parity_se3 import _st_cross, _st_from_vector


Normalization = Literal["none", "mass", "one_plus_mass"]


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _segment_sum(
    value: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    output = value.new_zeros((num_graphs, *value.shape[1:]))
    output.index_add_(0, batch, value)
    return output


@dataclass(frozen=True, slots=True)
class ImplicitSpatialKernelConfig:
    """Configuration for an edge-free isotropic spatial-kernel reference.

    The implemented reference supports an order-two Gaussian--Taylor feature
    map. For each fixed scale it has feature rank ten and yields a
    pointwise-positive approximation to an isotropic Gaussian kernel. No edge
    list, neighbor list, or node-pair matrix is constructed.
    """

    scales: tuple[float, ...] = (1.0, 2.0, 4.0)
    order: int = 2
    exclude_self: bool = True
    normalization: Normalization = "one_plus_mass"
    learnable_scale_weights: bool = False
    chunk_size: int = 2048
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if not isinstance(self.scales, tuple) or not self.scales:
            raise ValueError("scales must be a non-empty tuple")
        for scale in self.scales:
            _positive_real("scales", scale)
        if self.order not in {0, 2}:
            raise ValueError("implicit spatial kernel supports order 0 or 2")
        if not isinstance(self.exclude_self, bool):
            raise TypeError("exclude_self must be a bool")
        if self.normalization not in {"none", "mass", "one_plus_mass"}:
            raise ValueError(
                "normalization must be 'none', 'mass', or 'one_plus_mass'"
            )
        if not isinstance(self.learnable_scale_weights, bool):
            raise TypeError("learnable_scale_weights must be a bool")
        _positive_integer("chunk_size", self.chunk_size)
        _positive_real("eps", self.eps)

    @property
    def feature_rank_per_scale(self) -> int:
        return 1 if self.order == 0 else 10

    @property
    def feature_rank(self) -> int:
        return len(self.scales) * self.feature_rank_per_scale

    def complexity_contract(self) -> dict[str, object]:
        return {
            "edge_input": False,
            "neighbor_construction": False,
            "pair_matrix": False,
            "kernel": "multiscale_isotropic_gaussian_taylor",
            "taylor_order": self.order,
            "feature_rank": self.feature_rank,
            "chunk_size": self.chunk_size,
            "arithmetic": "O(N * feature_rank * value_width)",
            "working_memory": (
                "O(N*(feature_rank+value_width) + "
                "G*feature_rank*value_width + "
                "chunk_size*feature_rank*value_width)"
            ),
            "exact_radius_graph": False,
        }


@dataclass(frozen=True)
class ImplicitSpatialContext:
    """Prepared node features for edge-free spatial transport."""

    positions: torch.Tensor
    centered_positions: torch.Tensor
    batch: torch.Tensor
    num_graphs: int
    features: torch.Tensor
    feature_sum: torch.Tensor
    self_kernel: torch.Tensor


@dataclass(frozen=True)
class ImplicitSpatialTransport:
    """One edge-free kernel transport result.

    ``output`` follows the input value dtype. ``mass`` and ``self_kernel`` retain
    the geometry accumulation dtype (FP32 for FP16/BF16 inputs) so downstream
    diagnostics and repeated normalization do not quantize the receiver mass.
    """

    output: torch.Tensor
    mass: torch.Tensor
    self_kernel: torch.Tensor


@dataclass(frozen=True)
class ImplicitSpatialMoments:
    """Approximate receiver-centered moments without explicit neighbors."""

    mass: torch.Tensor
    relative_vector: torch.Tensor
    relative_tensor: torch.Tensor


class ImplicitGaussianSpatialKernel(nn.Module):
    r"""Edge-free multiscale Gaussian--Taylor kernel.

    For centered coordinate ``z=x-mu_g`` and length scale ``sigma``, the exact
    isotropic Gaussian factorizes as

    .. math::

        e^{-\|z_i-z_j\|^2/(2\sigma^2)}
        =e^{-\|u_i\|^2/2}e^{-\|u_j\|^2/2}e^{u_i^T u_j},

    where ``u=z/sigma``. The implementation truncates the final exponential at
    degree two,

    .. math::

        e^t \approx 1+t+t^2/2,

    and realizes the polynomial as an explicit ten-dimensional feature map per
    scale. The polynomial is strictly positive for every real ``t``. Translation
    invariance follows from graph centering; rotations act orthogonally on the
    feature blocks, so their dot product is invariant.
    """

    def __init__(self, config: ImplicitSpatialKernelConfig) -> None:
        super().__init__()
        if not isinstance(config, ImplicitSpatialKernelConfig):
            raise TypeError("config must be an ImplicitSpatialKernelConfig")
        self.config = config
        self.register_buffer(
            "_scales",
            torch.tensor(config.scales, dtype=torch.float32),
            persistent=True,
        )
        raw_weights = torch.zeros(len(config.scales), dtype=torch.float32)
        if config.learnable_scale_weights:
            self.raw_scale_weights = nn.Parameter(raw_weights)
        else:
            self.register_buffer(
                "raw_scale_weights",
                raw_weights,
                persistent=True,
            )

    @property
    def feature_rank(self) -> int:
        return self.config.feature_rank

    @staticmethod
    def _work_dtype(value: torch.Tensor) -> torch.dtype:
        return torch.float64 if value.dtype == torch.float64 else torch.float32

    def _validate_inputs(
        self,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> int:
        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if batch.shape != (positions.shape[0],):
            raise ValueError("batch must have shape (N,)")
        if batch.dtype != torch.long:
            raise TypeError("batch must use torch.long")
        if batch.device != positions.device:
            raise ValueError("positions and batch must share one device")
        if positions.numel() and not torch.isfinite(positions).all():
            raise ValueError("positions must be finite")
        if batch.numel() == 0:
            return 0
        if int(batch.min().item()) < 0:
            raise ValueError("batch indices must be nonnegative")
        return int(batch.max().item()) + 1

    def _center(
        self,
        positions: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        dtype = self._work_dtype(positions)
        work = positions.to(dtype=dtype)
        counts = _segment_sum(
            work.new_ones((work.shape[0], 1)),
            batch,
            num_graphs,
        ).clamp_min(1.0)
        center = _segment_sum(work, batch, num_graphs) / counts
        return work - center[batch]

    def _single_scale_features(
        self,
        centered: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        coordinate = centered / scale
        square = coordinate.square().sum(dim=-1, keepdim=True)
        envelope = torch.exp(-0.5 * square)
        if self.config.order == 0:
            return envelope

        # The quadratic block realizes (u_i dot u_j)^2 / 2 exactly:
        # 1/2 ST(u_i):ST(u_j) + |u_i|^2 |u_j|^2 / 6.
        compact_st = _st_from_vector(coordinate)
        orthonormal_st = _st_orthonormal(compact_st) / sqrt(2.0)
        trace = square / sqrt(6.0)
        polynomial = torch.cat(
            [
                coordinate.new_ones((coordinate.shape[0], 1)),
                coordinate,
                orthonormal_st,
                trace,
            ],
            dim=-1,
        )
        return envelope * polynomial

    def feature_map(
        self,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return centered coordinates and the finite spatial feature map."""

        num_graphs = self._validate_inputs(positions, batch)
        if positions.shape[0] == 0:
            return (
                positions,
                positions.new_zeros((0, self.feature_rank)),
                num_graphs,
            )
        centered = self._center(positions, batch, num_graphs)
        dtype = centered.dtype
        scales = self._scales.to(device=positions.device, dtype=dtype)
        weights = torch.softmax(
            self.raw_scale_weights.to(device=positions.device, dtype=dtype),
            dim=0,
        )
        blocks = [
            torch.sqrt(weights[index])
            * self._single_scale_features(centered, scales[index])
            for index in range(scales.numel())
        ]
        features = torch.cat(blocks, dim=-1)
        return centered, features, num_graphs

    def prepare(
        self,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> ImplicitSpatialContext:
        centered, features, num_graphs = self.feature_map(positions, batch)
        feature_sum = _segment_sum(features, batch, num_graphs)
        self_kernel = features.square().sum(dim=-1)
        return ImplicitSpatialContext(
            positions=positions,
            centered_positions=centered,
            batch=batch,
            num_graphs=num_graphs,
            features=features,
            feature_sum=feature_sum,
            self_kernel=self_kernel,
        )

    def _raw_transport(
        self,
        values: torch.Tensor,
        context: ImplicitSpatialContext,
    ) -> ImplicitSpatialTransport:
        if values.shape[0] != context.positions.shape[0]:
            raise ValueError("values and positions must have the same node count")
        if values.device != context.positions.device:
            raise ValueError("values and context must share one device")
        if values.numel() and not torch.isfinite(values).all():
            raise ValueError("values must be finite")

        original_shape = values.shape
        flat = values.reshape(values.shape[0], -1)
        if flat.shape[0] == 0:
            return ImplicitSpatialTransport(
                output=values.clone(),
                mass=context.features.new_zeros((0,)),
                self_kernel=context.features.new_zeros((0,)),
            )

        dtype = torch.promote_types(context.features.dtype, flat.dtype)
        features = context.features.to(dtype=dtype)
        flat_work = flat.to(dtype=dtype)
        feature_sum = context.feature_sum.to(dtype=dtype)
        statistics = flat_work.new_zeros(
            (context.num_graphs, features.shape[-1], flat_work.shape[-1])
        )

        # Chunked sufficient-statistic accumulation is O(NFD) and avoids both
        # an N x N pair matrix and an N x F x D full-node outer tensor.
        chunk = self.config.chunk_size
        for start in range(0, flat_work.shape[0], chunk):
            stop = min(start + chunk, flat_work.shape[0])
            phi = features[start:stop]
            value = flat_work[start:stop]
            outer = phi.unsqueeze(-1) * value.unsqueeze(-2)
            statistics.index_add_(0, context.batch[start:stop], outer)

        output_chunks: list[torch.Tensor] = []
        mass_chunks: list[torch.Tensor] = []
        for start in range(0, flat_work.shape[0], chunk):
            stop = min(start + chunk, flat_work.shape[0])
            phi = features[start:stop]
            batch_chunk = context.batch[start:stop]
            value = flat_work[start:stop]
            graph_statistics = statistics[batch_chunk]
            graph_output = torch.einsum("cf,cfd->cd", phi, graph_statistics)
            graph_mass = (phi * feature_sum[batch_chunk]).sum(dim=-1)
            if self.config.exclude_self:
                diagonal = context.self_kernel[start:stop].to(dtype=dtype)
                graph_output = graph_output - diagonal.unsqueeze(-1) * value
                graph_mass = graph_mass - diagonal
            output_chunks.append(graph_output)
            mass_chunks.append(graph_mass.clamp_min(0.0))

        output = torch.cat(output_chunks, dim=0)
        mass = torch.cat(mass_chunks, dim=0)
        return ImplicitSpatialTransport(
            output=output.reshape(original_shape),
            mass=mass,
            self_kernel=context.self_kernel.to(dtype=dtype),
        )

    def _denominator(self, mass: torch.Tensor) -> torch.Tensor:
        if self.config.normalization == "none":
            return torch.ones_like(mass)
        if self.config.normalization == "mass":
            return mass.clamp_min(self.config.eps)
        return 1.0 + mass

    def transport_prepared(
        self,
        values: torch.Tensor,
        context: ImplicitSpatialContext,
    ) -> ImplicitSpatialTransport:
        raw = self._raw_transport(values, context)
        denominator = self._denominator(raw.mass)
        output = raw.output / denominator.reshape(
            denominator.shape[0],
            *((1,) * (raw.output.ndim - 1)),
        )
        return ImplicitSpatialTransport(
            output=output.to(dtype=values.dtype),
            mass=raw.mass,
            self_kernel=raw.self_kernel,
        )

    def forward(
        self,
        values: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> ImplicitSpatialTransport:
        return self.transport_prepared(values, self.prepare(positions, batch))

    def moments_prepared(
        self,
        context: ImplicitSpatialContext,
    ) -> ImplicitSpatialMoments:
        centered = context.centered_positions
        first = self._raw_transport(centered, context)
        centered_tensor = _st_from_vector(centered)
        second = self._raw_transport(centered_tensor, context)
        mass = first.mass.to(dtype=centered.dtype)

        relative_vector = first.output.to(dtype=centered.dtype) - mass.unsqueeze(-1) * centered
        relative_tensor = (
            second.output.to(dtype=centered.dtype)
            + mass.unsqueeze(-1) * centered_tensor
            - 2.0 * _st_cross(first.output.to(dtype=centered.dtype), centered)
        )
        denominator = self._denominator(mass)
        relative_vector = relative_vector / denominator.unsqueeze(-1)
        relative_tensor = relative_tensor / denominator.unsqueeze(-1)
        return ImplicitSpatialMoments(
            mass=mass,
            relative_vector=relative_vector,
            relative_tensor=relative_tensor,
        )

    def moments(
        self,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> ImplicitSpatialMoments:
        return self.moments_prepared(self.prepare(positions, batch))


class ImplicitSpatialStateTransport(nn.Module):
    """Apply one edge-free invariant kernel to every hidden irrep sector."""

    def __init__(self, kernel: ImplicitGaussianSpatialKernel) -> None:
        super().__init__()
        if not isinstance(kernel, ImplicitGaussianSpatialKernel):
            raise TypeError("kernel must be an ImplicitGaussianSpatialKernel")
        self.kernel = kernel

    def forward(
        self,
        state: UnifiedSE3State,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> UnifiedSE3State:
        context = self.kernel.prepare(positions, batch)

        def apply(value: torch.Tensor) -> torch.Tensor:
            return self.kernel.transport_prepared(value, context).output

        return UnifiedSE3State(
            even_scalar=apply(state.even_scalar),
            odd_scalar=apply(state.odd_scalar),
            polar_vector=apply(state.polar_vector),
            axial_vector=apply(state.axial_vector),
            even_tensor=apply(state.even_tensor),
            odd_tensor=apply(state.odd_tensor),
        )


__all__ = [
    "ImplicitGaussianSpatialKernel",
    "ImplicitSpatialContext",
    "ImplicitSpatialKernelConfig",
    "ImplicitSpatialMoments",
    "ImplicitSpatialStateTransport",
    "ImplicitSpatialTransport",
]
