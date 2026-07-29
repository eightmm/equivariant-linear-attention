"""Bounded transient high-order reference paths.

The optimized model keeps persistent Cartesian state at ``l <= 2``.  This
module provides the deliberately narrower high-order operation used by the
``high_order`` profile:

``sender 1o x edge Y2 -> transient 3o``
``aggregate 3o and 2e independently at the receiver``
``aggregated 3o x aggregated 2e -> output 1o``.

The ``l=3`` value is therefore an activation local to one call.  It is never
returned as persistent node state and never appears in a checkpoint.
"""

from __future__ import annotations

from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from .reference_irreps import (
    ReferenceTensorProductPath,
    tensor_product_path,
)
from .spherical import (
    cartesian_to_real_l1,
    real_l1_to_cartesian,
    real_spherical_harmonics,
)


_REAL_DTYPES = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
)
_INTEGER_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
)


class TransientL3Workspace(nn.Module):
    """Aggregate-then-project ``1o -> 3o -> 1o`` reference interaction.

    Optional edge scalars only modulate the lift and the independently
    aggregated quadrupole through invariant gates.  They cannot change the
    transformation law.  Half and bfloat16 projections accumulate in FP32;
    float64 remains float64 and ordinary PyTorch autograd supports gradgrad.
    """

    workspace_degree = 3
    persistent_max_degree = 2
    supports_gradgrad = True

    def __init__(
        self,
        input_vector_channels: int,
        workspace_channels: int,
        output_vector_channels: int,
        *,
        edge_scalar_dim: int = 0,
        normalization: str = "sqrt_degree",
        device: torch.device | str | int | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        for name, value, minimum in (
            ("input_vector_channels", input_vector_channels, 1),
            ("workspace_channels", workspace_channels, 1),
            ("output_vector_channels", output_vector_channels, 1),
            ("edge_scalar_dim", edge_scalar_dim, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < minimum:
                relation = "positive" if minimum == 1 else "nonnegative"
                raise ValueError(f"{name} must be {relation}")
        if normalization not in {"none", "sqrt_degree", "mass_damped"}:
            raise ValueError(
                "normalization must be 'none', 'sqrt_degree', or 'mass_damped'"
            )
        self.input_vector_channels = input_vector_channels
        self.workspace_channels = workspace_channels
        self.output_vector_channels = output_vector_channels
        self.edge_scalar_dim = edge_scalar_dim
        self.normalization = normalization
        factory_kwargs = {"device": device, "dtype": dtype}
        self.lift_weight = nn.Parameter(
            torch.empty(
                workspace_channels,
                input_vector_channels,
                1,
                **factory_kwargs,
            )
        )
        self.project_weight = nn.Parameter(
            torch.empty(
                output_vector_channels,
                workspace_channels,
                1,
                **factory_kwargs,
            )
        )
        self.edge_gate = (
            nn.Linear(edge_scalar_dim, 2, **factory_kwargs)
            if edge_scalar_dim
            else None
        )
        self._lift_path = ReferenceTensorProductPath.parse("1o", "2e", "3o")
        self._project_path = ReferenceTensorProductPath.parse("3o", "2e", "1o")
        self.reset_parameters()

    def reset_parameters(self) -> None:
        lift_bound = 1.0 / sqrt(self.input_vector_channels)
        project_bound = 1.0 / sqrt(self.workspace_channels)
        nn.init.uniform_(self.lift_weight, -lift_bound, lift_bound)
        nn.init.uniform_(self.project_weight, -project_bound, project_bound)
        if self.edge_gate is not None:
            nn.init.zeros_(self.edge_gate.weight)
            nn.init.zeros_(self.edge_gate.bias)

    def forward(
        self,
        node_vectors: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_invariants: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(
            node_vectors,
            pos,
            edge_index,
            edge_invariants,
        )
        receiver, sender = edge_index.to(dtype=torch.long).unbind(dim=0)
        computation_dtype = self._computation_dtype(
            node_vectors,
            pos,
            edge_invariants,
        )
        displacement = (
            pos.index_select(0, sender) - pos.index_select(0, receiver)
        ).to(dtype=computation_dtype)
        quadrupole = real_spherical_harmonics(
            2,
            displacement,
            normalize=True,
        ).unsqueeze(-2)
        sender_vectors = cartesian_to_real_l1(
            node_vectors.index_select(0, sender).to(dtype=computation_dtype)
        )
        lifted = tensor_product_path(
            sender_vectors,
            quadrupole,
            self._lift_path,
            weights=self.lift_weight.to(dtype=computation_dtype),
        )
        lift_gate, quadrupole_gate = self._edge_gates(
            edge_invariants,
            edge_count=edge_index.shape[1],
            dtype=computation_dtype,
            device=node_vectors.device,
        )
        lifted = lifted * lift_gate[:, None, None]
        quadrupole = quadrupole * quadrupole_gate[:, None, None]

        num_nodes = node_vectors.shape[0]
        workspace = torch.zeros(
            num_nodes,
            self.workspace_channels,
            7,
            dtype=computation_dtype,
            device=node_vectors.device,
        )
        receiver_quadrupole = torch.zeros(
            num_nodes,
            1,
            5,
            dtype=computation_dtype,
            device=node_vectors.device,
        )
        workspace.index_add_(0, receiver, lifted.to(dtype=computation_dtype))
        receiver_quadrupole.index_add_(
            0,
            receiver,
            quadrupole.to(dtype=computation_dtype),
        )
        if self.normalization != "none":
            degree = torch.zeros(
                num_nodes,
                dtype=computation_dtype,
                device=node_vectors.device,
            )
            degree.index_add_(
                0,
                receiver,
                torch.ones_like(receiver, dtype=computation_dtype),
            )
            if self.normalization == "sqrt_degree":
                denominator = degree.clamp_min(1.0).sqrt()
            else:
                denominator = 1.0 + degree
            workspace = workspace / denominator[:, None, None]
            receiver_quadrupole = (
                receiver_quadrupole / denominator[:, None, None]
            )

        projected = tensor_product_path(
            workspace,
            receiver_quadrupole,
            self._project_path,
            weights=self.project_weight.to(dtype=computation_dtype),
        )
        return real_l1_to_cartesian(projected).to(dtype=node_vectors.dtype)

    def _edge_gates(
        self,
        edge_invariants: torch.Tensor | None,
        *,
        edge_count: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.edge_gate is None:
            ones = torch.ones(edge_count, dtype=dtype, device=device)
            return ones, ones
        if edge_invariants is None:
            raise RuntimeError("validated edge invariants are missing")
        logits = F.linear(
            edge_invariants.to(dtype=dtype),
            self.edge_gate.weight.to(dtype=dtype),
            self.edge_gate.bias.to(dtype=dtype),
        )
        gates = 1.0 + torch.tanh(logits)
        return gates.unbind(dim=-1)

    def _computation_dtype(
        self,
        node_vectors: torch.Tensor,
        pos: torch.Tensor,
        edge_invariants: torch.Tensor | None,
    ) -> torch.dtype:
        values = [
            node_vectors,
            pos,
            self.lift_weight,
            self.project_weight,
        ]
        if edge_invariants is not None:
            values.append(edge_invariants)
        dtype = values[0].dtype
        for value in values[1:]:
            dtype = torch.promote_types(dtype, value.dtype)
        return torch.float64 if dtype == torch.float64 else torch.float32

    def _validate_inputs(
        self,
        node_vectors: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_invariants: torch.Tensor | None,
    ) -> None:
        if not isinstance(node_vectors, torch.Tensor):
            raise TypeError("node_vectors must be a tensor")
        if node_vectors.dtype not in _REAL_DTYPES:
            raise TypeError("node_vectors must use a real floating dtype")
        expected_vector_shape = (
            node_vectors.shape[0],
            self.input_vector_channels,
            3,
        )
        if node_vectors.shape != expected_vector_shape:
            raise ValueError(
                "node_vectors must have shape "
                f"(N, {self.input_vector_channels}, 3)"
            )
        if not isinstance(pos, torch.Tensor):
            raise TypeError("pos must be a tensor")
        if pos.dtype not in _REAL_DTYPES:
            raise TypeError("pos must use a real floating dtype")
        if pos.shape != (node_vectors.shape[0], 3):
            raise ValueError("pos must have shape (N, 3)")
        if pos.device != node_vectors.device:
            raise ValueError("pos and node_vectors must share one device")
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError("edge_index must be a tensor")
        if edge_index.dtype not in _INTEGER_DTYPES:
            raise TypeError("edge_index must use an integer dtype")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if edge_index.device != node_vectors.device:
            raise ValueError("edge_index and node_vectors must share one device")
        if edge_index.numel():
            edge_long = edge_index.to(dtype=torch.long)
            if bool((edge_long < 0).any().item()) or bool(
                (edge_long >= node_vectors.shape[0]).any().item()
            ):
                raise ValueError("edge_index values are out of range")
        if self.edge_scalar_dim:
            if not isinstance(edge_invariants, torch.Tensor):
                raise TypeError(
                    "edge_invariants must be a tensor when edge_scalar_dim is positive"
                )
            if edge_invariants.dtype not in _REAL_DTYPES:
                raise TypeError(
                    "edge_invariants must use a real floating dtype"
                )
            if edge_invariants.shape != (
                edge_index.shape[1],
                self.edge_scalar_dim,
            ):
                raise ValueError(
                    "edge_invariants must have shape "
                    f"(E, {self.edge_scalar_dim})"
                )
            if edge_invariants.device != node_vectors.device:
                raise ValueError(
                    "edge_invariants and node_vectors must share one device"
                )
        elif edge_invariants is not None:
            raise ValueError(
                "edge_invariants must be omitted when edge_scalar_dim is zero"
            )
