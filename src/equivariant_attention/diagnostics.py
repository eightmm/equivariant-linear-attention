"""Pure, bounded diagnostics for attention and kernel inspection.

The helpers in this module detach their inputs and return only Python scalar
values.  They do not retain autograd graphs, mutate modules, or switch model
training state.  Matrix effective rank is deliberately opt-in and size-bounded
because it requires a full singular-value decomposition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real

import torch


ScalarSummary = dict[str, float | int]


def attention_weight_summary(
    weights: torch.Tensor,
    *,
    include_effective_rank: bool = False,
    effective_rank_max_size: int = 512,
) -> ScalarSummary:
    """Summarize one query-by-key attention matrix with JSON-safe scalars.

    Rows are normalized before computing the diagnostics, so callers may pass
    either probabilities or nonnegative unnormalized attention weights.  The
    reported max weight and effective support are means over query rows.  For a
    singleton key set, normalized entropy is defined as one rather than the
    indeterminate ratio ``0 / log(1)``.
    """

    probabilities = _normalized_attention(weights)
    num_queries, num_keys = probabilities.shape

    log_probabilities = torch.where(
        probabilities > 0,
        probabilities.log(),
        torch.zeros_like(probabilities),
    )
    row_entropy = -(probabilities * log_probabilities).sum(dim=-1)
    if num_keys == 1:
        normalized_entropy = 1.0
    else:
        normalized_entropy = float(
            (
                row_entropy.mean()
                / torch.tensor(
                    num_keys, dtype=torch.float64, device=probabilities.device
                ).log()
            ).item()
        )

    column_mass = probabilities.sum(dim=0)
    column_mean = column_mass.mean()
    column_cv = float((column_mass.std(correction=0) / column_mean).item())

    summary: ScalarSummary = {
        "num_queries": int(num_queries),
        "num_keys": int(num_keys),
        "entropy_over_log_n": normalized_entropy,
        "max_weight": float(probabilities.amax(dim=-1).mean().item()),
        "effective_support": float(row_entropy.exp().mean().item()),
        "column_cv": column_cv,
    }
    if include_effective_rank:
        summary["effective_rank"] = matrix_effective_rank(
            probabilities,
            max_matrix_size=effective_rank_max_size,
        )
    return summary


def kernel_component_quantiles(
    components: Mapping[str, torch.Tensor | Real],
    *,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> dict[str, float]:
    """Return flattened, JSON-safe quantiles for optional kernel components."""

    levels = _validated_quantiles(quantiles)
    summary: dict[str, float] = {}
    for name, component in components.items():
        if not isinstance(name, str) or not name:
            raise ValueError("kernel component names must be nonempty strings")
        values = _finite_vector(component, name=f"kernel component {name!r}")
        component_quantiles = torch.quantile(
            values, torch.tensor(levels, dtype=torch.float64, device=values.device)
        )
        for level, value in zip(levels, component_quantiles, strict=True):
            key = f"{name}.{_quantile_label(level)}"
            if key in summary:
                raise ValueError(f"quantile labels are not unique: {key}")
            summary[key] = float(value.item())
    return summary


def kernel_parameter_summary(
    beta: torch.Tensor | Real,
    gamma: torch.Tensor | Real,
) -> dict[str, float]:
    """Report finite min/mean/max scalars for the kernel beta and gamma."""

    summary: dict[str, float] = {}
    for name, parameter in (("beta", beta), ("gamma", gamma)):
        values = _finite_vector(parameter, name=name)
        summary[f"{name}.min"] = float(values.min().item())
        summary[f"{name}.mean"] = float(values.mean().item())
        summary[f"{name}.max"] = float(values.max().item())
    return summary


def dense_kernel_attention_summary(
    query_scalar: torch.Tensor,
    key_scalar: torch.Tensor,
    query_vector: torch.Tensor,
    key_vector: torch.Tensor,
    *,
    beta: torch.Tensor | Real,
    gamma: torch.Tensor | Real,
    kernel_floor: float,
    kernel_floor_mode: str = "fixed",
    graph_size: int | None = None,
    alignment_linear_term: bool,
    balanced: bool,
    pair_gate: torch.Tensor | None = None,
    include_effective_rank: bool = False,
    max_nodes: int = 512,
) -> dict[str, float | int]:
    """Diagnose one dense head without making dense work a model dependency.

    This helper is deliberately limited to small diagnostic graphs.  It mirrors
    the registered kernel and exactly one optional key-balancing cycle, then
    returns component, mass, denominator, and normalized-attention summaries.
    In ``inverse_graph_size`` mode, the positive baseline ``c + beta(1+dt)``
    is divided by graph size while content and the quadratic angular term are
    left unchanged, matching the registered global kernel.
    """

    tensors = (query_scalar, key_scalar, query_vector, key_vector)
    if any(not isinstance(value, torch.Tensor) or value.ndim != 2 for value in tensors):
        raise ValueError(
            "query/key scalar and vector features must be two-dimensional tensors"
        )
    num_queries = query_scalar.shape[0]
    num_keys = key_scalar.shape[0]
    if num_queries == 0 or num_keys == 0:
        raise ValueError("diagnostic query and key sets must be nonempty")
    if max(num_queries, num_keys) > max_nodes:
        raise ValueError("dense kernel diagnostics exceed the explicit max_nodes bound")
    if query_scalar.shape[1] != key_scalar.shape[1]:
        raise ValueError("query and key scalar feature dimensions must match")
    if query_vector.shape != (num_queries, 3) or key_vector.shape != (num_keys, 3):
        raise ValueError("query and key vectors must have shape (N, 3)")
    if not isinstance(alignment_linear_term, bool) or not isinstance(balanced, bool):
        raise TypeError("alignment_linear_term and balanced must be bools")
    floor = float(kernel_floor)
    if not isfinite(floor) or floor <= 0.0:
        raise ValueError("kernel_floor must be finite and positive")
    if kernel_floor_mode not in {"fixed", "inverse_graph_size"}:
        raise ValueError(f"unknown kernel_floor_mode: {kernel_floor_mode}")
    if kernel_floor_mode == "inverse_graph_size":
        if (
            isinstance(graph_size, bool)
            or not isinstance(graph_size, int)
            or graph_size <= 0
        ):
            raise ValueError(
                "graph_size must be a positive integer for inverse_graph_size"
            )
        if graph_size != num_queries or graph_size != num_keys:
            raise ValueError(
                "graph_size must match both diagnostic query and key counts"
            )
        if balanced:
            raise ValueError(
                "inverse_graph_size diagnostics are not registered with key balancing"
            )
        baseline_scale = 1.0 / graph_size
    else:
        baseline_scale = 1.0

    dtype = torch.float64
    q0 = query_scalar.detach().to(dtype=dtype)
    k0 = key_scalar.detach().to(dtype=dtype, device=q0.device)
    q1 = query_vector.detach().to(dtype=dtype, device=q0.device)
    k1 = key_vector.detach().to(dtype=dtype, device=q0.device)
    beta_value = _finite_vector(beta, name="beta").to(device=q0.device)
    gamma_value = _finite_vector(gamma, name="gamma").to(device=q0.device)
    if beta_value.numel() != 1 or gamma_value.numel() != 1:
        raise ValueError("dense diagnostics accept one scalar beta and gamma per head")
    beta_scalar = beta_value[0]
    gamma_scalar = gamma_value[0]

    content = q0 @ k0.T
    angular = q1 @ k1.T
    floor_component = torch.full_like(content, floor * baseline_scale)
    beta_constant = torch.ones_like(content) * beta_scalar * baseline_scale
    beta_dot = (
        beta_scalar * baseline_scale * angular
        if alignment_linear_term
        else torch.zeros_like(angular)
    )
    quadratic = gamma_scalar * angular.square()
    kernel = floor_component + content + beta_constant + beta_dot + quadratic
    if bool((kernel <= 0.0).any().item()) or not bool(
        torch.isfinite(kernel).all().item()
    ):
        raise ValueError("diagnostic kernel must be finite and strictly positive")
    if pair_gate is None:
        weighted = kernel
    else:
        if pair_gate.shape != kernel.shape:
            raise ValueError("pair_gate must match the query-by-key kernel shape")
        gate = pair_gate.detach().to(dtype=dtype, device=q0.device)
        if not bool(torch.isfinite(gate).all().item()) or bool(
            (gate < 0.0).any().item()
        ):
            raise ValueError("pair_gate must be finite and nonnegative")
        weighted = kernel * gate
    key_mass = weighted.sum(dim=0)
    if bool((key_mass <= 0.0).any().item()):
        raise ValueError("every diagnostic key must have positive mass")
    balanced_weight = weighted / key_mass.unsqueeze(0) if balanced else weighted
    row_denominator = balanced_weight.sum(dim=-1)
    if bool((row_denominator <= 0.0).any().item()):
        raise ValueError("every diagnostic query must have a positive denominator")
    weights = balanced_weight / row_denominator.unsqueeze(-1)

    summary: dict[str, float | int] = {}
    component_summary = kernel_component_quantiles(
        {
            "floor": floor_component,
            "content": content,
            "beta_constant": beta_constant,
            "beta_dot": beta_dot,
            "quadratic": quadratic,
            "kernel": kernel,
        },
        quantiles=(0.0, 0.01, 0.5, 0.99, 1.0),
    )
    summary.update(component_summary)
    summary.update(kernel_parameter_summary(beta_scalar, gamma_scalar))
    summary.update(_range_summary("key_mass", key_mass))
    summary.update(_range_summary("row_denominator", row_denominator))
    summary.update(
        {
            f"attention.{name}": value
            for name, value in attention_weight_summary(
                weights,
                include_effective_rank=include_effective_rank,
                effective_rank_max_size=max_nodes,
            ).items()
        }
    )
    return summary


def memory_assignment_summary(assignment: torch.Tensor) -> dict[str, float | int]:
    """Summarize finite soft-memory occupancy and assignment entropy."""

    if not isinstance(assignment, torch.Tensor) or assignment.ndim != 3:
        raise ValueError("assignment must have shape (nodes, heads, memories)")
    if 0 in assignment.shape:
        raise ValueError("assignment dimensions must be positive")
    probabilities = assignment.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(probabilities).all().item()) or bool(
        (probabilities < 0.0).any().item()
    ):
        raise ValueError("assignment must be finite and nonnegative")
    row_mass = probabilities.sum(dim=-1, keepdim=True)
    if bool((row_mass <= 0.0).any().item()):
        raise ValueError("each assignment row must have positive mass")
    probabilities = probabilities / row_mass
    occupancy = probabilities.sum(dim=0)
    entropy = -(
        probabilities
        * torch.where(
            probabilities > 0.0,
            probabilities.log(),
            torch.zeros_like(probabilities),
        )
    ).sum(dim=-1)
    memory_count = probabilities.shape[-1]
    normalized_entropy = (
        1.0
        if memory_count == 1
        else float(
            (
                entropy.mean()
                / torch.tensor(
                    float(memory_count),
                    dtype=torch.float64,
                    device=probabilities.device,
                ).log()
            ).item()
        )
    )
    return {
        "memory_count": int(memory_count),
        "occupancy.min": float(occupancy.min().item()),
        "occupancy.mean": float(occupancy.mean().item()),
        "occupancy.max": float(occupancy.max().item()),
        "assignment_entropy_over_log_m": normalized_entropy,
    }


def matrix_effective_rank(matrix: torch.Tensor, *, max_matrix_size: int) -> float:
    """Return entropy effective rank after an explicitly bounded full SVD.

    The function refuses matrices whose largest dimension exceeds
    ``max_matrix_size``.  This makes the otherwise-quadratic/cubic diagnostic a
    deliberate small-matrix operation instead of a hidden large-graph cost.
    """

    if (
        isinstance(max_matrix_size, bool)
        or not isinstance(max_matrix_size, int)
        or max_matrix_size <= 0
    ):
        raise ValueError("max_matrix_size must be a positive integer")
    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2:
        raise ValueError("matrix must be a two-dimensional tensor")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("matrix dimensions must be positive")
    if max(matrix.shape) > max_matrix_size:
        raise ValueError(
            f"matrix shape {tuple(matrix.shape)} exceeds max_matrix_size={max_matrix_size}; "
            "effective rank requires a full SVD"
        )
    if matrix.is_complex():
        raise ValueError("matrix must be real-valued")

    detached = matrix.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(detached).all().item()):
        raise ValueError("matrix must contain only finite values")
    singular_values = torch.linalg.svdvals(detached)
    total = singular_values.sum()
    if float(total.item()) == 0.0:
        return 0.0
    probabilities = singular_values / total
    log_probabilities = torch.where(
        probabilities > 0,
        probabilities.log(),
        torch.zeros_like(probabilities),
    )
    return float((-(probabilities * log_probabilities).sum()).exp().item())


def _normalized_attention(weights: torch.Tensor) -> torch.Tensor:
    if not isinstance(weights, torch.Tensor) or weights.ndim != 2:
        raise ValueError("attention weights must be a two-dimensional tensor")
    if weights.shape[0] == 0 or weights.shape[1] == 0:
        raise ValueError("attention weight dimensions must be positive")
    if weights.is_complex():
        raise ValueError("attention weights must be real-valued")

    detached = weights.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(detached).all().item()):
        raise ValueError("attention weights must contain only finite values")
    if bool((detached < 0).any().item()):
        raise ValueError("attention weights must be nonnegative")
    row_mass = detached.sum(dim=-1, keepdim=True)
    if bool((row_mass <= 0).any().item()):
        raise ValueError("every attention row must have positive mass")
    return detached / row_mass


def _finite_vector(values: torch.Tensor | Real, *, name: str) -> torch.Tensor:
    if isinstance(values, torch.Tensor) and values.is_complex():
        raise ValueError(f"{name} must be real-valued")
    if not isinstance(values, (torch.Tensor, Real)):
        raise TypeError(f"{name} must be a tensor or real scalar")
    vector = torch.as_tensor(values).detach().to(dtype=torch.float64).reshape(-1)
    if vector.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not bool(torch.isfinite(vector).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _validated_quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    if not quantiles:
        raise ValueError("at least one quantile is required")
    levels: list[float] = []
    for quantile in quantiles:
        if isinstance(quantile, bool) or not isinstance(quantile, Real):
            raise TypeError("quantiles must be real numbers")
        level = float(quantile)
        if not isfinite(level) or not 0.0 <= level <= 1.0:
            raise ValueError("quantiles must be finite values in [0, 1]")
        levels.append(level)
    labels = [_quantile_label(level) for level in levels]
    if len(labels) != len(set(labels)):
        raise ValueError("quantile levels must have unique labels")
    return tuple(levels)


def _quantile_label(level: float) -> str:
    percentage = f"{level * 100.0:.6f}".rstrip("0").rstrip(".")
    whole, separator, fraction = percentage.partition(".")
    whole = whole.zfill(2) if len(whole) < 3 else whole
    return f"q{whole}{f'p{fraction}' if separator else ''}"


def _range_summary(name: str, values: torch.Tensor) -> dict[str, float]:
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    minimum = values.min()
    maximum = values.max()
    ratio = maximum / minimum
    if not bool(torch.isfinite(ratio).item()):
        raise ValueError(f"{name} dynamic range is not representable in float64")
    return {
        f"{name}.min": float(minimum.item()),
        f"{name}.median": float(values.median().item()),
        f"{name}.max": float(maximum.item()),
        f"{name}.max_over_min": float(ratio.item()),
    }


__all__ = [
    "attention_weight_summary",
    "dense_kernel_attention_summary",
    "kernel_component_quantiles",
    "kernel_parameter_summary",
    "matrix_effective_rank",
    "memory_assignment_summary",
]
