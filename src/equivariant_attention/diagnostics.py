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


def local_attention_summary(
    receiver: torch.Tensor,
    sender: torch.Tensor,
    weights: torch.Tensor,
    squared_distance: torch.Tensor,
    *,
    num_nodes: int,
) -> dict[str, float | int | str]:
    """Summarize sparse local attention within receiver/head domains.

    ``squared_distance`` is the model's squared cutoff-normalized distance,
    so its square root is the physical distance divided by the cutoff.
    """
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int):
        raise TypeError("num_nodes must be an integer")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if not all(isinstance(index, torch.Tensor) for index in (receiver, sender)):
        raise TypeError("receiver and sender must be tensors")
    if receiver.ndim != 1 or sender.shape != receiver.shape:
        raise ValueError("receiver and sender must be one-dimensional with equal shape")
    if weights.ndim != 2 or weights.shape[0] != receiver.numel():
        raise ValueError("weights must have shape (edges, heads)")
    if squared_distance.shape != receiver.shape:
        raise ValueError("squared_distance must have shape (edges,)")
    if receiver.numel() == 0 or weights.shape[1] == 0:
        raise ValueError("local diagnostics require at least one edge and head")
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if receiver.dtype not in integer_dtypes or sender.dtype not in integer_dtypes:
        raise TypeError("receiver and sender must use integer dtypes")
    receiver = receiver.detach().to(dtype=torch.long)
    sender = sender.detach().to(device=receiver.device, dtype=torch.long)
    if bool((receiver < 0).any().item()) or bool((sender < 0).any().item()):
        raise ValueError("edge indices must be nonnegative")
    if int(receiver.max().item()) >= num_nodes or int(sender.max().item()) >= num_nodes:
        raise ValueError("edge indices must be smaller than num_nodes")

    probabilities = weights.detach().to(device=receiver.device, dtype=torch.float64)
    distance_square = squared_distance.detach().to(
        device=receiver.device, dtype=torch.float64
    )
    if not bool(torch.isfinite(probabilities).all().item()) or bool(
        (probabilities < 0.0).any().item()
    ):
        raise ValueError("weights must be finite and nonnegative")
    if not bool(torch.isfinite(distance_square).all().item()) or bool(
        (distance_square < 0.0).any().item()
    ):
        raise ValueError("squared_distance must be finite and nonnegative")

    head_count = probabilities.shape[1]
    row_mass = probabilities.new_zeros((num_nodes, head_count)).index_add(
        0, receiver, probabilities
    )
    if bool((row_mass <= 0.0).any().item()):
        raise ValueError("every receiver/head domain must have positive mass")
    row_mass_error = (row_mass - 1.0).abs().max()
    probabilities = probabilities / row_mass[receiver]
    degree = torch.bincount(receiver, minlength=num_nodes)
    if bool((degree <= 0).any().item()):
        raise ValueError("every receiver must have at least one local edge")

    log_probability = torch.where(
        probabilities > 0.0,
        probabilities.log(),
        torch.zeros_like(probabilities),
    )
    entropy = probabilities.new_zeros((num_nodes, head_count)).index_add(
        0, receiver, -(probabilities * log_probability)
    )
    degree_log = degree.to(dtype=torch.float64).log().unsqueeze(-1)
    normalized_entropy = torch.where(
        degree_log > 0.0,
        entropy / degree_log,
        torch.ones_like(entropy),
    )
    maximum = probabilities.new_zeros((num_nodes, head_count))
    maximum = maximum.scatter_reduce(
        0,
        receiver.unsqueeze(-1).expand_as(probabilities),
        probabilities,
        reduce="amax",
        include_self=True,
    )
    effective_support = entropy.exp()
    distance_over_cutoff = distance_square.sqrt()
    distance_quantiles = torch.quantile(
        distance_over_cutoff,
        distance_over_cutoff.new_tensor([0.0, 0.25, 0.5, 0.75, 0.95, 1.0]),
    )
    degree_float = degree.to(dtype=torch.float64)
    return {
        "scope": "receiver_by_head",
        "num_nodes": num_nodes,
        "head_count": int(head_count),
        "edge_count": int(receiver.numel()),
        "degree.min": int(degree.min().item()),
        "degree.mean": float(degree_float.mean().item()),
        "degree.median": float(degree_float.median().item()),
        "degree.max": int(degree.max().item()),
        "self_edge_fraction": float((receiver == sender).double().mean().item()),
        "attention.row_mass_max_abs_error": float(row_mass_error.item()),
        "attention.entropy_over_log_degree.min": float(
            normalized_entropy.min().item()
        ),
        "attention.entropy_over_log_degree.mean": float(
            normalized_entropy.mean().item()
        ),
        "attention.entropy_over_log_degree.max": float(
            normalized_entropy.max().item()
        ),
        "attention.max_weight.mean": float(maximum.mean().item()),
        "attention.max_weight.max": float(maximum.max().item()),
        "attention.effective_support.min": float(effective_support.min().item()),
        "attention.effective_support.mean": float(effective_support.mean().item()),
        "attention.effective_support.max": float(effective_support.max().item()),
        "distance_over_cutoff.q00": float(distance_quantiles[0].item()),
        "distance_over_cutoff.q25": float(distance_quantiles[1].item()),
        "distance_over_cutoff.q50": float(distance_quantiles[2].item()),
        "distance_over_cutoff.q75": float(distance_quantiles[3].item()),
        "distance_over_cutoff.q95": float(distance_quantiles[4].item()),
        "distance_over_cutoff.q100": float(distance_quantiles[5].item()),
    }


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
    """Summarize occupancy and marginal/conditional assignment information."""

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
    if memory_count == 1:
        conditional_entropy = 1.0
        marginal_entropy = 1.0
        mutual_information = 0.0
    else:
        log_memory_count = torch.tensor(
            float(memory_count),
            dtype=torch.float64,
            device=probabilities.device,
        ).log()
        conditional_entropy = float((entropy.mean() / log_memory_count).item())
        marginal = probabilities.mean(dim=0)
        marginal_log = torch.where(
            marginal > 0.0,
            marginal.log(),
            torch.zeros_like(marginal),
        )
        marginal_entropy = float(
            (-(marginal * marginal_log).sum(dim=-1).mean() / log_memory_count).item()
        )
        mutual_information = max(0.0, marginal_entropy - conditional_entropy)
    return {
        "memory_count": int(memory_count),
        "occupancy.min": float(occupancy.min().item()),
        "occupancy.mean": float(occupancy.mean().item()),
        "occupancy.max": float(occupancy.max().item()),
        "assignment_entropy_over_log_m": conditional_entropy,
        "conditional_entropy_over_log_m": conditional_entropy,
        "marginal_entropy_over_log_m": marginal_entropy,
        "mutual_information_over_log_m": mutual_information,
    }


def memory_center_summary(
    centers: torch.Tensor,
    *,
    interaction_cutoff: float,
) -> dict[str, object]:
    """Report invariant per-head memory-center spread and distance scales."""

    if not isinstance(centers, torch.Tensor) or centers.ndim != 3:
        raise ValueError("centers must have shape (heads, memories, 3)")
    if centers.shape[-1] != 3 or 0 in centers.shape:
        raise ValueError("centers must have positive heads/memories and dimension 3")
    if centers.is_complex():
        raise ValueError("centers must be real-valued")
    if isinstance(interaction_cutoff, bool) or not isinstance(interaction_cutoff, Real):
        raise TypeError("interaction_cutoff must be a real number")
    cutoff = float(interaction_cutoff)
    if not isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("interaction_cutoff must be finite and positive")

    values = centers.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("centers must contain only finite values")
    heads, memories, _ = values.shape
    head_summaries: list[dict[str, float | int]] = []
    for head in range(heads):
        head_centers = values[head]
        centered = head_centers - head_centers.mean(dim=0, keepdim=True)
        spread = centered.square().sum(dim=-1).mean().sqrt()
        distance = torch.linalg.vector_norm(
            head_centers.unsqueeze(-2) - head_centers.unsqueeze(-3),
            dim=-1,
        )
        if memories == 1:
            minimum = median = maximum = 0.0
        else:
            off_diagonal = distance[
                ~torch.eye(memories, dtype=torch.bool, device=distance.device)
            ]
            minimum = float(off_diagonal.min().item())
            median = float(off_diagonal.median().item())
            maximum = float(off_diagonal.max().item())
        head_summaries.append(
            {
                "head_index": int(head),
                "center_spread_rms": float(spread.item()),
                "offdiagonal_distance.min": minimum,
                "offdiagonal_distance.median": median,
                "offdiagonal_distance.max": maximum,
                "distance_over_cutoff.q00": minimum / cutoff,
                "distance_over_cutoff.q50": median / cutoff,
                "distance_over_cutoff.q100": maximum / cutoff,
            }
        )
    return {
        "scope": "single_graph_per_head",
        "head_count": int(heads),
        "memory_count": int(memories),
        "heads": head_summaries,
    }


def pair_gate_summary(
    pair_gate: torch.Tensor,
    *,
    nonconstant_relative_tolerance: float = 1e-3,
    symmetry_tolerance: float = 1e-6,
) -> dict[str, float]:
    """Summarize one graph/head HEMM pair gate without pooling domains.

    The current shared read/write assignment and symmetric coupling imply a
    square, symmetric gate in ``[0, 1]``. Scale-free variation is computed
    after division by the maximum gate value, which prevents underflow for
    tiny but finite gates without adding an ``eps`` that would change the
    statistic. Quantiles use linear interpolation.
    """

    if not isinstance(pair_gate, torch.Tensor) or pair_gate.ndim != 2:
        raise ValueError("pair_gate must be a two-dimensional tensor")
    if 0 in pair_gate.shape:
        raise ValueError("pair_gate dimensions must be positive")
    if pair_gate.shape[0] != pair_gate.shape[1]:
        raise ValueError("pair_gate must be square within one graph/head domain")
    if pair_gate.is_complex():
        raise ValueError("pair_gate must be real-valued")
    tolerance = _open_unit_interval(
        "nonconstant_relative_tolerance",
        nonconstant_relative_tolerance,
    )
    symmetry_limit = _open_unit_interval(
        "symmetry_tolerance",
        symmetry_tolerance,
    )

    values = pair_gate.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("pair_gate must contain only finite values")
    if bool((values < 0.0).any().item()):
        raise ValueError("pair_gate must be nonnegative")
    maximum = values.max()
    if float(maximum.item()) <= 0.0:
        raise ValueError("pair_gate must have positive mass")
    if float(maximum.item()) > 1.0 + symmetry_limit:
        raise ValueError("pair_gate values must be at most one")
    values = values.clamp_max(1.0)
    maximum = values.max()
    scaled = values / maximum
    symmetry_error = (scaled - scaled.T).abs().max()
    if float(symmetry_error.item()) > symmetry_limit:
        raise ValueError("pair_gate must be symmetric")

    mean = scaled.mean()
    centered = scaled - mean
    cv = centered.square().mean().sqrt() / mean
    centered_frobenius_ratio = torch.linalg.vector_norm(
        centered
    ) / torch.linalg.vector_norm(scaled)
    nonconstant_fraction = (
        (centered.abs() > tolerance * mean).to(dtype=torch.float64).mean()
    )
    quantiles = torch.quantile(
        values.reshape(-1),
        values.new_tensor([0.0, 0.01, 0.5, 0.99, 1.0]),
        interpolation="linear",
    )
    return {
        "min": float(quantiles[0].item()),
        "p01": float(quantiles[1].item()),
        "median": float(quantiles[2].item()),
        "p99": float(quantiles[3].item()),
        "max": float(quantiles[4].item()),
        "mean": float(values.mean().item()),
        "cv": float(cv.item()),
        "centered_frobenius_ratio": float(centered_frobenius_ratio.item()),
        "nonconstant_fraction": float(nonconstant_fraction.item()),
        "nonconstant_relative_tolerance": tolerance,
        "symmetry_relative_max_error": float(symmetry_error.item()),
    }


def memory_pair_gate_summary(
    assignment: torch.Tensor,
    coupling: torch.Tensor,
    *,
    nonconstant_relative_tolerance: float = 1e-3,
) -> dict[str, object]:
    """Report per-head gates for one graph and conservative all-head minima.

    Graphs must be selected by the caller before this function is called. This
    prevents between-graph constants or cross-graph zeros from creating a false
    positive coefficient of variation.
    """

    if not isinstance(assignment, torch.Tensor) or assignment.ndim != 3:
        raise ValueError("assignment must have shape (nodes, heads, memories)")
    if not isinstance(coupling, torch.Tensor) or coupling.ndim != 3:
        raise ValueError("coupling must have shape (heads, memories, memories)")
    if 0 in assignment.shape or 0 in coupling.shape:
        raise ValueError("assignment and coupling dimensions must be positive")
    nodes, heads, memories = assignment.shape
    if coupling.shape != (heads, memories, memories):
        raise ValueError("coupling shape must match assignment heads and memories")
    if assignment.is_complex() or coupling.is_complex():
        raise ValueError("assignment and coupling must be real-valued")
    probabilities = assignment.detach().to(dtype=torch.float64)
    interactions = coupling.detach().to(
        dtype=torch.float64, device=probabilities.device
    )
    if not bool(torch.isfinite(probabilities).all().item()) or not bool(
        torch.isfinite(interactions).all().item()
    ):
        raise ValueError("assignment and coupling must contain only finite values")
    if bool((probabilities < 0.0).any().item()) or bool(
        (interactions < 0.0).any().item()
    ):
        raise ValueError("assignment and coupling must be nonnegative")
    if bool((interactions > 1.0 + 1e-6).any().item()):
        raise ValueError("coupling values must be at most one")
    if not torch.allclose(
        interactions,
        interactions.transpose(-1, -2),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("coupling must be symmetric")
    row_mass = probabilities.sum(dim=-1)
    if not torch.allclose(row_mass, torch.ones_like(row_mass), atol=1e-6, rtol=1e-6):
        raise ValueError("assignment rows must lie on the probability simplex")

    head_summaries: list[dict[str, object]] = []
    for head in range(heads):
        head_assignment = probabilities[:, head]
        head_coupling = interactions[head]
        gate = torch.einsum(
            "im,mn,jn->ij",
            head_assignment,
            head_coupling,
            head_assignment,
        )
        assignment_summary = memory_assignment_summary(
            probabilities[:, head : head + 1]
        )
        assignment_summary["occupancy_fraction.min"] = (
            float(assignment_summary["occupancy.min"]) / nodes
        )
        assignment_summary["occupancy_fraction.mean"] = (
            float(assignment_summary["occupancy.mean"]) / nodes
        )
        assignment_summary["occupancy_fraction.max"] = (
            float(assignment_summary["occupancy.max"]) / nodes
        )
        coupling_summary = kernel_component_quantiles(
            {"coupling": head_coupling},
            quantiles=(0.0, 0.5, 1.0),
        )
        if memories == 1:
            off_diagonal_nonunit_fraction = 0.0
        else:
            off_diagonal = head_coupling[
                ~torch.eye(
                    memories,
                    dtype=torch.bool,
                    device=head_coupling.device,
                )
            ]
            off_diagonal_nonunit_fraction = float(
                ((off_diagonal - 1.0).abs() > nonconstant_relative_tolerance)
                .to(dtype=torch.float64)
                .mean()
                .item()
            )
        coupling_summary["off_diagonal_nonunit_fraction"] = (
            off_diagonal_nonunit_fraction
        )
        coupling_summary["centered_frobenius_ratio"] = pair_gate_summary(
            head_coupling,
            nonconstant_relative_tolerance=nonconstant_relative_tolerance,
        )["centered_frobenius_ratio"]
        head_summaries.append(
            {
                "head_index": int(head),
                "assignment": assignment_summary,
                "coupling": coupling_summary,
                "pair_gate": pair_gate_summary(
                    gate,
                    nonconstant_relative_tolerance=nonconstant_relative_tolerance,
                ),
            }
        )

    return {
        "scope": "single_graph_per_head",
        "node_count": int(nodes),
        "head_count": int(heads),
        "memory_count": int(memories),
        "heads": head_summaries,
        "worst_case": {
            "assignment_entropy.min": min(
                float(head["assignment"]["assignment_entropy_over_log_m"])
                for head in head_summaries
            ),
            "assignment_entropy.max": max(
                float(head["assignment"]["assignment_entropy_over_log_m"])
                for head in head_summaries
            ),
            "occupancy_fraction.min": min(
                float(head["assignment"]["occupancy_fraction.min"])
                for head in head_summaries
            ),
            "coupling.q00.max": max(
                float(head["coupling"]["coupling.q00"]) for head in head_summaries
            ),
            "coupling.off_diagonal_nonunit_fraction.min": min(
                float(head["coupling"]["off_diagonal_nonunit_fraction"])
                for head in head_summaries
            ),
            "pair_gate.cv": min(
                float(head["pair_gate"]["cv"]) for head in head_summaries
            ),
            "pair_gate.centered_frobenius_ratio": min(
                float(head["pair_gate"]["centered_frobenius_ratio"])
                for head in head_summaries
            ),
            "pair_gate.nonconstant_fraction": min(
                float(head["pair_gate"]["nonconstant_fraction"])
                for head in head_summaries
            ),
        },
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


def _open_unit_interval(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not isfinite(numeric) or not 0.0 < numeric < 1.0:
        raise ValueError(f"{name} must be finite and lie strictly between zero and one")
    return numeric


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
    "local_attention_summary",
    "matrix_effective_rank",
    "memory_assignment_summary",
    "memory_center_summary",
    "memory_pair_gate_summary",
    "pair_gate_summary",
]
