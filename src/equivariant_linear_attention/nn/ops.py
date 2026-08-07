from __future__ import annotations

from math import sqrt

import torch

INTEGER_DTYPES = frozenset({torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8})


def work_dtype(*values: torch.Tensor) -> torch.dtype:
    return torch.float64 if any(v.dtype == torch.float64 for v in values) else torch.float32


def segment_sum(value: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    output = value.new_zeros((num_segments, *value.shape[1:]))
    if value.shape[0]:
        output.index_add_(0, index.to(dtype=torch.long), value)
    return output


def segment_count(index: torch.Tensor, num_segments: int, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    count = torch.bincount(index.to(dtype=torch.long), minlength=num_segments)
    return count if dtype is None else count.to(dtype=dtype)


def segment_mean(value: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    count = segment_count(index, num_segments, dtype=value.dtype).clamp_min(1.0)
    return segment_sum(value, index, num_segments) / count.reshape(
        num_segments, *((1,) * (value.ndim - 1))
    )


def canonical_batch(
    batch: torch.Tensor | None,
    *,
    num_nodes: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    if batch is None:
        index = torch.zeros(num_nodes, device=device, dtype=torch.long)
    else:
        if not isinstance(batch, torch.Tensor):
            raise TypeError("batch must be a tensor")
        if batch.shape != (num_nodes,):
            raise ValueError("batch must have shape (N,)")
        if batch.device != device:
            raise ValueError("batch and nodes must share one device")
        if batch.dtype not in INTEGER_DTYPES:
            raise TypeError("batch must use an integer dtype")
        index = batch.to(dtype=torch.long)
        if index.numel() and bool((index < 0).any().item()):
            raise ValueError("batch values must be nonnegative")
    if num_nodes == 0:
        return index, 0, torch.zeros(0, device=device, dtype=torch.long)
    num_graphs = int(index.max().item()) + 1
    counts = torch.bincount(index, minlength=num_graphs)
    if bool((counts == 0).any().item()):
        raise ValueError("batch IDs must be contiguous from zero")
    return index, num_graphs, counts


def interaction_index(
    batch: torch.Tensor,
    group: torch.Tensor | None,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    if group is None:
        num = 0 if batch.numel() == 0 else int(batch.max().item()) + 1
        return batch, num, torch.bincount(batch, minlength=num)
    if group.shape != batch.shape or group.device != batch.device:
        raise ValueError("group must have shape (N,) on the batch device")
    if group.dtype not in INTEGER_DTYPES:
        raise TypeError("group must use an integer dtype")
    group = group.to(dtype=torch.long)
    if group.numel() and bool((group < 0).any().item()):
        raise ValueError("group values must be nonnegative")
    pair = torch.stack((batch, group), dim=-1)
    _, inverse = torch.unique(pair, dim=0, sorted=True, return_inverse=True)
    inverse = inverse.to(dtype=torch.long)
    num = 0 if inverse.numel() == 0 else int(inverse.max().item()) + 1
    return inverse, num, torch.bincount(inverse, minlength=num)


def centered_geometry(
    positions: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    center = segment_mean(positions, index, num_segments)
    centered = positions - center[index]
    radius_square = segment_mean(centered.square().sum(dim=-1), index, num_segments)
    radius = torch.sqrt(radius_square + eps)
    normalized = centered / radius[index, None]
    return centered, radius, normalized


def unit_ball(value: torch.Tensor, eps: float) -> torch.Tensor:
    return value / torch.sqrt(1.0 + value.square().sum(dim=-1, keepdim=True) + eps)


def bounded_scalar(value: torch.Tensor, eps: float) -> torch.Tensor:
    return value / torch.sqrt(1.0 + value.square() + eps)


def st_from_vector(value: torch.Tensor) -> torch.Tensor:
    x, y, z = value.unbind(dim=-1)
    trace = (x.square() + y.square() + z.square()) / 3.0
    return torch.stack((x.square() - trace, y.square() - trace, x * y, x * z, y * z), dim=-1)


def st_cross(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lx, ly, lz = left.unbind(dim=-1)
    rx, ry, rz = right.unbind(dim=-1)
    trace = (lx * rx + ly * ry + lz * rz) / 3.0
    return torch.stack(
        (
            lx * rx - trace,
            ly * ry - trace,
            0.5 * (lx * ry + ly * rx),
            0.5 * (lx * rz + lz * rx),
            0.5 * (ly * rz + lz * ry),
        ),
        dim=-1,
    )


def st_to_matrix(value: torch.Tensor) -> torch.Tensor:
    xx, yy, xy, xz, yz = value.unbind(dim=-1)
    zz = -xx - yy
    return torch.stack(
        (
            torch.stack((xx, xy, xz), dim=-1),
            torch.stack((xy, yy, yz), dim=-1),
            torch.stack((xz, yz, zz), dim=-1),
        ),
        dim=-2,
    )


def matrix_to_st(value: torch.Tensor) -> torch.Tensor:
    symmetric = 0.5 * (value + value.transpose(-1, -2))
    trace = symmetric.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
    identity = torch.eye(3, device=value.device, dtype=value.dtype)
    projected = symmetric - trace[..., None, None] * identity
    return torch.stack(
        (
            projected[..., 0, 0],
            projected[..., 1, 1],
            projected[..., 0, 1],
            projected[..., 0, 2],
            projected[..., 1, 2],
        ),
        dim=-1,
    )


def st_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lm = st_to_matrix(left)
    rm = st_to_matrix(right)
    return (lm * rm).sum(dim=(-2, -1))


def st_square(value: torch.Tensor) -> torch.Tensor:
    return st_inner(value, value)


def bounded_st(value: torch.Tensor, eps: float) -> torch.Tensor:
    return value / torch.sqrt(1.0 + st_square(value).unsqueeze(-1) / 5.0 + eps)


def normalize_st(value: torch.Tensor, eps: float) -> torch.Tensor:
    return value / torch.sqrt(st_square(value).unsqueeze(-1) + eps)


def st_orthonormal(value: torch.Tensor) -> torch.Tensor:
    xx, yy, xy, xz, yz = value.unbind(dim=-1)
    zz = -xx - yy
    return torch.stack(
        (
            (xx - yy) / sqrt(2.0),
            (xx + yy - 2.0 * zz) / sqrt(6.0),
            sqrt(2.0) * xy,
            sqrt(2.0) * xz,
            sqrt(2.0) * yz,
        ),
        dim=-1,
    )


def st_commutator_vector(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    commutator = st_to_matrix(left) @ st_to_matrix(right) - st_to_matrix(right) @ st_to_matrix(left)
    return torch.stack(
        (commutator[..., 2, 1], commutator[..., 0, 2], commutator[..., 1, 0]),
        dim=-1,
    )


def st_jordan_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lm = st_to_matrix(left)
    rm = st_to_matrix(right)
    return matrix_to_st(0.5 * (lm @ rm + rm @ lm))


def vector_tensor_l1_l2(vector: torch.Tensor, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tm = st_to_matrix(tensor)
    l1 = torch.einsum("...ab,...b->...a", tm, vector)
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    cross = torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )
    l2 = matrix_to_st(cross @ tm - tm @ cross)
    return l1, l2


def stf3(value: torch.Tensor) -> torch.Tensor:
    trace = torch.einsum("...aac->...c", value)
    identity = torch.eye(3, device=value.device, dtype=value.dtype)
    correction = (
        torch.einsum("ab,...c->...abc", identity, trace)
        + torch.einsum("ac,...b->...abc", identity, trace)
        + torch.einsum("bc,...a->...abc", identity, trace)
    ) / 5.0
    return value - correction


def bounded_stf3(value: torch.Tensor, eps: float) -> torch.Tensor:
    square = value.square().sum(dim=(-3, -2, -1), keepdim=True) / 7.0
    return value / torch.sqrt(1.0 + square + eps)


def stf4(value: torch.Tensor) -> torch.Tensor:
    trace2 = torch.einsum("...aakl->...kl", value)
    trace0 = torch.einsum("...aabb->...", value)
    identity = torch.eye(3, device=value.device, dtype=value.dtype)
    single = (
        torch.einsum("ij,...kl->...ijkl", identity, trace2)
        + torch.einsum("ik,...jl->...ijkl", identity, trace2)
        + torch.einsum("il,...jk->...ijkl", identity, trace2)
        + torch.einsum("jk,...il->...ijkl", identity, trace2)
        + torch.einsum("jl,...ik->...ijkl", identity, trace2)
        + torch.einsum("kl,...ij->...ijkl", identity, trace2)
    )
    double = trace0[..., None, None, None, None] * (
        torch.einsum("ij,kl->ijkl", identity, identity)
        + torch.einsum("ik,jl->ijkl", identity, identity)
        + torch.einsum("il,jk->ijkl", identity, identity)
    )
    return value - single / 7.0 + double / 35.0


def bounded_stf4(value: torch.Tensor, eps: float) -> torch.Tensor:
    square = value.square().sum(dim=(-4, -3, -2, -1), keepdim=True) / 9.0
    return value / torch.sqrt(1.0 + square + eps)


def positive_feature(value: torch.Tensor) -> torch.Tensor:
    return (torch.nn.functional.elu(value) + 1.0) / sqrt(max(1, value.shape[-1]))
