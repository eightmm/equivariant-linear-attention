from __future__ import annotations

from math import factorial, sqrt

import torch

INTEGER_DTYPES = frozenset(
    {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
)


def _symmetric_exponents(max_degree: int = 4) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (x_degree, y_degree, degree - x_degree - y_degree)
        for degree in range(max_degree + 1)
        for x_degree in range(degree, -1, -1)
        for y_degree in range(degree - x_degree, -1, -1)
    )


SYMMETRIC_EXPONENTS = _symmetric_exponents(4)
SYMMETRIC_DEGREES = tuple(sum(exponent) for exponent in SYMMETRIC_EXPONENTS)
SYMMETRIC_MULTINOMIAL_SQRT = tuple(
    sqrt(
        factorial(sum(exponent))
        / (factorial(exponent[0]) * factorial(exponent[1]) * factorial(exponent[2]))
    )
    for exponent in SYMMETRIC_EXPONENTS
)
SYMMETRIC_DEGREE_SLICES = (
    slice(0, 1),
    slice(1, 4),
    slice(4, 10),
    slice(10, 20),
    slice(20, 35),
)


def work_dtype(*values: torch.Tensor) -> torch.dtype:
    return (
        torch.float64
        if any(value.dtype == torch.float64 for value in values)
        else torch.float32
    )


def segment_sum(
    value: torch.Tensor, index: torch.Tensor, num_segments: int
) -> torch.Tensor:
    output = value.new_zeros((num_segments, *value.shape[1:]))
    if value.shape[0]:
        output.index_add_(0, index.to(dtype=torch.long), value)
    return output


def segment_count(
    index: torch.Tensor,
    num_segments: int,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    count = torch.bincount(index.to(dtype=torch.long), minlength=num_segments)
    return count if dtype is None else count.to(dtype=dtype)


def segment_mean(
    value: torch.Tensor, index: torch.Tensor, num_segments: int
) -> torch.Tensor:
    count = segment_count(index, num_segments, dtype=value.dtype).clamp_min(1.0)
    return segment_sum(value, index, num_segments) / count.reshape(
        num_segments,
        *((1,) * (value.ndim - 1)),
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


def symmetric_monomials(
    position: torch.Tensor,
    *,
    orthonormal: bool = False,
) -> torch.Tensor:
    """Complete symmetric Cartesian monomials through degree four.

    The returned order is grouped by degree with dimensions ``1, 3, 6, 10,
    15``.  With ``orthonormal=True`` each monomial is multiplied by the square
    root of its multinomial multiplicity, so the degree-k block has inner
    product ``(x dot y)^k`` exactly.
    """

    if position.ndim != 2 or position.shape[-1] != 3:
        raise ValueError("position must have shape (N,3)")
    x, y, z = position.unbind(dim=-1)
    one = torch.ones_like(x)
    x_power = (one, x, x.square(), x.square() * x, x.square().square())
    y_power = (one, y, y.square(), y.square() * y, y.square().square())
    z_power = (one, z, z.square(), z.square() * z, z.square().square())
    output = torch.stack(
        tuple(
            x_power[x_degree] * y_power[y_degree] * z_power[z_degree]
            for x_degree, y_degree, z_degree in SYMMETRIC_EXPONENTS
        ),
        dim=-1,
    )
    if not orthonormal:
        return output
    coefficient = output.new_tensor(SYMMETRIC_MULTINOMIAL_SQRT)
    return output * coefficient


def unit_ball(value: torch.Tensor, eps: float) -> torch.Tensor:
    return value / torch.sqrt(1.0 + value.square().sum(dim=-1, keepdim=True) + eps)


def bounded_scalar(value: torch.Tensor, eps: float) -> torch.Tensor:
    return value / torch.sqrt(1.0 + value.square() + eps)


def st_from_vector(value: torch.Tensor) -> torch.Tensor:
    x, y, z = value.unbind(dim=-1)
    trace = (x.square() + y.square() + z.square()) / 3.0
    return torch.stack(
        (x.square() - trace, y.square() - trace, x * y, x * z, y * z),
        dim=-1,
    )


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


def symmetric2_to_matrix(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] != 6:
        raise ValueError("value must end with six symmetric rank-two components")
    xx, xy, xz, yy, yz, zz = value.unbind(dim=-1)
    return torch.stack(
        (
            torch.stack((xx, xy, xz), dim=-1),
            torch.stack((xy, yy, yz), dim=-1),
            torch.stack((xz, yz, zz), dim=-1),
        ),
        dim=-2,
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
    lxx, lyy, lxy, lxz, lyz = left.unbind(dim=-1)
    rxx, ryy, rxy, rxz, ryz = right.unbind(dim=-1)
    lzz = -lxx - lyy
    rzz = -rxx - ryy
    return lxx * rxx + lyy * ryy + lzz * rzz + 2.0 * (lxy * rxy + lxz * rxz + lyz * ryz)


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


def st_matvec(tensor: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ab,...b->...a", st_to_matrix(tensor), vector)


def st_commutator_vector(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_matrix = st_to_matrix(left)
    right_matrix = st_to_matrix(right)
    commutator = left_matrix @ right_matrix - right_matrix @ left_matrix
    return torch.stack(
        (commutator[..., 2, 1], commutator[..., 0, 2], commutator[..., 1, 0]),
        dim=-1,
    )


def st_jordan_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_matrix = st_to_matrix(left)
    right_matrix = st_to_matrix(right)
    return matrix_to_st(0.5 * (left_matrix @ right_matrix + right_matrix @ left_matrix))


def vector_tensor_l1_l2(
    vector: torch.Tensor,
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor_matrix = st_to_matrix(tensor)
    l1 = torch.einsum("...ab,...b->...a", tensor_matrix, vector)
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
    l2 = matrix_to_st(cross @ tensor_matrix - tensor_matrix @ cross)
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


def compact_stf3(value: torch.Tensor) -> torch.Tensor:
    """Project ten symmetric degree-three monomials to seven STF components."""

    if value.shape[-1] != 10:
        raise ValueError("value must end with ten symmetric rank-three components")
    xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz = value.unbind(dim=-1)
    trace_x = xxx + xyy + xzz
    trace_y = xxy + yyy + yzz
    trace_z = xxz + yyz + zzz
    return torch.stack(
        (
            xxx - 3.0 * trace_x / 5.0,
            xxy - trace_y / 5.0,
            xxz - trace_z / 5.0,
            xyy - trace_x / 5.0,
            xyz,
            yyy - 3.0 * trace_y / 5.0,
            yyz - trace_z / 5.0,
        ),
        dim=-1,
    )


def compact_stf3_square(value: torch.Tensor) -> torch.Tensor:
    a, c, d, e, f, b, g = value.unbind(dim=-1)
    xzz = -a - e
    yzz = -c - b
    zzz = -d - g
    return (
        a.square()
        + b.square()
        + zzz.square()
        + 3.0
        * (
            c.square()
            + d.square()
            + e.square()
            + xzz.square()
            + g.square()
            + yzz.square()
        )
        + 6.0 * f.square()
    )


def bounded_compact_stf3(value: torch.Tensor, eps: float) -> torch.Tensor:
    return value / torch.sqrt(
        1.0 + compact_stf3_square(value).unsqueeze(-1) / 7.0 + eps
    )


def stf3_contract_st(value: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    """Contract compact STF3 with ST2 to a vector, ``T_abc Q_bc``."""

    a, c, d, e, f, b, g = value.unbind(dim=-1)
    xzz = -a - e
    yzz = -c - b
    zzz = -d - g
    qxx, qyy, qxy, qxz, qyz = tensor.unbind(dim=-1)
    qzz = -qxx - qyy
    return torch.stack(
        (
            a * qxx + e * qyy + xzz * qzz + 2.0 * (c * qxy + d * qxz + f * qyz),
            c * qxx + b * qyy + yzz * qzz + 2.0 * (e * qxy + f * qxz + g * qyz),
            d * qxx + g * qyy + zzz * qzz + 2.0 * (f * qxy + xzz * qxz + yzz * qyz),
        ),
        dim=-1,
    )


def stf3_contract_vector(value: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Contract compact STF3 with a vector to an ST2 tensor."""

    a, c, d, e, f, b, g = value.unbind(dim=-1)
    xzz = -a - e
    yzz = -c - b
    vx, vy, vz = vector.unbind(dim=-1)
    return torch.stack(
        (
            a * vx + c * vy + d * vz,
            e * vx + b * vy + g * vz,
            c * vx + e * vy + f * vz,
            d * vx + f * vy + xzz * vz,
            f * vx + g * vy + yzz * vz,
        ),
        dim=-1,
    )


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


def compact_stf4(value: torch.Tensor) -> torch.Tensor:
    """Project fifteen symmetric degree-four monomials to nine STF components."""

    if value.shape[-1] != 15:
        raise ValueError("value must end with fifteen symmetric rank-four components")
    (
        xxxx,
        xxxy,
        xxxz,
        xxyy,
        xxyz,
        xxzz,
        xyyy,
        xyyz,
        xyzz,
        xzzz,
        yyyy,
        yyyz,
        yyzz,
        yzzz,
        zzzz,
    ) = value.unbind(dim=-1)
    trace_xx = xxxx + xxyy + xxzz
    trace_xy = xxxy + xyyy + xyzz
    trace_xz = xxxz + xyyz + xzzz
    trace_yy = xxyy + yyyy + yyzz
    trace_yz = xxyz + yyyz + yzzz
    trace_zz = xxzz + yyzz + zzzz
    trace0 = trace_xx + trace_yy + trace_zz
    return torch.stack(
        (
            xxxx - 6.0 * trace_xx / 7.0 + 3.0 * trace0 / 35.0,
            xxxy - 3.0 * trace_xy / 7.0,
            xxxz - 3.0 * trace_xz / 7.0,
            xxyy - (trace_xx + trace_yy) / 7.0 + trace0 / 35.0,
            xxyz - trace_yz / 7.0,
            xyyy - 3.0 * trace_xy / 7.0,
            xyyz - trace_xz / 7.0,
            yyyy - 6.0 * trace_yy / 7.0 + 3.0 * trace0 / 35.0,
            yyyz - 3.0 * trace_yz / 7.0,
        ),
        dim=-1,
    )


def _compact_stf4_components(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
    a, b, c, d, e, f, g, h, i = value.unbind(dim=-1)
    xxzz = -a - d
    xyzz = -b - f
    xzzz = -c - g
    yyzz = -d - h
    yzzz = -e - i
    zzzz = a + 2.0 * d + h
    return a, b, c, d, e, xxzz, f, g, xyzz, xzzz, h, i, yyzz, yzzz, zzzz


def compact_stf4_square(value: torch.Tensor) -> torch.Tensor:
    (
        a,
        b,
        c,
        d,
        e,
        xxzz,
        f,
        g,
        xyzz,
        xzzz,
        h,
        i,
        yyzz,
        yzzz,
        zzzz,
    ) = _compact_stf4_components(value)
    return (
        a.square()
        + 4.0 * b.square()
        + 4.0 * c.square()
        + 6.0 * d.square()
        + 12.0 * e.square()
        + 6.0 * xxzz.square()
        + 4.0 * f.square()
        + 12.0 * g.square()
        + 12.0 * xyzz.square()
        + 4.0 * xzzz.square()
        + h.square()
        + 4.0 * i.square()
        + 6.0 * yyzz.square()
        + 4.0 * yzzz.square()
        + zzzz.square()
    )


def bounded_compact_stf4(value: torch.Tensor, eps: float) -> torch.Tensor:
    return value / torch.sqrt(
        1.0 + compact_stf4_square(value).unsqueeze(-1) / 9.0 + eps
    )


def stf4_contract_st(value: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    """Contract compact STF4 with ST2 to ST2, ``T_abcd Q_cd``."""

    (
        a,
        b,
        c,
        d,
        e,
        xxzz,
        f,
        g,
        xyzz,
        xzzz,
        h,
        i,
        yyzz,
        yzzz,
        zzzz,
    ) = _compact_stf4_components(value)
    qxx, qyy, qxy, qxz, qyz = tensor.unbind(dim=-1)
    qzz = -qxx - qyy
    del zzzz  # the output trace is fixed by STF and is not stored explicitly.
    return torch.stack(
        (
            a * qxx + d * qyy + xxzz * qzz + 2.0 * (b * qxy + c * qxz + e * qyz),
            d * qxx + h * qyy + yyzz * qzz + 2.0 * (f * qxy + g * qxz + i * qyz),
            b * qxx + f * qyy + xyzz * qzz + 2.0 * (d * qxy + e * qxz + g * qyz),
            c * qxx + g * qyy + xzzz * qzz + 2.0 * (e * qxy + xxzz * qxz + xyzz * qyz),
            e * qxx + i * qyy + yzzz * qzz + 2.0 * (g * qxy + xyzz * qxz + yyzz * qyz),
        ),
        dim=-1,
    )


def stf4_contract_st_vector(
    value: torch.Tensor,
    tensor: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    return st_matvec(stf4_contract_st(value, tensor), vector)


def positive_feature(value: torch.Tensor) -> torch.Tensor:
    return (torch.nn.functional.elu(value) + 1.0) / sqrt(max(1, value.shape[-1]))


__all__ = [
    "INTEGER_DTYPES",
    "SYMMETRIC_DEGREES",
    "SYMMETRIC_DEGREE_SLICES",
    "SYMMETRIC_EXPONENTS",
    "SYMMETRIC_MULTINOMIAL_SQRT",
    "bounded_compact_stf3",
    "bounded_compact_stf4",
    "bounded_scalar",
    "bounded_st",
    "bounded_stf3",
    "bounded_stf4",
    "canonical_batch",
    "centered_geometry",
    "compact_stf3",
    "compact_stf3_square",
    "compact_stf4",
    "compact_stf4_square",
    "interaction_index",
    "matrix_to_st",
    "normalize_st",
    "positive_feature",
    "segment_count",
    "segment_mean",
    "segment_sum",
    "st_commutator_vector",
    "st_cross",
    "st_from_vector",
    "st_inner",
    "st_jordan_product",
    "st_matvec",
    "st_orthonormal",
    "st_square",
    "st_to_matrix",
    "stf3",
    "stf3_contract_st",
    "stf3_contract_vector",
    "stf4",
    "stf4_contract_st",
    "stf4_contract_st_vector",
    "symmetric2_to_matrix",
    "symmetric_monomials",
    "unit_ball",
    "vector_tensor_l1_l2",
    "work_dtype",
]
