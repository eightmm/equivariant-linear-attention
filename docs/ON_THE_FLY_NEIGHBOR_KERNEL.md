# Exact on-the-fly neighbor kernel: implementation contract

This is a handoff specification for a future fused backend. It is not an
implemented speed claim.

## 1. Goal

Expose an exact local operator that accepts positions, graph membership, and
node state but no user-provided `edge_index`:

```python
output = local_kernel(
    state,
    positions,
    batch,
    cell,
    pbc,
)
```

The kernel discovers candidate neighbors internally, evaluates the existing
single positive sparse-local equation, reduces messages by receiver, and does
not retain a materialized edge list after the call.

This is different from `ImplicitGaussianSpatialKernel`:

- on-the-fly backend: exact cutoff semantics, internal spatial indexing;
- implicit kernel: no neighbor discovery, approximate smooth low-rank kernel.

## 2. Mathematical contract

For the existing local cutoff and invariant score,

\[
w_{ijr}
=
f_c(\widetilde u_{ij})
\exp\left[3\tanh(a_{ijr}/3)\right],
\]

\[
m_{ir}=\sum_{j\in\mathcal N(i)}w_{ijr},
\]

\[
S_{ir}^{f}
=
\frac{
\sum_{j\in\mathcal N(i)}w_{ijr}\rho_{ijr}^{f}z_{jr}^{f}
}{1+m_{ir}}.
\]

The fused backend must reproduce the explicit receiver-CSR implementation within
the declared dtype tolerances. It may change arithmetic scheduling but not:

- cutoff definition;
- self-edge policy;
- relation-specific cutoff narrowing;
- radial basis;
- parity routing;
- receiver normalization;
- output ordering.

## 3. Spatial indexing

Recommended nonperiodic implementation:

1. quantize positions to cells of width at least the maximum cutoff;
2. generate a sortable cell key including graph ID;
3. radix-sort or bucket nodes by key;
4. for each receiver, visit its own and adjacent cells;
5. apply the exact distance/cutoff test inside the kernel;
6. accumulate all message families in FP32 registers or shared memory;
7. write only receiver outputs and optional diagnostics.

Periodic implementation additionally requires:

- `cell: [G,3,3]` or one shared cell;
- `pbc: [G,3]`;
- minimum-image or explicit cell-shift handling;
- differentiation through displacement, not through discrete cell assignment.

For moving coordinates, use a Verlet skin and rebuild condition

\[
2\max_i\lVert x_i-x_i^{\rm build}\rVert>r_{\rm skin}.
\]

## 4. Complexity

Under bounded density and fixed cutoff, expected construction plus traversal is

\[
O(N+E).
\]

Worst-case complexity remains quadratic when all points occupy one cell or the
cutoff spans the full graph. Public claims must therefore say “expected linear
under bounded density”, not unconditional linear time.

No stored edge tensor is required across layers, but temporary cell keys,
sorted indices, offsets, and optional Verlet metadata remain:

\[
O(N+G_{\rm cells})
\]

plus receiver output storage.

## 5. Forward kernel stages

A practical first implementation may separate:

1. cell-key generation and sort;
2. cell-offset construction;
3. receiver traversal and message reduction.

A later fused path may combine stages 2 and 3. The message kernel should avoid
materializing

```text
[E, R]
[E, R, D]
[E, R, 3]
[E, R, 5]
```

edge activations.

Each receiver program should accumulate:

- positive mass and squared mass;
- scalar values;
- polar and axial vector values;
- even and odd ST5 tensor values;
- three direction moments for chirality;
- any requested diagnostic counts.

## 6. Backward contract

Coordinate and feature gradients must match the explicit reference. The first
implementation may recompute neighbors in backward from the saved sorted cell
metadata. Discrete cell membership is treated as an indexing decision; gradients
flow through the selected pair displacements and continuous kernel values.

Requirements:

- feature gradient parity/equivariance;
- finite coordinate gradients;
- double-backward fallback or an explicit “first-order only” capability receipt;
- deterministic mode where the explicit reference is deterministic;
- no silent graph cross-talk.

## 7. Required equivalence tests

For random and adversarial fixtures, compare with the explicit receiver-CSR
operator:

1. FP64 forward values;
2. input-feature gradients;
3. coordinate gradients;
4. parameter gradients;
5. empty graph and singleton graph;
6. zero-neighbor receivers;
7. points exactly near cell and cutoff boundaries;
8. self-edge exclusion;
9. relation cutoff narrowing;
10. batched graphs with overlapping coordinates;
11. periodic cells and cell shifts;
12. coordinate updates with skin/no-rebuild and forced rebuild.

Recommended initial tolerances:

```text
FP64 forward:         atol <= 1e-9, rtol <= 1e-9
FP64 first gradients: atol <= 1e-8, rtol <= 1e-8
FP32 forward:         atol <= 2e-5, rtol <= 2e-5
BF16 smoke:           finite + task-specific tolerance
```

## 8. Required performance tests

Measure explicit CSR, on-the-fly, and implicit-kernel modes over:

```text
N:       128, 512, 2k, 8k, 32k, 128k
degree:  8, 16, 32, 64, 128
batch:   one large / uniform small / ragged / skewed
dtype:   FP32 / BF16
mode:    forward / backward / optimizer step
```

Report separately:

- index construction;
- local message kernel;
- end-to-end layer;
- rebuild frequency for coordinate refinement;
- peak allocated and reserved memory;
- edges visited per second;
- nodes processed per second.

Promotion requires both mathematical equivalence and a measured benefit. A
backend that saves edge memory but is slower should remain a memory-mode option,
not become the default.

## 9. Suggested API boundary

```python
@dataclass(frozen=True)
class OnTheFlyNeighborConfig:
    cutoff: float
    skin: float = 0.0
    periodic: bool = False
    deterministic: bool = False

@dataclass(frozen=True)
class OnTheFlyNeighborReceipt:
    exact_cutoff: bool
    topology_rebuilt: bool
    pairs_visited: int
    max_degree: int
    supports_backward: bool
    supports_double_backward: bool
```

The receipt must describe capabilities and execution facts only. It is not a
speed claim.
