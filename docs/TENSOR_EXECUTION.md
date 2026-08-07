# Tensor-fused execution

This document describes the execution form of the edge-free ELA layer. The
mathematical operator is unchanged; the implementation removes repeated small
sector operations and redundant Cartesian tensor expansion.

## Packed value carrier

A scalar relation acts identically on every O(3) sector. The transported value is
therefore stored as one tensor

$$
V_{\mathrm{packed}}\in\mathbb R^{N\times H\times(D_h+17)},
$$

containing `0e`, `0o`, `1o`, `1e`, `2e`, and `2o` components. One segmented
Gram contraction transports all sectors at once.

## Unified PSD feature

Content, Mercer, and atlas relations are each PSD factorizations. Their
nonnegative graph/head mixture is represented by one concatenated feature

$$
\Phi=
\left[
\sqrt{\alpha_c/t_c}F_c,
\sqrt{\alpha_m/t_m}F_m,
\sqrt{\alpha_a}F_a
\right],
$$

so the complete relation is exactly

$$
R=\Phi\Phi^\top,
\qquad
RV=\Phi(\Phi^\top V).
$$

The implementation evaluates `R`, `R²`, and `R³` with three packed segmented
Gram applications rather than applying three operators to six sectors
independently.

## Compact coordinate basis

Complete symmetric Cartesian powers through degree four have only

$$
1+3+6+10+15=35
$$

independent monomials, not `1+3+9+27+81=121`. Multinomial-normalized monomials
preserve

$$
\langle\psi_k(x),\psi_k(y)\rangle=(x\cdot y)^k
$$

exactly, so the Gaussian Mercer approximation is unchanged.

The same 35-dimensional coordinate basis is reused by the moment bank, Mercer
features, and atlas statistics. Weighted relative moments through degree four
are recovered from one packed segment reduction.

## Irreducible transient storage

Third- and fourth-order symmetric trace-free tensors are stored directly in
their irreducible dimensions:

```text
3o: 7 components instead of 27
4e: 9 components instead of 81
```

Fixed Cartesian contraction formulas couple these compact tensors to the
persistent `l <= 2` carrier without reconstructing full rank-three or rank-four
tensors.

## Grouped equivariant projections

Polar/axial and even/odd tensor channel projections use paired batched matrix
multiplication. Tensor closure converts each ST5 tensor to matrix form at most
once and reuses vector-tensor paths.

## Static-geometry reuse and compilation

When coordinate updates are disabled, normalized coordinates and their compact
monomial basis are built once per model forward and reused across all layers.
The inference helper enables dynamic-shape compilation by default, keeping the
packed ragged node axis dynamic while the tensor core remains traceable.

## Complexity

The asymptotic contract remains

$$
O(NC^2+NKC)
$$

with node-linear memory. The optimization reduces feature width, intermediate
storage, segment-reduction count, and kernel-launch overhead without adding
edges, pair state, padding, or an alternative execution path.

## Exactness and validation contract

The optimized path is the only execution path; there is no runtime fallback to
the earlier sector-wise implementation. Mathematical equivalence is enforced by
tests that compare:

- each compact symmetric degree block with the corresponding full Cartesian
  relative-moment tensor through degree four;
- the 35-dimensional multinomial basis with the polynomial kernel
  $(x\cdot y)^k$ for every retained degree;
- compact `3o` and `4e` contractions with full STF rank-three and rank-four
  Cartesian contractions;
- the packed unified relation with its explicitly materialized dense PSD action;
- packed carrier round trips, transient irreducible dimensions, O(3) behavior,
  gradients, and the full model forward/backward contract.

The complete CPU gate consists of Ruff, the entire test suite, and the float64
machine-learning smoke test. Floating-point reduction order may differ from the
reference algebra, but the represented function and gradients are preserved to
the test tolerances.
