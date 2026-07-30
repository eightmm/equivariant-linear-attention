# Unified 3D initialization addendum

This note is normative for `UnifiedEquivariantAttention` and refines the
zero-initialization paragraph in `UNIFIED_3D_CORE.md`.

All ordinary sparse rank-to-head maps remain zero initialized. The one
exception is the local pseudoscalar rank-to-head bridge. For head `h` and local
rank `r`, its initial matrix is the deterministic cyclic selector

\[
W^{0o}_{hr}(0)=\mathbf 1\{r=h\bmod R\}.
\]

This map is parity preserving and uses no random initialization. It is required
because the even scalar update reads the chiral sector through even products
such as

\[
\left(C_i^{0o}\right)^2.
\]

If the pseudoscalar projection and carrier both started exactly at zero, an
even-only objective such as energy or affinity would see

\[
\frac{\partial (C^{0o})^2}{\partial C^{0o}}\bigg|_{C^{0o}=0}=0
\]

and could not train the chiral branch on its first backward pass. The cyclic
bridge removes that dead start while preserving every SE(3)/parity
transformation law. The scalar, polar, axial, even-tensor, odd-tensor, and mass
local output maps otherwise retain their neutral initialization.

The implementation and regression test are in
`UnifiedEquivariantAttention._initialize_chiral_bridge` and
`tests/test_unified_chiral_bridge.py`.
