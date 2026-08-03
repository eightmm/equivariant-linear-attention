# ELA hidden-state initialization addendum

> **Internal implementation note.** This records the initialization rationale
> for a private carrier used beneath `ELA`; it is not a public architecture or
> API contract. See [CANONICAL_ELA.md](CANONICAL_ELA.md) and
> [API_POLICY.md](API_POLICY.md) for the normative contract.

The note describes the zero-initialization exception retained by the private
ELA runtime.

All ordinary sparse rank-to-head maps remain zero initialized. The one
exception is the local pseudoscalar rank-to-head bridge. For head `h` and local
rank `r`, its initial matrix is the deterministic cyclic selector

$$
W^{0o}_{hr}(0)=\mathbf 1\{r=h\bmod R\}.
$$

This map is parity preserving and uses no random initialization. It is required
because the even scalar update reads the chiral sector through even products
such as

$$
\left(C_i^{0o}\right)^2.
$$

If the pseudoscalar projection and carrier both started exactly at zero, an
even-only objective such as energy or affinity would see

$$
\frac{\partial (C^{0o})^2}{\partial C^{0o}}\bigg|_{C^{0o}=0}=0
$$

and could not train the chiral branch on its first backward pass. The cyclic
bridge removes that dead start while preserving every SE(3)/parity
transformation law. The scalar, polar, axial, even-tensor, odd-tensor, and mass
local output maps otherwise retain their neutral initialization.

The implementation is in `_ELARuntime._initialize_chiral_bridge`; the focused
regression remains in `tests/test_unified_chiral_bridge.py` for historical test
continuity.
