# Layer-level SE(3) API, conditioning, and coordinate refinement

This document is normative for the layer API introduced by
`UnifiedEquivariantLayer` and the layered implementation of
`UnifiedEquivariantAttention`.

## 1. Coordinates are geometry, not ordinary irrep features

The model receives

\[
X_i\in\operatorname{Irreps}_{\rm in},
\qquad
x_i\in\mathbb R^3,
\]

where `node_irreps` transforms homogeneously under rotations and `pos` transforms
affinely:

\[
X_i^{(l,p)}\mapsto D^{(l,p)}(R)X_i^{(l,p)},
\]

\[
x_i\mapsto Rx_i+t.
\]

Raw positions therefore remain a separate argument. They are used to construct
relative displacement, radial features, cutoff weights, graph-centered geometry,
and receiver-centered node multipoles.

A requested `1o` output is a polar vector such as a displacement, velocity, or
force-like quantity. It is not itself an absolute position. Absolute refined
coordinates are produced only through the affine residual

\[
x_i' = x_i + \Delta x_i.
\]

## 2. Public hidden state

One layer consumes and returns

\[
h_i=
\left(
 s_i^{0e},
 s_i^{0o},
 V_i^{1o},
 A_i^{1e},
 T_i^{2e},
 U_i^{2o}
\right).
\]

The public Python container is `UnifiedSE3State`. Its fields are:

```text
even_scalar:  [N, C0]
odd_scalar:   [N, H]
polar_vector: [N, H, 3]
axial_vector: [N, H, 3]
even_tensor:  [N, H, 5]
odd_tensor:   [N, H, 5]
```

The compact tensor basis is `[xx, yy, xy, xz, yz]` with `zz=-xx-yy`.

`UnifiedSE3Context` contains the current positions, normalized positions, sparse
geometry, node multipoles, batch vector, and graph layout. A prepared context can
be reused by multiple layers when positions are fixed.

## 3. Layer decomposition

Each layer is explicitly decomposed into an attention residual and an FFN
residual.

First, sector-wise pre-normalization is applied:

\[
\widehat h^\ell=\operatorname{IrrepPreNorm}(h^\ell).
\]

The exact global and sparse local operators are evaluated on the same normalized
state:

\[
G^\ell=\operatorname{ExactGlobal}_{l\le2}(\widehat h^\ell),
\]

\[
S^\ell=\operatorname{SparseLocal}_{l\le2}
(\widehat h^\ell,x^\ell,\mathcal E).
\]

The attention residual includes the parity-aware message update and the low-rank
node tensor-product closure:

\[
\widetilde h^\ell
=
 h^\ell
 +\Delta h_{\rm attn}^\ell
 +\Delta h_{\rm closure}^\ell.
\]

The pointwise equivariant FFN is then applied as a separate residual:

\[
h^{\ell+1}
=
\widetilde h^\ell
+
\Delta h_{\rm ffn}^\ell.
\]

The public methods are:

```python
after_attention = layer.attention_residual(state, context, condition)
after_ffn = layer.ffn_residual(after_attention, context, condition)
result = layer(state, context, condition)
```

`result.state` equals the composition of the first two calls.

## 4. DiT-style condition input

A condition is an invariant ordinary scalar feature:

\[
c_g\in\mathbb R^{C_{\rm cond}}
\quad\text{or}\quad
c_i\in\mathbb R^{C_{\rm cond}}.
\]

It may be supplied per graph, per node, or as one vector broadcast to the whole
batch. Typical uses include diffusion time embeddings, noise level, class
conditioning, experimental state, temperature, or another global control
variable.

The condition is not an arbitrary irrep tensor. Allowing a non-invariant
condition without declaring its transformation law would break equivariance.

For the attention and FFN branches separately, a zero-initialized projection
produces:

- even-scalar adaptive shift and scale;
- copy-wise scales for `0o`, `1o`, `1e`, `2e`, and `2o`;
- independent residual gates for every hidden copy.

For even scalars,

\[
\operatorname{Mod}(s^{0e};c)
=
(1+\gamma^{0e}(c))\odot s^{0e}+\beta^{0e}(c).
\]

For a non-scalar sector `X`,

\[
\operatorname{Mod}(X;c)
=
(1+\gamma^X(c))\odot X.
\]

There is no additive vector or tensor shift. Such a shift would require an
external equivariant direction rather than an invariant condition.

Residual gates use the neutral parameterization

\[
\Delta h(c)
=
\left[1+\tanh g(c)\right]\odot\Delta h.
\]

All condition projection weights and biases start at zero. Consequently adding a
condition dimension does not change the initial unconditioned function, while
the condition projection receives a gradient on the first backward pass.

Example:

```python
config = Unified3DConfig(
    input_irreps="32x0e + 4x1o",
    output_irreps="1x1o",
    condition_dim=256,
)
model = UnifiedEquivariantAttention(config)

# Graph-level condition: [G, 256]
output = model(node_irreps, pos, graph, condition=condition)
```

## 5. Coordinate residual

When `coordinate_updates=True`, each layer predicts one polar displacement from
the current hidden `1o` sector:

\[
r_i=W_xV_i^{1o},
\]

\[
\Delta x_i
=
\Delta_{\max}
\,\sigma(a_i)
\frac{r_i}{\sqrt{1+\|r_i\|^2}}.
\]

The channel projection `W_x` is zero initialized. Thus

\[
\Delta x_i(0)=0,
\]

and enabling coordinate refinement preserves the original function at
initialization.

The update obeys

\[
\|\Delta x_i\|<\Delta_{\max}
\]

per layer. Under a proper rigid motion,

\[
\Delta x_i(Rx+t)=R\Delta x_i(x),
\]

and therefore

\[
x_i'(Rx+t)=Rx_i'(x)+t.
\]

The full model always returns:

```text
node_irreps
 graph_irreps
 positions
 coordinate_delta
```

When coordinate updates are disabled, `positions` equals the input and
`coordinate_delta` is zero.

## 6. Geometry refresh and topology contract

With coordinate refinement enabled, distance, cutoff, RBF, normalized position,
and node multipoles are recomputed after every layer. The receiver/sender
candidate topology is not rediscovered inside the layer.

Thus the contract is

```text
fixed candidate topology + per-layer geometry refresh
```

For large updates or dynamic systems, the prepared candidate graph must include a
sufficient radius skin or be rebuilt by an outer loop. A pair absent from the
candidate graph cannot become a new neighbor merely because coordinates moved.
This separation keeps neighbor discovery outside the differentiable layer and
avoids a silent quadratic fallback.

## 7. When coordinate output is useful

The coordinate path is useful for:

- diffusion and flow denoising;
- molecular conformation or protein–ligand pose refinement;
- structure relaxation and learned optimizers;
- point-cloud registration and deformation;
- coarse-to-fine geometry generation;
- displacement or velocity-field prediction.

It is usually unnecessary for a fixed-geometry scalar property model. In that
case keep `coordinate_updates=False`; the model still uses coordinates to build
its geometry and may emit an explicit `1o` task output if desired.

## 8. Manual layer use

```python
state, context = model.embed_input(node_irreps, pos, graph)

for layer in model.layers:
    layer_output = layer(state, context, condition)
    state = layer_output.state

node_output = model.project_state(state)
```

If a manually assembled stack applies `layer_output.positions`, the caller must
refresh the context before the next layer:

```python
context = model.prepare_context(layer_output.positions, graph)
```

The high-level model performs this refresh automatically when
`coordinate_updates=True`.
