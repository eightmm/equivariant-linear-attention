# Unified parity-complete SE(3) core

`UnifiedEquivariantAttention` is the canonical generic 3D path. Users declare
flattened `input_irreps` and final `output_irreps`; internal angular degree,
parity, routing, normalization, and local/global composition are fixed by the
implementation.

```python
config = Unified3DConfig(
    input_irreps="32x0e",
    output_irreps="1x0e + 1x0o + 2x1o",
    hidden_dim=128,
    num_layers=6,
    num_heads=8,
    local_rank=4,
    local_cutoff=6.0,
)
model = UnifiedEquivariantAttention(config)

graph = prepare_3d_graph(batch, edge_index)
output = model(node_irreps, positions, graph)

node_output = output["node_irreps"]
graph_output = output["graph_irreps"]
blocks = model.split_output(node_output)
```

Input and output declarations are the representation-level user choices.
Width, depth, head count, rank, and cutoff are resource/capacity
hyperparameters.

## 1. Fixed internal representation

The persistent state of node \(i\) in block \(\ell\) is

\[
h_i^\ell =
\left(
s_i^{+,\ell},
s_i^{-,\ell},
V_i^{-,\ell},
A_i^{+,\ell},
T_i^{+,\ell},
U_i^{-,\ell}
\right),
\]

with

\[
s_i^+\in\mathbb R^{C_0},\qquad
s_i^-\in\mathbb R^{H},
\]

\[
V_i^-\in\mathbb R^{H\times3},\qquad
A_i^+\in\mathbb R^{H\times3},
\]

\[
T_i^+\in\mathbb R^{H\times5},\qquad
U_i^-\in\mathbb R^{H\times5}.
\]

The corresponding irreps are

\[
\boxed{
C_0\times0e
\oplus H\times0o
\oplus H\times1o
\oplus H\times1e
\oplus H\times2e
\oplus H\times2o
}.
\]

Thus persistent \(l\) is automatically bounded at \(l\le2\). This is a
performance choice, not a claim that higher degrees are mathematically
irrelevant. Chirality is generated from aggregated \(l=1\) moments, so a
persistent \(l=3\) carrier is unnecessary for the canonical path.

The public contract is SE(3)-equivariance. Internally the parity bookkeeping is
stronger: all sectors obey their O(3) transformation law. For an orthogonal
matrix \(R\),

\[
s^+\mapsto s^+,\qquad
s^-\mapsto\det(R)s^-,
\]

\[
V^-\mapsto RV^-,
\qquad
A^+\mapsto\det(R)RA^+,
\]

\[
T^+\mapsto RT^+R^T,
\qquad
U^-\mapsto\det(R)RU^-R^T.
\]

Consequently the same core can produce reflection-even, reflection-odd, polar,
axial, and mixed outputs without changing its hidden architecture.

## 2. Input and prepared graph

The public feature input is one flattened direct sum

\[
f_i\in
C_{0e}\,0e\oplus C_{0o}\,0o\oplus
C_{1e}\,1e\oplus C_{1o}\,1o\oplus
C_{2e}\,2e\oplus C_{2o}\,2o,
\]

and position remains the separate geometry input \(x_i\in\mathbb R^3\).
The optimized path rejects persistent \(l>2\). `input_irreps="0"` supplies no
external features and initializes from geometry and invariant metadata alone.
The Cartesian `l=1` basis is xyz; `l=2` is stored as
`[xx, yy, xy, xz, yz]` with `zz=-xx-yy`.

`Prepared3DGraph` stores one receiver-major CSR graph. For receiver \(i\), its
candidate edges occupy

\[
\operatorname{row\_ptr}[i]:
\operatorname{row\_ptr}[i+1].
\]

Graph isolation, index ranges, relation ordering, and graph layout are validated
before model execution. The forward path has no complete-graph fallback and no
neighbor-provider dispatch.

## 3. Initialization

For graph \(g\),

\[
\mu_g=\frac1{N_g}\sum_{i\in g}x_i,
\]

\[
r_g=
\sqrt{
\frac1{N_g}
\sum_{i\in g}\|x_i-\mu_g\|^2+\epsilon
},
\]

\[
\bar x_i=\frac{x_i-\mu_{g(i)}}{r_{g(i)}}.
\]

Each input sector is projected only into the matching hidden sector. Only the
`0e` projector has a bias. The geometry-derived state is then added within the
same parity sector. In particular, the even scalar state begins as

\[
s_i^{+,0}=W_{0e}f_i^{0e}+b_{0e}+e_{\rm role(i)}+d_i.
\]

The initial polar state is

\[
V_i^{-,0}
=
W_{1o}f_i^{1o}
+
\tanh(G_v s_i^{+,0})\odot\bar x_i.
\]

The initial even tensor state is

\[
T_i^{+,0}
=
W_{2e}f_i^{2e}
+
\tanh(G_Ts_i^{+,0})
\odot\operatorname{ST}(\bar x_i).
\]

External odd scalar, axial vector, and odd tensor inputs are projected directly
and summed with their geometry-derived multipoles:

\[
s_i^{-,0}=W_{0o}f_i^{0o}+\chi_i,\qquad
A_i^{+,0}=W_{1e}f_i^{1e}+A_i^{\rm geom},\qquad
U_i^{-,0}=W_{2o}f_i^{2o}+U_i^{\rm geom}.
\]

All non-`0e` input maps are bias-free.

## 4. Homogeneous recurrence

Every block uses exactly the same recurrence:

\[
G^\ell
=
\operatorname{ExactGlobal}(h^\ell,\bar x),
\]

\[
S^\ell
=
\operatorname{SparseLocal}(h^\ell,x,\mathcal E),
\]

\[
h^{\ell+1}
=
\operatorname{ParityUpdateFFN}
\left(
h^\ell,
G^\ell+S^\ell
\right).
\]

There is no LGL schedule, local-only block, refresh stride, local softmax, or
per-layer representation selection.

## 5. Exact positive global transport

Let

\[
\bar s_i=\operatorname{LN}(s_i^+).
\]

For head \(h\), positive scalar features are

\[
q^0_{ih}
=
\frac{\operatorname{ELU}(W_q\bar s_i)+1}{\sqrt{D}},
\]

\[
k^0_{ih}
=
\frac{\operatorname{ELU}(W_k\bar s_i)+1}{\sqrt{D}}.
\]

Bounded polar and axial query/key vectors are

\[
q^p_{ih}
=
\mathcal U\left(
W_q^pV_i^-\odot\tanh g_q^p(\bar s_i)
\right),
\]

\[
k^p_{ih}
=
\mathcal U\left(
W_k^pV_i^-\odot\tanh g_k^p(\bar s_i)
\right),
\]

\[
q^a_{ih}
=
\mathcal U\left(
W_q^aA_i^+\odot\tanh g_q^a(\bar s_i)
\right),
\]

\[
k^a_{ih}
=
\mathcal U\left(
W_k^aA_i^+\odot\tanh g_k^a(\bar s_i)
\right),
\]

where

\[
\mathcal U(z)=\frac{z}{\sqrt{1+\|z\|^2+\epsilon}}.
\]

With positive learned \(\beta_h^p,\beta_h^a\),

\[
\Phi^Q_{ih}
=
\operatorname{concat}
\left(
q^0_{ih},
\sqrt{1+\beta_h^p+\beta_h^a},
\sqrt{\beta_h^p}\,q^p_{ih},
\sqrt{\beta_h^a}\,q^a_{ih}
\right),
\]

\[
\Phi^K_{jh}
=
\operatorname{concat}
\left(
k^0_{jh},
\sqrt{1+\beta_h^p+\beta_h^a},
\sqrt{\beta_h^p}\,k^p_{jh},
\sqrt{\beta_h^a}\,k^a_{jh}
\right).
\]

Their dot product is

\[
K_{ijh}
=
q^0_{ih}\cdot k^0_{jh}
+
1+\beta_h^p+\beta_h^a
+
\beta_h^p q^p_{ih}\cdot k^p_{jh}
+
\beta_h^a q^a_{ih}\cdot k^a_{jh}.
\]

Because \(\|q\|,\|k\|<1\),

\[
K_{ijh}>0.
\]

Polar-polar and axial-axial contractions are parity-even. No polar-axial
contraction enters the attention weight.

### 5.1 One balancing cycle

For graph \(g\),

\[
Q_{gh}=\sum_{i\in g}\Phi^Q_{ih},
\]

\[
m_{jh}=(\Phi^K_{jh})^TQ_{g(j)h},
\]

\[
\widetilde\Phi^K_{jh}
=
\frac{\Phi^K_{jh}}{m_{jh}}.
\]

This single balancing cycle is always applied.

### 5.2 Augmented sufficient statistic

Let \(z_{jh}\) contain all six parity sectors plus coordinate moments. Append a
denominator coordinate:

\[
\tilde z_{jh}=[z_{jh},1].
\]

Then

\[
A_{gh}
=
\sum_{j\in g}
\widetilde\Phi^K_{jh}\otimes\tilde z_{jh},
\]

\[
y_{ih}
=
(\Phi^Q_{ih})^TA_{g(i)h},
\]

\[
G_{ih}
=
\frac{y_{ih,1:-1}}{y_{ih,-1}}.
\]

No \(N\times N\) pair matrix is constructed. The implementation evaluates the
same equation with direct GEMM, padded BMM, bucketed BMM, or grouped ragged
GEMM according to the prepared graph layout. These are arithmetic schedules,
not different model definitions.

The value payload transports

\[
0e,\ 0o,\ 1o,\ 1e,\ 2e,\ 2o
\]

states with the same invariant positive weights. Centered coordinate zeroth,
first, and second moments reconstruct relative polar and even rank-two geometry
exactly.

## 6. Positive sparse local transport

For edge \(j\to i\),

\[
d_{ij}=\frac{x_j-x_i}{R_c},
\qquad
\nu_{ij}=\|d_{ij}\|^2.
\]

A typed relation can only narrow the shared domain. If its cutoff is
\(R_{\rho_{ij}}\le R_c\),

\[
\tilde\nu_{ij}
=
\nu_{ij}
\left(
\frac{R_c}{R_{\rho_{ij}}}
\right)^2.
\]

Otherwise \(\tilde\nu_{ij}=\nu_{ij}\).

The common cutoff is

\[
f_c(\tilde\nu)
=
\begin{cases}
\frac12\left(1+\cos(\pi\tilde\nu)\right),
&0\le\tilde\nu<1,\\
0,&\tilde\nu\ge1.
\end{cases}
\]

For local rank \(r\), the parity-even score is

\[
\begin{aligned}
a_{ijr}
={}&
q_{ir}k_{jr}
+b_r
+B_r\operatorname{RBF}(\nu_{ij})\\
&+\alpha_r^p\langle u_{ir}^p,v_{jr}^p\rangle\\
&+\beta_r^p
\langle u_{ir}^p,d_{ij}\rangle
\langle v_{jr}^p,d_{ij}\rangle\\
&+\alpha_r^a\langle u_{ir}^a,v_{jr}^a\rangle\\
&+\beta_r^a
\langle u_{ir}^a,d_{ij}\rangle
\langle v_{jr}^a,d_{ij}\rangle\\
&+b_{\rho_{ij},r}.
\end{aligned}
\]

The individual axial-direction contractions are pseudoscalars, but their
product is even.

The bounded positive gate and edge weight are

\[
g_{ijr}
=
\exp\left(3\tanh(a_{ijr}/3)\right),
\]

\[
w_{ijr}
=
f_c(\tilde\nu_{ij})g_{ijr}.
\]

Receiver masses are

\[
m_{ir}=\sum_{j\to i}w_{ijr},
\]

\[
m_{ir}^{(2)}=\sum_{j\to i}w_{ijr}^2.
\]

Every transported family uses the same normalization:

\[
S_{ir}^{(f)}
=
\frac{
\sum_{j\to i}
w_{ijr}\rho_{ijr}^{(f)}z_{jr}^{(f)}
}{
1+m_{ir}
}.
\]

The additive one is part of the operator. A singleton edge therefore vanishes
continuously at the cutoff:

\[
\frac{w}{1+w}\longrightarrow0.
\]

All receiver sums use the same CSR segment semantics.

## 7. Chirality without explicit triplets

Three independently gated polar direction moments are aggregated:

\[
D_{ir}^{(a)}
=
\frac{
\sum_{j\to i}
w_{ijr}\rho_{ijr}^{(a)}
g_{jr}^{(a)}d_{ij}
}{
1+m_{ir}
},
\qquad a=1,2,3.
\]

The axial vector is

\[
C_{ir}^{1e}
=
D_{ir}^{(1)}\times D_{ir}^{(2)}.
\]

The pseudoscalar is

\[
C_{ir}^{0o}
=
C_{ir}^{1e}\cdot D_{ir}^{(3)}.
\]

The odd rank-two tensor is

\[
C_{ir}^{2o}
=
\operatorname{ST}
\left(
D_{ir}^{(3)},C_{ir}^{1e}
\right).
\]

These are computed after edge aggregation. Therefore there is no explicit
\((i,j,k)\) triplet tensor and no \(O(Ek)\) triplet enumeration. The cost remains

\[
O(ER)+O(NR).
\]

Under reflection,

\[
C^{1e}\mapsto\det(R)RC^{1e},
\]

\[
C^{0o}\mapsto\det(R)C^{0o},
\]

\[
C^{2o}\mapsto\det(R)RC^{2o}R^T.
\]

The local rank-to-head maps are zero-initialized. The initial function is thus a
pure exact-global model; chiral/local sectors wake through gradients without
perturbing initial outputs.

## 8. Parity-preserving update

Global and local messages are added sector by sector:

\[
M_i^\ell=G_i^\ell+S_i^\ell.
\]

Even scalar updates use only even invariants:

\[
\left[
M^{0e},
(M^{0o})^2,
\|M^{1o}\|^2,
\|M^{1e}\|^2,
\|M^{2e}\|_F^2,
\|M^{2o}\|_F^2,
(C^{0o})^2
\right].
\]

Odd scalar updates are linear combinations of odd bases with even
state-dependent coefficients:

\[
\left[
M^{0o},
M^{1o}\cdot M^{1e},
M^{2e}:M^{2o},
C^{0o}
\right].
\]

Representative polar and axial update bases are

\[
\Delta V^{1o}
\in
\operatorname{span}
\left\{
M^{1o},
M^{0o}M^{1e},
M^{2e}M^{1o},
M^{2o}M^{1e}
\right\},
\]

\[
\Delta A^{1e}
\in
\operatorname{span}
\left\{
M^{1e},
M^{0o}M^{1o},
M^{2e}M^{1e},
M^{2o}M^{1o}
\right\}.
\]

Even and odd tensor bases are

\[
\Delta T^{2e}
\in
\operatorname{span}
\left\{
M^{2e},
\operatorname{ST}(M^{1o},M^{1o}),
\operatorname{ST}(M^{1e},M^{1e}),
M^{0o}M^{2o}
\right\},
\]

\[
\Delta U^{2o}
\in
\operatorname{span}
\left\{
M^{2o},
\operatorname{ST}(M^{1o},M^{1e}),
M^{0o}M^{2e},
\ldots
\right\}.
\]

All vector and tensor residuals are bounded before addition. The pointwise FFN
uses the same parity rule: arbitrary nonlinearities act on even invariants,
while odd/equivariant states are multiplied by even gates and mixed only
through parity-valid products.

## 9. Output irreps

Users declare

```python
output_irreps="2x0e + 1x0o + 3x1o + 1x1e + 1x2e + 1x2o"
```

and nothing about hidden parity or hidden \(l\).

The output head selects the corresponding automatic carrier:

\[
0e\leftarrow s^+,
\qquad
0o\leftarrow s^-,
\]

\[
1o\leftarrow V^-,
\qquad
1e\leftarrow A^+,
\]

\[
2e\leftarrow T^+,
\qquad
2o\leftarrow U^-.
\]

Odd scalar projections have no bias, preserving pseudoscalar parity.

The forward returns flattened values in the declared layout:

```python
{
    "node_irreps": Tensor[N, output_layout.dim],
    "graph_irreps": Tensor[G, output_layout.dim],
}
```

Use

```python
model.split_output(output["node_irreps"])
```

to recover a dictionary keyed by `0e`, `0o`, `1o`, `1e`, `2e`, and `2o`.

Typical declarations are:

- invariant property or energy: `"1x0e"`
- signed chirality or optical pseudoscalar: `"1x0o"`
- generic scalar with even and odd components: `"1x0e + 1x0o"`
- coordinate/displacement field: `"C x1o"`
- axial direction: `"C x1e"`
- anisotropic even tensor: `"C x2e"`

For protein-ligand affinity, `"1x0e"` is normally appropriate. The output is
globally reflection-even, while internal \(0o/1e/2o\) sectors still encode
relative ligand-pocket chirality through even products such as

\[
\chi_{\rm ligand}\chi_{\rm pocket}.
\]

## 10. Complexity

At fixed width, head count, local rank, and output multiplicity,

\[
\boxed{
\text{time}=O\left(L(N+E)\right)
}
\]

and

\[
\boxed{
\text{persistent state}=O(N),\qquad
\text{neighbor metadata}=O(N+E).
}
\]

The global branch never forms an \(N\times N\) attention matrix. The local branch
uses supplied receiver-major CSR edges. Neighbor discovery remains outside the
layer and must be measured separately.

## 11. Current boundary

The optimized unified output supports \(l\le2\). This is deliberate:

- \(l=0,1,2\) with both parity sectors covers scalar, pseudoscalar, polar,
  axial, and anisotropic tensor outputs;
- local chirality is generated without persistent high-\(l\) activation;
- memory and edge payload remain bounded.

Higher output degree should be added only with a measured optimized executor,
not by silently falling back to a slow reference tensor-product path.
