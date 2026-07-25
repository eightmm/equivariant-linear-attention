# Architecture v3: capability and evidence boundary

## Current outcome

Architecture v3 is implemented as an opt-in extension of
`EquivariantAttention`; it is not the default. The implementation adds:

- an exact 15-component feature map for
  `kappa * (q_primary dot k_primary)^4`;
- two learned `1o` angular axes per head for the linear/quadratic kernel;
- public polar-vector `1o` and symmetric-traceless reflection-even `2e`
  inputs;
- invariant RMS normalization of hidden non-scalar irreps;
- optional gated-edge-MLP activation recomputation;
- pathwise pre-clip gradient measurements; and
- a compressed exact quadratic factorization with 6 components for one axis
  and 21 for two axes, instead of 9 and 36.

The focused CPU evidence is green: exact dense/factorized agreement,
O(3)/translation/permutation/batch contracts, external-input gradients,
disabled-path compatibility, checkpointed forward/backward equality, and
training-runner contracts pass. The subsequent strict-CUDA QM9 screen completed
all five registered arms. No arm passed the joint accuracy/resource gate, so
the extensions remain experimental and off by default.

## Kernel

For scalar content `a,b`, primary polar axes `q1,k1`, and optional two-axis
direct sums `qbar,kbar`, the combined v3 pair kernel is

```text
K_ijh = c + a_ih dot b_jh
        + beta_h + delta_h (qbar_ih dot kbar_jh)
        + gamma_h (qbar_ih dot kbar_jh)^2
        + kappa_h (q1_ih dot k1_jh)^4.
```

All learned coefficients are bounded so the kernel stays positive. The degree
two and degree four terms both have exact finite feature maps, so graph
summaries are accumulated before querying them. No `N x N` pair tensor is
materialized. At fixed widths, heads, and two-axis cap, the global path remains
`O(N)` in node count; a hybrid LGL model additionally pays `O(E_local)` for its
two gated local stages.

`angular_feature_rank=2` is deliberately narrow terminology: it is a direct
sum of two `l=1`/`1o` axes, not an `l=2` irrep. The actual `2e` representation
is the separate persistent symmetric-traceless tensor state.

## Functional comparison

The comparison below is architectural, not a claim that this implementation
matches the published accuracy or optimized kernels of another project.

| Family | Symmetry contract | Context and pair cost | Representation scope | Permutation contract |
|---|---|---|---|---|
| This v3 | full `O(3)` plus translation-invariant scalar prediction | exact global fixed-width `O(N)`; optional supplied-edge local `O(E)` | public/hidden `0e`, polar `1o`, reflection-even `2e`; no arbitrary `l` | exact node permutation equivariance |
| Private same-feature EGNN control | Euclidean equivariance | local message passing `O(E)`; complete graph is `O(N^2)` | scalar state plus relative-coordinate vector transport | exact graph permutation equivariance |
| [SE(3)-Transformer](https://proceedings.neurips.cc/paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html) | `SE(3)` | equivariant pair/neighborhood attention; all-pairs use is quadratic | irreducible equivariant features | graph/set attention is permutation equivariant |
| [Equiformer](https://arxiv.org/abs/2206.11990) / [EquiformerV2](https://arxiv.org/abs/2306.12059) | `SE(3)`/`E(3)` or `SO(3)` formulation | local graph attention and equivariant convolutions, proportional to supplied neighborhoods and tensor-product cost | general configured irreps; V2 is designed for higher degrees | graph permutation equivariant |
| [Euclidean Fast Attention](https://arxiv.org/abs/2412.08541) | Euclidean physical symmetries | global linear-scaling attention-like operator with Euclidean rotary encodings | model-specific equivariant atomic representations | designed for atomic sets |
| [SE(3)-Hyena](https://arxiv.org/abs/2407.01049) | `SE(3)` | global long convolution at subquadratic sequence cost | equivariant sequence features | a set-permutation guarantee is not automatic for an ordered long convolution |
| [Clebsch-Gordan Transformer](https://arxiv.org/abs/2509.24093) | `SO(3)` | global `O(N log N)` Clebsch-Gordan convolution | all representation orders | token permutation equivariance is optional in the published design |

The v3 combination is functionally stronger than a purely local equivariant
attention layer along one axis: every node can receive global geometric
context without a dense pair tensor or any global edge list. It is also
strictly linear rather than `O(N log N)` at fixed feature width. These facts do
not make it universally superior:

- Euclidean Fast Attention is also linear-scaling and has a purpose-built
  long-range positional encoding.
- EquiformerV2 and the Clebsch-Gordan Transformer support higher-order irreps
  beyond this model's `0e/1o/2e` subset.
- this model is reflection-even `O(3)` and cannot express
  chirality-sensitive scalar targets through parity-odd hidden channels;
- the kernel is a bounded finite polynomial kernel, not softmax attention;
- local gated transport still materializes edge activations, and its constant
  dominates on small protein-ligand and QM9 graphs; and
- no v3 real-data result yet establishes accuracy superiority over any
  published implementation.

## Why a nominally linear model can be slower or use more memory

Asymptotic scaling does not determine the small-graph crossover. The quartic
term adds 15 query/key features per head. The second angular axis adds two
channel projections and changes the degree-two summary from 6 to 21
components. Backward stores or recomputes these activations, while the LGL
route still runs two nonlinear edge MLP stages. On QM9-sized graphs and the
current LBA pockets, a compact EGNN can therefore have lower kernel-launch
overhead and less working memory even though its cost grows with every edge.

The compressed quadratic factorization removes the provably redundant part of
that cost (`9 -> 6`, `36 -> 21`). Activation recomputation can reduce the
gated-edge activation footprint, but it necessarily adds backward compute.
Only synchronized full train-step measurements can decide the net effect.

## Feature parity and representational inputs

Using identical raw atom/segment features, coordinates, split, topology, and
training policy is the right control for asking whether the learned
architecture is better. A model can still benefit from how it represents that
same information: radial bases, tensor products, vector/tensor channels,
normalization, and readout are architectural inductive biases.

External `1o`/`2e` fields are a separate capability. A fair benchmark may use
them only when every arm receives the same underlying information and the
comparison explicitly admits model-specific equivariant encoders. Giving only
v3 an additional physical descriptor would be a feature comparison, not a
same-feature architecture comparison.

## Real-data evidence

The preceding gated-plus-grouped v2 LGL has already been evaluated on the
complete official ATOM3D-LBA ID30 train/validation split. Its one-seed
validation RMSE was `1.550035 pK`, versus `1.592008 pK` for the matched
incumbent and `1.692812 pK` for the private EGNN. The paired interval versus
the incumbent crossed zero, all arms clipped more than 99% of updates, and no
test split was opened. This is evidence for the preceding hybrid, not v3.

The v3 protocol first screened individual additions on cached QM9 gap
validation. A candidate could advance only when it improved by at least
`0.010 eV` while keeping median train-step latency and CUDA peak allocation
within `1.25x` of the gated-plus-grouped incumbent.

| Arm | Validation MAE (eV) | Improvement (eV) | Step ratio | Peak ratio | Clip fraction |
|---|---:|---:|---:|---:|---:|
| incumbent | 0.646338 | — | 1.000 | 1.000 | 0.906 |
| irrep RMS normalization | 0.675463 | -0.029126 | 1.094 | 1.005 | 0.950 |
| quartic kernel | 0.673753 | -0.027415 | 0.995 | 1.135 | 0.908 |
| two angular axes | 0.703972 | -0.057634 | 1.024 | 1.155 | 0.912 |
| combined | 0.642692 | +0.003646 | 1.067 | 1.297 | 0.950 |

All values were finite and all arms shared the same paired base initialization,
raw features, coordinates, split, and sparse candidates. The combined arm was
the only arm better than the incumbent, but it missed the accuracy threshold
and exceeded the memory ceiling. The individual negative results plus the
small combined gain indicate a non-additive interaction, not evidence that any
individual mechanism is useful. The frozen decision is therefore “software
capability retained, performance promotion rejected.” The conditional v3 LBA
comparison did not run and no test labels were evaluated.

The earlier two-update CPU diagnostic attributed almost all clipping to the
readout. That observation did not generalize across training: over 500 CUDA
updates, the incumbent's largest mean path norm was the global/shared-update
bucket (`3.63`), while combined was dominated by input (`3.98`), global
(`3.33`), and local (`2.76`) buckets. Combined increased total mean pre-clip
norm from `4.88` to `6.67` and clipping from `90.6%` to `95.0%`. Clipping is
therefore a time-varying, multi-path optimization problem rather than a
readout-only or unnormalized-local-only defect.

## Completion boundary

This is a coherent completion point for an exact fixed-width `O(3)`
local/global encoder over `0e/1o/2e`. It is not a final general equivariant
Transformer. The next major model-class expansion would require an explicit
choice among:

1. parity-complete `O(3)` channels and chirality-sensitive `SE(3)` prediction;
2. general `l` with spherical-harmonic/Clebsch-Gordan products;
3. a production neighbor-list backend and force/energy conservation contract;
4. a new global spatial encoding beyond finite polynomial moments.

Those are different architectures, not small v3 switches. The current screen
shows that adding polynomial order or angular axes without a better
optimization and spatial-selectivity mechanism is not a productive next
iteration.
