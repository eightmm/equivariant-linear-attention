# Canonical branch-fusion study

## Scope

This packet evaluates the architecture as a generic operator over sparse 3D
graphs and point clouds. QM9 and ATOM3D-LBA are measurement tasks, not
architecture-specific concepts. No pocket/ligand role is part of canonical
ELA, and the retired LGL route is not an arm or baseline.

The intervention is the invariant, identity-initialized global/local branch
fusion in canonical ELA. Both downstream arms have the same ELA schema,
initial state, forward graph, optimizer parameter groups, data order, and
update budget:

- `identity_locked`: branch parameters remain in the optimizer but their
  gradients are zeroed;
- `trainable_fusion`: branch parameters learn normally.

Branch parameters use zero weight decay in both arms. At initialization both
models and predictions must be byte-identical.

## Preregistered falsification

### Mechanics

Before reading a downstream metric:

- generic proper and improper O(3), translation, node permutation, graph
  isolation, and edge-order tests pass;
- scalar and non-scalar input/output irreps through \(l=2\) transform correctly;
- input and coordinate double backward are finite;
- CUDA FP32 and BF16 forward/backward pass;
- all common outputs and gradients in the raw-control/canonical compatibility
  probe match exactly.

### Resource prerequisite

Five fresh-process AB/BA order pairs (exact seeds `0..4`) are run for each of
two task-like shapes:

- QM9-like: \(N=128,\ k=8,\ width=64,\ depth=3\);
- LBA-like: \(N=512,\ k=32,\ width=64,\ depth=3\).

Promotion requires, per shape:

- parameter ratio at most `1.05x`;
- median order-balanced inference ratio at most `1.10x`;
- median order-balanced optimizer-step ratio at most `1.15x`;
- maximum allocated-memory ratio at most `1.10x`;
- no individual latency ratio above `1.20x`.

The benchmark excludes neighbor discovery, graph packing, and host/device
preparation. It includes AdamW state allocation and optimizer update in the
training measurement. BF16 is a separate safety path; FP32 is the resource
decision path. Aggregation fails closed unless all receipts come from the same
clean commit and CUDA device fingerprint, use at least 10 warmups and 30 timed
repeats, have identical common-state/input hashes within each AB/BA pair, and
record finite nonzero branch gradients and CUDA memory.

### QM9 screen

- seed `42`, `500` updates;
- target `gap` (`target_index=4`);
- `110000/10000/remaining` train/validation/closed-test split;
- same `11x0e` input features, coordinates, sparse edges, graph-mean readout,
  train-only target normalization, batches, clipping, optimizer, and schedule;
- test is not evaluated.

Advance to multi-seed confirmation only when:

- validation MAE improvement (`identity_locked - trainable_fusion`) is at least
  `0.010 eV`;
- regression is no worse than `0.020 eV`;
- all branch gradients are finite;
- at least one layer/sector has router-weight RMS deviation at least `1e-3`
  and fused-message relative RMS at least `1e-5`.

### LBA capacity screen

- immutable ATOM3D-LBA revision
  `f93dd2d150a47c270f624620f84e07451a158705`;
- train rows `0..15` only, `1000` updates;
- validation and test are not loaded or evaluated;
- same `140x0e` features, bound coordinates, ligand readout mask, sparse
  topology, normalization, batches, clipping, and optimizer.

This is a memorization/capacity check only. Candidate train MAE must reach
`<=0.10 pK`, with the same finite/active router conditions. It cannot establish
generalization or affinity-model superiority.

## Claim boundary

A failed QM9 or LBA screen retains the minimal ELA API but demotes learned
branch routing from a canonical empirical claim to an experimental mechanism.
A passed one-seed screen still does not establish a result: confirmation would
require QM9 seeds `41..43` at 2,000 updates and, only after a separate
edge-topology identity receipt, LBA ID30 seeds `41..43`. QM9 test and LBA test
remain closed throughout.

## Results

### Mechanics and exactness

The final core passed generic proper/improper O(3), translation, permutation,
graph-isolation, edge-order, double-backward, CUDA FP32, and CUDA BF16 checks.
With strict determinism enabled, the compatibility probe reported exactly zero
node output, graph output, feature-gradient, coordinate-gradient, and
common-parameter-gradient differences for every FP32 resource receipt.

The full suite artifact is
`artifacts/canonical-ela-full-03dc01e/manifest.json` (SHA-256
`89bc9ff72631f34a7d7ca8b0aef90679f1a466391fbbea570541e12ce134bca8`).
The later changes were a strict benchmark fix, an algebraically equivalent
fusion implementation, and a stride-safe provenance hash; final repository
gates are reported below.

The clean final code-and-study commit
`36b5c948e8f926b4459d6a8505c8f03b2c81282b` passed:

- `scripts/check.sh fast`: 1,369 passed, 4 skipped, 86.56% coverage, CPU
  float64 smoke passed;
- `scripts/check.sh gpu`: CUDA BF16 and FP32 smoke passed;
- the canonical branch-fusion, zero-init, O(3)/permutation, double-backward,
  CUDA, migration, downstream, and resource-contract set: 40 passed.

The machine-readable receipt is
`artifacts/canonical-final-gates-36b5c94/receipt.json` (SHA-256
`1927a6f79844b31ef45189c77ce607c056f068f1a1ea5f38a62ccc4ed66681e3`).
The subsequent release documentation commit changes no source, test, script, or
CI file relative to this verified commit. Repository CI remains manual
`workflow_dispatch` only.

### FP32 CUDA resource decision

The registered 20-run AB/BA matrix is
`artifacts/canonical-ela-resource-daa49c5/aggregate.json` (SHA-256
`00fe14061112a31765995a020be87605125850dea4c7026edfa10a070ac1f0c0`).
It failed:

| shape | params | inference | optimizer step | max memory | max individual latency |
|---|---:|---:|---:|---:|---:|
| `N=128, k=8` | `1.03441x` | `1.09570x` | `1.14493x` | `1.00792x` | `3.71733x` |
| `N=512, k=32` | `1.03441x` | `1.10383x` | `1.11304x` | `1.00742x` | `1.13247x` |

The small-shape medians and both memory gates passed. Promotion still fails
because the large-shape inference median exceeded `1.10x` and the small-shape
matrix contained a large system-latency outlier. The outlier was retained; no
selective rerun or post-result gate relaxation was used. The coefficient-fusion
optimization reduced optimizer-step peak memory from about `1.017x` to about
`1.008x`, but did not establish the complete resource contract.

### Real-data diagnostic

Because the resource prerequisite failed, the exact registered QM9/LBA protocol
was run only with the explicit non-promotable diagnostic override. The receipt
is `artifacts/canonical-branch-fusion-diagnostic-580300b/summary.json`
(SHA-256
`c1391b93e1bd1f02aa7a46f0e652078ecc4e50d75851f208222b7f0ffc831d2f`).

| task / metric | identity locked | trainable fusion | candidate change |
|---|---:|---:|---:|
| QM9 validation MAE | `0.618511 eV` | `0.595026 eV` | `+0.023484 eV` |
| QM9 median step | `0.197069 s` | `0.210212 s` | `1.06669x` |
| LBA train-only MAE | `0.092693 pK` | `0.331238 pK` | `-0.238545 pK` |
| LBA median step | `0.195336 s` | `0.202510 s` | `1.03673x` |

Both arms started with byte-identical states and predictions. The locked
router stayed byte-identical; the candidate router had finite, nonzero
gradients on every update and became active. QM9 passed its architecture-effect
checks, but LBA failed the `<=0.10 pK` capacity gate while the locked control
passed it. Gradient clipping remained high: `99.6%` for both QM9 arms and
`93.3%/94.1%` for the LBA control/candidate.

QM9 test was never evaluated. LBA validation and test were not loaded or
evaluated. LBA remains a 16-complex memorization result, not a generalization
claim.

## Decision

The minimal ELA API, exact global operator, sparse local residual, invariant
router mechanics, migration path, and wrappers remain implemented and
supported. Learned branch routing is **not empirically promoted**: the resource
packet and LBA capacity gate both failed. It remains an experimental mechanism
inside the canonical module, initialized to the exact admitted `G + L`
function. No multi-seed confirmation is authorized from this packet and no
accuracy or efficiency superiority claim is made.
