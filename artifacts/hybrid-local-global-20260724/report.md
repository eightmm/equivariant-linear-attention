# Same-feature gated local/global architecture

Date: 2026-07-24

Packet: `hybrid-local-global-20260724`

Base commit: `c0a02e1243072fd2838ddd2591eae38673bb2720`

Branch: `codex/dynamic-coordinate-egnn-analysis`

## Question

Can a stronger short-range equivariant interaction improve the existing LGL
factorized-attention model without giving it privileged input features or
changing the exact edge-free `O(N)` global channel?

The causal comparison freezes raw node features, coordinates, targets, splits,
readout, optimizer, and candidate edges. Learned processing of those tensors
is part of the architecture and is allowed to differ. This separates
architecture quality from featurization quality; it does not imply that
featurization is unimportant for a practical molecular model.

## Intervention

Two opt-in changes were implemented:

1. a per-head gated local edge MLP over receiver/sender scalar state, existing
   RBF distance features, and invariant vector/relative-direction
   contractions. It emits scalar, receiver-vector, sender-vector,
   relative-vector, and symmetric-traceless rank-2 messages;
2. parameter-free normalization of scalar-message, angular, and
   persistent-tensor invariant families before the incumbent learned update.

Local sums use `1/sqrt(receiver_degree)` and expose log degree plus smooth
cutoff mass. Candidate-only module construction is isolated in an RNG fork so
every common incumbent parameter retains identical initialization. Both
options are false by default.

For the PDBBind-derived lane, a permutation-consistent candidate builder keeps
self edges and up to 16 same-segment plus 16 cross-segment neighbors inside
6 Angstrom, retaining all exact-distance ties. The three compared models
receive the same resulting topology.

## Verification

- Focused contracts: 71 passed.
- Repository fast gate: 491 passed, 88.65% coverage; ruff, compileall, and the
  CPU float64 ML smoke passed.
- Strict CUDA smoke: two updates, zero nonfinite gradient parameters, and
  7,048 nonzero gated-local gradient elements.
- Float64 tests cover disabled-state identity, common initialization, O(3),
  translation, node permutation, finite state/coordinate gradients, sparse
  topology permutation, cutoff, and self-edge behavior.
- No test labels were evaluated in either real-data lane.

## QM9 result

The frozen screen used cached QM9 `gap`, seed 42, strict deterministic CUDA,
500 updates, a 110,000/10,000 random-row train/validation split, FP32, and
identical precomputed 2.5-Angstrom candidates.

| arm | parameters | validation MAE | delta | mean pre-clip norm | clipped | peak CUDA |
|---|---:|---:|---:|---:|---:|---:|
| incumbent LGL | 153,285 | 0.709287 eV | 0 | 5.435 | 455/500 | 180.1 MB |
| gated local | 160,559 | 0.749135 eV | +0.039848 eV | 4.463 | 445/500 | 206.6 MB |
| gated + grouped | 160,559 | 0.683609 eV | -0.025678 eV | 3.771 | 455/500 | 208.0 MB |

The gated-only hypothesis was falsified. The combined package passed the
registered improvement and parameter gates: `0.025678 eV` lower MAE at
`1.04745x` parameters. It also reduced mean pre-clip norm by `30.61%`, while
the clipping fraction itself remained unchanged. Because grouped-only was not
run, the correct claim is an interaction/package effect, not that either
component is independently sufficient.

The screen is one model seed on an adaptively reused random-row validation
split. It supports follow-up, not a final accuracy or EGNN-superiority claim.

## ATOM3D-LBA/PDBBind train-only result

The selected package was compared with the incumbent and a near-parameter
private static EGNN on the cached first 16 ATOM3D-LBA train complexes. All
three arms used identical raw features, positions, ligand readout masks,
targets, deterministic batches, and 153,029 directed candidates. Validation
and test were not loaded.

| arm | parameters | threshold result | final observed train MAE | median step | peak CUDA |
|---|---:|---:|---:|---:|---:|
| incumbent LGL | 161,541 | pass at 1,800 / 49.82 s | 0.085251 pK | 25.00 ms | 268.9 MB |
| gated + grouped LGL | 168,815 | pass at 1,050 / 27.60 s | 0.098200 pK | 23.63 ms | 402.2 MB |
| private static EGNN | 167,260 | miss at 3,000 | 0.116225 pK | 4.26 ms | 326.1 MB |

The candidate crossed `0.10 pK` with `41.7%` fewer updates and `44.6%` less
time than the incumbent. A second strict run reproduced the same metrics and
final-state hashes. This repairs the earlier failure to overfit this frozen
subset and confirms that the data/readout/gradient path is functional.

It is not evidence of affinity generalization. The candidate also used
`1.50x` the incumbent peak allocation and remained `5.55x` slower per step
than EGNN on these 223--721-node complexes. The result therefore supports
capacity/convergence, not small-graph efficiency or official EGNN superiority.

## Verdict and remaining gaps

The architecture is materially stronger but not complete. The selected
gated-plus-grouped package is retained as the preferred experimental LGL
variant, while public defaults remain unchanged until:

1. a matched multi-seed, longer QM9 confirmation establishes that the
   `0.025678 eV` screen gain is seed-robust;
2. a leakage-controlled ATOM3D ID30 or protein/ligand-cluster-disjoint
   validation study establishes affinity generalization;
3. end-to-end neighbor construction is replaced or measured for large point
   clouds, since the current no-dependency segment-balanced builder uses a
   quadratic distance matrix outside the model hot path;
4. chirality-sensitive tasks receive an explicit parity-odd/pseudoscalar
   pathway; the current scalar outputs are O(3)-invariant and cannot
   distinguish isolated mirror pairs;
5. feature enrichment is studied only after the architecture is locked.
   Formal charge, aromaticity, bonds, residue identity, and protein embeddings
   could change practical affinity performance, but adding them inside this
   packet would confound the architecture comparison.

## Evidence

- `scope.md`: frozen hypothesis, gates, and compute boundary.
- `qm9-screen/summary.json`: QM9 decision and paired identities.
- `pdbbind-overfit.json`: train-only capacity, resources, topology, and hashes.
- `cpu-verification.json`, `cpu-smoke.json`, `cuda-smoke.json`: software and
  execution checks.
- `protocol-deviation.md`: the post-approval evidence-field repair and
  deterministic rerun.
- `results-summary.json`: compact machine-readable verdict.
