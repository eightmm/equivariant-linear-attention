# Deterministic sparse-topology contract (2026-07-27)

## Decision

The recorded cross-run ATOM3D-LBA topology defect is root-caused and repaired.
The candidate list is now built from exact float64 squared displacements against
the squared cutoff, so it is reproducible across processes, invariant under
translation and node permutation, and independent of the BLAS thread budget.

This unblocks the gate that every recent packet deferred: "before a cross-run or
multi-seed claim, the recorded one-edge topology reproducibility defect must be
repaired". It changes no model equation, no default, and no checkpoint schema.

## The defect

`segment_balanced_knn_edge_index` decided retention with
`torch.cdist(pos, pos) < cutoff`. For more than 25 points and `p=2`, `cdist`
defaults to the matrix-multiplication identity
`||a-b||^2 = ||a||^2 + ||b||^2 - 2 a . b`. In float32 that form loses accuracy
in proportion to the coordinate magnitude, and its reduction order depends on
the BLAS blocking, so the same coordinates give different bits in different
processes.

Measured on 400 uniformly placed points and the registered `6.0 Angstrom`
cutoff, against the exact float64 distance:

| coordinate offset | maximum `cdist` error | pairs wrongly admitted | pairs wrongly rejected |
| ---: | ---: | ---: | ---: |
| 0 A | 0.0110 A | 0 | 0 |
| 30 A | 0.0312 A | 0 | 0 |
| 100 A | 0.0884 A | 0 | 2 |
| 300 A | 0.2500 A | 6 | 26 |

Raw ATOM3D-LBA coordinates are not centered, so the operative regime is the
lower rows of that table, not the first. Two independent failures follow:

1. **Not translation invariant.** One complex and its translate produced
   different candidate lists, which contradicts the project's own local-edge
   contract.
2. **Not reproducible.** `cdist` output bytes differed across
   1, 2, 4, and 8 CPU threads for identical input, which is the mechanism behind
   the recorded 32,303,245 versus 32,303,244 edge-count drift at matching
   sample identity and seed.

The `kthvalue` tie handling was not the cause. Retaining every exact tie at the
kth boundary was already permutation safe and is kept.

## The frozen contract

For nodes `i, j` with float64 promotion of the stored coordinates:

```text
d2_ij   = sum_a (p_ja - p_ia)^2          # exact, fixed 3-term reduction
retain  = d2_ij < R_c^2                  # squared test, no square root
budget  = the k smallest d2 per relation, all exact ties retained
order   = receiver-major, then self, intra-segment, cross-segment,
          then ascending sender index
```

- The decision is evaluated in float64 whatever the storage dtype, so the
  candidate list does not depend on model precision.
- Self edges are always present. A receiver degree may exceed its budget only
  through exact ties, which is what keeps selection permutation equivariant.
- Retention uses squared distance, matching the frozen local-cutoff contract
  `||p_j - p_i||^2 < R_c^2`. The layer still applies its own strict cutoff to
  supplied candidates, so the retained set is a subset of this list.
- The implementation is chunked over receivers, materializes no `N x N`
  distance matrix, and is 1.2--4.4x faster than the replaced per-receiver loop
  at 500--5,000 nodes.

`topology_sha256` is now one shared definition in
`equivariant_attention.pdbbind`; the LBA runners delegate to it so no packet can
hash a candidate list its own way.

## Verification

`scripts/check.sh fast`: 624 tests, 88.66% coverage, ml smoke ok.

`tests/test_topology_contract.py` freezes: translation invariance, exact
agreement with the float64 cutoff graph when untruncated, no retained pair
outside the cutoff, survival of the layer's own cutoff filter, permutation
equivariance above the 25-point `cdist` threshold, thread-count invariance,
tie retention at the boundary, zero-budget and cross-segment budget behavior,
hash sensitivity to content and sample identity, and hash equality across fresh
subprocesses at different thread budgets.

Two of these tests failed before the repair and pass after it. The full official
ID30 identity was then rebuilt in fresh subprocesses:

```text
threads=1  edges=32302952  sha256=57f40fb1...  67.5 s
threads=4  edges=32302952  sha256=57f40fb1...  55.7 s
```

Frozen identity for the 3,507 train plus 466 validation complexes at revision
`f93dd2d150a47c270f624620f84e07451a158705`, `R_c = 6.0 A`, `intra_k = cross_k = 16`:

```text
edge_count      32302952
topology_sha256 57f40fb157e6416558db5507d95c3a5e4f828881e0bc92e142e1b85de802dc6c
```

Evidence: `artifacts/topology-contract-20260727/id30-identity.json`.

## Effect on recorded results

The repaired list has 293 fewer directed edges than the drifting `cdist` list
(32,302,952 versus 32,303,245), that is `9.1e-6` of the total. An earlier
exploratory squared-distance probe reported 32,302,953; it differed from this
contract by one edge because it did not promote to float64.

Consequences, stated plainly:

- Every historical LBA number stays valid **within** its own packet, because all
  arms in a packet consumed one shared in-memory list.
- Historical hashes `344158d8...` and `1eea0af8...` are legacy identities. They
  must not be used as expected values for new runs, and
  `scripts/run_lba_multiseed_confirmation.py` keeps its pinned legacy hash
  because it verifies already-recorded summaries rather than a fresh build.
- New LBA numbers are not bit-comparable with historical ones. Any new
  cross-packet comparison must rerun both arms under this contract.

`_make_synthetic_sample` also used the matrix-multiplication distance for its
target. It is now explicitly built with the direct form; below 26 nodes, which
covers the registered synthetic settings, the values are unchanged.

## Next experiment

The blocked confirmation can now run. `scripts/run_lba_clipping_confirmation.py`
implements it: seeds 41--43, clip 1 versus no clipping, one process, one hashed
topology, and thresholds fixed before any outcome is inspected (mean paired
improvement at least `0.020 pK`, at least two of three paired wins, worst paired
regression at most `0.050 pK`, latency and peak-allocation ratios at most
`1.05`). It aborts before training if `--expect-topology` does not match.

## Reproduction

```bash
uv run python scripts/verify_lba_topology.py \
  artifacts/topology-contract-20260727/id30-identity.json \
  --expect 57f40fb157e6416558db5507d95c3a5e4f828881e0bc92e142e1b85de802dc6c

uv run pytest tests/test_topology_contract.py -q
```
