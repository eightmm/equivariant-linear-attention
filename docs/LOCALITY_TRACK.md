# Locality track

Record of how contact-scale locality entered the edge-free core, what was
deliberately left out, and what is still open. The mathematics of what
landed is in `docs/ARCHITECTURE.md` sections 3 and 5; this file records
the decisions and their evidence.

## Diagnosis

Two independent defects made contact-scale locality inexpressible.

**Absolute scale was destroyed.** Per-segment RMS normalization removed
length entirely, so the core had no representation of "5 angstrom". Every
absolute-scale signal in the PSR runs that scored well came from the
offline dense neighbour-count feature, which lives outside the core.

**The Mercer sector was truncation-locked.** The global sector is a
degree-four truncation of `exp(-gamma ||x - y||^2)` expanded at the
segment centroid. The truncation is valid only while `2 gamma |x||y|`
stays small, and with `x` measured from the centroid `|x|` is of order
the segment radius `R`. Bandwidth is therefore floored near `R`; below
it the polynomial grows instead of decaying. The atlas, the other
spatial sector, was running at two charts.

## Landed

Two commits on `main`, both edge-free and node-linear.

- `39066f5` chart-recentered local relation and absolute scale. Adds the
  fourth relation sector, absolute coordinates in length-scale units,
  six-channel bounded radial invariants, and equivariant soft
  farthest-point chart seeding.
- `6b30c02` `chart_density`, a node-linear in-model soft neighbour-count
  channel.

Measured at initialization on real PSR structures, seeding rather than
learned logits alone moves chart centre dispersion from 3-5 to 15-16
angstrom, near/far kernel selectivity from 22-26 to 76-113, and
per-structure rank agreement with the offline 5 angstrom neighbour counts
from 0.44-0.49 to 0.62-0.66.

These are untrained initialization measurements. Accuracy is not
validated: the PSR runs that separate absolute scale (P0) from the local
sector (P1) are still pending, and `num_local_charts = 0` exists so the
scale-only ablation is available.

## Deliberately excluded

Branch `agent/local-geometry-mercer`, draft PR #7, holds a second and
incompatible attack on the same problem. Its later half
(`29dba5e`..`3cef4f9`) builds a genuine compactly supported local
geometry: per-segment `cdist` plus `topk` produce a kNN
`(source, receiver)` support, and on top of it sit Wendland C2 windows,
central cumulants through order four, a reproducing local jet, and a
local body algebra. It is conceptually richer than the Mercer sector and
has real hard cutoffs.

It cannot be the canonical path, because a kNN support is an edge list,
which breaks three non-negotiable invariants at once: no explicit or
inferred edges, no sparse gather/scatter path, and node-linear memory
(the support is `O(kN)` pairs). The per-segment Python loop also defeats
batching. There is no contract-preserving adaptation of it.

It is therefore integrated behind `local_points`, default `0`, and
declared non-canonical in `PROJECT.md`. At the default nothing is
constructed: a default-config model has the same 202 `state_dict`
tensors, the same 109641 parameters, and bit-identical seeded forward
output as before the integration. Its value is as an upper bound, which
is the number the edge-free sectors have to chase; `train_psr.py
--local-points K` makes that measurable.

The model refactor on that branch (`model/config.py`, `model/network.py`,
`nn/equivariant_ffn.py`) came with it, re-derived from this repository's
`ela.py` rather than taken as-is, and landed as a separate
behavior-preserving commit.

### What was not taken

The branch head is red, and three artefacts there are abandoned
work-in-progress rather than finished work.

- `nn/local_projection_v2.py` reads `body.fourth_scalar`, which
  `LocalBodyFeatures` does not define. Nothing imports it, so it has
  never executed. `local_projection.py` is the live implementation and
  is what was ported; it runs correctly at every width tried, including
  the default `width=128` where `moment_rank=8` and `body_rank=6`
  differ, which is the case the abandoned rewrite was meant to address.
- `tests/test_local_projection.py` imports `LocalCumulants`, a type
  that was never implemented, and constructs it with a `scale` field
  that `MomentFeatures` does not have. Patched to the real type it then
  fails against `local_projection.py` on an `eps` argument, because it
  was written for the v2 signature. It has never run.
- `contract()` on that branch renames `relative_moment_order` to
  `global_relative_moment_order`, which breaks `test_api.py`. The key
  is kept unrenamed here; the new keys are additive only.

The last coherent state of the branch is `8f5f9d5`.

## Deferred

**Re-derive the local moment bank on the chart-recentered kernel.** The
first half of PR #7 (`efa9b1d`..`bfb56ca`) added an edge-free local
Mercer *moment bank*, which is complementary to what landed: it puts
locality in the moment path rather than the relation path, so the two
compose. It is not cherry-picked as-is because its kernel expands at the
centroid and therefore carries exactly the truncation lock diagnosed
above; its own documentation records the limitation. Landing an
unmeasured sector with a known-flawed kernel would also confound the
pending P0/P1 ablations. The idea is worth redoing on the recentered
absolute-scale kernel once those ablations have reported.

Note that phase two rewrote both `nn/local_geometry.py` (362 lines to 74)
and its tests, so the moment-bank implementation exists only at the phase
one commits, not at the branch head.

**Reach hard-cutoff bandwidths.** `chart_density` cannot currently
reproduce the offline neighbour counts, because the recentered envelope
floors the bandwidth at the chart radius and the seeding degenerates
above roughly 32 charts. Tiling more finely without degenerate seeds is
the open problem.
