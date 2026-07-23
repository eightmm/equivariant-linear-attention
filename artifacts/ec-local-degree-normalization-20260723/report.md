# Edge-conditioned local degree normalization

## Verdict

The implementation contract passed, but the registered clipping hypothesis
failed. The square-root receiver-degree option remains opt-in and the default
remains the original sum.

| Measure | EC-LGL sum | EC-LGL sum/√degree | Candidate − baseline |
|---|---:|---:|---:|
| Validation MAE (eV) | 0.744964 | 0.715997 | -0.028967 |
| Clipping fraction | 0.920 | 0.916 | -0.004 |
| Clipped updates | 460/500 | 458/500 | -2 |
| Mean pre-clip norm | 6.1541 | 6.7255 | +0.5715 |
| Maximum pre-clip norm | 44.1013 | 53.5068 | +9.4056 |
| Training/evaluation wall time (s) | 16.561 | 16.739 | +0.178 |
| Peak CUDA allocation (bytes) | 168440320 | 168450560 | +10240 |

The frozen gate required at least `0.05` absolute clipping reduction and no
more than `+0.020 eV` validation regression. The accuracy guard passed, but
clipping fell by only `0.004`; therefore the combined gate failed.

## Implementation

For each receiver `i`, the opt-in path divides all four non-self
edge-conditioned local message sums—scalar, sender vector, relative vector,
and symmetric-traceless tensor—by
`sqrt(max(nonself_incoming_candidate_count_i, 1))`. The degree is shared across
heads and message types and introduces no parameter.

The unnormalized path remains the default. Default versus explicit-disabled
models have identical parameter schema, initialization, and outputs. Float64
tests cover the explicit normalized reference, zero-degree receivers, O(3),
translation, node permutation, edge order, graph relabeling/isolation, and
finite nonzero gradients.

## Interpretation

Simple receiver-degree normalization is not a sufficient repair for the
high-clipping pathology in this 500-step EC-LGL screen. The higher mean and
maximum pre-clip norms directly contradict a broad claim that the intervention
stabilized overall gradient scale.

The `0.028967 eV` validation improvement is encouraging but descriptive: it is
one strict deterministic seed on the adaptively reused random-row validation
split and the primary diagnostic gate failed. It justifies retaining the
option for a future independently registered multi-seed comparison, not making
it the default or claiming improved generalization.

A plausible next hypothesis is that clipping is dominated by another parameter
group or that learning compensates for reduced local message scale. That
hypothesis was not measured here and is not a conclusion.

## Evidence boundary

- Cached QM9 `gap`, random-row split seed 42.
- One model seed, FP32, strict CUDA, precomputed 2.5-Angstrom candidates.
- Identical source, initial state, state schema, data hashes, and split hashes.
- Both raw runs share execution-time aggregate source SHA-256 `c4b28d15…`;
  the final tree's aggregate hash differs because `PROJECT.md` gained the
  post-outcome verdict. Material Python code was not changed after execution
  and its final file hashes are recorded in the manifest.
- Train-only target normalization.
- Validation used for the registered screen; test evaluation disabled.
- No EGNN, multi-seed, cross-hardware, or generalization claim.
