# Soft normalization v2 and grouped-only attribution

Confirmed follow-up scope for the 2026-07-24 hybrid local/global packet.

## Question

Does replacing the cutoff-squared normalizer with a cutoff-mass normalizer fix
singleton-edge attenuation without breaking equivariance or numerical
stability, and was the prior QM9 gain attributable to grouped invariant
normalization rather than the gated local transport?

## Architecture intervention

- For every applicable local or interaction aggregation, define
  `C_i = sum_j c_ij` and use
  `m_i = sum_j c_ij phi_ij / sqrt(1 + C_i)`.
- Retain `S_i = sum_j c_ij^2` only as a smooth diagnostic/mass feature where it
  is already exposed. It must not determine the message divisor.
- Apply the rule to normalized edge-conditioned transport, gated transport,
  pairwise local content, and ligand-pocket interaction aggregation.
- Cast stable-normalized interaction-readout states back to the model scalar
  dtype before low-precision linear layers.
- Defaults, public flags, raw features, topology, exact global linear
  attention, and readout semantics otherwise remain unchanged.

## Required software evidence

- A singleton message decreases monotonically as distance approaches the
  cutoff and is exactly zero at the cutoff.
- Equal active neighbors scale as `d / sqrt(1 + d)` at unit cutoff weight.
- Outputs and coordinate gradients are finite at cutoff ratios
  `0.5, 0.7, 0.9, 0.95, 0.99, 1.0`.
- Existing O(3), translation, permutation, graph-isolation, and cutoff
  continuity checks remain green.
- Direct bfloat16 interaction-readout forward/backward is finite on CUDA.

## Frozen real-data screen

Run a strict deterministic QM9 `gap` architecture screen with the cached
random-row split, 110,000 train rows, 10,000 validation rows, batch size 64,
seed 42, static LGL routing, precomputed identical local candidates, FP32,
500 updates, and no test evaluation. Compare the exact 2x2:

1. incumbent: gated off, grouped off;
2. grouped only: gated off, grouped on;
3. gated only: gated on, grouped off;
4. combined: gated on, grouped on.

All arms receive identical raw features, coordinates, targets, split, batches,
and candidate edges. Initial common parameter tensors must hash identically.

## Decision rule

- This is a one-seed attribution screen, not a promotion or generalization
  claim.
- If grouped-only validation MAE is within `0.005 eV` of combined, prefer the
  simpler grouped-only intervention.
- Attribute an interaction benefit to the combined model only if combined
  beats grouped-only by at least `0.010 eV`.
- Record validation MAE, clipping fraction, mean/max pre-clip gradient norm,
  parameters, runtime, and peak CUDA allocation for every arm.
- Multi-seed confirmation is deferred to a separately registered packet and
  only applies to a winning arm.

## Boundaries and compute envelope

- Cached QM9 data only; no download, dependency change, validation retuning, or
  test-label access.
- One local GPU, at most 300 seconds cumulative GPU wall time for the four-arm
  screen plus a focused low-precision smoke.
- Ligand-centered interaction readout v2, chirality features, parity-complete
  hidden irreps, and production neighbor rebuilding are explicitly deferred.
- Automatic GitHub Actions remain disabled; verification is local.
