# Diagnostic and repair implementation evidence

## Changed mechanism support

- Assignment diagnostics now separate marginal entropy, conditional entropy,
  and normalized mutual information while retaining the historical conditional
  entropy field for compatibility.
- Center diagnostics report per-head RMS spread and off-diagonal distance and
  cutoff-normalized quantiles. Coupling diagnostics report centered-Frobenius
  variation in addition to nonunit off-diagonal fraction.
- The Stage-0 probe now evaluates ones, radial, identity, and fixed residual
  couplings through the actual middle global transport. It captures individual
  middle message terms, post-middle scalar/vector/tensor state, fixed-probe
  gradients with respect to scalar/vector/position inputs, and full output.
- Four deterministic graph roles, widths 16/64, seeds 401--403, and M=4/M=8
  can be executed and aggregated by one suite command.
- The feedback's shared invariant MLP router is allocated for every route and
  M, uses fixed invariant DCT slot codes, is nonzero initialized, and remains
  an exact M=1 execution bypass. The identity-residual coupling primitive keeps
  bounds, symmetry, exact unit diagonal, and slot-permutation covariance.

No residual mix was added to the public model configuration because none of
the registered candidates passed the frozen Stage-0 matrix. Thus there is no
post-hoc lambda or inactive public knob and no memory arm is admitted.

## TDD evidence

The initial focused run failed during collection because the specified center
diagnostic did not exist. Subsequent red phases captured the missing graph
suite/RMS helpers, mechanism argument, suite aggregator, runtime-router
diagnostic connection, and lane width metadata. After minimal implementation:

```text
uv run --locked pytest -q \
  tests/test_diagnostics.py \
  tests/test_local_memory.py \
  tests/test_probe_memory_activation.py \
  tests/test_train_compare.py

117 passed in 2.06s
```

Focused coverage includes entropy limiting cases, center invariance, coupling
variation, residual bounds/diagonal/permutation, shared router state schema,
actual transport/gradient capture and helper restoration, strict JSON, and
bounded training diagnostics. Broader exact forward/gradient, M=1, node/slot
permutation, O(3), translation, and cutoff tests remain in the project suite and
are run again at the final verification gate.
