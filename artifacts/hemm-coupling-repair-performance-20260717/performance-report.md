# Registered performance decision

The M=1 `lgl` route passes the preregistered QM9 validation and CUDA resource
rules against matched M=1 `ggg` on clean commit `a8bda618`.

| seed | GGG MAE (eV) | LGL MAE (eV) | improvement (eV) |
|---:|---:|---:|---:|
| 41 | 0.530218 | 0.507433 | 0.022785 |
| 42 | 0.589037 | 0.572078 | 0.016958 |
| 43 | 0.631137 | 0.512559 | 0.118578 |


- Mean improvement: `0.052774 eV` (threshold `0.010 eV`).
- Positive seeds: `3/3` (threshold `2/3`).
- Worst seed improvement: `0.016958 eV` (floor `-0.020 eV`).
- Mean measured training time: GGG `48.861s`, LGL `42.353s`
  (`-13.32%`).
- Clean five-process CUDA benchmark: worst latency change
  `-7.08%`; worst peak-memory change
  `6.31%`, both below `+20%`.
- All six rows have matching source/data/split/schema/parameter provenance;
  each paired seed has an identical initial-state hash.
- Gradient accounting is stable across seeds: both routes report
  `151,400` parameter elements with a gradient, while exact
  nonzero-gradient elements are `151,080` for GGG and
  `150,504` for LGL. This route-dependent difference is
  recorded and is not a parameter/schema mismatch.
- `test_evaluated=false` in all rows and no test MAE/RMSE key is present.
- GPU usage is a conservative operator bound, not an exact timer sum: saved
  artifact timers total `496.168s`; the reported ceiling is
  `570s` (`9.5 min`), leaving `73.832s` for the GPU
  check and process overhead whose raw duration receipt was not recorded.

Interacting M=4/M=8 remains blocked by Stage-0. This is adaptive three-seed
validation-only evidence on a random-row warm split, not test-set or
cold-molecule evidence and not an automatic public-default change.
