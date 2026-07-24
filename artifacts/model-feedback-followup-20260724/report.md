# Model-feedback follow-up report

Date: 2026-07-24

Reference base: `626275a1c785b15c6016af678782b2b08a7f1e22`

## Decision

The review correctly identified a cutoff discontinuity, an underspecified
dynamic-neighbor contract, and a missing task-specific protein-ligand readout.
Those three items were implemented behind compatibility-preserving defaults.
The packed local reduction reduced scatter calls but did not improve profiled
time or memory. The interaction readout trained correctly on real cached data
but did not beat ligand mean pooling, so it remains experimental.

Automatic GitHub Actions triggers are disabled. The existing CPU workflow is
manual-only through `workflow_dispatch`; local `scripts/check.sh fast` remains
the release gate.

## Implemented

1. **Smooth local normalization**
   - Replaced raw receiver degree with
     `S_i = sum_j f_cutoff(u_ij)^2` in gated local, normalized
     edge-conditioned local, and pairwise local content.
   - Mass features are now
     `[log1p(sum f_cutoff), log1p(sum f_cutoff^2)]`.
   - The cutoff value and first coordinate derivative therefore remain
     continuous as a candidate enters or leaves the retained edge set.

2. **Packed receiver reduction**
   - Scalar, polar-vector, relative-vector, symmetric-traceless tensor, and
     mass contributions share one `index_add` per gated local stage.
   - This preserves `O(E)` local transport at fixed width.

3. **Dynamic-neighbor policy**
   - `error` (default): reject moving coordinates plus external candidates.
   - `fixed`: explicitly accept a fixed, approximate candidate topology.
   - `rebuild`: discard external candidates and rebuild complete same-graph
     candidates from current coordinates before every local stage.

4. **Task-dependent readout**
   - Added `readout_mode=mean|sum|interaction`.
   - `interaction` pools ligand, pocket, and cross-interface content.
   - Six learned polar interface moments form two pseudoscalars; only their
     parity-even products reach the scalar property head.
   - The final interaction projection is zero initialized, so its initial
     prediction is exactly the ligand mean baseline.
   - Interface transport is `O(E)` once candidates are supplied; the exact
     no-edge fallback still uses quadratic candidate discovery.

5. **Reproducible real-data runner**
   - `scripts/run_interaction_readout_pdbbind.py` compares only the readout on a
     common gated+grouped backbone.
   - It reads cached ATOM3D-LBA train rows 0--15 and never evaluates validation
     or test labels.

## CUDA reduction diagnostic

Command shape:

```bash
uv run python scripts/profile_train_step.py \
  --model gated_static --nodes 2048 --edge-multiplier 64 \
  --device cuda --warmup 2 --repeats 3
```

Both runs used initial-state SHA-256
`21ca283b0536c7d75fceb24658374111df16f143b20f98b7bd311dc850f3023c`.

| metric | base `626275a` | follow-up | change |
|---|---:|---:|---:|
| `aten::index_add` calls | 75 | 45 | -40.0% |
| forward device time | 15.423 ms | 15.782 ms | +2.3% |
| peak CUDA allocation | 1,257,324,544 B | 1,298,436,608 B | +3.3% |

The call-count objective passed and both resource changes stayed inside the
predeclared 10% guard. This is not evidence of a speed or memory win: packing
adds a temporary concatenated edge tensor, and profiler time includes profiler
overhead.

Raw files:

- `profile-gated-main-626275a.json`
- `profile-gated-fused.json`

## ATOM3D-LBA train-only diagnostic

The strict-deterministic run used revision
`f93dd2d150a47c270f624620f84e07451a158705`, train rows 0--15, 153,029 common
directed candidates, batch size 2, AdamW at `1e-3`, clipping at 1.0, and 1,000
updates. Initial predictions were bitwise identical:
`f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b`.

| metric | mean | interaction |
|---|---:|---:|
| parameters | 168,815 | 173,611 |
| initial train MAE | 2.025625 pK | 2.025625 pK |
| best observed train MAE | 0.088952 pK | 0.111377 pK |
| final train MAE | 0.088952 pK | 0.183435 pK |
| median train step | 24.01 ms | 26.99 ms |
| peak CUDA allocation | 424,491,008 B | 423,822,848 B |

The head is wired, differentiable, deterministic, and capable of fitting the
small set. It did not improve the registered train-capacity proxy and increased
step latency by 12.4%, so it is not promoted. Because this is a 16-row
train-only run, neither arm has affinity-generalization evidence.

Raw file: `pdbbind-interaction-readout.json`.

## Repository verification

- Focused cutoff/coordinate/readout suite: `53 passed`.
- Local release gate: `scripts/check.sh fast` passed with `500 passed`,
  88.78% coverage, Ruff, compile, and CPU float64 ML smoke.
- CUDA smoke: bf16 and fp32 both passed.
- No automatic hosted CI run is required or triggered by push/PR.

## Review items not promoted

- A full parity-complete `0o/1e/2o` or arbitrary-`l` hidden backbone was not
  added. The new head is parity-aware but its scalar output remains O(3)
  invariant; chirality benefit still requires a target-disjoint validation
  study.
- A cell-list/Verlet-skin neighbor backend remains absent. Exact `rebuild` is
  correctness-first and quadratic per graph.
- Chemistry-enriched features were not mixed into this architecture lane.
  ATOM3D-LBA features remain identical across arms so this result isolates the
  readout change; a practical affinity model should test richer features in a
  separate lane.
- The persistent clipping rate and EGNN small-graph speed gap remain open.
  Packed scatter alone cannot remove the edge-MLP/autograd cost.

## Claim boundary

This packet supports cutoff regularity, explicit dynamic-neighbor semantics,
O(3)/translation/permutation behavior, reduced scatter count, and train-only
readout wiring. It does not support QM9 accuracy improvement, PDBBind
generalization, EGNN superiority, chirality improvement, or universal
speed/memory gains.
