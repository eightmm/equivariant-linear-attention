# Training

The implemented probe uses one AdamW optimizer with constant learning rate.
Defaults are `lr=3e-4`, `weight_decay=0.01`, and gradient-norm clipping at 1.0.
No Muon optimizer or WSD schedule is implemented or claimed.

Targets are normalized with statistics fitted on training samples only.
Validation and optional test predictions are transformed back to original units
before MAE/RMSE aggregation.

```bash
uv run python scripts/train_compare.py \
  --dataset synthetic \
  --steps 100 \
  --hidden-dim 64 \
  --num-layers 3 \
  --num-heads 4 \
  --routing ggg \
  --memory-count 1 \
  --split-seed 42 \
  --model-seed 42
```

The public defaults are `ggg`, one memory, memory interaction off, radial trace
off, fixed pair floor, one balancing cycle, and the alignment-linear term on.
The registered architecture flags are:

```text
--routing {ggg,lgl,lll}
--memory-count {1,4,8}
--memory-interaction
--radial-trace
--no-alignment-linear-term
--no-key-balancing
--kernel-floor-mode {fixed,inverse_graph_size}
--evaluate-test
```

Interacting multi-memory runs require three-layer `lgl`. The
`inverse_graph_size` kernel-baseline mode requires `--no-key-balancing`; it
scales `(c + beta + delta*beta*t)` by `1/N_g`, not only `c`. Test evaluation is
off unless `--evaluate-test` is supplied, and it must remain off throughout
adaptive selection.

Before scheduling any M=4/M=8 interacting-memory training, run the label-free
mechanism gate:

```bash
uv run python scripts/probe_memory_activation.py \
  --memory-counts 4 8 --device cpu --dtype float64
```

The current implementation returns `block_interacting_memory_arms`; the flags
remain useful for bounded diagnostics but are not eligible for broader training
until a separately preregistered mechanism passes Stage 0.

`--amp-dtype bf16` enables autocast while coordinates remain float32. CUDA
mixed-precision claims require the GPU smoke and run-specific environment
metadata. Long training, schedulers, distributed execution, checkpoint
selection, and resume are outside the current prototype contract.

The bounded ATOM3D-LBA capacity check is a separate train-only runner:

```bash
uv sync --locked --extra qm9 --extra pdbbind
uv run --locked --extra qm9 --extra pdbbind python \
  scripts/run_registered_pdbbind_overfit.py \
  artifacts/pdbbind-overfit-persistent2e-20260723/registered-result.json
```

It freezes 16 train complexes, batch size 2, AdamW at `1e-3`, no weight decay,
gradient clipping at 1.0, at most 3,000 updates, and at most 1,800 cumulative
seconds. The attention arm is edge-free GGG with four persistent `2e`
channels; the private static EGNN gets directed 6-Angstrom radius candidates
and the closest parameter-matched width. Success is train MAE at most
`0.10 pK`. This is an overfit/wiring check only: it performs no validation or
test evaluation and cannot support a generalization claim.
