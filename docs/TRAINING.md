# Training

The implemented probe uses one AdamW optimizer with constant learning rate.
Defaults are `lr=3e-4`, `weight_decay=0.01`, and gradient-norm clipping at 1.0.
No Muon optimizer or WSD schedule is implemented or claimed.

Targets are normalized with statistics fitted on training samples only.
Validation and test predictions are transformed back to original units before
MAE/RMSE aggregation.

```bash
uv run python scripts/train_compare.py \
  --dataset synthetic \
  --steps 100 \
  --hidden-dim 64 \
  --num-layers 3 \
  --num-heads 4 \
  --split-seed 42 \
  --model-seed 42 \
  --skip-test-eval
```

`--amp-dtype bf16` enables autocast. CUDA mixed-precision claims require the GPU
smoke and run-specific environment metadata. Long training, schedulers,
distributed execution, checkpoint selection, and resume are outside the current
prototype contract.

For the registered P1 screen, `--no-linear-kernel` selects the quadratic-only
kernel and `--no-key-balancing` selects direct row normalization. The four
combinations share one implementation and are recorded in `run_config`.
