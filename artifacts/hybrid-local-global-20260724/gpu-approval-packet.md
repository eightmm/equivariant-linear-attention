# Local GPU approval packet

- Target: local CUDA device 0, reported by project PyTorch as NVIDIA RTX PRO
  6000 Blackwell Max-Q Workstation Edition.
- Environment: existing `uv.lock`, `uv run --locked`, PyTorch
  `2.12.1+cu130`; no package or lock change.
- Inputs: cached `data/qm9` and cached immutable ATOM3D-LBA revision
  `f93dd2d150a47c270f624620f84e07451a158705`; no network, download, private
  transfer, validation/test label, or checkpoint publication.
- Code:
  - `scripts/run_hybrid_local_global_qm9.py`,
    SHA-256 `dce99813444fcdc73617a5ff439d86fa28d1ea53957c51af9dacfbadedb574d2`;
  - `scripts/run_hybrid_local_global_pdbbind.py`,
    SHA-256 `cee7ede6170420f2f383c01e8ba4d0b4ddf18cdac1b374c70bdb8bfb40d4d271`;
  - `src/equivariant_attention/moment.py`,
    SHA-256 `4d286ecdacec1f6006e7c071d3aeca51b4f11eda4b616a668fcd67f1f49f174f`;
  - `src/equivariant_attention/pdbbind.py`,
    SHA-256 `7809a766033384e42e464e96f620cf3f7303ac8e06f6fc10b3cda7dd483d9ecd`.
- Smoke command:
  `uv run --locked python scripts/train_compare.py --dataset synthetic
  --num-samples 16 --train-size 10 --val-size 3 --batch-size 4 --steps 2
  --hidden-dim 64 --num-layers 3 --num-heads 4 --routing lgl
  --gated-local-transport --grouped-invariant-normalization
  --determinism strict --device cuda --skip-test-eval
  --metrics-out artifacts/hybrid-local-global-20260724/cuda-smoke.json`.
- QM9 command:
  `uv run --locked python scripts/run_hybrid_local_global_qm9.py
  artifacts/hybrid-local-global-20260724/qm9-screen --device cuda
  --steps 500 --budget-seconds 300`.
- Conditional ATOM3D-LBA command:
  `uv run --locked python scripts/run_hybrid_local_global_pdbbind.py
  artifacts/hybrid-local-global-20260724/pdbbind-overfit.json --device cuda
  --max-steps 3000 --budget-seconds 600 --candidate <QM9 diagnostic arm>`.
- Resource ceiling: one local GPU, at most 900 cumulative GPU-wall seconds,
  expected peak below 2 GiB, CPU preprocessing below 2 minutes, artifact growth
  below 100 MiB.
- Validation: finite two-update smoke; QM9 improvement/regression/parameter
  gates from `scope.md`; ATOM3D-LBA best train MAE `<=0.10 pK`; test remains
  disabled.
- Cancellation: each orchestrator launches a distinct process group, first
  sends `SIGTERM`, waits two seconds, then targets only that group with
  `SIGKILL` if necessary. Every completed arm writes its JSON before the next
  arm starts.
