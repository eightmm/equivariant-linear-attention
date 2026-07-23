# Exploratory post-screen diagnosis

Date: 2026-07-23

This diagnosis is post-outcome and cannot promote C4/C5. It uses the same
strict-CUDA QM9 split, seed 42, 500 updates, and disabled test evaluation.

The frozen tensor arm regressed by `0.0809 eV`. Two additional arms separate
the persistent-state effect from the shifted-kernel initial constant:

1. `hidden_tensor_dim=4`, unit scalar content, tensor-product kernel disabled;
2. the same model with the tensor-product kernel enabled but
   `tensor_kernel_init=0.001`.

Prediction: if arm 1 is already poor, persistent `2e` state rather than the
shifted kernel dominates the regression. If arm 1 recovers but arm 2 remains
poor, even a small shifted constant is harmful. If arm 2 recovers toward arm 1,
the registered `eta=0.05` caused excessive early dilution. No threshold will
be changed and no confirmation follows from these exploratory arms.
