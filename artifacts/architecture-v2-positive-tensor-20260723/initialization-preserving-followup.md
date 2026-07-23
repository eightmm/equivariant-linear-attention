# Initialization-preserving architecture-v2.1 follow-up

Date: 2026-07-23

## Trigger

The registered architecture-v2 screen remains a valid negative result for its
exact source hash. Post-screen diagnostics found a design confound: enabling
the tensor kernel constructs two random `_ChannelMix` modules before incumbent
modules later in each layer. Those allocations advance the global RNG, so the
same model seed no longer gives the tensor candidate the same initial values
for its common incumbent parameters. Changing `eta` from `0.05` to `0.001`
therefore cannot compare the tensor option with the persistent-only arm under
paired common initialization.

## Single repair

Construct the optional tensor query/key projections inside a CPU RNG fork that
restores the incumbent generator state immediately afterward. This preserves
the existing state-dict order while preventing the new random allocations from
shifting any incumbent initialization in the current or later layers. Names,
shapes, forward equations, parameter count, defaults, and the disabled state
schema remain unchanged. A new test requires every common state entry of a
persistent-`2e` incumbent and tensor-kernel candidate to be bit-identical for
the same seed.

## Re-evaluation

After focused/fast/strict-CUDA gates, repeat the original four-arm QM9
500-update screen in a fresh `v2.1` artifact directory with the original
thresholds unchanged. Any original C4/C5 claim remains rejected; the follow-up
gets a separate verdict. If and only if the repaired candidate passes the same
screen gate, run the original five-seed confirmation. Re-run the train-only
ATOM3D-LBA three-arm comparison because initialization affects that result too.
No test labels, new data, new dependency, or extra architecture lever is
admitted.
