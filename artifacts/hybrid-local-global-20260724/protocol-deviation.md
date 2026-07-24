# Post-approval evidence-only runner repair

The approved PDBBind runner SHA-256 was
`cee7ede6170420f2f383c01e8ba4d0b4ddf18cdac1b374c70bdb8bfb40d4d271`.
Its reused legacy `_run_arm` function reported `edge_count_with_self=0` for
every arm whose name was not exactly `egnn`, even though `collate_graphs` and
`predict_graph_scalar` forwarded the supplied edge tensor to all models.

After the first completed run exposed this mismatch, the new runner was changed
only to overwrite that evidence field with the sum of its already frozen input
edge counts for every arm. The repaired runner SHA-256 is
`7e20d9d23aef12b61a4bd48352a7bc04eeeeb3b37bdbe9588b4f7bfa7103a75e`.
No model, optimizer, data, topology, seed, threshold, or resource setting
changed.

The full three-arm run was repeated under the original 900-second cumulative
ceiling. Its model metrics and all three final-state hashes were identical to
the first run. The corrected artifact records 153,029 input edges for every
arm. The first and repaired runs used 94.131 and 94.392 GPU-wall seconds,
respectively; total packet GPU wall, including CUDA smoke and QM9, remained
approximately 289.655 seconds.
