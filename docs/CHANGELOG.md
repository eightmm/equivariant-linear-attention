# Changelog

## Unreleased

### Unified graph API

- [api] expose exactly `ELA` and `ELAGraph` from the package root;
- [api] standardize all execution on `ELAGraph -> ELA -> ELAGraph`;
- [api] use one public source-to-target edge convention;
- [api] move batching to `ELAGraph.collate`;
- [api] place node, graph, coordinate, and displacement predictions on the
  returned graph instead of a separate output class;
- [api] declare coordinate updates with `update_positions` at model construction;
- [api] remove public raw-tensor, dictionary, PyG-object, padded, refiner, and
  prepared-batch entry points;
- [api] retain packed receiver-major CSR only as a private numerical IR.

### Correctness and architecture

- [geometry] omit automatic self edges;
- [geometry] record preparation provenance and invalidate stale radius topology;
- [geometry] support exact Verlet-skin reuse in advanced configuration;
- [architecture] restore fixed `global + local` canonical fusion;
- [cleanup] remove the retired learned branch-router implementation and its
  standalone tests instead of carrying an unreachable architecture variant;
- [architecture] scale geometric carrier capacity with model width;
- [architecture] add per-copy equivariant normalization, pseudoscalar-aware
  global routing, fixed radial shells, unit-direction local angular features,
  relation-conditioned transport, and second-moment chiral carriers;
- [architecture] make coordinate refinement stagewise: each selected layer
  boundary updates geometry while preserving the hidden state, and the public
  switch selects every layer boundary;
- [irreps] add physically correct compact symmetric-traceless tensor metrics.

### Performance

- [performance] keep PyTorch as the automatic backend;
- [performance] retain Triton as an explicit contract-tested backend;
- [performance] make neighbor policy part of reproducible geometry configuration;
- [validation] add a source-bound, exact-argv completion runner with separate
  GPU, data, and CPU-finalization authority phases and a hard G1 stop gate;
- [validation] defer current CUDA and real-data receipts while the workstation
  GPU is occupied, without converting CPU evidence into a speed claim.
