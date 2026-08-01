# ELABatch

`ELA.collate` and the package-level `collate_graphs` return `ELABatch`, a plain
`dict` subclass with device-transfer convenience. It is not a framework-specific
graph object and has no dependency beyond PyTorch.

```python
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    pin_memory=True,
    collate_fn=ELA.collate,
)

for batch in loader:
    batch = batch.to(
        "cuda",
        dtype=torch.bfloat16,
        non_blocking=True,
    )
    output = model(batch)
    loss = criterion(output["graph_irreps"], batch["target"])
```

## Transfer semantics

```python
batch.to(device)
```

moves tensors without changing floating dtypes.

```python
batch.to(device, dtype=torch.bfloat16)
```

converts node features, targets, conditions, and floating semantic-order values
to BF16 while keeping positions in FP32 by default.

```python
batch.to(
    device,
    dtype=torch.float32,
    geometry_dtype=torch.float64,
)
```

explicitly controls geometry precision.

Integer graph metadata, edge indices, relation IDs, groups, and boolean masks
retain their dtypes.

`batch.pin_memory()` pins CPU tensors when a pinned-memory allocator is
available. Non-tensor sample IDs remain unchanged.

## Metadata

```python
batch.num_nodes
batch.num_graphs
batch["sample_ids"]
batch["target"]
```

`ELABatch` remains an ordinary mapping, so standard dictionary operations,
serialization choices, and custom training-loop fields continue to work. The
model accepts known execution fields and ignores `target` and `sample_ids`.
Unknown keys fail closed to catch misspelled model inputs.
