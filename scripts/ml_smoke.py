"""CPU-only model smoke for the project verification contract."""

import sys

import torch

from equivariant_attention import (
    EquivariantAttention,
    EquivariantAttentionConfig,
    RichEquivariantAttention,
    RichEquivariantAttentionConfig,
    prepare_for_inference,
)


def main() -> int:
    device_name = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    use_bf16 = "bf16" in sys.argv[2:]
    use_compile = "compile" in sys.argv[2:]
    if device_name == "cuda" and not torch.cuda.is_available():
        print("ml_smoke: cuda requested but unavailable", file=sys.stderr)
        return 1
    device = torch.device(device_name)
    torch.manual_seed(23)
    dtype = torch.bfloat16 if use_bf16 else torch.float64
    modes = ["linear", "linear_sh", "local", "dense"]
    for mode in modes:
        if not run_mode(mode, device, dtype, use_compile):
            return 1
    for mode in ("linear", "local"):
        if not run_rich_mode(mode, device, dtype, use_compile):
            return 1
    print(f"ml_smoke: ok ({device}, dtype={dtype})")
    return 0


def run_mode(mode: str, device: torch.device, dtype: torch.dtype, use_compile: bool) -> bool:
    edge_dim = 2 if mode == "dense" else 0
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=5,
            edge_dim=edge_dim,
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            attention_mode=mode,  # type: ignore[arg-type]
            local_radius=4.0,
            max_neighbors=4,
        )
    ).to(device=device, dtype=dtype)
    node_feats = torch.randn(6, 5, device=device, dtype=dtype)
    pos = torch.randn(6, 3, device=device, dtype=dtype)
    edge_feats = torch.randn(6, 6, 2, device=device, dtype=dtype) if edge_dim > 0 else None
    batch = torch.tensor([0, 0, 0, 1, 1, 1], device=device, dtype=torch.long)
    neighbor_index = make_neighbors(batch, width=3) if mode == "local" else None
    neighbor_mask = torch.ones_like(neighbor_index, dtype=torch.bool) if neighbor_index is not None else None

    out = model(
        node_feats,
        pos,
        edge_feats=edge_feats,
        batch=batch,
        neighbor_index=neighbor_index,
        neighbor_mask=neighbor_mask,
    )
    loss = (
        out["node_scalar"].square().mean()
        + out["node_vector"].square().mean()
        + out["graph_scalar"].square().mean()
        + out["graph_vector"].square().mean()
        + out["node_tensor"].square().mean()
        + out["graph_tensor"].square().mean()
    )
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            print(f"ml_smoke: non-finite gradient in {name}", file=sys.stderr)
            return False

    infer_model = prepare_for_inference(model, device=device, dtype=dtype, compile_model=use_compile)
    infer_batch = None if use_compile else batch
    infer_neighbor_index = None if use_compile else neighbor_index
    infer_neighbor_mask = None if use_compile else neighbor_mask
    with torch.inference_mode():
        infer_model(
            node_feats,
            pos,
            edge_feats=edge_feats,
            batch=infer_batch,
            neighbor_index=infer_neighbor_index,
            neighbor_mask=infer_neighbor_mask,
        )
    return True


def run_rich_mode(mode: str, device: torch.device, dtype: torch.dtype, use_compile: bool) -> bool:
    model = RichEquivariantAttention(
        RichEquivariantAttentionConfig(
            node_dim=5,
            hidden_irreps="12x0e + 4x1o + 2x2e",
            output_irreps="2x0e + 1x1o + 1x2e",
            num_layers=2,
            num_heads=4,
            attention_mode=mode,  # type: ignore[arg-type]
        )
    ).to(device=device, dtype=dtype)
    node_feats = torch.randn(6, 5, device=device, dtype=dtype)
    pos = torch.randn(6, 3, device=device, dtype=dtype)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], device=device, dtype=torch.long)
    neighbor_index = make_neighbors(batch, width=3) if mode == "local" else None
    neighbor_mask = torch.ones_like(neighbor_index, dtype=torch.bool) if neighbor_index is not None else None

    out = model(node_feats, pos, batch=batch, neighbor_index=neighbor_index, neighbor_mask=neighbor_mask)
    loss = sum(value.square().mean() for value in out.values() if torch.is_tensor(value))
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            print(f"ml_smoke: non-finite rich gradient in {name}", file=sys.stderr)
            return False

    infer_compile = use_compile and mode == "linear"
    infer_model = prepare_for_inference(model, device=device, dtype=dtype, compile_model=infer_compile)
    infer_batch = None if infer_compile else batch
    infer_neighbor_index = None if infer_compile else neighbor_index
    infer_neighbor_mask = None if infer_compile else neighbor_mask
    with torch.inference_mode():
        infer_model(
            node_feats,
            pos,
            batch=infer_batch,
            neighbor_index=infer_neighbor_index,
            neighbor_mask=infer_neighbor_mask,
        )
    return True


def make_neighbors(batch: torch.Tensor, width: int) -> torch.Tensor:
    rows = []
    for i, graph_id in enumerate(batch.tolist()):
        members = torch.nonzero(batch == graph_id, as_tuple=False).flatten().tolist()
        start = members.index(i)
        ordered = members[start:] + members[:start]
        while len(ordered) < width:
            ordered.append(i)
        rows.append(ordered[:width])
    return torch.tensor(rows, device=batch.device, dtype=torch.long)


if __name__ == "__main__":
    sys.exit(main())
