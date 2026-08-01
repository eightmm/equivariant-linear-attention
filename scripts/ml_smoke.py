"""One-batch canonical ELA forward, backward, batching, and inference smoke."""

from __future__ import annotations

import sys

import torch

from equivariant_attention import ELA
from equivariant_attention.inference import autocast_dtype, prepare_for_inference


def _comparison_tolerance(dtype: torch.dtype, *, automatic: bool) -> float:
    if dtype == torch.bfloat16:
        return 2e-2
    if dtype == torch.float16:
        return 1e-2
    return 5e-3 if automatic else 1e-9


def _fixed_degree_edges(
    nodes: int,
    degree: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    receiver = torch.arange(nodes, device=device).repeat_interleave(degree)
    offset = torch.arange(degree, device=device).repeat(nodes)
    sender = (receiver + offset) % nodes
    return torch.stack([receiver, sender])


def _finite_gradients(model: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def main() -> int:
    device_name = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    use_bf16 = "bf16" in sys.argv[2:]
    use_auto = "auto" in sys.argv[2:]
    use_compile = "compile" in sys.argv[2:]
    if device_name == "cuda" and not torch.cuda.is_available():
        print("ml_smoke: cuda requested but unavailable", file=sys.stderr)
        return 1

    device = torch.device(device_name)
    parameter_dtype = (
        torch.float32
        if use_auto
        else (torch.bfloat16 if use_bf16 else torch.float64)
    )
    geometry_dtype = (
        torch.float64 if parameter_dtype == torch.float64 else torch.float32
    )
    torch.manual_seed(23)

    model = ELA.scalar(
        5,
        output_dim=2,
        width=32,
        depth=2,
        cutoff=10.0,
        num_rbf=8,
    ).to(device=device, dtype=parameter_dtype)
    nodes = 7
    node_irreps = torch.randn(
        nodes,
        5,
        device=device,
        dtype=parameter_dtype,
        requires_grad=True,
    )
    positions = torch.randn(
        nodes,
        3,
        device=device,
        dtype=geometry_dtype,
        requires_grad=True,
    )
    batch = torch.tensor(
        [0, 0, 0, 1, 1, 1, 1],
        device=device,
        dtype=torch.long,
    )
    graph = model.prepare_graph(
        positions.detach(),
        batch=batch,
        edge_index=torch.cat(
            [
                _fixed_degree_edges(3, 3, device=device),
                _fixed_degree_edges(4, 4, device=device) + torch.tensor(
                    [[3], [3]],
                    device=device,
                ),
            ],
            dim=1,
        ),
    )

    output = model(node_irreps, positions, graph)
    required = {"node_irreps", "graph_irreps", "positions", "coordinate_delta"}
    if not required.issubset(output):
        print("ml_smoke: canonical output contract failed", file=sys.stderr)
        return 1
    loss = output["node_irreps"].float().square().mean()
    loss = loss + output["graph_irreps"].float().square().mean()
    loss.backward()
    if not _finite_gradients(model):
        print("ml_smoke: non-finite parameter gradient", file=sys.stderr)
        return 1
    if node_irreps.grad is None or not torch.isfinite(node_irreps.grad).all():
        print("ml_smoke: non-finite node gradient", file=sys.stderr)
        return 1
    if positions.grad is None or not torch.isfinite(positions.grad).all():
        print("ml_smoke: non-finite coordinate gradient", file=sys.stderr)
        return 1

    # Public padded batch path: no PyG object and no explicit graph metadata.
    padded_nodes = torch.zeros(
        2,
        4,
        5,
        device=device,
        dtype=parameter_dtype,
    )
    padded_positions = torch.zeros(
        2,
        4,
        3,
        device=device,
        dtype=geometry_dtype,
    )
    mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]],
        device=device,
    )
    padded_nodes[mask] = node_irreps.detach()
    padded_positions[mask] = positions.detach()
    model.eval()
    with torch.inference_mode():
        padded_output = model(
            padded_nodes,
            padded_positions,
            mask=mask,
        )
    if padded_output["node_irreps"].shape != (2, 4, 2):
        print("ml_smoke: padded output shape failed", file=sys.stderr)
        return 1
    if not torch.isfinite(padded_output["node_irreps"]).all():
        print("ml_smoke: padded output is non-finite", file=sys.stderr)
        return 1

    with torch.inference_mode():
        eager_output = model(node_irreps.detach(), positions.detach(), graph)
    inference_model = prepare_for_inference(
        model,
        device=device,
        dtype="auto" if use_auto else parameter_dtype,
        compile_model=use_compile,
    )
    if use_auto and {parameter.dtype for parameter in inference_model.parameters()} != {
        torch.float32
    }:
        print(
            "ml_smoke: auto inference did not preserve fp32 parameters",
            file=sys.stderr,
        )
        return 1
    with torch.inference_mode():
        inference_output = inference_model(
            node_irreps.detach().to(dtype=torch.float32 if use_auto else parameter_dtype),
            positions.detach(),
            graph,
        )
    if not all(
        torch.isfinite(value).all() for value in inference_output.values()
    ):
        print("ml_smoke: non-finite inference output", file=sys.stderr)
        return 1

    comparison_dtype = autocast_dtype(device) if use_auto else parameter_dtype
    tolerance = _comparison_tolerance(comparison_dtype, automatic=use_auto)
    for key in required:
        if not torch.allclose(
            eager_output[key].float(),
            inference_output[key].float(),
            atol=tolerance,
            rtol=tolerance,
        ):
            print(f"ml_smoke: eager/inference mismatch in {key}", file=sys.stderr)
            return 1

    print(f"ml_smoke: ok ({device}, dtype={parameter_dtype})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
