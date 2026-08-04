"""One-batch smoke for the public ELAGraph -> ELA -> ELAGraph contract."""

from __future__ import annotations

import sys

import torch

from equivariant_linear_attention import ELA, ELAGraph
from equivariant_linear_attention.inference import autocast_dtype, prepare_for_inference


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
    # Public convention: row 0 is sender, row 1 is receiver.
    return torch.stack([sender, receiver])


def _finite_gradients(model: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def _output_tensors(graph: ELAGraph) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {"x": graph.x, "pos": graph.pos}
    for name in ("graph_x", "graph_sum", "delta"):
        value = getattr(graph, name)
        if value is None:
            raise RuntimeError(f"missing model output: {name}")
        values[name] = value
    return values


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

    model = ELA(
        "5x0e",
        "2x0e",
        width=32,
        depth=2,
        cutoff=10.0,
    ).to(device=device, dtype=parameter_dtype)
    nodes = 7
    x = torch.randn(
        nodes,
        model.config.input_layout.dim,
        device=device,
        dtype=parameter_dtype,
        requires_grad=True,
    )
    pos = torch.randn(
        nodes,
        3,
        device=device,
        dtype=geometry_dtype,
        requires_grad=True,
    )
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], device=device)
    edge_index = torch.cat(
        [
            _fixed_degree_edges(3, 3, device=device),
            _fixed_degree_edges(4, 4, device=device) + 3,
        ],
        dim=1,
    )
    graph = ELAGraph(x=x, pos=pos, edge_index=edge_index, batch=batch)
    output = model(graph)
    output_values = _output_tensors(output)
    loss = output.x.float().square().mean()
    assert output.graph_x is not None
    loss = loss + output.graph_x.float().square().mean()
    loss.backward()
    if not _finite_gradients(model):
        print("ml_smoke: non-finite parameter gradient", file=sys.stderr)
        return 1
    if x.grad is None or not torch.isfinite(x.grad).all():
        print("ml_smoke: non-finite node gradient", file=sys.stderr)
        return 1
    if pos.grad is None or not torch.isfinite(pos.grad).all():
        print("ml_smoke: non-finite coordinate gradient", file=sys.stderr)
        return 1

    # DataLoader batching uses the same graph type; no padded or alternate API.
    first = ELAGraph(x.detach()[:3], pos.detach()[:3])
    second = ELAGraph(x.detach()[3:], pos.detach()[3:])
    collated = ELAGraph.collate([first, second])
    model.eval()
    with torch.inference_mode():
        collated_output = model(collated)
    if collated_output.x.shape != (7, 2):
        print("ml_smoke: collated output shape failed", file=sys.stderr)
        return 1
    if not torch.isfinite(collated_output.x).all():
        print("ml_smoke: collated output is non-finite", file=sys.stderr)
        return 1

    with torch.inference_mode():
        eager_output = model(graph)
    eager_values = _output_tensors(eager_output)
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
    inference_dtype = torch.float32 if use_auto else parameter_dtype
    inference_graph = graph.to(
        device,
        dtype=inference_dtype,
        geometry_dtype=geometry_dtype,
    )
    with torch.inference_mode():
        inference_output = inference_model(inference_graph)
    if not isinstance(inference_output, ELAGraph):
        print("ml_smoke: inference did not return ELAGraph", file=sys.stderr)
        return 1
    inference_values = _output_tensors(inference_output)
    if not all(torch.isfinite(value).all() for value in inference_values.values()):
        print("ml_smoke: non-finite inference output", file=sys.stderr)
        return 1

    comparison_dtype = autocast_dtype(device) if use_auto else parameter_dtype
    tolerance = _comparison_tolerance(comparison_dtype, automatic=use_auto)
    for key, eager in eager_values.items():
        if not torch.allclose(
            eager.float(),
            inference_values[key].float(),
            atol=tolerance,
            rtol=tolerance,
        ):
            print(f"ml_smoke: eager/inference mismatch in {key}", file=sys.stderr)
            return 1

    # Keep this lookup live so the smoke checks every output tensor above.
    assert output_values.keys() == eager_values.keys()
    print(f"ml_smoke: ok ({device}, dtype={parameter_dtype})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
