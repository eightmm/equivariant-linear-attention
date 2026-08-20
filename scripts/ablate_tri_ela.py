"""Run controlled component interventions on one trained TriELA instance.

The script does not construct alternate public architectures.  It fits one
canonical model briefly, freezes its weights, and then uses temporary forward
hooks to remove exactly one contribution at a time: pair transition, outgoing
triangle, incoming triangle, pair-to-node injection, global transport, or
local transport.  Every requested intervention must bind to a real module;
missing stage structure is an error rather than a silent skip.

By default a deterministic synthetic batch is used.  ``--input`` accepts a
``torch.save``-produced tensor dictionary with required ``x`` and ``pos`` and
optional ``batch``, ``group``, ``condition``, ``order``, ``update_mask``, and
``y`` entries.
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import torch
from benchmark_triangle import (
    emit_json,
    parse_choice_grid,
    resolve_device,
    resolve_dtype,
)
from torch import nn

ARMS = (
    "full",
    "minus_pair_ffn",
    "minus_outgoing",
    "minus_incoming",
    "minus_pair_to_node",
    "minus_global",
    "minus_local",
)


def _module_sequence(value: object, *, path: str) -> tuple[nn.Module, ...]:
    if isinstance(value, (nn.ModuleList, nn.Sequential)):
        modules = tuple(value)
    elif isinstance(value, nn.Module):
        modules = (value,)
    else:
        raise TypeError(f"{path} must be a module or module collection")
    if not modules:
        raise RuntimeError(f"{path} is empty; intervention would be a silent no-op")
    return modules


def _zero_tree(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return torch.zeros_like(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: _zero_tree(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.init
        }
        return dataclasses.replace(value, **updates)
    if isinstance(value, tuple):
        items = tuple(_zero_tree(item) for item in value)
        if hasattr(value, "_fields"):
            return type(value)(*items)
        return items
    if isinstance(value, list):
        return [_zero_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _zero_tree(item) for key, item in value.items()}
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    raise RuntimeError(
        f"cannot zero module output of type {type(value).__name__} explicitly"
    )


def _zero_output(
    _module: nn.Module,
    _inputs: tuple[object, ...],
    output: object,
) -> object:
    return _zero_tree(output)


def _looks_like_state(value: object) -> bool:
    return all(
        hasattr(value, name)
        for name in (
            "even_scalar",
            "odd_scalar",
            "polar_vector",
            "axial_vector",
        )
    )


def _identity_state(
    _module: nn.Module,
    inputs: tuple[object, ...],
    output: object,
) -> object:
    state = next((value for value in inputs if _looks_like_state(value)), None)
    if state is None:
        raise RuntimeError("transport intervention could not identify its input state")
    if type(output) is type(state):
        return state
    if dataclasses.is_dataclass(output) and not isinstance(output, type):
        fields = {field.name for field in dataclasses.fields(output)}
        if "state" in fields:
            return dataclasses.replace(output, state=state)
    if isinstance(output, tuple) and output and _looks_like_state(output[0]):
        return (state, *output[1:])
    raise RuntimeError(
        "transport intervention requires a state output or a dataclass with `.state`"
    )


def _model_stages(model: nn.Module) -> tuple[nn.Module, ...]:
    stages = getattr(model, "stages", None)
    return _module_sequence(stages, path="TriELA.stages")


def _intervention_targets(
    model: nn.Module,
    arm: str,
) -> tuple[tuple[nn.Module, Callable[..., object]], ...]:
    if arm == "full":
        return ()
    targets: list[tuple[nn.Module, Callable[..., object]]] = []
    for stage_index, stage in enumerate(_model_stages(model)):
        stage_path = f"TriELA.stages[{stage_index}]"
        if arm in {"minus_pair_ffn", "minus_outgoing", "minus_incoming"}:
            blocks = _module_sequence(
                getattr(stage, "pair_blocks", None),
                path=f"{stage_path}.pair_blocks",
            )
            attribute = {
                "minus_pair_ffn": "transition",
                "minus_outgoing": "outgoing",
                "minus_incoming": "incoming",
            }[arm]
            for block_index, block in enumerate(blocks):
                module = getattr(block, attribute, None)
                if not isinstance(module, nn.Module):
                    raise TypeError(
                        f"{stage_path}.pair_blocks[{block_index}].{attribute} "
                        "must be a module"
                    )
                targets.append((module, _zero_output))
        elif arm == "minus_pair_to_node":
            modules = _module_sequence(
                getattr(stage, "pair_to_node", None),
                path=f"{stage_path}.pair_to_node",
            )
            targets.extend((module, _zero_output) for module in modules)
        elif arm in {"minus_global", "minus_local"}:
            attribute = "global_blocks" if arm == "minus_global" else "local_blocks"
            modules = _module_sequence(
                getattr(stage, attribute, None),
                path=f"{stage_path}.{attribute}",
            )
            targets.extend((module, _identity_state) for module in modules)
        else:
            raise ValueError(f"unsupported ablation arm: {arm}")

    if not targets:
        raise RuntimeError(f"{arm} did not bind any module")
    unique: dict[int, tuple[nn.Module, Callable[..., object]]] = {}
    for module, hook in targets:
        unique[id(module)] = (module, hook)
    return tuple(unique.values())


@contextmanager
def _apply_intervention(model: nn.Module, arm: str) -> Iterator[None]:
    handles = []
    try:
        for module, hook in _intervention_targets(model, arm):
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _tensor_stats(value: torch.Tensor) -> dict[str, object]:
    numeric = value.detach().float()
    return {
        "shape": list(value.shape),
        "mean": float(numeric.mean().item()),
        "rms": float(numeric.square().mean().sqrt().item()),
        "l2": float(numeric.norm().item()),
        "max_abs": float(numeric.abs().max().item()),
    }


def _output_tensors(graph: object) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name in ("x", "graph_x", "pos", "delta"):
        value = getattr(graph, name, None)
        if value is not None:
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"TriELA output {name} must be a tensor or None")
            tensors[name] = value
    if "x" not in tensors or "graph_x" not in tensors:
        raise RuntimeError("TriELA output must provide node x and graph_x")
    return tensors


def _compare_outputs(
    candidate: Mapping[str, torch.Tensor],
    baseline: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    if candidate.keys() != baseline.keys():
        raise RuntimeError("ablation and baseline returned different output fields")
    comparison: dict[str, dict[str, float]] = {}
    for name, baseline_value in baseline.items():
        candidate_value = candidate[name]
        if candidate_value.shape != baseline_value.shape:
            raise RuntimeError(f"ablation changed {name} shape")
        difference = (candidate_value - baseline_value).detach().float()
        comparison[name] = {
            "rms": float(difference.square().mean().sqrt().item()),
            "max_abs": float(difference.abs().max().item()),
        }
    return comparison


def _normalize_target(
    target: torch.Tensor,
    *,
    num_graphs: int,
    output_dim: int,
) -> torch.Tensor:
    if target.ndim == 0:
        target = target.reshape(1, 1)
    elif target.ndim == 1:
        if output_dim == 1 and target.shape[0] == num_graphs:
            target = target.unsqueeze(-1)
        elif num_graphs == 1 and target.shape[0] == output_dim:
            target = target.unsqueeze(0)
    if target.shape != (num_graphs, output_dim):
        raise ValueError(
            f"target must normalize to ({num_graphs},{output_dim}); "
            f"got {tuple(target.shape)}"
        )
    return target


def _synthetic_target(
    graph: object,
    *,
    output_dim: int,
) -> torch.Tensor:
    x = graph.x
    pos = graph.pos
    batch = graph.batch_index
    num_graphs = int(graph.num_graphs)
    projection = torch.linspace(
        -0.5,
        0.5,
        steps=x.shape[-1] * output_dim,
        device=x.device,
        dtype=x.dtype,
    ).reshape(x.shape[-1], output_dim)
    radial = pos.square().sum(dim=-1, keepdim=True)
    channel_scale = torch.linspace(
        0.05,
        0.15,
        steps=output_dim,
        device=x.device,
        dtype=x.dtype,
    ).unsqueeze(0)
    node_target = x @ projection + radial * channel_scale
    graph_sum = x.new_zeros((num_graphs, output_dim))
    graph_sum.index_add_(0, batch, node_target)
    counts = x.new_zeros((num_graphs,))
    counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
    return graph_sum / counts.clamp_min(1).unsqueeze(-1)


def _load_tensor_batch(
    path: str,
    *,
    graph_type: type,
    device: torch.device,
    dtype: torch.dtype,
) -> object:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("--input must contain a tensor dictionary")
    allowed = {
        "x",
        "pos",
        "batch",
        "group",
        "condition",
        "order",
        "update_mask",
        "y",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unsupported tensor-batch keys: {unknown}")
    if "x" not in payload or "pos" not in payload:
        raise ValueError("tensor batch requires x and pos")
    if any(
        value is not None and not isinstance(value, torch.Tensor)
        for value in payload.values()
    ):
        raise TypeError("all tensor-batch values must be tensors or None")

    floating = {"x", "pos", "condition", "order", "y"}
    values: dict[str, torch.Tensor | None] = {}
    for name, value in payload.items():
        if value is None:
            values[name] = None
        elif name in floating:
            values[name] = value.to(device=device, dtype=dtype)
        else:
            values[name] = value.to(device=device)
    return graph_type(**values)


def _synthetic_batch(
    *,
    graph_type: type,
    batch_size: int,
    nodes_per_graph: int,
    input_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> object:
    total_nodes = batch_size * nodes_per_graph
    x = torch.randn(total_nodes, input_dim, device=device, dtype=dtype)
    pos = torch.randn(total_nodes, 3, device=device, dtype=dtype)
    batch = torch.arange(batch_size, device=device).repeat_interleave(nodes_per_graph)
    return graph_type(x=x, pos=pos, batch=batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="optional torch tensor-dictionary batch")
    parser.add_argument("--input-irreps")
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--pair-width", type=int, default=8)
    parser.add_argument("--triangle-hidden", type=int, default=8)
    parser.add_argument("--num-stages", type=int, default=1)
    parser.add_argument("--pair-blocks-per-stage", type=int, default=1)
    parser.add_argument("--local-blocks-per-stage", type=int, default=1)
    parser.add_argument("--max-pair-tokens", type=int, default=128)
    parser.add_argument("--fit-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help=f"comma-separated subset of {','.join(ARMS)}; full is required",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arms = parse_choice_grid(args.arms, name="arms", choices=ARMS)
    if "full" not in arms:
        raise ValueError("arms must include the full baseline")
    positive = {
        "nodes": args.nodes,
        "batch-size": args.batch_size,
        "input-dim": args.input_dim,
        "output-dim": args.output_dim,
        "width": args.width,
        "pair-width": args.pair_width,
        "triangle-hidden": args.triangle_hidden,
        "num-stages": args.num_stages,
        "pair-blocks-per-stage": args.pair_blocks_per_stage,
        "local-blocks-per-stage": args.local_blocks_per_stage,
        "max-pair-tokens": args.max_pair_tokens,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"these arguments must be positive: {', '.join(invalid)}")
    if args.fit_steps < 0:
        raise ValueError("fit-steps must be nonnegative")
    if not torch.isfinite(torch.tensor(args.learning_rate)) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if args.input is None and args.nodes > args.max_pair_tokens:
        raise ValueError("nodes exceed max-pair-tokens")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device=device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    from equivariant_linear_attention import ELAGraph, TriELA, TriELAConfig

    graph = (
        _load_tensor_batch(
            args.input,
            graph_type=ELAGraph,
            device=device,
            dtype=dtype,
        )
        if args.input is not None
        else _synthetic_batch(
            graph_type=ELAGraph,
            batch_size=args.batch_size,
            nodes_per_graph=args.nodes,
            input_dim=args.input_dim,
            device=device,
            dtype=dtype,
        )
    )
    graph_x = graph.x
    input_irreps = args.input_irreps or f"{graph_x.shape[-1]}x0e"
    output_irreps = f"{args.output_dim}x0e"
    model = TriELA(
        input_irreps,
        output_irreps,
        width=args.width,
        pair_width=args.pair_width,
        triangle_hidden=args.triangle_hidden,
        num_stages=args.num_stages,
        pair_blocks_per_stage=args.pair_blocks_per_stage,
        local_blocks_per_stage=args.local_blocks_per_stage,
        pair_dropout=0.0,
        max_pair_tokens=args.max_pair_tokens,
    ).to(device=device, dtype=dtype)
    if not isinstance(model.config, TriELAConfig):
        raise TypeError("TriELA.config must expose the public TriELAConfig")

    supplied_target = getattr(graph, "y", None)
    target = (
        _normalize_target(
            supplied_target,
            num_graphs=int(graph.num_graphs),
            output_dim=args.output_dim,
        )
        if supplied_target is not None
        else _synthetic_target(graph, output_dim=args.output_dim)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    fit_losses: list[float] = []
    model.train()
    for _ in range(args.fit_steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(graph)
        prediction = output.graph_x
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("TriELA output must provide graph_x")
        loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
        loss.backward()
        optimizer.step()
        fit_losses.append(float(loss.detach().item()))

    model.eval()
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    metrics: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for arm in arms:
            with _apply_intervention(model, arm):
                result = model(graph)
            tensors = _output_tensors(result)
            outputs[arm] = tensors
            prediction = tensors["graph_x"].float()
            metrics[arm] = {
                "mse": float(
                    torch.nn.functional.mse_loss(prediction, target.float()).item()
                ),
                "mae": float(
                    torch.nn.functional.l1_loss(prediction, target.float()).item()
                ),
            }

    baseline = outputs["full"]
    rows = [
        {
            "arm": arm,
            "target_metrics": metrics[arm],
            "output_stats": {
                name: _tensor_stats(value) for name, value in outputs[arm].items()
            },
            "delta_from_full": _compare_outputs(outputs[arm], baseline),
        }
        for arm in arms
    ]
    emit_json(
        {
            "experiment": "tri_ela_controlled_component_ablation",
            "interpretation": (
                "functional intervention on one fitted weight set; not an "
                "accuracy or retrained architecture comparison"
            ),
            "model": {
                "input_irreps": input_irreps,
                "output_irreps": output_irreps,
                "width": args.width,
                "pair_width": args.pair_width,
                "triangle_hidden": args.triangle_hidden,
                "num_stages": args.num_stages,
                "pair_blocks_per_stage": args.pair_blocks_per_stage,
                "local_blocks_per_stage": args.local_blocks_per_stage,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
            },
            "batch": {
                "source": str(Path(args.input)) if args.input else "synthetic",
                "num_nodes": int(graph.num_nodes),
                "num_graphs": int(graph.num_graphs),
            },
            "fit": {
                "steps": args.fit_steps,
                "learning_rate": args.learning_rate,
                "losses": fit_losses,
            },
            "environment": {
                "torch_version": torch.__version__,
                "device": str(device),
                "dtype": args.dtype,
                "seed": args.seed,
            },
            "rows": rows,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
