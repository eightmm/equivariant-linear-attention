"""One-batch forward, backward, and inference smoke test."""

import sys

import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig, prepare_for_inference


def main() -> int:
    device_name = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    use_bf16 = "bf16" in sys.argv[2:]
    use_compile = "compile" in sys.argv[2:]
    if device_name == "cuda" and not torch.cuda.is_available():
        print("ml_smoke: cuda requested but unavailable", file=sys.stderr)
        return 1

    device = torch.device(device_name)
    dtype = torch.bfloat16 if use_bf16 else torch.float64
    geometry_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    torch.manual_seed(23)
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=5,
            hidden_irreps="16x0e + 4x1o",
            output_irreps="2x0e + 1x1o + 1x2e",
            num_layers=2,
            num_heads=4,
        )
    ).to(device=device, dtype=dtype)
    node_feats = torch.randn(7, 5, device=device, dtype=dtype)
    pos = torch.randn(7, 3, device=device, dtype=geometry_dtype)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], device=device)

    outputs = model(node_feats, pos, batch=batch)
    loss = sum(value.float().square().mean() for value in outputs.values())
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            print(f"ml_smoke: non-finite gradient in {name}", file=sys.stderr)
            return 1

    model.eval()
    with torch.inference_mode():
        eager_outputs = model(node_feats, pos, batch=batch)
    inference_model = prepare_for_inference(
        model,
        device=device,
        dtype=dtype,
        compile_model=use_compile,
    )
    with torch.inference_mode():
        inference_outputs = inference_model(node_feats, pos, batch=batch)
    if not all(torch.isfinite(value).all() for value in inference_outputs.values()):
        print("ml_smoke: non-finite inference output", file=sys.stderr)
        return 1
    tolerance = 5e-3 if dtype in {torch.float16, torch.bfloat16} else 1e-10
    for key in eager_outputs:
        if not torch.allclose(
            eager_outputs[key],
            inference_outputs[key],
            atol=tolerance,
            rtol=tolerance,
        ):
            print(f"ml_smoke: eager/inference mismatch in {key}", file=sys.stderr)
            return 1

    print(f"ml_smoke: ok ({device}, dtype={dtype})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
