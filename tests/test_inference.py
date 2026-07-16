import torch

from equivariant_attention import EquivariantAttention, EquivariantAttentionConfig
from equivariant_attention.inference import prepare_for_inference


def test_auto_inference_keeps_parameters_in_float32_on_cpu() -> None:
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=3,
            hidden_irreps="8x0e + 2x1o",
            num_layers=1,
            num_heads=2,
        )
    )

    prepared = prepare_for_inference(model, device="cpu", dtype="auto", compile_model=False)

    assert {parameter.dtype for parameter in prepared.parameters()} == {torch.float32}
    assert not any(parameter.requires_grad for parameter in prepared.parameters())


def test_explicit_low_precision_inference_remains_available() -> None:
    model = EquivariantAttention(
        EquivariantAttentionConfig(
            node_dim=3,
            hidden_irreps="8x0e + 2x1o",
            num_layers=1,
            num_heads=2,
        )
    )

    prepared = prepare_for_inference(model, device="cpu", dtype="bf16", compile_model=False)

    assert {parameter.dtype for parameter in prepared.parameters()} == {torch.bfloat16}
