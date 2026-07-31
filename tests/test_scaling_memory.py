from __future__ import annotations

from equivariant_attention import estimate_implicit_spatial_kernel


def test_implicit_memory_proxy_accounts_for_graph_statistics_and_chunk() -> None:
    one_graph = estimate_implicit_spatial_kernel(
        nodes=1024,
        graphs=1,
        feature_rank=30,
        value_width=64,
        applications=1,
        chunk_size=128,
    )
    many_graphs = estimate_implicit_spatial_kernel(
        nodes=1024,
        graphs=64,
        feature_rank=30,
        value_width=64,
        applications=1,
        chunk_size=128,
    )
    larger_chunk = estimate_implicit_spatial_kernel(
        nodes=1024,
        graphs=1,
        feature_rank=30,
        value_width=64,
        applications=1,
        chunk_size=512,
    )

    assert many_graphs.inference_memory_proxy > one_graph.inference_memory_proxy
    assert larger_chunk.inference_memory_proxy > one_graph.inference_memory_proxy
    assert one_graph.training_memory_proxy > one_graph.inference_memory_proxy
    assert one_graph.node_linear is True
