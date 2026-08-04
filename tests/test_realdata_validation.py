from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from equivariant_linear_attention import ELAGraph


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_realdata.py"
_SPEC = importlib.util.spec_from_file_location("_ela_validate_realdata", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
realdata = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = realdata
_SPEC.loader.exec_module(realdata)


def _lba_row(*, label: float = 7.25) -> dict[str, object]:
    return {
        # The type-0 node is the duplicated full-protein copy and must vanish.
        "input_ids": [9, 2, 3, 4, 5],
        "coords": [
            [99.0, 99.0, 99.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        "token_type_ids": [0, 1, 1, 2, 2],
        "labels": label,
    }


class _FakeLBADataset:
    def __init__(self) -> None:
        self.graphs = tuple(
            realdata._lba_row_to_graph(
                _lba_row(label=float(index + 1)),
                split="train",
                row_index=index,
                cutoff=3.0,
                intra_k=1,
                cross_k=1,
            )
            for index in range(4)
        )

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> ELAGraph:
        return self.graphs[index]

    def targets(self, indices: tuple[int, ...]) -> torch.Tensor:
        return torch.tensor(
            [self.graphs[index].y.item() for index in indices],
            dtype=torch.float64,
        )


def test_lba_test_split_is_rejected_before_any_path_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_if_probed(_path: Path) -> bool:
        raise AssertionError("a refused split must not probe a cache path")

    monkeypatch.setattr(Path, "is_file", fail_if_probed)
    with pytest.raises(ValueError, match="test shard is intentionally unreachable"):
        realdata._open_lba_split(tmp_path, "test")


def test_lba_features_filter_and_ligand_readout_use_input_indicator() -> None:
    first = realdata._lba_row_to_graph(
        _lba_row(),
        split="train",
        row_index=0,
        cutoff=3.0,
        intra_k=1,
        cross_k=1,
    )
    second = realdata._lba_row_to_graph(
        _lba_row(label=6.5),
        split="val",
        row_index=0,
        cutoff=3.0,
        intra_k=1,
        cross_k=1,
    )

    assert first.x.shape == (4, realdata.LBA_INPUT_DIM)
    assert first.pos.shape == (4, 3)
    assert first.y.item() == pytest.approx(7.25)
    assert torch.equal(first.x[:, -2:], torch.tensor([[1.0, 0.0]] * 2 + [[0.0, 1.0]] * 2))
    assert first.x[:, :138].sum(dim=1).tolist() == [1.0] * 4
    assert not bool((first.pos == 99.0).any().item())

    packed = ELAGraph.collate((first, second))
    node_prediction = torch.arange(8, dtype=torch.float32)[:, None]
    readout = realdata._ligand_mean(node_prediction, packed)
    assert torch.equal(readout[:, 0], torch.tensor([2.5, 6.5]))


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("input_ids", [0, 2, 3, 4, 5], "tokens must lie"),
        ("labels", float("nan"), "one finite scalar"),
        ("labels", [1.0, 2.0], "one finite scalar"),
    ),
)
def test_lba_rejects_invalid_tokens_and_labels(
    field: str,
    value: object,
    match: str,
) -> None:
    row = _lba_row()
    row[field] = value
    with pytest.raises(ValueError, match=match):
        realdata._lba_row_to_graph(row, split="train", row_index=0)


def test_segment_balanced_topology_is_deterministic_and_sender_first() -> None:
    graph = realdata._lba_row_to_graph(
        _lba_row(),
        split="train",
        row_index=0,
        cutoff=3.0,
        intra_k=1,
        cross_k=1,
    )
    repeated = realdata._lba_row_to_graph(
        _lba_row(),
        split="train",
        row_index=0,
        cutoff=3.0,
        intra_k=1,
        cross_k=1,
    )
    assert torch.equal(graph.edge_index, repeated.edge_index)
    assert torch.equal(graph.edge_type, repeated.edge_type)
    assert graph.edge_index is not None
    assert graph.edge_type is not None

    sender, receiver = graph.edge_index
    ligand = graph.x[:, -1].bool()
    expected_relation = torch.where(
        ligand[sender] != ligand[receiver],
        torch.full_like(sender, 2),
        torch.where(ligand[sender], torch.ones_like(sender), torch.zeros_like(sender)),
    )
    assert torch.equal(graph.edge_type, expected_relation)
    self_receivers = receiver[sender == receiver]
    assert sorted(self_receivers.tolist()) == list(range(graph.num_nodes))

    first_manifest = realdata._topology_manifest(((graph,),))
    second_manifest = realdata._topology_manifest(((repeated,),))
    assert first_manifest == second_manifest
    assert set(first_manifest) == {
        "sample_identity_sha256",
        "edge_index_sha256",
        "edge_relation_sha256",
        "edge_topology_sha256",
        "joint_sha256",
        "graphs",
        "directed_edges_with_self",
    }

    asymmetric_edges, _ = realdata._segment_balanced_topology(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        torch.zeros(3, dtype=torch.bool),
        intra_k=1,
        cross_k=0,
        cutoff=10.0,
    )
    directed_pairs = set(zip(*asymmetric_edges.tolist(), strict=True))
    assert (1, 2) in directed_pairs  # sender 1 is nearest to receiver 2
    assert (2, 1) not in directed_pairs


def test_static_ablation_arms_share_schema_state_and_initial_prediction() -> None:
    models, pairing = realdata._build_arm_models(
        input_dim=realdata.LBA_INPUT_DIM,
        edge_types=realdata.LBA_RELATIONS,
        width=16,
        depth=1,
        cutoff=3.0,
        arms=realdata.STATIC_ARMS,
        include_stagewise=True,
        seed=17,
    )
    graph = realdata._lba_row_to_graph(
        _lba_row(),
        split="train",
        row_index=0,
        cutoff=3.0,
        intra_k=1,
        cross_k=1,
    )

    hashes: set[str] = set()
    predictions: list[torch.Tensor] = []
    for arm in realdata.STATIC_ARMS:
        model = models[arm].eval()
        assert pairing["controls"][arm]["paired_schema"] is True
        assert pairing["controls"][arm]["base_schema_sha256"] == pairing["base_schema_sha256"]
        hashes.add(pairing["controls"][arm]["initial_state_sha256"])
        with torch.no_grad():
            prediction, _ = realdata._predict(model, graph, "ligand_mean")
        predictions.append(prediction)

    assert len(hashes) == 1
    assert all(torch.equal(predictions[0], value) for value in predictions[1:])
    assert pairing["controls"]["no-relation"]["disabled_lane_parameters"]
    assert pairing["controls"]["no-cg12"]["disabled_lane_parameters"]
    assert pairing["controls"]["no-multiscale"]["disabled_lane_parameters"]
    assert pairing["controls"]["stagewise"]["paired_schema"] is False
    stagewise = models["stagewise"]
    coordinate_before = realdata._coordinate_state_sha256(stagewise)
    assert coordinate_before is not None
    with torch.no_grad():
        assert stagewise.coordinate_head is not None
        stagewise.coordinate_head.base_weight.add_(0.25)
    assert realdata._coordinate_state_sha256(stagewise) != coordinate_before


def test_one_update_fake_lba_training_emits_consistent_receipt() -> None:
    dataset = _FakeLBADataset()
    indices = tuple(range(len(dataset)))
    normalizer = realdata._TargetNormalizer.fit(dataset.targets(indices))
    task = realdata._TaskData(
        train_dataset=dataset,
        evaluation_dataset=dataset,
        train_indices=indices,
        evaluation_indices=indices,
        input_dim=realdata.LBA_INPUT_DIM,
        edge_types=realdata.LBA_RELATIONS,
        prediction="ligand_mean",
        evaluation_split="train",
        normalizer=normalizer,
        data={},
        split={},
        topology={},
        access={},
        limitations=[],
    )
    models, _ = realdata._build_arm_models(
        input_dim=realdata.LBA_INPUT_DIM,
        edge_types=realdata.LBA_RELATIONS,
        width=16,
        depth=1,
        cutoff=3.0,
        arms=("full",),
        include_stagewise=False,
        seed=23,
    )
    model = models["full"]
    initial_state = realdata._state_sha256(model)
    result = realdata._run_arm(
        arm="full",
        model=model,
        task=task,
        steps=1,
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        order_seed=23,
    )

    assert result["updates_completed"] == 1
    assert result["initial_state_sha256"] == initial_state
    assert result["final_state_sha256"] != initial_state
    assert result["evaluation"]["count"] == len(dataset)
    assert result["initial_evaluation"]["count"] == len(dataset)
    assert result["evaluation"]["normalized_mse"] >= 0.0
    assert result["initial_evaluation"]["normalized_mse"] >= 0.0
    assert result["initial_coordinate_state_sha256"] is None
    assert result["final_coordinate_state_sha256"] is None
    assert result["evaluation"]["coordinate_delta_max"] == 0.0
    assert result["peak_cuda_memory_bytes"] is None


def test_id30_identity_gate_uses_split_receipts_and_exact_counts() -> None:
    train = {
        "graphs": realdata.LBA_ID30_TRAIN_SIZE,
        "directed_edges_with_self": 30_000_000,
        "edge_topology_sha256": "train",
    }
    validation = {
        "graphs": realdata.LBA_ID30_VALIDATION_SIZE,
        "directed_edges_with_self": (
            realdata.LBA_ID30_DIRECTED_EDGES_WITH_SELF - 30_000_000
        ),
        "edge_topology_sha256": "validation",
    }
    gate = realdata._lba_id30_identity_gate(train, validation)
    assert gate["passed"] is True
    combined = realdata._combine_topology_manifests(train, validation)
    assert combined["graphs"] == 3507 + 466
    assert combined["directed_edges_with_self"] == 32_302_952

    wrong = dict(validation, graphs=465)
    with pytest.raises(ValueError, match="frozen identity mismatch"):
        realdata._lba_id30_identity_gate(train, wrong)


def test_cli_defaults_are_bounded_and_qm9_rejects_relation_arm() -> None:
    qm9 = realdata.parse_args(("qm9", "result.json"))
    overfit = realdata.parse_args(("lba-overfit", "result.json"))
    id30 = realdata.parse_args(("lba-id30", "result.json"))
    assert qm9.steps == 100
    assert overfit.steps == 250
    assert id30.steps == 220
    assert qm9.arms == ("full", "no-cg12", "no-multiscale")
    assert overfit.arms == realdata.STATIC_ARMS
    assert id30.arms == realdata.STATIC_ARMS

    invalid = realdata.parse_args(
        ("qm9", "result.json", "--arms", "full", "no-relation")
    )
    with pytest.raises(ValueError, match="inapplicable"):
        realdata._validate_args(invalid)
    assert realdata._prediction_pairing_status(("only-arm",)) is None
    assert realdata._prediction_pairing_status(("same", "same")) is True
    assert realdata._prediction_pairing_status(("left", "right")) is False
