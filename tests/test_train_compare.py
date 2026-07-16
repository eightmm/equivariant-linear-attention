from pathlib import Path
import runpy

import pytest


def _script_symbols() -> dict[str, object]:
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_compare.py"
    return runpy.run_path(script)


def test_qm9_data_identity_accepts_expected_hashes(tmp_path: Path) -> None:
    symbols = _script_symbols()
    hash_file = symbols["_hash_file"]
    data_identity = symbols["_qm9_data_identity"]
    relative_paths = ["raw/gdb9.sdf", "raw/gdb9.sdf.csv", "processed/data_v3.pt"]
    expected = {}
    for index, relative in enumerate(relative_paths):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-{index}".encode("ascii"))
        expected[relative] = hash_file(path)

    assert data_identity(tmp_path, expected=expected) == expected


def test_qm9_data_identity_rejects_changed_data(tmp_path: Path) -> None:
    symbols = _script_symbols()
    data_identity = symbols["_qm9_data_identity"]
    path = tmp_path / "processed/data_v3.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="data identity mismatch"):
        data_identity(tmp_path, expected={"processed/data_v3.pt": "0" * 64})


def test_run_config_records_single_architecture() -> None:
    symbols = _script_symbols()
    args = symbols["parse_args"]([])

    config = symbols["_run_config"](
        args,
        split_seed=42,
        model_seed=43,
    )

    assert config["model"] == "factorized_moment"
    assert config["attention"] == "factorized_moment"
    assert config["balance_cycles"] == 1
    assert config["key_balancing"] is True
    assert config["linear_kernel_init"] > 0.0
    assert config["ffn_hidden_ratio"] == 2.0
