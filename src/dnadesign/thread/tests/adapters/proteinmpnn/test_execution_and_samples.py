"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/proteinmpnn/test_execution_and_samples.py

ProteinMPNN execution preflight and sample parsing tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from dnadesign.thread.adapters.proteinmpnn.execution import ProteinMpnnExecutionConfig, run_official_proteinmpnn_request
from dnadesign.thread.adapters.proteinmpnn.execution_preflight import validate_proteinmpnn_root
from dnadesign.thread.adapters.proteinmpnn.manifest import build_request_manifest, request_hash
from dnadesign.thread.adapters.proteinmpnn.samples import (
    parse_proteinmpnn_fasta_samples,
    validate_sample_table,
    write_sample_table,
)
from dnadesign.thread.adapters.proteinmpnn.sidecars import write_jsonl


def test_proteinmpnn_preflight_rejects_missing_official_scripts(tmp_path: Path) -> None:
    issues = validate_proteinmpnn_root(tmp_path)

    check_ids = {issue.check_id for issue in issues}
    assert "thread.proteinmpnn.tool_missing_script" in check_ids
    assert "thread.proteinmpnn.tool_missing_weights" in check_ids


def test_proteinmpnn_preflight_accepts_required_scripts_and_weights(tmp_path: Path) -> None:
    for rel_path in (
        "protein_mpnn_run.py",
        "helper_scripts/parse_multiple_chains.py",
        "helper_scripts/assign_fixed_chains.py",
        "helper_scripts/make_fixed_positions_dict.py",
        "vanilla_model_weights/v_48_020.pt",
    ):
        path = tmp_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")

    assert validate_proteinmpnn_root(tmp_path) == []


def test_parse_proteinmpnn_fasta_samples_extracts_seed_temperature_and_scores(tmp_path: Path) -> None:
    fasta_path = tmp_path / "seqs" / "chain_a_backbone.fa"
    fasta_path.parent.mkdir()
    fasta_path.write_text(
        "\n".join(
            [
                (
                    ">chain_a_backbone, score=1.0000, global_score=2.0000, fixed_chains=[], "
                    "designed_chains=['A'], model_name=v_48_020, git_hash=abc123, seed=101"
                ),
                "AAA",
                ">T=0.1, sample=1, score=0.5000, global_score=1.5000, seq_recovery=0.9000",
                "AAC",
                ">T=0.3, sample=1, score=0.7000, global_score=1.7000, seq_recovery=0.8000",
                "AAD",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = parse_proteinmpnn_fasta_samples(
        fasta_path,
        backend_run_id="run-1",
        request_hash="sha256:request",
        seed=101,
        sequence_length=3,
    )

    assert [row["temperature"] for row in rows] == [0.1, 0.3]
    assert [row["sequence"] for row in rows] == ["AAC", "AAD"]
    assert rows[0]["sample_id"] == "run-1__seed101__temp0.1__sample1"
    assert rows[0]["score"] == 0.5
    assert rows[0]["global_score"] == 1.5
    assert rows[0]["status"] == "accepted"


def test_validate_sample_table_rejects_stale_sequence_hash(tmp_path: Path) -> None:
    sample_table_path = tmp_path / "sample_table.parquet"
    write_sample_table(
        sample_table_path,
        [
            {
                "sample_id": "sample-1",
                "backend_run_id": "run-1",
                "request_hash": "sha256:request",
                "seed": 101,
                "temperature": 0.1,
                "sample_index": 1,
                "sequence": "AAAA",
                "sequence_hash": "sha256:stale",
                "score": 0.5,
                "global_score": 1.5,
                "seq_recovery": 0.9,
                "backend_result_hash": "sha256:result",
                "status": "accepted",
            }
        ],
        request_hash="sha256:request",
    )

    issues = validate_sample_table(
        sample_table_path,
        request_hash="sha256:request",
        expected_sample_count=1,
        sequence_length=4,
    )

    assert {issue.check_id for issue in issues} == {"thread.proteinmpnn.sample_table_sequence_hash_mismatch"}


def test_proteinmpnn_execution_config_derives_expected_samples() -> None:
    config = ProteinMpnnExecutionConfig(batch_id="eco1_rt_n96", num_seq_per_target=16, batch_size=4)

    assert config.expected_sample_count(seed_count=3, temperature_count=2) == 96
    assert config.run_dir_name == "eco1_rt_n96"


def test_proteinmpnn_execution_config_rejects_nondivisible_batch_size() -> None:
    with pytest.raises(ValueError, match="num_seq_per_target must be divisible"):
        ProteinMpnnExecutionConfig(batch_id="bad", num_seq_per_target=10, batch_size=3)


def test_run_official_proteinmpnn_request_rejects_invalid_manifest_before_writing_outputs(tmp_path: Path) -> None:
    proteinmpnn_root = _write_fake_proteinmpnn_root(tmp_path / "ProteinMPNN")
    manifest_path = tmp_path / "request_manifest.yaml"
    manifest_path.write_text("schema_id: wrong\n", encoding="utf-8")
    output_dir = tmp_path / "proteinmpnn_outputs"

    with pytest.raises(ValueError, match="request manifest validation failed"):
        run_official_proteinmpnn_request(
            request_manifest_path=manifest_path,
            proteinmpnn_root=proteinmpnn_root,
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_run_official_proteinmpnn_request_resolves_colocated_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proteinmpnn_root = _write_fake_proteinmpnn_root(tmp_path / "ProteinMPNN")
    request_dir = tmp_path / "proteinmpnn_request"
    request_dir.mkdir()
    backbone_path = request_dir / "chain_a_backbone.pdb"
    parsed_path = request_dir / "parsed_pdbs.jsonl"
    assigned_path = request_dir / "assigned_chains.jsonl"
    fixed_path = request_dir / "fixed_positions.jsonl"
    backbone_path.write_text("END\n", encoding="utf-8")
    parsed_payload = {"name": "target", "num_of_chains": 1, "seq_chain_A": "ABC"}
    assigned_payload = {"target": [["A"], []]}
    fixed_payload = {"target": {"A": [1, 3]}}
    write_jsonl(parsed_path, parsed_payload)
    write_jsonl(assigned_path, assigned_payload)
    write_jsonl(fixed_path, fixed_payload)
    stale_dir = tmp_path / "other_host" / "proteinmpnn_request"
    stale_dir.mkdir(parents=True)
    write_jsonl(stale_dir / "parsed_pdbs.jsonl", {"name": "target", "seq_chain_A": "ZZZ"})
    write_jsonl(stale_dir / "assigned_chains.jsonl", {"target": [["B"], []]})
    write_jsonl(stale_dir / "fixed_positions.jsonl", {"target": {"A": [2]}})
    manifest_without_hash = build_request_manifest(
        artifact_id="test.portable_request",
        created_by="test",
        profile_id=None,
        mask_policy_id=None,
        target_name="target",
        chain_id="A",
        sidecar_paths={
            "chain_a_backbone_pdb": backbone_path,
            "parsed_pdbs_jsonl": parsed_path,
            "assigned_chains_jsonl": assigned_path,
            "fixed_positions_jsonl": fixed_path,
        },
        upstream_artifact_hashes={},
        source_thread_plan={"path": "/other/host/thread_plan.yaml"},
        canonical_to_mpnn={1: 1, 2: 2, 3: 3},
        fixed_positions=[1, 3],
        mutable_positions=[2],
        excluded_positions=[],
        seed_set=[101],
        temperatures=[0.1],
        omit_aas=[],
        batch_id="portable",
        num_seq_per_target=4,
        batch_size=2,
        expected_sample_count=4,
    )
    manifest_without_hash["sidecar_paths"] = {
        name: str(stale_dir / Path(str(sidecar_path)).name)
        for name, sidecar_path in manifest_without_hash["sidecar_paths"].items()
    }
    manifest = {"request_hash": request_hash(manifest_without_hash), **manifest_without_hash}
    manifest_path = request_dir / "request_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    observed_run_commands: list[list[str]] = []
    observed_parse_inputs: list[str] = []

    def fake_run(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if "parse_multiple_chains.py" in argv[1]:
            observed_parse_inputs.append(argv[argv.index("--input_path") + 1])
            write_jsonl(Path(argv[argv.index("--output_path") + 1]), parsed_payload)
        elif "assign_fixed_chains.py" in argv[1]:
            write_jsonl(Path(argv[argv.index("--output_path") + 1]), assigned_payload)
        elif "make_fixed_positions_dict.py" in argv[1]:
            write_jsonl(Path(argv[argv.index("--output_path") + 1]), fixed_payload)
        elif "protein_mpnn_run.py" in argv[1]:
            observed_run_commands.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    run_official_proteinmpnn_request(
        request_manifest_path=Path("proteinmpnn_request/request_manifest.yaml"),
        proteinmpnn_root=proteinmpnn_root,
        output_dir=tmp_path / "proteinmpnn_outputs",
    )

    assert len(observed_run_commands) == 1
    assert observed_parse_inputs == [request_dir.as_posix().rstrip("/") + "/"]
    command = observed_run_commands[0]
    assert command[command.index("--jsonl_path") + 1] == str(parsed_path)
    assert command[command.index("--chain_id_jsonl") + 1] == str(assigned_path)
    assert command[command.index("--fixed_positions_jsonl") + 1] == str(fixed_path)
    assert command[command.index("--num_seq_per_target") + 1] == "4"
    assert command[command.index("--batch_size") + 1] == "2"
    backend_manifest = yaml.safe_load(
        (tmp_path / "proteinmpnn_outputs" / "batches" / "portable" / "backend_run_manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert backend_manifest["expected_sample_count"] == 4


def _write_fake_proteinmpnn_root(root: Path) -> Path:
    for rel_path in (
        "protein_mpnn_run.py",
        "helper_scripts/parse_multiple_chains.py",
        "helper_scripts/assign_fixed_chains.py",
        "helper_scripts/make_fixed_positions_dict.py",
        "vanilla_model_weights/v_48_020.pt",
    ):
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    return root
