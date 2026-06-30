"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/proteinmpnn/test_request_adapter.py

ProteinMPNN request adapter contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from dnadesign.thread.adapters.proteinmpnn import (
    assigned_chains_payload,
    build_request_manifest,
    fixed_positions_payload,
    proteinmpnn_run_commands,
    request_hash,
    validate_request_manifest,
    write_jsonl,
)


def test_proteinmpnn_manifest_validator_accepts_helper_sidecars(tmp_path: Path) -> None:
    parsed_path = tmp_path / "parsed_pdbs.jsonl"
    assigned_path = tmp_path / "assigned_chains.jsonl"
    fixed_path = tmp_path / "fixed_positions.jsonl"
    pdb_path = tmp_path / "target.pdb"
    pdb_path.write_text("END\n", encoding="utf-8")
    parsed_payload = {
        "name": "target",
        "num_of_chains": 1,
        "seq": "ACD",
        "seq_chain_A": "ACD",
        "coords_chain_A": {"N_chain_A": [], "CA_chain_A": [], "C_chain_A": [], "O_chain_A": []},
    }
    write_jsonl(parsed_path, parsed_payload)
    write_jsonl(assigned_path, assigned_chains_payload(target_name="target", chain_id="A"))
    write_jsonl(fixed_path, fixed_positions_payload(target_name="target", chain_id="A", fixed_positions=[1, 3]))
    manifest_without_hash = build_request_manifest(
        artifact_id="test.proteinmpnn_request",
        created_by="test",
        profile_id="profile",
        mask_policy_id="mask",
        target_name="target",
        chain_id="A",
        sidecar_paths={
            "chain_a_backbone_pdb": pdb_path,
            "parsed_pdbs_jsonl": parsed_path,
            "assigned_chains_jsonl": assigned_path,
            "fixed_positions_jsonl": fixed_path,
        },
        upstream_artifact_hashes={},
        source_thread_plan={"path": "thread_plan.yaml"},
        canonical_to_mpnn={3: 1, 4: 2, 5: 3},
        fixed_positions=[1, 3],
        mutable_positions=[2],
        excluded_positions=[],
        seed_set=[101],
        temperatures=[0.1, 0.3],
        batch_id="test_batch",
        num_seq_per_target=1,
        batch_size=1,
        expected_sample_count=2,
    )
    manifest = {"request_hash": request_hash(manifest_without_hash), **manifest_without_hash}
    manifest_path = tmp_path / "request_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    assert validate_request_manifest(manifest_path) == []


def test_proteinmpnn_manifest_validator_accepts_colocated_sidecars_from_another_host(tmp_path: Path) -> None:
    parsed_path = tmp_path / "parsed_pdbs.jsonl"
    assigned_path = tmp_path / "assigned_chains.jsonl"
    fixed_path = tmp_path / "fixed_positions.jsonl"
    pdb_path = tmp_path / "target.pdb"
    pdb_path.write_text("END\n", encoding="utf-8")
    write_jsonl(
        parsed_path,
        {
            "name": "target",
            "num_of_chains": 1,
            "seq": "ACD",
            "seq_chain_A": "ACD",
            "coords_chain_A": {"N_chain_A": [], "CA_chain_A": [], "C_chain_A": [], "O_chain_A": []},
        },
    )
    write_jsonl(assigned_path, assigned_chains_payload(target_name="target", chain_id="A"))
    write_jsonl(fixed_path, fixed_positions_payload(target_name="target", chain_id="A", fixed_positions=[1, 3]))
    manifest_without_hash = build_request_manifest(
        artifact_id="test.proteinmpnn_request",
        created_by="test",
        profile_id="profile",
        mask_policy_id="mask",
        target_name="target",
        chain_id="A",
        sidecar_paths={
            "chain_a_backbone_pdb": pdb_path,
            "parsed_pdbs_jsonl": parsed_path,
            "assigned_chains_jsonl": assigned_path,
            "fixed_positions_jsonl": fixed_path,
        },
        upstream_artifact_hashes={},
        source_thread_plan={"path": "/other/host/thread_plan.yaml"},
        canonical_to_mpnn={3: 1, 4: 2, 5: 3},
        fixed_positions=[1, 3],
        mutable_positions=[2],
        excluded_positions=[],
        seed_set=[101],
        temperatures=[0.1],
        batch_id="test_batch",
        num_seq_per_target=1,
        batch_size=1,
        expected_sample_count=1,
    )
    manifest_without_hash["sidecar_paths"] = {
        name: f"/other/host/proteinmpnn_request/{Path(path).name}"
        for name, path in manifest_without_hash["sidecar_paths"].items()
    }
    manifest = {"request_hash": request_hash(manifest_without_hash), **manifest_without_hash}
    manifest_path = tmp_path / "request_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    assert validate_request_manifest(manifest_path) == []


def test_proteinmpnn_manifest_validator_rejects_rehashed_wrong_fixed_sidecar(tmp_path: Path) -> None:
    parsed_path = tmp_path / "parsed_pdbs.jsonl"
    assigned_path = tmp_path / "assigned_chains.jsonl"
    fixed_path = tmp_path / "fixed_positions.jsonl"
    pdb_path = tmp_path / "target.pdb"
    pdb_path.write_text("END\n", encoding="utf-8")
    write_jsonl(parsed_path, {"name": "target", "num_of_chains": 1})
    write_jsonl(assigned_path, assigned_chains_payload(target_name="target", chain_id="A"))
    write_jsonl(fixed_path, fixed_positions_payload(target_name="target", chain_id="A", fixed_positions=[1, 3]))
    manifest_without_hash = build_request_manifest(
        artifact_id="test.proteinmpnn_request",
        created_by="test",
        profile_id="profile",
        mask_policy_id="mask",
        target_name="target",
        chain_id="A",
        sidecar_paths={
            "chain_a_backbone_pdb": pdb_path,
            "parsed_pdbs_jsonl": parsed_path,
            "assigned_chains_jsonl": assigned_path,
            "fixed_positions_jsonl": fixed_path,
        },
        upstream_artifact_hashes={},
        source_thread_plan={"path": "thread_plan.yaml"},
        canonical_to_mpnn={3: 1, 4: 2, 5: 3},
        fixed_positions=[1, 3],
        mutable_positions=[2],
        excluded_positions=[],
        seed_set=[101],
        temperatures=[0.1],
        batch_id="test_batch",
        num_seq_per_target=1,
        batch_size=1,
        expected_sample_count=1,
    )
    manifest = {"request_hash": request_hash(manifest_without_hash), **manifest_without_hash}
    fixed_path.write_text(json.dumps({"target": {"A": [1]}}) + "\n", encoding="utf-8")
    manifest["sidecar_hashes"]["fixed_positions_jsonl"] = _sha256_uri(fixed_path)
    manifest["request_hash"] = request_hash({key: value for key, value in manifest.items() if key != "request_hash"})
    manifest_path = tmp_path / "request_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    issues = validate_request_manifest(manifest_path)

    assert [issue.check_id for issue in issues] == ["thread.proteinmpnn.sidecar_payload_mismatch"]


def test_proteinmpnn_request_hash_ignores_provenance_only_fields(tmp_path: Path) -> None:
    manifest_without_hash = _minimal_request_manifest(tmp_path)
    changed_provenance = dict(manifest_without_hash)
    changed_provenance["source_thread_plan"] = {
        "path": "thread_plan.yaml",
        "hash": "sha256:" + "1" * 64,
        "request_hash": "sha256:" + "2" * 64,
    }
    changed_provenance["upstream_artifact_hashes"] = {"mask_set": "sha256:" + "3" * 64}
    changed_provenance["created_by"] = "other.materializer"

    assert request_hash(changed_provenance) == request_hash(manifest_without_hash)


def test_proteinmpnn_request_hash_changes_for_executable_fields(tmp_path: Path) -> None:
    manifest_without_hash = _minimal_request_manifest(tmp_path)
    changed_positions = dict(manifest_without_hash)
    changed_positions["fixed_positions_jsonl"] = {"target": {"A": [1]}}

    assert request_hash(changed_positions) != request_hash(manifest_without_hash)


def test_proteinmpnn_run_commands_use_requested_chain_id() -> None:
    commands = proteinmpnn_run_commands(seed_set=[101], temperatures=[0.1], chain_id="B")

    assign_command = next(command for command in commands if command["name"] == "assign_fixed_chains")

    chain_arg_index = assign_command["argv"].index("--chain_list") + 1
    assert assign_command["argv"][chain_arg_index] == "B"


def _minimal_request_manifest(tmp_path: Path) -> dict[str, object]:
    parsed_path = tmp_path / "parsed_pdbs.jsonl"
    assigned_path = tmp_path / "assigned_chains.jsonl"
    fixed_path = tmp_path / "fixed_positions.jsonl"
    pdb_path = tmp_path / "target.pdb"
    pdb_path.write_text("END\n", encoding="utf-8")
    write_jsonl(parsed_path, {"name": "target", "num_of_chains": 1})
    write_jsonl(assigned_path, assigned_chains_payload(target_name="target", chain_id="A"))
    write_jsonl(fixed_path, fixed_positions_payload(target_name="target", chain_id="A", fixed_positions=[1, 3]))
    return build_request_manifest(
        artifact_id="test.proteinmpnn_request",
        created_by="test",
        profile_id="profile",
        mask_policy_id="mask",
        target_name="target",
        chain_id="A",
        sidecar_paths={
            "chain_a_backbone_pdb": pdb_path,
            "parsed_pdbs_jsonl": parsed_path,
            "assigned_chains_jsonl": assigned_path,
            "fixed_positions_jsonl": fixed_path,
        },
        upstream_artifact_hashes={"mask_set": "sha256:" + "0" * 64},
        source_thread_plan={"path": "thread_plan.yaml"},
        canonical_to_mpnn={3: 1, 4: 2, 5: 3},
        fixed_positions=[1, 3],
        mutable_positions=[2],
        excluded_positions=[],
        seed_set=[101],
        temperatures=[0.1],
        batch_id="test_batch",
        num_seq_per_target=1,
        batch_size=1,
        expected_sample_count=1,
    )


def _sha256_uri(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
