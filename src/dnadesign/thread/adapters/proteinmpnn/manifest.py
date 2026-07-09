"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/proteinmpnn/manifest.py

ProteinMPNN request manifest construction and command declaration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri
from dnadesign.thread.adapters.proteinmpnn.sidecars import fixed_positions_payload

SCHEMA_ID = "proteinmpnn.fixed_backbone_request"
POSITION_BASIS = "chain_local_1_indexed_after_export"
FALLBACK_POLICY = "explicit_no_fallback"
_EXECUTABLE_REQUEST_FIELDS = (
    "schema_id",
    "schema_version",
    "backend_kind",
    "proteinmpnn_name",
    "proteinmpnn_design_chain",
    "proteinmpnn_position_basis",
    "canonical_position_count",
    "fixed_position_count",
    "mutable_position_count",
    "excluded_missing_backbone_positions",
    "omit_aas",
    "fallback_policy",
    "seed_set",
    "temperature_schedule",
    "batch_id",
    "num_seq_per_target",
    "batch_size",
    "expected_sample_count",
    "canonical_to_proteinmpnn_position",
    "fixed_positions_jsonl",
    "mutable_positions_by_chain",
    "sidecar_hashes",
    "run_commands",
)


def build_request_manifest(
    *,
    artifact_id: str,
    created_by: str,
    profile_id: str | None,
    mask_policy_id: str | None,
    target_name: str,
    chain_id: str,
    sidecar_paths: Mapping[str, Path],
    upstream_artifact_hashes: Mapping[str, str],
    source_thread_plan: Mapping[str, Any],
    canonical_to_mpnn: Mapping[int, int],
    fixed_positions: list[int],
    mutable_positions: list[int],
    excluded_positions: list[int],
    seed_set: list[int],
    temperatures: list[float],
    batch_id: str,
    num_seq_per_target: int,
    batch_size: int,
    expected_sample_count: int,
) -> dict[str, Any]:
    """Build a hash-linked ProteinMPNN request manifest without its self-hash."""

    serialized_sidecar_paths = {name: str(path) for name, path in sidecar_paths.items()}
    sidecar_hashes = {name: sha256_uri(path) for name, path in sidecar_paths.items()}
    fixed_payload = fixed_positions_payload(
        target_name=target_name,
        chain_id=chain_id,
        fixed_positions=fixed_positions,
    )
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": 1,
        "artifact_id": artifact_id,
        "status": "materialized",
        "execution_status": "planned_not_run",
        "created_by": created_by,
        "backend_kind": "proteinmpnn",
        "profile_id": profile_id,
        "mask_policy_id": mask_policy_id,
        "proteinmpnn_name": target_name,
        "proteinmpnn_design_chain": chain_id,
        "proteinmpnn_position_basis": POSITION_BASIS,
        "canonical_position_count": len(canonical_to_mpnn),
        "fixed_position_count": len(fixed_positions),
        "mutable_position_count": len(mutable_positions),
        "excluded_missing_backbone_positions": excluded_positions,
        "omit_aas": ["C"],
        "fallback_policy": FALLBACK_POLICY,
        "seed_set": seed_set,
        "temperature_schedule": temperatures,
        "batch_id": batch_id,
        "num_seq_per_target": num_seq_per_target,
        "batch_size": batch_size,
        "expected_sample_count": expected_sample_count,
        "source_thread_plan": dict(source_thread_plan),
        "canonical_to_proteinmpnn_position": {str(key): value for key, value in sorted(canonical_to_mpnn.items())},
        "fixed_positions_jsonl": fixed_payload,
        "mutable_positions_by_chain": {chain_id: mutable_positions},
        "sidecar_paths": serialized_sidecar_paths,
        "sidecar_hashes": sidecar_hashes,
        "upstream_artifact_hashes": dict(upstream_artifact_hashes),
        "run_commands": proteinmpnn_run_commands(
            seed_set=seed_set,
            temperatures=temperatures,
            chain_id=chain_id,
            fixed_positions=fixed_positions,
            num_seq_per_target=num_seq_per_target,
            batch_size=batch_size,
        ),
    }


def proteinmpnn_run_commands(
    *,
    seed_set: Sequence[int],
    temperatures: Sequence[float],
    chain_id: str,
    fixed_positions: Sequence[int],
    num_seq_per_target: int = 1,
    batch_size: int = 1,
) -> list[dict[str, Any]]:
    """Return the official ProteinMPNN helper/run command shape without executing it."""

    temp_text = " ".join(f"{temperature:g}" for temperature in temperatures)
    commands = [
        {
            "name": "parse_multiple_chains",
            "argv": [
                "python",
                "helper_scripts/parse_multiple_chains.py",
                "--input_path",
                "proteinmpnn_request/",
                "--output_path",
                "proteinmpnn_request/parsed_pdbs.jsonl",
            ],
        },
        {
            "name": "assign_fixed_chains",
            "argv": [
                "python",
                "helper_scripts/assign_fixed_chains.py",
                "--input_path",
                "proteinmpnn_request/parsed_pdbs.jsonl",
                "--output_path",
                "proteinmpnn_request/assigned_chains.jsonl",
                "--chain_list",
                chain_id,
            ],
        },
        {
            "name": "make_fixed_positions",
            "argv": [
                "python",
                "helper_scripts/make_fixed_positions_dict.py",
                "--input_path",
                "proteinmpnn_request/parsed_pdbs.jsonl",
                "--output_path",
                "proteinmpnn_request/fixed_positions.jsonl",
                "--chain_list",
                chain_id,
                "--position_list",
                " ".join(str(position) for position in fixed_positions),
            ],
        },
    ]
    for seed in seed_set:
        commands.append(
            {
                "name": f"protein_mpnn_run_seed_{seed}",
                "argv": [
                    "python",
                    "protein_mpnn_run.py",
                    "--jsonl_path",
                    "proteinmpnn_request/parsed_pdbs.jsonl",
                    "--chain_id_jsonl",
                    "proteinmpnn_request/assigned_chains.jsonl",
                    "--fixed_positions_jsonl",
                    "proteinmpnn_request/fixed_positions.jsonl",
                    "--out_folder",
                    "proteinmpnn_outputs",
                    "--num_seq_per_target",
                    str(num_seq_per_target),
                    "--sampling_temp",
                    temp_text,
                    "--seed",
                    str(seed),
                    "--batch_size",
                    str(batch_size),
                    "--omit_AAs",
                    "C",
                    "--save_score",
                    "1",
                ],
            }
        )
    return commands


def request_hash(payload: Mapping[str, Any]) -> str:
    """Hash the executable ProteinMPNN request payload.

    Provenance fields such as upstream artifact hashes, source paths, and author
    metadata stay in the manifest but do not define the backend execution.
    """

    executable_payload = {field: payload[field] for field in _EXECUTABLE_REQUEST_FIELDS if field in payload}
    encoded = json.dumps(executable_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
