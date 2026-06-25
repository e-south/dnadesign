"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/proteinmpnn/validation.py

Generic ProteinMPNN request manifest validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri
from dnadesign.thread.adapters.proteinmpnn.manifest import FALLBACK_POLICY, POSITION_BASIS, SCHEMA_ID, request_hash
from dnadesign.thread.adapters.proteinmpnn.models import ProteinMpnnRequestIssue
from dnadesign.thread.adapters.proteinmpnn.sidecars import resolve_manifest_sidecar_path


def validate_request_manifest(path: Path) -> list[ProteinMpnnRequestIssue]:
    """Validate generic ProteinMPNN request manifest and helper sidecars."""

    issues: list[ProteinMpnnRequestIssue] = []
    manifest = _load_yaml(path)
    _validate_metadata(issues, manifest=manifest, path=path)
    _validate_sidecars(issues, manifest=manifest, path=path)
    _validate_sidecar_payloads(issues, manifest=manifest, path=path)
    _validate_request_hash(issues, manifest=manifest, path=path)
    return issues


def _validate_metadata(issues: list[ProteinMpnnRequestIssue], *, manifest: Mapping[str, Any], path: Path) -> None:
    expected = {
        "schema_id": SCHEMA_ID,
        "schema_version": 1,
        "status": "materialized",
        "execution_status": "planned_not_run",
        "backend_kind": "proteinmpnn",
        "proteinmpnn_position_basis": POSITION_BASIS,
        "fallback_policy": FALLBACK_POLICY,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            issues.append(
                ProteinMpnnRequestIssue(
                    check_id="thread.proteinmpnn.request_metadata_mismatch",
                    message=f"ProteinMPNN request field {field!r} must equal {value!r}",
                    path=str(path),
                )
            )
    if manifest.get("omit_aas") != ["C"]:
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.proteinmpnn.omit_aas_mismatch",
                message="ProteinMPNN request must declare omit_aas: [C] for no-new-cysteine policy",
                path=str(path),
            )
        )
    batch_id = manifest.get("batch_id")
    num_seq_per_target = manifest.get("num_seq_per_target")
    batch_size = manifest.get("batch_size")
    if not isinstance(batch_id, str) or not batch_id.strip():
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.proteinmpnn.invalid_batch_id",
                message="ProteinMPNN request must declare a non-empty batch_id",
                path=str(path),
            )
        )
    if not isinstance(num_seq_per_target, int) or isinstance(num_seq_per_target, bool) or num_seq_per_target <= 0:
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.proteinmpnn.invalid_num_seq_per_target",
                message="ProteinMPNN request must declare positive num_seq_per_target",
                path=str(path),
            )
        )
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.proteinmpnn.invalid_batch_size",
                message="ProteinMPNN request must declare positive batch_size",
                path=str(path),
            )
        )
    if (
        isinstance(num_seq_per_target, int)
        and not isinstance(num_seq_per_target, bool)
        and isinstance(batch_size, int)
        and not isinstance(batch_size, bool)
        and batch_size > 0
        and num_seq_per_target % batch_size != 0
    ):
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.proteinmpnn.invalid_batch_divisibility",
                message="ProteinMPNN num_seq_per_target must be divisible by batch_size",
                path=str(path),
            )
        )


def _validate_sidecars(issues: list[ProteinMpnnRequestIssue], *, manifest: Mapping[str, Any], path: Path) -> None:
    sidecar_paths = manifest.get("sidecar_paths")
    sidecar_hashes = manifest.get("sidecar_hashes")
    if not isinstance(sidecar_paths, Mapping) or not isinstance(sidecar_hashes, Mapping):
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.proteinmpnn.missing_sidecar_hashes",
                message="ProteinMPNN request must declare sidecar_paths and sidecar_hashes",
                path=str(path),
            )
        )
        return
    for name, sidecar in sidecar_paths.items():
        sidecar_path = resolve_manifest_sidecar_path(path, sidecar)
        if not sidecar_path.exists():
            issues.append(
                ProteinMpnnRequestIssue(
                    check_id="thread.proteinmpnn.sidecar_missing",
                    message=f"ProteinMPNN request sidecar {name!r} is missing",
                    path=str(path),
                )
            )
            continue
        if sidecar_hashes.get(name) != sha256_uri(sidecar_path):
            issues.append(
                ProteinMpnnRequestIssue(
                    check_id="thread.proteinmpnn.sidecar_hash_mismatch",
                    message=f"ProteinMPNN request sidecar {name!r} hash must match current file",
                    path=str(path),
                )
            )


def _validate_sidecar_payloads(
    issues: list[ProteinMpnnRequestIssue], *, manifest: Mapping[str, Any], path: Path
) -> None:
    sidecar_paths = manifest.get("sidecar_paths")
    target_name = str(manifest.get("proteinmpnn_name", ""))
    chain_id = str(manifest.get("proteinmpnn_design_chain", ""))
    if not isinstance(sidecar_paths, Mapping):
        return
    fixed_payload = manifest.get("fixed_positions_jsonl")
    fixed_path = resolve_manifest_sidecar_path(path, sidecar_paths.get("fixed_positions_jsonl", ""))
    fixed_record = _jsonl_record_or_issue(issues, sidecar_path=fixed_path, sidecar_name="fixed_positions", path=path)
    if fixed_record is not None and fixed_record != fixed_payload:
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.proteinmpnn.sidecar_payload_mismatch",
                message="fixed_positions.jsonl must match the fixed_positions_jsonl manifest payload",
                path=str(path),
            )
        )
    assigned_path = resolve_manifest_sidecar_path(path, sidecar_paths.get("assigned_chains_jsonl", ""))
    assigned_record = _jsonl_record_or_issue(
        issues, sidecar_path=assigned_path, sidecar_name="assigned_chains", path=path
    )
    if assigned_record is not None and assigned_record != {target_name: [[chain_id], []]}:
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.proteinmpnn.sidecar_payload_mismatch",
                message="assigned_chains.jsonl must select the design chain and no fixed-context chains",
                path=str(path),
            )
        )
    parsed_path = resolve_manifest_sidecar_path(path, sidecar_paths.get("parsed_pdbs_jsonl", ""))
    parsed_payload = _jsonl_record_or_issue(issues, sidecar_path=parsed_path, sidecar_name="parsed_pdbs", path=path)
    if parsed_payload is not None:
        if parsed_payload.get("name") != target_name or parsed_payload.get("num_of_chains") != 1:
            issues.append(
                ProteinMpnnRequestIssue(
                    check_id="thread.proteinmpnn.sidecar_payload_mismatch",
                    message="parsed_pdbs.jsonl must describe the exported single-chain ProteinMPNN target",
                    path=str(path),
                )
            )


def _validate_request_hash(issues: list[ProteinMpnnRequestIssue], *, manifest: Mapping[str, Any], path: Path) -> None:
    observed = manifest.get("request_hash")
    payload = {key: value for key, value in manifest.items() if key != "request_hash"}
    expected = request_hash(payload)
    if observed != expected:
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.proteinmpnn.request_hash_mismatch",
                message="ProteinMPNN request_hash must match the canonical request payload",
                path=str(path),
            )
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded


def _load_jsonl_record(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return records[0]


def _jsonl_record_or_issue(
    issues: list[ProteinMpnnRequestIssue], *, sidecar_path: Path, sidecar_name: str, path: Path
) -> dict[str, Any] | None:
    if not sidecar_path.exists() or not sidecar_path.is_file():
        return None
    try:
        return _load_jsonl_record(sidecar_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.proteinmpnn.sidecar_payload_invalid",
                message=f"ProteinMPNN request sidecar {sidecar_name!r} must contain one JSON object: {exc}",
                path=str(path),
            )
        )
        return None
