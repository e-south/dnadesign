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

_PROTEINMPNN_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYX")


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
    omit_aas = manifest.get("omit_aas")
    if (
        not isinstance(omit_aas, list)
        or any(not isinstance(aa, str) or aa not in _PROTEINMPNN_AMINO_ACIDS for aa in omit_aas)
        or len(set(omit_aas)) != len(omit_aas)
    ):
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.proteinmpnn.invalid_omit_aas",
                message="ProteinMPNN omit_aas must be a unique list of supported one-letter amino-acid codes",
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
    omit_sidecar = sidecar_paths.get("omit_AA_jsonl")
    if omit_sidecar:
        omit_path = resolve_manifest_sidecar_path(path, omit_sidecar)
        omit_payload = _jsonl_record_or_issue(
            issues,
            sidecar_path=omit_path,
            sidecar_name="omit_AA_jsonl",
            path=path,
        )
        if omit_payload is not None and parsed_payload is not None:
            _validate_omit_aa_payload(
                issues,
                omit_payload=omit_payload,
                parsed_payload=parsed_payload,
                manifest=manifest,
                target_name=target_name,
                chain_id=chain_id,
                path=path,
            )


def _validate_omit_aa_payload(
    issues: list[ProteinMpnnRequestIssue],
    *,
    omit_payload: Mapping[str, Any],
    parsed_payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    target_name: str,
    chain_id: str,
    path: Path,
) -> None:
    errors: list[str] = []
    if set(omit_payload) != {target_name}:
        errors.append(f"target keys must equal [{target_name!r}]")
    target_payload = omit_payload.get(target_name)
    if not isinstance(target_payload, Mapping) or set(target_payload) != {chain_id}:
        errors.append(f"target payload must contain only chain {chain_id!r}")
        _append_omit_aa_issue(issues, errors=errors, path=path)
        return
    groups = target_payload.get(chain_id)
    if not isinstance(groups, list):
        errors.append("chain payload must be a list of [positions, omitted_amino_acids] groups")
        _append_omit_aa_issue(issues, errors=errors, path=path)
        return
    sequence = parsed_payload.get(f"seq_chain_{chain_id}")
    if not isinstance(sequence, str) or not sequence:
        errors.append(f"parsed_pdbs.jsonl must contain a non-empty seq_chain_{chain_id}")
        _append_omit_aa_issue(issues, errors=errors, path=path)
        return
    mutable_by_chain = manifest.get("mutable_positions_by_chain")
    mutable_values = mutable_by_chain.get(chain_id) if isinstance(mutable_by_chain, Mapping) else None
    if not isinstance(mutable_values, list) or any(
        not isinstance(position, int) or isinstance(position, bool) for position in mutable_values
    ):
        errors.append("manifest mutable_positions_by_chain must contain integer positions for the design chain")
        _append_omit_aa_issue(issues, errors=errors, path=path)
        return
    expected_positions = set(mutable_values)
    observed_positions: set[int] = set()
    for group_index, group in enumerate(groups):
        if not isinstance(group, list) or len(group) != 2:
            errors.append(f"group {group_index} must contain positions and one omitted-amino-acid string")
            continue
        positions, omitted = group
        if not isinstance(positions, list) or not positions:
            errors.append(f"group {group_index} positions must be a non-empty list")
            continue
        if not isinstance(omitted, str) or not omitted:
            errors.append(f"group {group_index} omitted amino acids must be a non-empty string")
            continue
        if len(set(omitted)) != len(omitted) or any(aa not in _PROTEINMPNN_AMINO_ACIDS for aa in omitted):
            errors.append(f"group {group_index} contains duplicate or unsupported amino-acid codes")
        for position in positions:
            if not isinstance(position, int) or isinstance(position, bool):
                errors.append(f"group {group_index} positions must be integers")
                continue
            if position < 1 or position > len(sequence):
                errors.append(f"position {position} is outside the exported chain length {len(sequence)}")
                continue
            if position not in expected_positions:
                errors.append(f"position {position} is not mutable in the request manifest")
            if position in observed_positions:
                errors.append(f"position {position} appears in more than one omit group")
            observed_positions.add(position)
    extra_positions = sorted(observed_positions - expected_positions)
    if extra_positions:
        errors.append(f"omit sidecar contains non-mutable positions {extra_positions}")
    _append_omit_aa_issue(issues, errors=errors, path=path)


def _append_omit_aa_issue(
    issues: list[ProteinMpnnRequestIssue],
    *,
    errors: list[str],
    path: Path,
) -> None:
    if not errors:
        return
    issues.append(
        ProteinMpnnRequestIssue(
            check_id="thread.proteinmpnn.invalid_omit_aa_jsonl",
            message="Invalid ProteinMPNN omit_AA_jsonl payload: " + "; ".join(errors),
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
