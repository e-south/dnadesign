"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/sampling/sample_table.py

ProteinMPNN sample-table validators for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue
from dnadesign.thread.adapters.proteinmpnn.samples import BACKEND_RUN_SCHEMA_ID, validate_sample_table

_THREAD_TO_ECO1_ISSUES = {
    "thread.proteinmpnn.sample_table_missing_columns": "eco1_rt.sampling.sample_table_missing_columns",
    "thread.proteinmpnn.sample_table_metadata_mismatch": "eco1_rt.sampling.sample_table_metadata_mismatch",
    "thread.proteinmpnn.sample_table_count_mismatch": "eco1_rt.sampling.sample_table_count_mismatch",
    "thread.proteinmpnn.sample_table_duplicate_ids": "eco1_rt.sampling.sample_table_duplicate_ids",
    "thread.proteinmpnn.sample_table_request_hash_mismatch": "eco1_rt.sampling.sample_table_request_hash_mismatch",
    "thread.proteinmpnn.sample_table_sequence_length_mismatch": (
        "eco1_rt.sampling.sample_table_sequence_length_mismatch"
    ),
}


def validate_sample_table_content(path: Path, *, output_root: Path) -> list[ContractIssue]:
    """Validate Eco1 ProteinMPNN sample table and backend-run manifest."""

    issues: list[ContractIssue] = []
    request_manifest_path = output_root / "proteinmpnn_request/request_manifest.yaml"
    backend_run_manifest_path = output_root / "proteinmpnn_outputs/backend_run_manifest.yaml"
    request_manifest = _load_yaml(request_manifest_path)
    issues.extend(
        _adapt_thread_issue(issue)
        for issue in validate_sample_table(
            path,
            request_hash=str(request_manifest["request_hash"]),
            expected_sample_count=int(request_manifest["expected_sample_count"]),
            sequence_length=int(request_manifest["canonical_position_count"]),
        )
    )
    if not backend_run_manifest_path.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.backend_run_manifest_not_materialized",
                message="Phase 2 backend ingest requires proteinmpnn_outputs/backend_run_manifest.yaml",
                path=str(backend_run_manifest_path),
            )
        )
    else:
        issues.extend(
            _validate_backend_run_manifest(
                backend_run_manifest_path,
                request_manifest_path=request_manifest_path,
                request_hash=str(request_manifest["request_hash"]),
                request_manifest=request_manifest,
            )
        )
    return issues


def _validate_backend_run_manifest(
    path: Path, *, request_manifest_path: Path, request_hash: str, request_manifest: dict[str, Any]
) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    manifest = _load_yaml(path)
    expected = {
        "schema_id": BACKEND_RUN_SCHEMA_ID,
        "status": "materialized",
        "backend_kind": "proteinmpnn",
        "request_hash": request_hash,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.sampling.backend_run_manifest_field_mismatch",
                    message=f"ProteinMPNN backend run manifest field {field!r} must match the request",
                    path=f"{path}:{field}",
                )
            )
    if not str(manifest.get("request_manifest_path", "")).strip():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.backend_run_manifest_field_mismatch",
                message="ProteinMPNN backend run manifest field 'request_manifest_path' must be non-empty",
                path=f"{path}:request_manifest_path",
            )
        )
    if manifest.get("request_manifest_hash") != "sha256:" + _sha256(request_manifest_path):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.backend_run_manifest_field_mismatch",
                message="ProteinMPNN backend run manifest field 'request_manifest_hash' must match the request",
                path=f"{path}:request_manifest_hash",
            )
        )
    for field in ("batch_id", "num_seq_per_target", "batch_size", "expected_sample_count"):
        if field not in manifest:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.sampling.backend_run_manifest_missing_batch_field",
                    message=f"ProteinMPNN backend run manifest must declare {field!r}",
                    path=f"{path}:{field}",
                )
            )
        elif manifest.get(field) != request_manifest.get(field):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.sampling.backend_run_manifest_batch_field_mismatch",
                    message=f"ProteinMPNN backend run manifest field {field!r} must match the request",
                    path=f"{path}:{field}",
                )
            )
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.backend_run_manifest_missing_runs",
                message="ProteinMPNN backend run manifest must include backend run records",
                path=f"{path}:runs",
            )
        )
    return issues


def _adapt_thread_issue(issue: Any) -> ContractIssue:
    return ContractIssue(
        check_id=_THREAD_TO_ECO1_ISSUES.get(issue.check_id, "eco1_rt.sampling.sample_table_invalid"),
        message=issue.message,
        path=issue.path,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
