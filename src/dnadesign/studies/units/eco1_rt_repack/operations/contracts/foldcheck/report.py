"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/foldcheck/report.py

Eco1 fold-check report validator.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.common import _load_yaml
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    resolve_output_root,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_report.constants import (
    REFERENCE_BACKBONE_RELATIVE_PATH,
    RESIDUE_MAP_FILE_NAME,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_report.reference import (
    mapped_reference_positions,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_request.constants import (
    WT_SEQUENCE_ID,
)
from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT
from dnadesign.thread.adapters.colabfold.manifest import file_sha256_uri, ordered_positions_hash
from dnadesign.thread.adapters.colabfold.outputs import MAPPED_REFERENCE_COORDINATE_BASIS
from dnadesign.thread.foldcheck import validate_foldcheck_report

_V2_REFERENCE_LINEAGE_FIELDS = {
    "reference_structure_hash",
    "reference_mobile_positions_hash",
    "reference_coordinate_basis",
}


def validate_foldcheck_report_content(
    path: Path,
    *,
    repo_root: Path,
    output_root: Path,
    source_output_root: Path | None = None,
) -> list[ContractIssue]:
    """Validate the Eco1 fold-check report against the current request and candidate table."""

    request_manifest_path = output_root / "foldcheck_request/foldcheck_request_manifest.yaml"
    if not request_manifest_path.exists():
        return [
            ContractIssue(
                check_id="eco1_rt.foldcheck_report.request_manifest_missing",
                message="fold-check report validation requires the current fold-check request manifest",
                path=str(request_manifest_path),
            )
        ]
    manifest = _load_yaml(request_manifest_path)
    candidate_table_path = _first_existing_path(
        output_root / "candidate_table.parquet",
        output_root / "candidate_pool.parquet",
    )
    expected_candidate_ids = _expected_candidate_ids(candidate_table_path, manifest=manifest)
    issues = _candidate_authority_issues(candidate_table_path, output_root=output_root)
    issues.extend(
        ContractIssue(check_id=issue.check_id, message=issue.message, path=issue.path)
        for issue in validate_foldcheck_report(
            path,
            request_hash=str(manifest.get("request_hash", "")),
            expected_candidate_ids=expected_candidate_ids,
            wt_candidate_id=WT_SEQUENCE_ID,
        )
    )
    issues.extend(_validate_report_sequence_hashes(path, manifest))
    report_schema = pq.read_schema(path)
    if (report_schema.metadata or {}).get(b"schema_version", b"1") == b"2" and _V2_REFERENCE_LINEAGE_FIELDS <= set(
        report_schema.names
    ):
        issues.extend(
            _validate_v2_reference_lineage(
                path,
                repo_root=repo_root,
                output_root=output_root,
                source_output_root=source_output_root,
            )
        )
    return issues


def _validate_v2_reference_lineage(
    path: Path,
    *,
    repo_root: Path,
    output_root: Path,
    source_output_root: Path | None,
) -> list[ContractIssue]:
    source_root = resolve_output_root(repo_root, source_output_root or DEFAULT_THREAD_OUTPUT_ROOT)
    reference_path = _first_existing_path(
        output_root / REFERENCE_BACKBONE_RELATIVE_PATH,
        source_root / REFERENCE_BACKBONE_RELATIVE_PATH,
    )
    residue_map_path = _first_existing_path(
        output_root / RESIDUE_MAP_FILE_NAME,
        source_root / RESIDUE_MAP_FILE_NAME,
    )
    if reference_path is None or residue_map_path is None:
        return [
            ContractIssue(
                check_id="eco1_rt.foldcheck_report.reference_lineage_authority_missing",
                message="v2 fold-check validation requires the bound reference backbone and residue map",
                path=str(path),
            )
        ]
    expected = {
        "reference_structure_hash": file_sha256_uri(reference_path),
        "reference_mobile_positions_hash": ordered_positions_hash(mapped_reference_positions(residue_map_path)),
        "reference_coordinate_basis": MAPPED_REFERENCE_COORDINATE_BASIS,
    }
    rows = pq.read_table(
        path,
        columns=sorted(expected),
    ).to_pylist()
    issues: list[ContractIssue] = []
    for field, expected_value in expected.items():
        observed = {str(row.get(field, "")) for row in rows}
        if observed != {expected_value}:
            issues.append(
                ContractIssue(
                    check_id=f"eco1_rt.foldcheck_report.{field}_mismatch",
                    message=f"v2 fold-check {field} does not match the current study authority",
                    path=str(path),
                )
            )
    return issues


def _first_existing_path(*paths: Path) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def _expected_candidate_ids(candidate_table_path: Path | None, *, manifest: Mapping[str, object]) -> set[str]:
    expected = {WT_SEQUENCE_ID}
    if candidate_table_path is None:
        return expected | set(_manifest_sequence_hashes(manifest))
    for row in pq.read_table(candidate_table_path).to_pylist():
        if str(row.get("status")) == "accepted":
            expected.add(str(row["candidate_id"]))
    return expected


def _candidate_authority_issues(candidate_table_path: Path | None, *, output_root: Path) -> list[ContractIssue]:
    if candidate_table_path is not None:
        return []
    return [
        ContractIssue(
            check_id="eco1_rt.foldcheck_report.candidate_authority_missing",
            message="fold-check report validation requires candidate_table.parquet or candidate_pool.parquet",
            path=str(output_root),
        )
    ]


def _validate_report_sequence_hashes(path: Path, manifest: Mapping[str, object]) -> list[ContractIssue]:
    expected_hashes = _manifest_sequence_hashes(manifest)
    issues: list[ContractIssue] = []
    for index, row in enumerate(pq.read_table(path).to_pylist()):
        candidate_id = str(row.get("candidate_id", ""))
        expected_hash = expected_hashes.get(candidate_id)
        if expected_hash is None:
            continue
        if str(row.get("input_sequence_hash", "")) != expected_hash:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.foldcheck_report.sequence_hash_mismatch",
                    message=f"fold-check row {candidate_id!r} does not match the current request sequence hash",
                    path=f"{path}:row[{index}]",
                )
            )
    return issues


def _manifest_sequence_hashes(manifest: Mapping[str, object]) -> dict[str, str]:
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list):
        return {}
    return {
        str(row["sequence_id"]): str(row["sequence_hash"])
        for row in sequences
        if isinstance(row, dict) and "sequence_id" in row and "sequence_hash" in row
    }
