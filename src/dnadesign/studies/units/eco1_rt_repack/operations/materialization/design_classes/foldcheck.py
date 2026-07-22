"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/design_classes/foldcheck.py

Expanded fold-check request materialization for Eco1 RT design classes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    load_yaml,
    write_yaml,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.constants import (
    CANDIDATE_POOL_FILE_NAME,
    DEFAULT_DESIGN_CLASSES_ROOT,
    DEFAULT_SOURCE_OUTPUT_ROOT,
    FOLDCHECK_REQUEST_DIR_NAME,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.models import (
    MaterializedDesignClassFoldCheckRequest,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_request.constants import (
    BACKEND_KIND,
    REFERENCE_STRUCTURE_ID,
    RUNTIME_KIND,
    STORAGE_POLICY,
    THRESHOLD_POLICY_ID,
    THRESHOLD_VALUES,
    WT_SEQUENCE_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_request.sequences import (
    build_foldcheck_sequence_records,
)
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri
from dnadesign.thread.foldcheck import build_foldcheck_request_manifest, write_foldcheck_fasta


def materialize_design_class_foldcheck_request(
    *,
    repo_root: Path,
    output_root: Path | None = None,
    source_output_root: Path | None = None,
    created_at: str = "2026-07-01T00:00:00Z",
    require_generated_candidates: bool = True,
) -> MaterializedDesignClassFoldCheckRequest:
    """Write a ColabFold request for the nonredundant expanded candidate pool."""

    root = repo_root.expanduser().resolve()
    class_root = _resolve(root, output_root or DEFAULT_DESIGN_CLASSES_ROOT)
    source_root = _resolve(root, source_output_root or DEFAULT_SOURCE_OUTPUT_ROOT)
    candidate_pool_path = class_root / CANDIDATE_POOL_FILE_NAME
    candidate_pool_manifest_path = class_root / "candidate_pool_manifest.yaml"
    residue_map_path = source_root / "residue_map.parquet"
    manifest_path = class_root / "design_class_manifest.yaml"
    for required in (candidate_pool_path, candidate_pool_manifest_path, residue_map_path, manifest_path):
        if not required.exists():
            raise FileNotFoundError(required)
    pool_manifest = _load_yaml(candidate_pool_manifest_path)
    if require_generated_candidates and int(pool_manifest.get("generated_candidate_table_count") or 0) == 0:
        raise ValueError("expanded fold-check request requires at least one generated design-class candidate table")
    records = build_foldcheck_sequence_records(
        candidate_table_path=candidate_pool_path,
        residue_map_path=residue_map_path,
        wt_sequence_id=WT_SEQUENCE_ID,
    )
    request_root = class_root / FOLDCHECK_REQUEST_DIR_NAME
    fasta_path = request_root / "input_sequences.fasta"
    write_foldcheck_fasta(fasta_path, records)
    request_manifest = build_foldcheck_request_manifest(
        artifact_id="eco1_rt_design_classes_v1.foldcheck_request",
        created_by="dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes",
        created_at=created_at,
        backend_kind=BACKEND_KIND,
        runtime_kind=RUNTIME_KIND,
        execution_status="planned_not_run",
        input_fasta_path=Path("input_sequences.fasta"),
        output_root=Path("colabfold_outputs"),
        sequence_records=records,
        wt_sequence_id=WT_SEQUENCE_ID,
        reference_structure_id=REFERENCE_STRUCTURE_ID,
        threshold_policy_id=THRESHOLD_POLICY_ID,
        threshold_values=THRESHOLD_VALUES,
        upstream_artifact_hashes={
            "candidate_pool": sha256_uri(candidate_pool_path),
            "residue_map": sha256_uri(residue_map_path),
            "design_class_manifest": sha256_uri(manifest_path),
        },
        storage_policy=STORAGE_POLICY,
    )
    request_manifest_path = request_root / "foldcheck_request_manifest.yaml"
    write_yaml(request_manifest_path, request_manifest)
    return MaterializedDesignClassFoldCheckRequest(
        input_fasta_path=fasta_path,
        request_manifest_path=request_manifest_path,
    )


def _resolve(repo_root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (repo_root / expanded).resolve()


def _load_yaml(path: Path) -> dict[str, object]:
    return load_yaml(path)
