"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_request/pipeline.py

Materialize an Eco1 ColabFold-planned fold-check request.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    find_repo_root,
    resolve_output_root,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_request.constants import (
    ARTIFACT_ID,
    BACKEND_KIND,
    CREATED_BY,
    DEFAULT_OUTPUT_ROOT,
    EXECUTION_STATUS,
    REFERENCE_STRUCTURE_ID,
    REQUEST_DIR_NAME,
    RUNTIME_KIND,
    STORAGE_POLICY,
    THRESHOLD_POLICY_ID,
    THRESHOLD_VALUES,
    WT_SEQUENCE_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_request.models import (
    MaterializedFoldCheckRequestArtifacts,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_request.sequences import (
    build_foldcheck_sequence_records,
)
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri
from dnadesign.thread.foldcheck import build_foldcheck_request_manifest, write_foldcheck_fasta


def materialize_foldcheck_request(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    created_at: str = "2026-06-25T00:00:00Z",
) -> MaterializedFoldCheckRequestArtifacts:
    """Materialize fold-check FASTA and manifest without running ColabFold."""

    root = (repo_root or find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = resolve_output_root(root, output_root or DEFAULT_OUTPUT_ROOT)
    request_root = out_root / REQUEST_DIR_NAME
    request_root.mkdir(parents=True, exist_ok=True)

    candidate_table_path = out_root / "candidate_table.parquet"
    residue_map_path = out_root / "residue_map.parquet"
    proteinmpnn_request_path = out_root / "proteinmpnn_request/request_manifest.yaml"
    for required_path in (candidate_table_path, residue_map_path, proteinmpnn_request_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    records = build_foldcheck_sequence_records(
        candidate_table_path=candidate_table_path,
        residue_map_path=residue_map_path,
        wt_sequence_id=WT_SEQUENCE_ID,
    )
    fasta_path = request_root / "input_sequences.fasta"
    write_foldcheck_fasta(fasta_path, records)
    manifest = build_foldcheck_request_manifest(
        artifact_id=ARTIFACT_ID,
        created_by=CREATED_BY,
        created_at=created_at,
        backend_kind=BACKEND_KIND,
        runtime_kind=RUNTIME_KIND,
        execution_status=EXECUTION_STATUS,
        input_fasta_path=fasta_path,
        output_root=request_root / "colabfold_outputs",
        sequence_records=records,
        wt_sequence_id=WT_SEQUENCE_ID,
        reference_structure_id=REFERENCE_STRUCTURE_ID,
        threshold_policy_id=THRESHOLD_POLICY_ID,
        threshold_values=THRESHOLD_VALUES,
        upstream_artifact_hashes={
            "candidate_table": sha256_uri(candidate_table_path),
            "residue_map": sha256_uri(residue_map_path),
            "proteinmpnn_request": sha256_uri(proteinmpnn_request_path),
        },
        storage_policy=STORAGE_POLICY,
    )
    manifest_path = request_root / "foldcheck_request_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return MaterializedFoldCheckRequestArtifacts(
        input_fasta_path=fasta_path,
        request_manifest_path=manifest_path,
    )
