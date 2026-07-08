"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/generation_policies/foldcheck.py

Materialize Eco1 RT v2 generation-policy fold-check inputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import find_repo_root
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_request.constants import (
    BACKEND_KIND,
    EXECUTION_STATUS,
    REFERENCE_STRUCTURE_ID,
    REQUEST_DIR_NAME,
    RUNTIME_KIND,
    THRESHOLD_POLICY_ID,
    THRESHOLD_VALUES,
    WT_SEQUENCE_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_request.sequences import (
    build_foldcheck_sequence_records,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.candidate_pool import (
    materialize_generation_policy_candidate_pool,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    DEFAULT_CREATED_AT,
    DEFAULT_GENERATION_POLICIES_ROOT,
    DEFAULT_SOURCE_OUTPUT_ROOT,
    GENERATION_POLICY_VERSION,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.manifest_io import (
    load_valid_generation_policy_manifest,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.models import (
    MaterializedGenerationPolicyFoldCheckRequest,
)
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri
from dnadesign.thread.foldcheck import build_foldcheck_request_manifest, write_foldcheck_fasta

_ARTIFACT_ID = "eco1_rt_generation_policies_v2.foldcheck_request"
_CREATED_BY = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.foldcheck"
_LOCAL_STORAGE_POLICY = {
    "raw_fold_outputs": "local_runtime_storage",
    "preferred_runtime_locus": "local_julius_colabfold",
    "sync_scope": "foldcheck_request_manifest_and_colabfold_outputs",
}


def materialize_generation_policy_foldcheck_request(
    *,
    repo_root: Path | None = None,
    generation_policy_root: Path | None = None,
    source_output_root: Path | None = None,
    created_at: str = DEFAULT_CREATED_AT,
) -> MaterializedGenerationPolicyFoldCheckRequest:
    """Materialize a ColabFold-ready FASTA and request manifest for v2 candidates."""

    root = (repo_root or find_repo_root(Path.cwd())).expanduser().resolve()
    policy_root = _resolve_path(root, generation_policy_root or DEFAULT_GENERATION_POLICIES_ROOT)
    source_root = _resolve_path(root, source_output_root or DEFAULT_SOURCE_OUTPUT_ROOT)
    policy_manifest_path = policy_root / "generation_policy_manifest.yaml"
    load_valid_generation_policy_manifest(policy_manifest_path)

    pool_result = materialize_generation_policy_candidate_pool(
        repo_root=root,
        generation_policy_root=policy_root,
        created_at=created_at,
    )
    residue_map_path = source_root / "residue_map.parquet"
    if not residue_map_path.exists():
        raise FileNotFoundError(residue_map_path)

    request_root = policy_root / REQUEST_DIR_NAME
    request_root.mkdir(parents=True, exist_ok=True)
    records = build_foldcheck_sequence_records(
        candidate_table_path=pool_result.candidate_pool_path,
        residue_map_path=residue_map_path,
        wt_sequence_id=WT_SEQUENCE_ID,
    )
    fasta_path = request_root / "input_sequences.fasta"
    write_foldcheck_fasta(fasta_path, records)
    request_manifest = build_foldcheck_request_manifest(
        artifact_id=_ARTIFACT_ID,
        created_by=_CREATED_BY,
        created_at=created_at,
        backend_kind=BACKEND_KIND,
        runtime_kind=RUNTIME_KIND,
        execution_status=EXECUTION_STATUS,
        input_fasta_path=Path("input_sequences.fasta"),
        output_root=Path("colabfold_outputs"),
        sequence_records=records,
        wt_sequence_id=WT_SEQUENCE_ID,
        reference_structure_id=REFERENCE_STRUCTURE_ID,
        threshold_policy_id=THRESHOLD_POLICY_ID,
        threshold_values=THRESHOLD_VALUES,
        upstream_artifact_hashes={
            "generation_policy_manifest": sha256_uri(policy_manifest_path),
            "candidate_pool": sha256_uri(pool_result.candidate_pool_path),
            "candidate_pool_manifest": sha256_uri(pool_result.manifest_path),
            "residue_map": sha256_uri(residue_map_path),
        },
        storage_policy={
            **_LOCAL_STORAGE_POLICY,
            "generation_policy_version": GENERATION_POLICY_VERSION,
        },
    )
    request_manifest_path = request_root / "foldcheck_request_manifest.yaml"
    request_manifest_path.write_text(yaml.safe_dump(request_manifest, sort_keys=False), encoding="utf-8")
    return MaterializedGenerationPolicyFoldCheckRequest(
        candidate_pool_path=pool_result.candidate_pool_path,
        candidate_pool_manifest_path=pool_result.manifest_path,
        input_fasta_path=fasta_path,
        request_manifest_path=request_manifest_path,
    )


def _resolve_path(repo_root: Path, path: Path) -> Path:
    resolved = path.expanduser()
    return resolved if resolved.is_absolute() else (repo_root / resolved).resolve()
