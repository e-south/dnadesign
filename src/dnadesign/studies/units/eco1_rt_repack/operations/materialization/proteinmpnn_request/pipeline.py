"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/proteinmpnn_request/pipeline.py

Materialize helper-compatible ProteinMPNN request sidecars for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    find_repo_root,
    load_yaml,
    require_hash,
    require_mapping,
    require_text,
    resolve_output_root,
    resolve_source_ref,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.structure_io import (
    load_first_model,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.proteinmpnn_request.constants import (
    ARTIFACT_ID,
    CHAIN_ID,
    CREATED_BY,
    DEFAULT_OUTPUT_ROOT,
    PROTEINMPNN_NAME,
    REQUEST_DIR_NAME,
    STRUCTURE_SOURCES,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.proteinmpnn_request.models import (
    MaterializedProteinMpnnRequestArtifacts,
)
from dnadesign.thread.adapters.proteinmpnn import (
    assigned_chains_payload,
    build_request_manifest,
    export_chain_backbone,
    fixed_positions_payload,
    mapped_chain_rows,
    request_hash,
    require_int_list,
    require_missing_backbone_excluded,
    to_proteinmpnn_positions,
    write_jsonl,
)
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri


def materialize_proteinmpnn_request(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
) -> MaterializedProteinMpnnRequestArtifacts:
    """Materialize ProteinMPNN request sidecars without running ProteinMPNN."""

    root = (repo_root or find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = resolve_output_root(root, output_root or DEFAULT_OUTPUT_ROOT)
    request_root = out_root / REQUEST_DIR_NAME
    request_root.mkdir(parents=True, exist_ok=True)

    thread_plan_path = out_root / "thread_plan.yaml"
    residue_map_path = out_root / "residue_map.parquet"
    backbone_bundle_path = out_root / "backbone_bundle.yaml"
    for required_path in (thread_plan_path, residue_map_path, backbone_bundle_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    thread_plan = load_yaml(thread_plan_path)
    structure_sources = load_yaml(root / STRUCTURE_SOURCES)
    selected_source = require_mapping(structure_sources.get("selected_source"), "selected_source")
    model_path = resolve_source_ref(root, require_text(selected_source, "ec86kit_model_ref"))
    require_hash(model_path, require_text(selected_source, "ec86kit_model_sha256"))
    model = load_first_model(model_path)

    mapped_rows = mapped_chain_rows(residue_map_path, chain_id=CHAIN_ID, expected_mapped_count=309)
    chain_pdb_path = request_root / f"{PROTEINMPNN_NAME}.pdb"
    export = export_chain_backbone(
        model=model,
        mapped_residue_rows=mapped_rows,
        chain_id=CHAIN_ID,
        output_path=chain_pdb_path,
        target_name=PROTEINMPNN_NAME,
    )
    fixed_positions = to_proteinmpnn_positions(
        thread_plan.get("fixed_positions"),
        export.canonical_to_proteinmpnn_position,
        "fixed_positions",
    )
    mutable_positions = to_proteinmpnn_positions(
        thread_plan.get("mutable_positions"),
        export.canonical_to_proteinmpnn_position,
        "mutable_positions",
    )
    excluded_positions = require_int_list(
        thread_plan.get("excluded_non_fixed_missing_backbone_positions"),
        "excluded_non_fixed_missing_backbone_positions",
    )
    require_missing_backbone_excluded(excluded_positions, export.canonical_to_proteinmpnn_position)

    parsed_pdbs_path = request_root / "parsed_pdbs.jsonl"
    assigned_chains_path = request_root / "assigned_chains.jsonl"
    fixed_positions_path = request_root / "fixed_positions.jsonl"
    write_jsonl(parsed_pdbs_path, export.parsed_payload)
    write_jsonl(assigned_chains_path, assigned_chains_payload(target_name=PROTEINMPNN_NAME, chain_id=CHAIN_ID))
    write_jsonl(
        fixed_positions_path,
        fixed_positions_payload(
            target_name=PROTEINMPNN_NAME,
            chain_id=CHAIN_ID,
            fixed_positions=fixed_positions,
        ),
    )

    seed_set = require_int_list(thread_plan.get("seed_set"), "seed_set")
    temperatures = [float(value) for value in _require_number_list(thread_plan.get("temperature_schedule"))]
    batch_id = _require_text(thread_plan.get("batch_id"), "batch_id")
    num_seq_per_target = _require_positive_int(thread_plan.get("num_seq_per_target"), "num_seq_per_target")
    batch_size = _require_positive_int(thread_plan.get("batch_size"), "batch_size")
    source_thread_plan = {
        "path": str(thread_plan_path),
        "hash": sha256_uri(thread_plan_path),
        "request_hash": thread_plan.get("request_hash"),
        "fixed_positions": list(thread_plan.get("fixed_positions", [])),
        "mutable_positions": list(thread_plan.get("mutable_positions", [])),
    }
    manifest_without_hash = build_request_manifest(
        artifact_id=ARTIFACT_ID,
        created_by=CREATED_BY,
        profile_id=str(thread_plan.get("profile_id", "")),
        mask_policy_id=str(thread_plan.get("mask_policy_id", "")),
        target_name=PROTEINMPNN_NAME,
        chain_id=CHAIN_ID,
        sidecar_paths={
            "chain_a_backbone_pdb": chain_pdb_path,
            "parsed_pdbs_jsonl": parsed_pdbs_path,
            "assigned_chains_jsonl": assigned_chains_path,
            "fixed_positions_jsonl": fixed_positions_path,
        },
        upstream_artifact_hashes={
            "thread_plan": sha256_uri(thread_plan_path),
            "residue_map": sha256_uri(residue_map_path),
            "backbone_bundle": sha256_uri(backbone_bundle_path),
            "structure_sources_yaml": sha256_uri(root / STRUCTURE_SOURCES),
            "ec86kit_model": sha256_uri(model_path),
        },
        source_thread_plan=source_thread_plan,
        canonical_to_mpnn=export.canonical_to_proteinmpnn_position,
        fixed_positions=fixed_positions,
        mutable_positions=mutable_positions,
        excluded_positions=excluded_positions,
        seed_set=seed_set,
        temperatures=temperatures,
        batch_id=batch_id,
        num_seq_per_target=num_seq_per_target,
        batch_size=batch_size,
        expected_sample_count=int(thread_plan.get("expected_sample_count", 0)),
    )
    manifest = {"request_hash": request_hash(manifest_without_hash), **manifest_without_hash}
    manifest_path = request_root / "request_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return MaterializedProteinMpnnRequestArtifacts(
        chain_a_backbone_pdb_path=chain_pdb_path,
        parsed_pdbs_path=parsed_pdbs_path,
        assigned_chains_path=assigned_chains_path,
        fixed_positions_path=fixed_positions_path,
        request_manifest_path=manifest_path,
    )


def _require_number_list(value: object) -> list[float | int]:
    if not isinstance(value, list) or not value:
        raise ValueError("temperature_schedule must be a non-empty list")
    for item in value:
        if not isinstance(item, int | float) or isinstance(item, bool):
            raise ValueError("temperature_schedule must contain numbers")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()
