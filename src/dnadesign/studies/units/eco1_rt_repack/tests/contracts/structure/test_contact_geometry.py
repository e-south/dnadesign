"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/structure/test_contact_geometry.py

Contact-geometry artifact contract tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.structure import (
    validate_contact_geometry_profile_content,
)


def test_contact_geometry_validator_rejects_missing_atom_class_columns(tmp_path: Path) -> None:
    path = tmp_path / "contact_geometry_profile.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "canonical_position": 1,
                "wt_aa": "M",
                "mapping_status": "unresolved_structure",
            }
        ],
        schema=pa.schema(
            [
                pa.field("canonical_position", pa.int32(), nullable=False),
                pa.field("wt_aa", pa.string(), nullable=False),
                pa.field("mapping_status", pa.string(), nullable=False),
            ],
            metadata={
                b"schema_id": b"eco1_rt_repack.contact_geometry_profile",
                b"status": b"materialized",
            },
        ),
    )
    pq.write_table(table, path)

    issues = validate_contact_geometry_profile_content(path)

    assert "eco1_rt.structure.contact_geometry_missing_columns" in {issue.check_id for issue in issues}


def test_contact_geometry_validator_rejects_stale_upstream_hash(tmp_path: Path) -> None:
    upstream = tmp_path / "residue_map.parquet"
    upstream.write_text("current residue map bytes", encoding="utf-8")
    path = tmp_path / "contact_geometry_profile.parquet"
    table = pa.Table.from_pylist(
        [_unresolved_geometry_row()],
        schema=pa.schema(
            [
                pa.field("canonical_position", pa.int32(), nullable=False),
                pa.field("wt_aa", pa.string(), nullable=False),
                pa.field("structure_chain_id", pa.string(), nullable=False),
                pa.field("structure_residue_id", pa.int32(), nullable=True),
                pa.field("mapping_status", pa.string(), nullable=False),
                pa.field("nearest_context_atom_distance_angstrom", pa.float64(), nullable=True),
                pa.field("nearest_sidechain_context_distance_angstrom", pa.float64(), nullable=True),
                pa.field("nearest_backbone_context_distance_angstrom", pa.float64(), nullable=True),
                pa.field("nearest_dna_distance_angstrom", pa.float64(), nullable=True),
                pa.field("nearest_rna_distance_angstrom", pa.float64(), nullable=True),
                pa.field("nearest_context_chain_id", pa.string(), nullable=False),
                pa.field("nearest_context_molecule_type", pa.string(), nullable=False),
                pa.field("nearest_context_atom_name", pa.string(), nullable=False),
                pa.field("sidechain_atom_status", pa.string(), nullable=False),
                pa.field("contact_atom_count_within_4a", pa.int32(), nullable=False),
                pa.field("contact_atom_count_within_6a", pa.int32(), nullable=False),
                pa.field("contact_atom_count_within_8a", pa.int32(), nullable=False),
                pa.field("contact_atom_count_within_10a", pa.int32(), nullable=False),
                pa.field("contact_atom_count_within_12a", pa.int32(), nullable=False),
                pa.field("contact_atom_count_within_15a", pa.int32(), nullable=False),
                pa.field("contact_atom_count_within_20a", pa.int32(), nullable=False),
                pa.field("retained_context_chain_count_within_8a", pa.int32(), nullable=False),
                pa.field("retained_context_chain_count_within_12a", pa.int32(), nullable=False),
                pa.field("retained_context_chain_count_within_15a", pa.int32(), nullable=False),
                pa.field("retained_context_chain_count_within_20a", pa.int32(), nullable=False),
            ],
            metadata={
                b"schema_id": b"eco1_rt_repack.contact_geometry_profile",
                b"schema_version": b"1",
                b"artifact_id": b"eco1_rt_conservative_v1.contact_geometry_profile",
                b"status": b"materialized",
                b"created_by": b"test",
                b"created_at": b"2026-06-22T00:00:00Z",
                b"geometry_backend_id": b"biopython_mmcif_atom_geometry_v1",
                b"upstream_artifact_hashes": json.dumps({"residue_map": "sha256:stale"}).encode("utf-8"),
            },
        ),
    )
    pq.write_table(table, path)

    issues = validate_contact_geometry_profile_content(
        path,
        upstream_artifact_paths={"residue_map": upstream},
    )

    assert "eco1_rt.structure.contact_geometry_upstream_hash_mismatch" in {issue.check_id for issue in issues}


def _unresolved_geometry_row() -> dict[str, object]:
    return {
        "canonical_position": 1,
        "wt_aa": "M",
        "structure_chain_id": "",
        "structure_residue_id": None,
        "mapping_status": "unresolved_structure",
        "nearest_context_atom_distance_angstrom": None,
        "nearest_sidechain_context_distance_angstrom": None,
        "nearest_backbone_context_distance_angstrom": None,
        "nearest_dna_distance_angstrom": None,
        "nearest_rna_distance_angstrom": None,
        "nearest_context_chain_id": "",
        "nearest_context_molecule_type": "",
        "nearest_context_atom_name": "",
        "sidechain_atom_status": "unresolved_structure",
        "contact_atom_count_within_4a": 0,
        "contact_atom_count_within_6a": 0,
        "contact_atom_count_within_8a": 0,
        "contact_atom_count_within_10a": 0,
        "contact_atom_count_within_12a": 0,
        "contact_atom_count_within_15a": 0,
        "contact_atom_count_within_20a": 0,
        "retained_context_chain_count_within_8a": 0,
        "retained_context_chain_count_within_12a": 0,
        "retained_context_chain_count_within_15a": 0,
        "retained_context_chain_count_within_20a": 0,
    }
