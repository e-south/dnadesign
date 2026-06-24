"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_geometry/writer.py

Parquet writer for Eco1 RT contact-geometry profiles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.constants import (
    _CHAIN_COUNT_THRESHOLDS,
    _CONTACT_THRESHOLDS,
    _CREATED_BY,
    _GEOMETRY_BACKEND_ID,
    threshold_id,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import require_text


def write_geometry_profile(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    upstream_hashes: Mapping[str, str],
    selected_source: Mapping[str, Any],
    created_at: str,
) -> None:
    """Write a hash-linked contact_geometry_profile.parquet artifact."""

    fields = [
        pa.field("canonical_position", pa.int32(), nullable=False),
        pa.field("wt_aa", pa.string(), nullable=False),
        pa.field("structure_chain_id", pa.string(), nullable=False),
        pa.field("structure_residue_id", pa.int32(), nullable=True),
        pa.field("pdb_insertion_code", pa.string(), nullable=False),
        pa.field("mapping_status", pa.string(), nullable=False),
        pa.field("nearest_context_atom_distance_angstrom", pa.float64(), nullable=True),
        pa.field("nearest_sidechain_context_distance_angstrom", pa.float64(), nullable=True),
        pa.field("nearest_backbone_context_distance_angstrom", pa.float64(), nullable=True),
        pa.field("nearest_dna_distance_angstrom", pa.float64(), nullable=True),
        pa.field("nearest_rna_distance_angstrom", pa.float64(), nullable=True),
        pa.field("nearest_context_chain_id", pa.string(), nullable=False),
        pa.field("nearest_context_molecule_type", pa.string(), nullable=False),
        pa.field("nearest_context_residue_id", pa.int32(), nullable=True),
        pa.field("nearest_context_residue_name", pa.string(), nullable=False),
        pa.field("nearest_context_atom_name", pa.string(), nullable=False),
        pa.field("sidechain_atom_status", pa.string(), nullable=False),
    ]
    fields.extend(
        pa.field(f"contact_atom_count_within_{threshold_id(threshold)}", pa.int32(), nullable=False)
        for threshold in _CONTACT_THRESHOLDS
    )
    fields.extend(
        pa.field(f"retained_context_chain_count_within_{threshold_id(threshold)}", pa.int32(), nullable=False)
        for threshold in _CHAIN_COUNT_THRESHOLDS
    )
    schema = pa.schema(
        fields,
        metadata={
            b"schema_id": b"eco1_rt_repack.contact_geometry_profile",
            b"schema_version": b"1",
            b"artifact_id": b"eco1_rt_conservative_v1.contact_geometry_profile",
            b"status": b"materialized",
            b"created_by": _CREATED_BY.encode("utf-8"),
            b"created_at": created_at.encode("utf-8"),
            b"geometry_backend_id": _GEOMETRY_BACKEND_ID.encode("utf-8"),
            b"selected_structure_source_id": require_text(selected_source, "source_id").encode("utf-8"),
            b"reference_sequence_hash": require_text(selected_source, "reference_sequence_hash").encode("utf-8"),
            b"upstream_artifact_hashes": json.dumps(dict(upstream_hashes), sort_keys=True).encode("utf-8"),
        },
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
