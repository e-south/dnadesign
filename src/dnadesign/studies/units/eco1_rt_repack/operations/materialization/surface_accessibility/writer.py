"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/surface_accessibility/writer.py

Parquet writer for Eco1 RT surface-accessibility profiles.

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

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import require_text
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.surface_accessibility.constants import (
    _CREATED_BY,
    _SHRAKE_RUPLEY_N_POINTS,
    _SURFACE_BACKEND_ID,
)


def write_surface_accessibility_profile(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    upstream_hashes: Mapping[str, str],
    selected_source: Mapping[str, Any],
    created_at: str,
) -> None:
    """Write a hash-linked surface_accessibility_profile.parquet artifact."""

    schema = pa.schema(
        [
            pa.field("canonical_position", pa.int32(), nullable=False),
            pa.field("wt_aa", pa.string(), nullable=False),
            pa.field("structure_chain_id", pa.string(), nullable=False),
            pa.field("structure_residue_id", pa.int32(), nullable=True),
            pa.field("pdb_insertion_code", pa.string(), nullable=False),
            pa.field("mapping_status", pa.string(), nullable=False),
            pa.field("residue_sasa_angstrom2", pa.float64(), nullable=True),
            pa.field("sidechain_sasa_angstrom2", pa.float64(), nullable=True),
            pa.field("backbone_sasa_angstrom2", pa.float64(), nullable=True),
            pa.field("surface_accessibility_class", pa.string(), nullable=False),
            pa.field("sidechain_surface_status", pa.string(), nullable=False),
        ],
        metadata={
            b"schema_id": b"eco1_rt_repack.surface_accessibility_profile",
            b"schema_version": b"1",
            b"artifact_id": b"eco1_rt_conservative_v1.surface_accessibility_profile",
            b"status": b"materialized",
            b"created_by": _CREATED_BY.encode("utf-8"),
            b"created_at": created_at.encode("utf-8"),
            b"surface_backend_id": _SURFACE_BACKEND_ID.encode("utf-8"),
            b"shrake_rupley_n_points": str(_SHRAKE_RUPLEY_N_POINTS).encode("utf-8"),
            b"selected_structure_source_id": require_text(selected_source, "source_id").encode("utf-8"),
            b"reference_sequence_hash": require_text(selected_source, "reference_sequence_hash").encode("utf-8"),
            b"upstream_artifact_hashes": json.dumps(dict(upstream_hashes), sort_keys=True).encode("utf-8"),
        },
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
