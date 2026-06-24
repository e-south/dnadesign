"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/surface_accessibility/test_materialization.py

Surface-accessibility evidence tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure import (
    materialize_structure_authority,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure_preprocessing import (
    materialize_structure_preprocessing_manifest,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.surface_accessibility import (
    materialize_surface_accessibility_profile,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import ec86kit_source_artifacts_available, repo_root

pytestmark = pytest.mark.skipif(
    not ec86kit_source_artifacts_available(),
    reason="requires sibling ec86kit structure-authority artifacts",
)


def test_surface_accessibility_profile_materializes_sasa_features(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    materialize_structure_preprocessing_manifest(repo_root=repo_root(), output_root=tmp_path)

    result = materialize_surface_accessibility_profile(repo_root=repo_root(), output_root=tmp_path)

    table = pq.read_table(result.surface_accessibility_profile_path)
    assert table.num_rows == 320
    metadata = table.schema.metadata or {}
    assert metadata[b"schema_id"] == b"eco1_rt_repack.surface_accessibility_profile"
    assert metadata[b"surface_backend_id"] == b"biopython_shrake_rupley_sasa_v1"
    rows = table.to_pylist()
    by_position = {row["canonical_position"]: row for row in rows}

    unresolved = by_position[1]
    assert unresolved["mapping_status"] == "unresolved_structure"
    assert unresolved["residue_sasa_angstrom2"] is None
    assert unresolved["surface_accessibility_class"] == "unresolved_structure"

    exposed = by_position[301]
    assert exposed["mapping_status"] == "mapped"
    assert exposed["surface_accessibility_class"] == "surface_exposed"
    assert exposed["residue_sasa_angstrom2"] == pytest.approx(87.3, abs=0.3)
    assert exposed["sidechain_sasa_angstrom2"] == pytest.approx(83.7, abs=0.3)

    limited = by_position[112]
    assert limited["mapping_status"] == "mapped"
    assert limited["surface_accessibility_class"] == "buried_or_limited_access"
    assert limited["residue_sasa_angstrom2"] < 30.0


def test_surface_accessibility_profile_rejects_missing_preprocessing_manifest(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="structure_preprocessing_manifest.yaml"):
        materialize_surface_accessibility_profile(repo_root=repo_root(), output_root=tmp_path)
