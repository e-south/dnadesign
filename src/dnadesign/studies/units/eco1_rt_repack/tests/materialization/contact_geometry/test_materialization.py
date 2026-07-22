"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/contact_geometry/test_materialization.py

Atom-class contact-geometry tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry import (
    materialize_contact_geometry_profile,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure import (
    materialize_structure_authority,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure_preprocessing import (
    materialize_structure_preprocessing_manifest,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import ec86kit_source_artifacts_available, repo_root

pytestmark = pytest.mark.skipif(
    not ec86kit_source_artifacts_available(),
    reason="requires sibling ec86kit structure-authority artifacts",
)


def test_contact_geometry_profile_materializes_atom_class_features(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    materialize_structure_preprocessing_manifest(repo_root=repo_root(), output_root=tmp_path)

    result = materialize_contact_geometry_profile(repo_root=repo_root(), output_root=tmp_path)

    table = pq.read_table(result.contact_geometry_profile_path)
    assert table.num_rows == 320
    metadata = table.schema.metadata or {}
    assert metadata[b"schema_id"] == b"eco1_rt_repack.contact_geometry_profile"
    assert metadata[b"status"] == b"materialized"
    assert metadata[b"geometry_backend_id"] == b"biopython_mmcif_atom_geometry_v1"
    rows = table.to_pylist()
    by_position = {row["canonical_position"]: row for row in rows}

    unresolved = by_position[1]
    assert unresolved["mapping_status"] == "unresolved_structure"
    assert unresolved["nearest_context_atom_distance_angstrom"] is None
    assert unresolved["sidechain_atom_status"] == "unresolved_structure"

    mapped = by_position[3]
    assert mapped["mapping_status"] == "mapped"
    assert mapped["structure_chain_id"] == "A"
    assert mapped["nearest_context_atom_distance_angstrom"] == pytest.approx(16.731, abs=0.002)
    assert mapped["nearest_dna_distance_angstrom"] == pytest.approx(16.731, abs=0.002)
    assert mapped["nearest_rna_distance_angstrom"] == pytest.approx(35.343, abs=0.002)
    assert mapped["nearest_backbone_context_distance_angstrom"] > 0
    assert mapped["sidechain_atom_status"] in {"materialized", "glycine_no_sidechain"}
    assert mapped["retained_context_chain_count_within_20a"] >= 1
    assert mapped["contact_atom_count_within_20a"] >= 1


def test_contact_geometry_profile_rejects_missing_preprocessing_manifest(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="structure_preprocessing_manifest.yaml"):
        materialize_contact_geometry_profile(repo_root=repo_root(), output_root=tmp_path)
