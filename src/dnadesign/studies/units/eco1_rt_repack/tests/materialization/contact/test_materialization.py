"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/contact/test_materialization.py

Contact-profile materialization tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.suite import validate_checked_in_contracts
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact import materialize_contact_profile
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure import (
    materialize_structure_authority,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import ec86kit_source_artifacts_available, repo_root

pytestmark = pytest.mark.skipif(
    not ec86kit_source_artifacts_available(),
    reason="requires sibling ec86kit structure-authority artifacts",
)


def test_contact_materializer_writes_thresholded_retained_context_profile(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    result = materialize_contact_profile(repo_root=repo_root(), output_root=tmp_path)

    table = pq.read_table(result.contact_profile_path)
    assert table.num_rows == 320
    metadata = table.schema.metadata or {}
    assert metadata[b"schema_id"] == b"thread.contact_profile"
    assert metadata[b"status"] == b"materialized"
    assert metadata[b"contact_threshold_angstrom"] == b"5.0"
    rows = table.to_pylist()
    assert rows[0]["canonical_position"] == 1
    assert rows[0]["mapping_status"] == "unresolved_structure"
    assert rows[0]["nearest_context_atom_distance_angstrom"] is None
    assert rows[0]["passes_contact_mask"] is False

    position_3 = next(row for row in rows if row["canonical_position"] == 3)
    assert position_3["wt_aa"] == "S"
    assert position_3["retained_context_id"] == "retained_nucleic_acid_context"
    assert position_3["nearest_dna_chain"] == "D"
    assert position_3["nearest_rna_chain"] == "E"
    assert position_3["nearest_context_chain_id"] == "D"
    assert position_3["nearest_context_atom_distance_angstrom"] == 16.731
    assert position_3["contact_threshold_angstrom"] == 5.0
    assert position_3["passes_contact_mask"] is False
    assert position_3["source_hash"] == ("sha256:29fb97933658cc6f62f0cdbb10f86bbad60f5115d56ad017e6cc7222ec640a49")


def test_phase1_rejects_contact_profile_source_hash_mismatch(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    result = materialize_contact_profile(repo_root=repo_root(), output_root=tmp_path)
    table = pq.read_table(result.contact_profile_path)
    rows = table.to_pylist()
    for row in rows:
        row["source_hash"] = "sha256:not-the-distance-profile"
    pq.write_table(table.from_pylist(rows, schema=table.schema), result.contact_profile_path)

    report = validate_checked_in_contracts(repo_root=repo_root(), phase="phase1_thread_contract", output_root=tmp_path)

    assert report.passed is False
    assert "eco1_rt.evidence.contact_profile_source_hash_mismatch" in {issue.check_id for issue in report.issues}
