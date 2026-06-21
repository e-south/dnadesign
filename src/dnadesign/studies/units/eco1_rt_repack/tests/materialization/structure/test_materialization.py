"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/structure/test_materialization.py

Structure-materialization tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure import (
    materialize_structure_authority,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import ec86kit_source_artifacts_available, repo_root

pytestmark = pytest.mark.skipif(
    not ec86kit_source_artifacts_available(),
    reason="requires sibling ec86kit structure-authority artifacts",
)


def test_materializer_writes_thread_ready_structure_artifacts(tmp_path: Path) -> None:
    result = materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)

    bundle = yaml.safe_load(result.backbone_bundle_path.read_text(encoding="utf-8"))
    assert bundle["schema_id"] == "thread.backbone_bundle"
    assert bundle["status"] == "materialized"
    assert bundle["source_ref"] == "pdb:7v9u"
    assert bundle["rt_chain_id"] == "A"
    chain_roles = {row["chain_id"]: row for row in bundle["chain_inventory"]}
    assert chain_roles["A"]["molecule_type"] == "protein"
    assert chain_roles["A"]["thread_role"] == "design_backbone"
    assert chain_roles["D"]["molecule_type"] == "dna"
    assert chain_roles["E"]["molecule_type"] == "rna"
    assert chain_roles["F"]["molecule_type"] == "rna"

    table = pq.read_table(result.residue_map_path)
    assert table.num_rows == 320
    metadata = table.schema.metadata or {}
    assert metadata[b"schema_id"] == b"thread.residue_map"
    assert metadata[b"status"] == b"materialized"
    assert metadata[b"created_at"]
    assert metadata[b"upstream_artifact_hashes"]
    rows = table.to_pylist()
    mapped = [row for row in rows if row["mapping_status"] == "mapped"]
    unresolved = [row for row in rows if row["mapping_status"] == "unresolved_structure"]
    assert len(mapped) == 309
    assert [row["canonical_position"] for row in unresolved] == [1, 2, 312, 313, 314, 315, 316, 317, 318, 319, 320]
    assert all(row["unresolved_policy"] == "fixed" for row in unresolved)
    assert all(row["is_designable_initially"] is False for row in rows)
