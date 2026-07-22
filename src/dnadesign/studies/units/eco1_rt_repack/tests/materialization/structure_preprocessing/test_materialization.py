"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/structure_preprocessing/test_materialization.py

Structure-preprocessing provenance tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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


def test_structure_preprocessing_manifest_records_protomer_chain_roles(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)

    result = materialize_structure_preprocessing_manifest(repo_root=repo_root(), output_root=tmp_path)

    manifest = yaml.safe_load(result.structure_preprocessing_manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_id"] == "eco1_rt_repack.structure_preprocessing_manifest"
    assert manifest["status"] == "materialized"
    assert manifest["raw_structure"]["source_ref"] == "pdb:7v9u"
    assert manifest["selected_protomer"]["source_id"] == "ec86kit_7v9u_protomer1"
    assert manifest["selected_protomer"]["protomer_id"] == 1
    assert manifest["selected_protomer"]["rt_chain_id"] == "A"
    assert manifest["selected_protomer"]["retained_context_chains"] == ["D", "E", "F"]
    assert manifest["selected_protomer"]["excluded_context"] == ["ec86kit_protomer2_assignment"]
    assert manifest["design_objectives"]["preserve_paired_protomer_dimerization"] is False
    assert (
        manifest["design_objectives"]["paired_protomer_interface_policy"]
        == "not_mask_authoritative_unless_retained_context_manual_or_conserved"
    )

    chain_roles = {row["chain_id"]: row for row in manifest["chain_inventory"]}
    assert chain_roles["A"]["molecule_type"] == "protein"
    assert chain_roles["A"]["thread_role"] == "design_backbone"
    assert chain_roles["D"]["molecule_type"] == "dna"
    assert chain_roles["E"]["molecule_type"] == "rna"
    assert chain_roles["F"]["molecule_type"] == "rna"
    assert manifest["upstream_artifact_hashes"]["ec86kit_model"].startswith("sha256:")
