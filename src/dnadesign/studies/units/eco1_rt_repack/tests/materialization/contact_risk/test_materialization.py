"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/contact_risk/test_materialization.py

Contact-risk profile materialization tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.contact_risk import (
    validate_contact_risk_profile_content,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.conservation import (
    materialize_conservation_profile,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact import materialize_contact_profile
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry import (
    materialize_contact_geometry_profile,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_risk import (
    materialize_contact_risk_profile,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.mask_set import materialize_mask_set
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


def test_contact_risk_profile_materializer_records_available_and_missing_evidence(tmp_path: Path) -> None:
    _materialize_upstream_artifacts(tmp_path)

    result = materialize_contact_risk_profile(repo_root=repo_root(), output_root=tmp_path)

    profile = _load_yaml(result.contact_risk_profile_path)
    assert profile["schema_id"] == "eco1_rt_repack.contact_risk_profile"
    assert profile["status"] == "materialized"
    assert profile["sampling_decision"]["status"] == "not_sampling_authoritative"
    assert profile["sampling_decision"]["reason"] == (
        "contact-risk profile audits the selected simple mask and does not create backend requests"
    )

    evidence = profile["evidence_availability"]
    assert evidence["nearest_context_atom_distance"]["status"] == "materialized"
    assert evidence["sidechain_context_distance"]["status"] == "materialized"
    assert evidence["backbone_context_distance"]["status"] == "materialized"
    assert evidence["contact_atom_density"]["status"] == "materialized"
    assert evidence["retained_context_chain_count"]["status"] == "materialized"

    summary = profile["summary"]
    assert summary["total_positions"] == 320
    assert summary["manual_mask_position_count"] == 12
    assert summary["wang_candidate_prior_position_count"] == 8
    assert summary["direct_contact_fixed_position_count"] > 0
    assert summary["selected_mask_non_fixed_mapped_position_count"] > 0

    rows = profile["residues"]
    assert len(rows) == 320
    by_position = {row["canonical_position"]: row for row in rows}
    assert by_position[195]["manual_mask"] is True
    assert by_position[195]["contact_risk_class"] == "motif_anchor_protected"
    assert by_position[257]["wang_candidate_prior"] is True
    assert by_position[257]["wang_candidate_prior_status"] == "active_direct_contact_mask_prior"
    assert by_position[4]["selected_mask_non_fixed"] is True
    assert by_position[3]["sidechain_context_distance_angstrom"] is not None
    assert by_position[3]["backbone_context_distance_angstrom"] is not None
    assert by_position[3]["contact_atom_count_within_20a"] >= 1

    assert validate_contact_risk_profile_content(result.contact_risk_profile_path) == []


def test_contact_risk_profile_validator_rejects_missing_evidence_status(tmp_path: Path) -> None:
    _materialize_upstream_artifacts(tmp_path)
    result = materialize_contact_risk_profile(repo_root=repo_root(), output_root=tmp_path)
    profile = _load_yaml(result.contact_risk_profile_path)
    del profile["evidence_availability"]["contact_atom_density"]["status"]
    result.contact_risk_profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

    issues = validate_contact_risk_profile_content(result.contact_risk_profile_path)

    assert {issue.check_id for issue in issues} == {"eco1_rt.contact_risk.evidence_availability_missing_status"}


def _materialize_upstream_artifacts(tmp_path: Path) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    materialize_structure_preprocessing_manifest(repo_root=repo_root(), output_root=tmp_path)
    materialize_contact_profile(repo_root=repo_root(), output_root=tmp_path)
    materialize_contact_geometry_profile(repo_root=repo_root(), output_root=tmp_path)
    materialize_conservation_profile(
        repo_root=repo_root(),
        output_root=tmp_path,
        alignment_root=_write_alignment_inputs(tmp_path),
    )
    materialize_mask_set(repo_root=repo_root(), output_root=tmp_path)


def _write_alignment_inputs(tmp_path: Path) -> Path:
    residue_rows = pq.read_table(tmp_path / "residue_map.parquet").to_pylist()
    target = "".join(str(row["wt_aa"]) for row in residue_rows)
    root = tmp_path / "alignments"
    root.mkdir()
    for profile_id in ("ec86_clade9_conservation_v1", "ec86_iia3_cluster42_1_conservation_v1"):
        records = [("eco1_rt_ec86kit_reference", target)]
        for index in range(20):
            sequence = list(target)
            sequence[2] = "S" if index < 14 else "A"
            sequence[3] = "G" if target[3] != "G" else "A"
            records.append((f"{profile_id}_homolog_{index + 1:02d}", "".join(sequence)))
        _write_fasta(root / f"{profile_id}.aligned.fasta", records)
    return root


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.write_text(
        "".join(f">{record_id}\n{sequence}\n" for record_id, sequence in records),
        encoding="utf-8",
    )


def _load_yaml(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded
