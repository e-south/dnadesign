"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/generation_policies/test_policy_manifests.py

Generation-policy config and manifest tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies import (
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    GENERATION_POLICY_VERSION,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
    PRIMARY_POLICY_IDS,
    build_default_generation_policy_config,
    materialize_generation_policies,
    validate_generation_policy_config,
)

from ._candidate_tables import write_generation_policy_source_inputs


def test_generation_policy_config_rejects_legacy_design_class_ids() -> None:
    config = build_default_generation_policy_config()
    config["generation_policies"] = {
        "eco1_rt_clade9_plurality25_contact8a_v1": {"enabled": True, "requested_variants": 336}
    }

    with pytest.raises(ValueError, match="legacy design-class id"):
        validate_generation_policy_config(config)


def test_default_generation_policy_config_accepts_only_primary_policy_ids() -> None:
    config = validate_generation_policy_config(build_default_generation_policy_config())

    assert config.generation_policy_version == GENERATION_POLICY_VERSION
    assert tuple(policy.policy_id for policy in config.enabled_policies) == PRIMARY_POLICY_IDS
    assert sum(policy.requested_variants for policy in config.enabled_policies) == 1008


def test_generation_policy_manifest_materializes_active_policy_boundary(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    write_generation_policy_source_inputs(source_root)
    result = materialize_generation_policies(repo_root=Path.cwd(), output_root=tmp_path, source_output_root=source_root)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    positions = pq.read_table(result.positions_path).to_pylist()
    alphabets = pq.read_table(result.alphabets_path).to_pylist()

    assert manifest["schema_id"] == "eco1_rt.generation_policy_manifest"
    assert manifest["generation_policy_version"] == GENERATION_POLICY_VERSION
    assert manifest["policy_manifest_hash"].startswith("sha256:")
    assert {row["policy_id"] for row in manifest["generation_policies"]} == set(PRIMARY_POLICY_IDS)
    assert "eco1_rt_clade9_plurality25_contact8a_v1" not in {
        row["policy_id"] for row in manifest["generation_policies"]
    }

    assert {row["policy_id"] for row in positions} == set(PRIMARY_POLICY_IDS)
    assert {row["policy_id"] for row in alphabets} == set(PRIMARY_POLICY_IDS)
    assert "post_generation_filter" not in {row["alphabet_enforcement_mode"] for row in alphabets}
    for row in positions:
        if row["is_open_position"]:
            assert row["protected_reason_codes"] == []
        if row["is_wang_thumb_track"]:
            assert row["protected_reason_codes"]
            assert not row["is_open_position"]
        if row["is_c_terminal_thumb_context"]:
            assert row["protected_reason_codes"]
            assert not row["is_open_position"]


def test_peripheral_policy_alphabet_is_position_specific_and_upstream_enforced(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    write_generation_policy_source_inputs(source_root)
    result = materialize_generation_policies(repo_root=Path.cwd(), output_root=tmp_path, source_output_root=source_root)
    rows = pq.read_table(result.alphabets_path).to_pylist()

    near_rows = [
        row
        for row in rows
        if row["policy_id"] == NEAR_DNA_RNA_ACID_FREE_POLICY_ID
        and row["alphabet_scope"] == "near_dna_rna_gt5_le10_excluding_protected"
    ]
    combined_near_rows = [
        row
        for row in rows
        if row["policy_id"] == COMBINED_NEAR_PLUS_DISTAL_POLICY_ID
        and row["alphabet_scope"] == "near_dna_rna_gt5_le10_excluding_protected"
    ]

    assert near_rows
    assert combined_near_rows
    assert {row["alphabet_enforcement_mode"] for row in near_rows + combined_near_rows} == {"upstream_omit_AA_jsonl"}
    for row in near_rows + combined_near_rows:
        assert isinstance(row["eco1_position"], int)
        assert "D" in row["disallowed_amino_acids"]
        assert "E" in row["disallowed_amino_acids"]
        assert set(row["allowed_amino_acids"]).isdisjoint(set(row["disallowed_amino_acids"]))


def test_combined_policy_open_set_is_union_of_distal_and_near_sets(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    write_generation_policy_source_inputs(source_root)
    result = materialize_generation_policies(repo_root=Path.cwd(), output_root=tmp_path, source_output_root=source_root)
    rows = pq.read_table(result.positions_path).to_pylist()
    open_by_policy = {
        policy_id: {
            int(row["eco1_position"]) for row in rows if row["policy_id"] == policy_id and row["is_open_position"]
        }
        for policy_id in PRIMARY_POLICY_IDS
    }

    distal = open_by_policy["distal_scaffold_repack_v1"]
    near = open_by_policy[NEAR_DNA_RNA_ACID_FREE_POLICY_ID]
    combined = open_by_policy[COMBINED_NEAR_PLUS_DISTAL_POLICY_ID]
    assert distal
    assert near
    assert distal.isdisjoint(near)
    assert combined == distal | near
    for position in near:
        near_row = next(
            row
            for row in rows
            if row["policy_id"] == NEAR_DNA_RNA_ACID_FREE_POLICY_ID and int(row["eco1_position"]) == position
        )
        assert near_row["is_near_region_gt5_le10a"]
        assert not near_row["is_direct_contact_le_5a"]
        assert not near_row["is_c_terminal_thumb_context"]


def test_primary_policies_fix_declared_255_311_context_without_retroactively_fixing_c233(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    write_generation_policy_source_inputs(source_root)
    result = materialize_generation_policies(repo_root=Path.cwd(), output_root=tmp_path, source_output_root=source_root)
    rows = pq.read_table(result.positions_path).to_pylist()

    for policy_id in PRIMARY_POLICY_IDS:
        by_position = {int(row["eco1_position"]): row for row in rows if row["policy_id"] == policy_id}
        for position in (255, 311):
            row = by_position[position]
            assert row["is_c_terminal_thumb_context"]
            assert "c_terminal_thumb_context_255_311" in row["protected_reason_codes"]
            assert not row["is_open_position"]
        for position in (230, 233, 254):
            row = by_position[position]
            assert not row["is_c_terminal_thumb_context"]
            assert "c_terminal_thumb_context_255_311" not in row["protected_reason_codes"]


def test_v3_no_cysteine_rule_can_force_an_open_wt_cysteine_to_change(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    write_generation_policy_source_inputs(source_root)
    residue_path = source_root / "residue_map.parquet"
    residue_rows = pq.read_table(residue_path).to_pylist()
    next(row for row in residue_rows if int(row["canonical_position"]) == 20)["wt_aa"] = "C"
    pq.write_table(pa.Table.from_pylist(residue_rows), residue_path)

    result = materialize_generation_policies(repo_root=Path.cwd(), output_root=tmp_path, source_output_root=source_root)
    rows = pq.read_table(result.alphabets_path).to_pylist()
    wt_cys_row = next(
        row for row in rows if row["policy_id"] == NEAR_DNA_RNA_ACID_FREE_POLICY_ID and row["eco1_position"] == 20
    )

    assert "C" not in wt_cys_row["allowed_amino_acids"]
    assert "C" in wt_cys_row["disallowed_amino_acids"]
    assert "force an open WT cysteine to change" in wt_cys_row["interpretation_limit"]
    assert "preserve WT" not in wt_cys_row["interpretation_limit"]
