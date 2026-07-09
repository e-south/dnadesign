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

import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies import (
    GENERATION_POLICY_VERSION,
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


def test_generation_policy_manifest_materializes_v2_boundary(tmp_path: Path) -> None:
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


def test_near_policy_alphabet_is_position_specific_and_upstream_enforced(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    write_generation_policy_source_inputs(source_root)
    result = materialize_generation_policies(repo_root=Path.cwd(), output_root=tmp_path, source_output_root=source_root)
    rows = pq.read_table(result.alphabets_path).to_pylist()

    near_rows = [
        row
        for row in rows
        if row["policy_id"] == "near_dna_rna_acid_free_v1"
        and row["alphabet_scope"] == "near_dna_rna_gt5_le10_excluding_protected"
    ]
    combined_near_rows = [
        row
        for row in rows
        if row["policy_id"] == "combined_near_acid_free_plus_distal_v1"
        and row["alphabet_scope"] == "near_dna_rna_gt5_le10_excluding_protected"
    ]

    assert near_rows
    assert combined_near_rows
    assert {row["alphabet_enforcement_mode"] for row in near_rows} == {"upstream_omit_AA_jsonl"}
    assert {row["alphabet_enforcement_mode"] for row in combined_near_rows} == {"upstream_omit_AA_jsonl"}
    for row in near_rows + combined_near_rows:
        assert isinstance(row["eco1_position"], int)
        assert row["wt_aa"] in row["allowed_amino_acids"]
        assert "D" in row["disallowed_amino_acids"]
        assert "E" in row["disallowed_amino_acids"]
        assert set(row["allowed_amino_acids"]).isdisjoint(set(row["disallowed_amino_acids"]))


def test_combined_policy_open_set_is_union_of_distal_and_near(tmp_path: Path) -> None:
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

    assert open_by_policy["distal_scaffold_repack_v1"]
    assert open_by_policy["near_dna_rna_acid_free_v1"]
    assert open_by_policy["combined_near_acid_free_plus_distal_v1"] == (
        open_by_policy["distal_scaffold_repack_v1"] | open_by_policy["near_dna_rna_acid_free_v1"]
    )
    for position in open_by_policy["near_dna_rna_acid_free_v1"]:
        near_row = next(
            row
            for row in rows
            if row["policy_id"] == "near_dna_rna_acid_free_v1" and int(row["eco1_position"]) == position
        )
        assert near_row["is_near_region_gt5_le10a"]
        assert not near_row["is_direct_contact_le_5a"]
        assert not near_row["is_c_terminal_thumb_context"]
