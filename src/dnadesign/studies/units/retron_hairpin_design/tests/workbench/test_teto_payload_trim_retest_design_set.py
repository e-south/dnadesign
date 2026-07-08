"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/workbench/test_teto_payload_trim_retest_design_set.py

Tests for the tetO payload trim retest workbench design set.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import yaml

from dnadesign.studies.units.retron_hairpin_design.catalog.compiler_spec import load_msd_compiler_spec

from ..support.paths import repo_root_from


def test_teto_payload_trim_retest_design_set_is_payload_family_based_and_compiler_ready() -> None:
    root = repo_root_from(__file__)
    study_dir = root / "docs" / "studies" / "retron_hairpin_design"
    design_set_path = study_dir / "workbench" / "design_sets" / "teto_payload_trim_retest_v1.yaml"
    spec_path = study_dir / "compiler" / "inputs" / "teto_payload_trim_retest_v1.spec.yaml"

    design_set = yaml.safe_load(design_set_path.read_text(encoding="utf-8"))
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    resolved = load_msd_compiler_spec(spec_path, study_dir=study_dir)

    assert design_set["contract"] == "retron_msd_design_set_v1"
    assert design_set["design_set_id"] == "teto_payload_trim_retest_v1"
    assert design_set["payload_source_policy"] == "payload_family_catalog_terms_not_variant_specific_semantics"
    assert design_set["source_refs"]["payload_binding_catalog"].endswith(
        "workbench/ontology/payload_binding_sites.yaml"
    )
    assert design_set["parent_payload"]["payload_family_id"] == "tetO_ecoli_working"
    assert design_set["parent_payload"]["parent_payload_id"] == "tetO_ecoli_working_w00_19"
    assert set(design_set["payload_trims"]) == {
        "tetO_ecoli_working_w02_17",
        "tetO_ecoli_working_w03_16",
    }
    assert {trim["exact_sequence_5to3"] for trim in design_set["payload_trims"].values()} == {
        "CCTATCAGTGATAGA",
        "CTATCAGTGATAG",
    }
    assert {design["payload_family_id"] for design in design_set["designs"]} == {"tetO_ecoli_working"}
    assert {design["source_precedent_id"] for design in design_set["designs"]} == {
        "pES-retron-26",
        "pES-retron-180",
    }
    assert {design["scaffold_context"] for design in design_set["designs"]} == {"retron26", "retron180"}
    assert {design["construct_id"] for design in design_set["designs"]} == {
        "pES-teto-r26-w02-17",
        "pES-teto-r26-w03-16",
        "pES-teto-r180-w02-17",
        "pES-teto-r180-w03-16",
    }
    assert spec["allow_non_ligatable_s0"] is True
    assert set(spec["payload_sequences"]) == set(design_set["payload_trims"])
    assert set(spec["cap_sequences"]) == {"C26", "C172"}
    assert len(resolved.catalog.records) == 4
    assert [record.variant_metadata.rt_mode for record in resolved.catalog.records] == ["wt_eco1"] * 4
